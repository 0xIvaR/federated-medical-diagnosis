import base64
import json
import zlib
import numpy as np
from utils.fipca import project_delta, flatten_weights
from utils.differential_privacy import laplacian_noise
import torch
from flwr.client import NumPyClient
from collections import OrderedDict

class FlowerClient(NumPyClient):
    _cached_basis = None
    _cached_basis_version = -1

    def __init__(self, model, train_loader, val_loader, device, local_epochs=1):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss()
        self.local_epochs = local_epochs
        
        labels = np.concatenate([target.numpy().astype(int) for _, target in self.train_loader]).flatten()
        self.label_dist = (np.bincount(labels, minlength=9) / len(labels)).tolist()
        self.global_weights_flat = None

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.head.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.head.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.head.load_state_dict(state_dict, strict=True)
        self.global_weights_flat = flatten_weights([val.cpu().numpy() for _, val in self.model.head.state_dict().items()])

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        self.model.backbone.eval()
        optimizer = torch.optim.Adam(self.model.head.parameters(), lr=0.001)

        for _ in range(self.local_epochs):
            for images, labels in self.train_loader:
                if images.size(0) == 0 or labels.size(0) == 0:
                    continue
                images = images.to(self.device)
                labels = labels.to(self.device).squeeze(-1).long()
                optimizer.zero_grad()
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss.backward()
                optimizer.step()

        raw_weights = self.get_parameters(config={})
        bytes_before = int(self.global_weights_flat.nbytes) if self.global_weights_flat is not None else int(sum(a.nbytes for a in raw_weights))
        pca_fitted = config.get("pca_fitted", "0") == "1"
        basis_version = int(config.get("basis_version", "-1"))

        if pca_fitted and basis_version > FlowerClient._cached_basis_version and "pca_components" in config:
            n_k = int(config["n_components"])
            d = int(config["original_dim"])
            raw = zlib.decompress(base64.b64decode(config["pca_components"]))
            components = np.frombuffer(raw[:n_k * d * 4], dtype=np.float32).reshape(n_k, d).copy()
            mean_ = np.frombuffer(raw[n_k * d * 4:], dtype=np.float32).copy()
            FlowerClient._cached_basis = (components, mean_)
            FlowerClient._cached_basis_version = basis_version

        if pca_fitted and FlowerClient._cached_basis is not None and self.global_weights_flat is not None:
            components, mean_ = FlowerClient._cached_basis
            local_flat = flatten_weights(raw_weights)
            delta = local_flat - self.global_weights_flat
            scores = project_delta(delta, components, mean_).astype(np.float32)
            dp_epsilon = float(config.get("dp_epsilon", "0"))
            dp_clip = float(config.get("dp_clip_norm", "1.0"))
            if dp_epsilon > 0:
                scores = laplacian_noise(scores, dp_epsilon, dp_clip)
            payload = [scores]
            bytes_after = int(scores.nbytes)
            compressed = "1"
        else:
            dp_epsilon = 0.0
            dp_clip = 1.0
            payload = raw_weights
            bytes_after = bytes_before
            compressed = "0"

        return payload, len(self.train_loader.dataset), {
            "label_dist":   json.dumps(self.label_dist),
            "compressed":   compressed,
            "bytes_before": str(bytes_before),
            "bytes_after":  str(bytes_after),
            "dp_epsilon":   str(dp_epsilon),
            "dp_clip_norm": str(dp_clip),
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                if images.size(0) == 0 or labels.size(0) == 0:
                    continue
                images = images.to(self.device)
                labels = labels.to(self.device).squeeze(-1).long()
                logits = self.model(images)
                loss += self.criterion(logits, labels).item()
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        avg_loss = loss / len(self.val_loader)
        return avg_loss, total, {"accuracy": accuracy}