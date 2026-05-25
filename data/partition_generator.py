import os
import pickle
import numpy as np
from medmnist import PathMNIST

def generate_partitions(num_hospitals=3, alpha=0.5, save_dir="data/partitions"):
    os.makedirs(save_dir, exist_ok=True)

    train_ds = PathMNIST(split="train", download=False, root="data/raw")
    images = np.array(train_ds.imgs)
    labels = np.array([int(l) for l in train_ds.labels]).flatten()

    num_classes = 9
    hospital_indices = [[] for _ in range(num_hospitals)]

    for c in range(num_classes):
        class_idx = np.where(labels == c)[0]
        np.random.shuffle(class_idx)
        proportions = np.random.dirichlet(np.repeat(alpha, num_hospitals))
        proportions = np.maximum(proportions, 1e-6)
        proportions /= proportions.sum()
        split_points = np.cumsum(proportions * len(class_idx)).astype(int)
        split_points[-1] = len(class_idx)

        start = 0
        for h in range(num_hospitals):
            end = split_points[h]
            hospital_indices[h].extend(class_idx[start:end])
            start = end

    for h in range(num_hospitals):
        idx = np.array(hospital_indices[h])
        partition = {"images": images[idx], "labels": labels[idx]}
        path = os.path.join(save_dir, f"hospital_{h+1}.pkl")
        with open(path, "wb") as f:
            pickle.dump(partition, f)

        counts = np.bincount(labels[idx], minlength=num_classes)
        print(f"Hospital {h+1}: {len(idx)} samples | Distribution: {counts}")

def create_noniid_partitions(num_clients=3, alpha=0.5, seed=42):

    np.random.seed(seed)
    
    train_ds = PathMNIST(split="train", download=False, root="data/raw")
    labels = np.array([int(l) for l in train_ds.labels]).flatten()
    num_classes = 9

    
    client_indices = [[] for _ in range(num_clients)]

    
    for c in range(num_classes):
        class_idx = np.where(labels == c)[0]
        np.random.shuffle(class_idx)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = np.maximum(proportions, 1e-6)
        proportions /= proportions.sum()
        split_points = np.cumsum(proportions * len(class_idx)).astype(int)
        split_points[-1] = len(class_idx)
        start = 0
        for h in range(num_clients):
            end = split_points[h]
            client_indices[h].extend(class_idx[start:end])
            start = end
    
    client_indices = [np.array(idx) for idx in client_indices]
    return client_indices, None, None

def get_client_class_distribution(client_indices, labels, num_classes=9):

    distribution = {}
    for client_id, indices in enumerate(client_indices):
        client_labels = labels[indices]
        dist = np.zeros(num_classes)
        for cls in range(num_classes):
            dist[cls] = np.sum(client_labels == cls) / len(client_labels) if len(client_labels) > 0 else 0.0
        distribution[client_id] = dist
    return distribution

from torch.utils.data import Subset, DataLoader
from data.dataloader import get_data_loaders

def get_noniid_client_loaders(client_id, num_clients=3, alpha=0.5, batch_size=32, seed=42):
    client_indices, _, _ = create_noniid_partitions(num_clients, alpha, seed)
    _, _, _, train_dataset = get_data_loaders(batch_size=batch_size)
    client_subset = Subset(train_dataset, client_indices[client_id])
    train_loader = DataLoader(client_subset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    _, val_loader, _, _ = get_data_loaders(batch_size=batch_size)
    return train_loader, val_loader

if __name__ == "__main__":
    np.random.seed(42)
    generate_partitions()
