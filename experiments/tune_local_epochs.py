import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flwr as fl
import torch
from models.full_model import PathologyClassifier
from data.partition_data import get_client_loaders
from federated.client import FlowerClient
from federated.strategies.fed_avg import get_fedavg_strategy

def client_fn(cid: str, local_epochs=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PathologyClassifier(num_classes=9).to(device)
    client_id = int(cid)
    train_loader, val_loader = get_client_loaders(client_id, num_clients=3, batch_size=32)
    return FlowerClient(model, train_loader, val_loader, device, local_epochs)

def test_local_epochs(epochs):
    print(f"\nTesting local_epochs={epochs}")
    print("=" * 60)

    def wrapped_client_fn(cid: str):
        return client_fn(cid, local_epochs=epochs)

    strategy = get_fedavg_strategy()

    fl.simulation.start_simulation(
        client_fn=wrapped_client_fn,
        num_clients=3,
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
        client_resources={"num_cpus": 2, "num_gpus": 0.3}
    )

if __name__ == "__main__":
    for epochs in [1, 3, 5]:
        test_local_epochs(epochs)