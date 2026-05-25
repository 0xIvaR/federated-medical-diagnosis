import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flwr as fl
import torch
from models.full_model import PathologyClassifier
from data.partition_data import get_client_loaders
from federated.client import FlowerClient
from federated.strategies.fed_avg import get_fedavg_strategy

def client_fn(cid: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PathologyClassifier(num_classes=9).to(device)
    client_id = int(cid)
    train_loader, val_loader = get_client_loaders(client_id, num_clients=5, batch_size=32)
    return FlowerClient(model, train_loader, val_loader, device)

def main():
    print("=" * 60)
    print("FEDERATED LEARNING - 5 CLIENTS (IID)")
    print("=" * 60)

    strategy = get_fedavg_strategy()

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=5,
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.2}
    )

    print("\n5-client training complete")

if __name__ == "__main__":
    main()