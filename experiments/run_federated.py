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
    train_loader, val_loader = get_client_loaders(client_id, num_clients=3, batch_size=32)
    
    return FlowerClient(model, train_loader, val_loader, device)

def main():
    num_rounds = 10
    num_clients = 3
    
    print("="*60)
    print("FEDERATED LEARNING - IID PARTITIONING")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Clients: {num_clients}")
    print(f"  Rounds: {num_rounds}")
    print(f"  Strategy: FedAvg")
    print(f"  Data: IID (equal random splits)")
    print("\n" + "="*60)
    
    strategy = get_fedavg_strategy()
    
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 2, "num_gpus": 0.3}
    )
    
    print("\n" + "="*60)
    print("FEDERATED TRAINING COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()