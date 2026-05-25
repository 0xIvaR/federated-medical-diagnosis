import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import flwr as fl
import torch
from models.full_model import PathologyClassifier
from data.partition_generator import get_noniid_client_loaders
from federated.client import FlowerClient
from federated.strategies.distance_aware_agg import DistanceAwareAggregation
from federated.strategies.fed_avg import weighted_average

ALPHA = 0.5
NUM_CLIENTS = 3
NUM_ROUNDS = 10
SIMILARITY_METRIC = "wasserstein"

def client_fn(cid: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PathologyClassifier(num_classes=9).to(device)
    client_id = int(cid)

    train_loader, val_loader = get_noniid_client_loaders(
        client_id=client_id,
        num_clients=NUM_CLIENTS,
        alpha=ALPHA,
        batch_size=32
    )

    return FlowerClient(model, train_loader, val_loader, device)

def main():
    print("=" * 60)
    print("ABLATION: WASSERSTEIN DISTANCE-AWARE AGGREGATION")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Clients: {NUM_CLIENTS}")
    print(f"  Rounds: {NUM_ROUNDS}")
    print(f"  Alpha (skew): {ALPHA}")
    print(f"  Strategy: DistanceAwareAggregation")
    print(f"  Similarity Metric: {SIMILARITY_METRIC}")
    print("\n" + "=" * 60)

    strategy = DistanceAwareAggregation(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        evaluate_metrics_aggregation_fn=weighted_average,
        similarity_metric=SIMILARITY_METRIC,
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 2, "num_gpus": 0.3}
    )

    print("\n" + "=" * 60)
    print("WASSERSTEIN ABLATION COMPLETE")
    print("Record final round accuracy from terminal output")
    print("=" * 60)

if __name__ == "__main__":
    main()
