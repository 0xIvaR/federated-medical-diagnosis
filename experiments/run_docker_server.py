import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flwr as fl
from federated.strategies.distance_aware_agg import DistanceAwareAggregation
from federated.strategies.fed_avg import weighted_average

# Extract configuration from environment or defaults
NUM_CLIENTS = int(os.environ.get("NUM_CLIENTS", 5))
NUM_ROUNDS = int(os.environ.get("NUM_ROUNDS", 10))
EPSILON = float(os.environ.get("DP_EPSILON", 5.0))
CLIP_NORM = float(os.environ.get("DP_CLIP_NORM", 1.0))
SERVER_PORT = os.environ.get("SERVER_PORT", "0.0.0.0:8080")
TARGET_COMPONENTS = int(os.environ.get("TARGET_COMPONENTS", 20))
FIPCA_WARMUP_ROUNDS = int(os.environ.get("FIPCA_WARMUP_ROUNDS", 6))
FIPCA_MIN_COMPONENTS = int(os.environ.get("FIPCA_MIN_COMPONENTS", 8))
FIPCA_MAX_RECON_ERROR = float(os.environ.get("FIPCA_MAX_RECON_ERROR", 0.15))
DAA_DISTANCE_STRENGTH = float(os.environ.get("DAA_DISTANCE_STRENGTH", 0.35))
DAA_MIN_WEIGHT = float(os.environ.get("DAA_MIN_WEIGHT", 0.10))
DAA_MAX_WEIGHT = float(os.environ.get("DAA_MAX_WEIGHT", 0.35))
SERVER_LR_COMPRESSED = float(os.environ.get("SERVER_LR_COMPRESSED", 0.50))


def main():
    print("=" * 60)
    print("STARTING STANDALONE FLOWER SERVER FOR DOCKER SIMULATION")
    print("=" * 60)
    print(f"  Listening on: {SERVER_PORT}")
    print(f"  Target Clients: {NUM_CLIENTS}")
    print(f"  Rounds: {NUM_ROUNDS}")
    print(f"  Strategy: DistanceAwareAggregation + FIPCA (K={TARGET_COMPONENTS}) + DP")
    print(f"  DP Epsilon: {EPSILON} | Clip Norm: {CLIP_NORM}")
    print(f"  FIPCA Warmup: {FIPCA_WARMUP_ROUNDS} | Min Components: {FIPCA_MIN_COMPONENTS}")
    print(f"  Reconstruction Error Threshold: {FIPCA_MAX_RECON_ERROR}")
    print(f"  Client Weight Bounds: [{DAA_MIN_WEIGHT}, {DAA_MAX_WEIGHT}]")
    print("=" * 60)

    strategy = DistanceAwareAggregation(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        evaluate_metrics_aggregation_fn=weighted_average,
        target_components=TARGET_COMPONENTS,
        dp_epsilon=EPSILON,
        dp_clip_norm=CLIP_NORM,
        compression_warmup_rounds=FIPCA_WARMUP_ROUNDS,
        min_pca_components=FIPCA_MIN_COMPONENTS,
        max_recon_error=FIPCA_MAX_RECON_ERROR,
        distance_strength=DAA_DISTANCE_STRENGTH,
        min_weight=DAA_MIN_WEIGHT,
        max_weight=DAA_MAX_WEIGHT,
        server_lr_compressed=SERVER_LR_COMPRESSED,
    )

    fl.server.start_server(
        server_address=SERVER_PORT,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
