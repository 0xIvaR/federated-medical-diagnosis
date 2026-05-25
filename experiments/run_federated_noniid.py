import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random
import numpy as np
import torch
import yaml

# --- Precedence & Configuration Setup ---
"""
Configuration precedence (highest → lowest):
1. Environment variables (DP_EPSILON, DP_CLIP_NORM, etc.)
2. experiments/config.yaml values
3. Hardcoded defaults in script
"""

DEFAULTS = {
    "ALPHA": 0.3,
    "NUM_CLIENTS": 3,
    "NUM_ROUNDS": 10,
    "DP_EPSILON": 5.0,
    "DP_CLIP_NORM": 1.0,
    "RANDOM_SEED": 42,
    "TARGET_COMPONENTS": 500
}

# 1. Load config.yaml if it exists
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
yaml_config = {}
if os.path.exists(config_path):
    try:
        with open(config_path, "r") as f:
            yaml_config = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Warning: Failed to load config.yaml: {e}")

# 2. Extract values based on priority (Env Var -> Config file -> Default)
def get_config_val(key, default):
    # Try environment variable first
    env_val = os.environ.get(key)
    if env_val is not None:
        if isinstance(default, float):
            return float(env_val)
        if isinstance(default, int):
            return int(env_val)
        return env_val
    # Try yaml config second
    if key in yaml_config:
        return yaml_config[key]
    # Fallback to defaults
    return default

ALPHA = get_config_val("ALPHA", DEFAULTS["ALPHA"])
NUM_CLIENTS = get_config_val("NUM_CLIENTS", DEFAULTS["NUM_CLIENTS"])
NUM_ROUNDS = get_config_val("NUM_ROUNDS", DEFAULTS["NUM_ROUNDS"])
EPSILON = get_config_val("DP_EPSILON", DEFAULTS["DP_EPSILON"])
CLIP_NORM = get_config_val("DP_CLIP_NORM", DEFAULTS["DP_CLIP_NORM"])
SEED = get_config_val("RANDOM_SEED", DEFAULTS["RANDOM_SEED"])
TARGET_COMPONENTS = get_config_val("TARGET_COMPONENTS", DEFAULTS["TARGET_COMPONENTS"])

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

import flwr as fl
from models.full_model import PathologyClassifier
from data.partition_generator import get_noniid_client_loaders
from federated.client import FlowerClient
from federated.strategies.distance_aware_agg import DistanceAwareAggregation
from federated.strategies.fed_avg import weighted_average


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
    print("="*60)
    print("M6B: FEDERATED LEARNING + FIPCA + DIFFERENTIAL PRIVACY")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Clients: {NUM_CLIENTS}")
    print(f"  Rounds: {NUM_ROUNDS}")
    print(f"  Alpha (skew): {ALPHA}")
    print(f"  Strategy: DistanceAware + FIPCA (K={TARGET_COMPONENTS}) + DP")
    print(f"  Epsilon: {EPSILON}")
    print(f"  Clip Norm: {CLIP_NORM}")
    print(f"  M6A Baseline Accuracy: 83.24%")
    print(f"  Accuracy Floor (M6A - 1%): >= 82.24%")
    print("\n" + "="*60)
    
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
    )
    
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 2, "num_gpus": 0.3}
    )
    
    print("\n" + "="*60)
    print("M6B: DP + FIPCA COMPLETE")
    print(f"  Epsilon: {EPSILON} | Clip: {CLIP_NORM}")
    print(f"  Privacy budget (naive, T={NUM_ROUNDS}): eps_total = {EPSILON * NUM_ROUNDS}")
    print(f"  Accuracy floor: >= 82.24% (M6A - 1%)")
    print("  Record final round accuracy from logs above")
    print("="*60)

if __name__ == "__main__":
    main()
