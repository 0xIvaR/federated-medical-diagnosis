import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def estimate_param_bytes(num_params):
    return num_params * 4

def log_communication_stats(num_clients, num_rounds, params_per_client):
    bytes_per_param = 4
    upload_per_client = params_per_client * bytes_per_param
    download_per_client = upload_per_client
    total_per_round = (upload_per_client + download_per_client) * num_clients
    total_simulation = total_per_round * num_rounds

    print("\n" + "=" * 60)
    print("COMMUNICATION STATISTICS")
    print("=" * 60)
    print(f"Trainable parameters per client: {params_per_client:,}")
    print(f"Upload per client per round: {upload_per_client / 1024:.2f} KB")
    print(f"Download per client per round: {download_per_client / 1024:.2f} KB")
    print(f"\nTotal per round (all clients): {total_per_round / 1024:.2f} KB")
    print(f"\nTotal for {num_rounds} rounds: {total_simulation / (1024**2):.2f} MB")
    print("=" * 60)

if __name__ == "__main__":
    log_communication_stats(num_clients=3, num_rounds=10, params_per_client=133641)