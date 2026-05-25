import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.utils.data import Subset
import numpy as np
from data.dataloader import get_data_loaders

def create_iid_partitions(num_clients=3):
    _, _, _, train_dataset = get_data_loaders(batch_size=32)
    
    num_samples = len(train_dataset)
    indices = np.random.permutation(num_samples)
    
    split_size = num_samples // num_clients
    
    client_partitions = []
    
    for i in range(num_clients):
        start_idx = i * split_size
        end_idx = (i + 1) * split_size if i < num_clients - 1 else num_samples
        
        client_indices = indices[start_idx:end_idx]
        client_partitions.append(client_indices)
    
    return client_partitions, train_dataset

def get_client_loaders(client_id, num_clients=3, batch_size=32):
    partitions, train_dataset = create_iid_partitions(num_clients)
    
    client_train_dataset = Subset(train_dataset, partitions[client_id])
    
    train_loader = torch.utils.data.DataLoader(
        client_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    _, val_loader, _, _ = get_data_loaders(batch_size=batch_size)
    
    return train_loader, val_loader

if __name__ == "__main__":
    num_clients = 3
    partitions, full_dataset = create_iid_partitions(num_clients)
    
    print("="*60)
    print("IID PARTITION VERIFICATION")
    print("="*60)
    
    total_samples = len(full_dataset)
    print(f"\nTotal training samples: {total_samples:,}")
    print(f"Number of clients: {num_clients}")
    print(f"\nPartition sizes:")
    
    partition_sum = 0
    for i, partition in enumerate(partitions):
        size = len(partition)
        partition_sum += size
        percentage = (size / total_samples) * 100
        print(f"  Client {i}: {size:6,} samples ({percentage:5.2f}%)")
    
    print(f"\nTotal partitioned: {partition_sum:,}")
    print(f"Match original: {partition_sum == total_samples}")
    
    print("\n" + "="*60)
    print("Testing DataLoader creation...")
    print("="*60)
    
    for client_id in range(num_clients):
        train_loader, val_loader = get_client_loaders(client_id, num_clients, batch_size=64)
        print(f"\nClient {client_id}:")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
    
    print("\n" + "="*60)
    print("VERIFICATION COMPLETE")
    print("="*60)