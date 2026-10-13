import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from data.dataloader import get_data_loaders

def test_gpu_memory():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on: {device}")
    if torch.cuda.is_available():
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")

    train_loader, _, _, _ = get_data_loaders(batch_size=128)

    print("\nProcessing 5 batches to verify GPU memory...")

    for i, (images, labels) in enumerate(train_loader):
        if i >= 5:
            break
        
        images = images.to(device)
        labels = labels.to(device)
        
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated() / 1024**2
            mem_reserved = torch.cuda.memory_reserved() / 1024**2
            print(f"Batch {i+1}: Shape {images.shape} | Allocated: {mem_allocated:.1f}MB | Reserved: {mem_reserved:.1f}MB")
        else:
            print(f"Batch {i+1}: Shape {images.shape} (CPU execution)")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("\n✓ GPU batch test passed successfully - no memory overflow.")

if __name__ == "__main__":
    test_gpu_memory()
