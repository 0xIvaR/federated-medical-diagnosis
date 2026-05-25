import torch
from dataloader import get_data_loaders

def test_gpu_memory():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on: {device}")

    train_loader, _, _, _ = get_data_loaders(batch_size=128)

    print("\nProcessing 5 batches to test GPU memory...")

    for i, (images, labels) in enumerate(train_loader):
        if i >= 5:
            break
        
        images = images.to(device)
        labels = labels.to(device)
        
        mem_allocated = torch.cuda.memory_allocated() / 1024**2
        mem_reserved = torch.cuda.memory_reserved() / 1024**2
        
        print(f"Batch {i+1}: Shape {images.shape} | "
              f"Allocated: {mem_allocated:.1f}MB | "
              f"Reserved: {mem_reserved:.1f}MB")

    torch.cuda.empty_cache()
    print("\n✓ GPU memory test passed - no crashes")

if __name__ == "__main__":
    test_gpu_memory()