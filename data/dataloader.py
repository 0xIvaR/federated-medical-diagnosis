import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from medmnist import PathMNIST
import numpy as np

def get_data_loaders(batch_size=32, num_workers=0):
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = PathMNIST(
        split='train',
        transform=transform,
        download=False,
        root='./data/raw'
    )
    
    val_dataset = PathMNIST(
        split='val',
        transform=transform,
        download=False,
        root='./data/raw'
    )
    
    test_dataset = PathMNIST(
        split='test',
        transform=transform,
        download=False,
        root='./data/raw'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_dataset

if __name__ == "__main__":
    print("="*60)
    print("TESTING DATALOADER")
    print("="*60)
    
    train_loader, val_loader, test_loader, _ = get_data_loaders(batch_size=64)
    
    print(f"\n✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    print(f"✓ Test batches: {len(test_loader)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✓ Device: {device}")
    
    images, labels = next(iter(train_loader))
    images = images.to(device)
    labels = labels.to(device)
    
    print(f"\n✓ Batch shape: {images.shape}")
    print(f"✓ Labels shape: {labels.shape}")
    print(f"✓ Data on GPU: {images.is_cuda}")
    print(f"✓ Memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
    
    print("\n" + "="*60)
    print("DATALOADER WORKING CORRECTLY")
    print("="*60)