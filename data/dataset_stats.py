from medmnist import PathMNIST
import numpy as np
import torch

train_dataset = PathMNIST(split='train', download=False, root='./data/raw')
val_dataset = PathMNIST(split='val', download=False, root='./data/raw')
test_dataset = PathMNIST(split='test', download=False, root='./data/raw')

def compute_stats(dataset, split_name):
    labels = dataset.labels.flatten()
    unique, counts = np.unique(labels, return_counts=True)
    
    print(f"\n{split_name} Set:")
    print(f"  Total samples: {len(dataset):,}")
    print(f"  Classes: {len(unique)}")
    print(f"  Min samples per class: {counts.min():,}")
    print(f"  Max samples per class: {counts.max():,}")
    print(f"  Imbalance ratio: {counts.max()/counts.min():.2f}x")
    
    return unique, counts

print("="*60)
print("DATASET STATISTICS")
print("="*60)

train_unique, train_counts = compute_stats(train_dataset, "Training")
val_unique, val_counts = compute_stats(val_dataset, "Validation")
test_unique, test_counts = compute_stats(test_dataset, "Test")

print("\n" + "="*60)
print("Per-Class Breakdown (Training Set)")
print("="*60)

class_names = [
    'Adipose', 'Background', 'Debris', 'Lymphocytes',
    'Mucus', 'Smooth Muscle', 'Normal Colon', 
    'Cancer Stroma', 'Adenocarcinoma'
]

for cls, count in zip(train_unique, train_counts):
    print(f"  {class_names[cls]:25s} {count:6,} ({count/len(train_dataset)*100:5.2f}%)")

print("\n" + "="*60)