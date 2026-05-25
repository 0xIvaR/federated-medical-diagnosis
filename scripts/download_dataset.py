import os
import medmnist
from medmnist import INFO
import numpy as np

def main():
    print("="*60)
    print("DOWNLOADING PathMNIST DATASET")
    print("="*60)

    data_flag = 'pathmnist'
    info = INFO[data_flag]

    print(f"\nDataset: {info['python_class']}")
    print(f"Task: {info['task']}")
    print(f"Number of Classes: {len(info['label'])}")
    print(f"Image Size: 28x28x3 (RGB)")

    # Ensure local raw directory exists
    os.makedirs("./data/raw", exist_ok=True)

    DataClass = getattr(medmnist, info['python_class'])

    # Download if not present
    train_dataset = DataClass(split='train', download=True, root='./data/raw')
    val_dataset = DataClass(split='val', download=True, root='./data/raw')
    test_dataset = DataClass(split='test', download=True, root='./data/raw')

    print(f"\n✓ Training samples: {len(train_dataset):,}")
    print(f"✓ Validation samples: {len(val_dataset):,}")
    print(f"✓ Test samples: {len(test_dataset):,}")

    print("\n" + "="*60)
    print("Class Distribution Analysis")
    print("="*60)

    train_labels = train_dataset.labels.flatten()
    unique, counts = np.unique(train_labels, return_counts=True)

    for cls, count in zip(unique, counts):
        print(f"Class {cls}: {count:6,} samples ({count/len(train_labels)*100:5.2f}%)")

    print("\n✓ Dataset downloaded and prepared successfully in: ./data/raw/")
    print("="*60)

if __name__ == "__main__":
    main()
