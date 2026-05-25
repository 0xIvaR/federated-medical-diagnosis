import sys
import torch
import torchvision
import flwr
import sklearn
import numpy as np
import pandas as pd
import matplotlib
import seaborn
import cv2
import PIL
import medmnist
import codecarbon
import yaml
import tqdm

packages = {
    "Python": sys.version.split()[0],
    "PyTorch": torch.__version__,
    "torchvision": torchvision.__version__,
    "Flower": flwr.__version__,
    "scikit-learn": sklearn.__version__,
    "NumPy": np.__version__,
    "Pandas": pd.__version__,
    "Matplotlib": matplotlib.__version__,
    "Seaborn": seaborn.__version__,
    "OpenCV": cv2.__version__,
    "Pillow": PIL.__version__,
    "MedMNIST": medmnist.__version__,
    "CodeCarbon": codecarbon.__version__,
}

print("\n" + "="*60)
print("FINAL DEPENDENCY CHECK")
print("="*60)
for pkg, version in packages.items():
    print(f"✓ {pkg:20s} {version}")

print(f"\n✓ CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")

print("="*60)
print("ALL DEPENDENCIES INSTALLED SUCCESSFULLY")
print("="*60)