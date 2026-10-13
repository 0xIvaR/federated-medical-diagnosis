import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from medmnist import PathMNIST

def main():
    os.makedirs('reports', exist_ok=True)
    train_dataset = PathMNIST(split='train', download=False, root='./data/raw')

    class_names = [
        'Adipose', 'Background', 'Debris', 'Lymphocytes',
        'Mucus', 'Smooth Muscle', 'Normal Colon', 
        'Cancer-associated Stroma', 'Colorectal Adenocarcinoma'
    ]

    fig, axes = plt.subplots(3, 6, figsize=(15, 8))
    fig.suptitle('PathMNIST Sample Images (Colon Pathology)', fontsize=16, fontweight='bold')

    indices = np.random.choice(len(train_dataset), 18, replace=False)

    for idx, ax in enumerate(axes.flat):
        img, label = train_dataset[indices[idx]]
        img_np = np.array(img)
        
        ax.imshow(img_np)
        ax.set_title(f"{class_names[int(np.asarray(label).item())]}", fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    output_path = 'reports/sample_images.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Sample visualization saved to: {output_path}")

if __name__ == "__main__":
    main()
