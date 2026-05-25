import medmnist
from medmnist import PathMNIST
import matplotlib.pyplot as plt
import numpy as np

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
    ax.set_title(f"{class_names[int(label)]}", fontsize=9)
    ax.axis('off')

plt.tight_layout()
plt.savefig('data/sample_images.png', dpi=150, bbox_inches='tight')
print("✓ Sample visualization saved to: data/sample_images.png")
plt.show()