import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from data.partition_generator import get_noniid_client_loaders
from data.dataloader import get_data_loaders

def plot_iid_vs_noniid(num_clients=3, alpha=0.5):
    # Load full training dataset to compute IID distribution
    _, _, _, train_dataset = get_data_loaders(batch_size=32)
    labels = np.array([train_dataset[i][1].item() for i in range(len(train_dataset))])

    # Simple round‑robin IID partitioning
    iid_partitions = {i: [] for i in range(num_clients)}
    for i in range(len(labels)):
        iid_partitions[i % num_clients].append(i)

    iid_distributions = {}
    for client_id, indices in iid_partitions.items():
        client_labels = labels[indices]
        distribution = np.zeros(9)
        for cls in range(9):
            distribution[cls] = np.sum(client_labels == cls) / len(client_labels)
        iid_distributions[client_id] = distribution

    # Non‑IID distributions using the per‑client data loaders
    noniid_distributions = {}
    for client_id in range(num_clients):
        train_loader, _ = get_noniid_client_loaders(
            client_id=client_id,
            num_clients=num_clients,
            alpha=alpha,
            batch_size=32
        )
        client_labels = []
        for _, target in train_loader:
            client_labels.extend(target.numpy().tolist())
        client_labels = np.array(client_labels)
        distribution = np.zeros(9)
        for cls in range(9):
            distribution[cls] = np.sum(client_labels == cls) / len(client_labels) if len(client_labels) > 0 else 0.0
        noniid_distributions[client_id] = distribution

    class_names = [
        'Adipose', 'Background', 'Debris', 'Lymphocytes',
        'Mucus', 'Smooth Muscle', 'Normal Colon',
        'Cancer Stroma', 'Adenocarcinoma'
    ]

    fig, axes = plt.subplots(2, num_clients, figsize=(16, 8))
    fig.suptitle(f'IID vs Non-IID Data Distribution (α={alpha})', fontsize=14, fontweight='bold')
    colors = plt.cm.Set3(np.linspace(0, 1, 9))

    for client_id in range(num_clients):
        ax_iid = axes[0][client_id]
        ax_iid.bar(range(9), iid_distributions[client_id], color=colors)
        ax_iid.set_title(f'IID - Client {client_id}')
        ax_iid.set_xticks(range(9))
        ax_iid.set_xticklabels(class_names, rotation=45, ha='right', fontsize=7)
        ax_iid.set_ylabel('Proportion')
        ax_iid.set_ylim(0, 1.0)
        ax_iid.grid(axis='y', alpha=0.3)

        ax_noniid = axes[1][client_id]
        ax_noniid.bar(range(9), noniid_distributions[client_id], color=colors)
        ax_noniid.set_title(f'Non-IID (α={alpha}) - Client {client_id}')
        ax_noniid.set_xticks(range(9))
        ax_noniid.set_xticklabels(class_names, rotation=45, ha='right', fontsize=7)
        ax_noniid.set_ylabel('Proportion')
        ax_noniid.set_ylim(0, 1.0)
        ax_noniid.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_dir = Path('experiments')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'iid_vs_noniid_distribution.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Distribution plot saved to {output_path}')
    plt.show()

if __name__ == "__main__":
    plot_iid_vs_noniid(num_clients=3, alpha=0.5)