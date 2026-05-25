import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import json

def plot_fl_vs_centralized():
    with open('experiments/baseline_results.json', 'r') as f:
        baseline = json.load(f)

    centralized_train = [h['accuracy'] for h in baseline['training_history']['train_history']]
    centralized_val = [h['accuracy'] for h in baseline['training_history']['val_history']]

    fl_rounds = list(range(1, 11))
    fl_accuracy = [0.7738, 0.7938, 0.8053, 0.8074, 0.8124, 0.8144, 0.8177, 0.8190, 0.8190, 0.8204]

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, 21), centralized_val, 'b-', linewidth=2, label='Centralized')
    plt.axhline(y=0.8326, color='b', linestyle='--', alpha=0.5, label='Centralized Test (83.26%)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Centralized Training (20 epochs)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(fl_rounds, fl_accuracy, 'r-o', linewidth=2, markersize=6, label='Federated (3 clients)')
    plt.axhline(y=0.8204, color='r', linestyle='--', alpha=0.5, label='FL Final (82.04%)')
    plt.axhline(y=0.8326, color='b', linestyle='--', alpha=0.5, label='Centralized Test (83.26%)')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.title('Federated Learning (10 rounds, IID)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('experiments/convergence_comparison.png', dpi=150, bbox_inches='tight')
    print("Plot saved to experiments/convergence_comparison.png")
    plt.show()

if __name__ == "__main__":
    plot_fl_vs_centralized()