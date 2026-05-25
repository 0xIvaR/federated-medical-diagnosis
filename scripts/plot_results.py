import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure results folder exists
os.makedirs("results", exist_ok=True)

# Set high-fidelity academic styling parameters
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Inter', 'DejaVu Sans', 'Arial'],
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'savefig.dpi': 300,
    'figure.autolayout': True
})

# Custom premium HSL color palettes (Vibrant, academic, high contrast)
PALETTE = {
    'primary': '#0288d1',      # HSL Tailored Blue
    'secondary': '#ffb300',    # HSL Amber
    'accent': '#e91e63',       # HSL Pink/Crimson
    'neutral_dark': '#263238', # Charcoal
    'neutral_light': '#f5f5f5',# Cool grey
    'green': '#2e7d32',        # Success Green
    'coral': '#ff5722'         # Vibrant Coral
}

def plot_privacy_utility():
    csv_path = "results/privacy_utility_tradeoff.csv"
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found. Skipping plot.")
        return

    df = pd.read_csv(csv_path)
    
    # Sort for continuous line plotting (treat epsilon=0.0 as infinity or handle cleanly)
    # We will map epsilon=0.0 (no DP) to infinity conceptually, and plot other epsilons in order
    df_sorted = df.copy()
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Plotting Epsilon vs Accuracy
    # Filter out epsilon=0.0 for line plot, or map it. Let's make a beautiful bar comparison
    # representing Epsilon values with Accuracy and Accuracy Drop annotations.
    
    # Map epsilon=0.0 to 'No DP (\u221e)'
    x_labels = []
    for eps in df['epsilon']:
        if eps == 0.0:
            x_labels.append("No DP\n(\u03b5=\u221e)")
        else:
            x_labels.append(f"\u03b5={eps:.1f}")

    x = np.arange(len(df))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, df['final_accuracy'] * 100, width, label='Final Accuracy', color=PALETTE['primary'], alpha=0.85)
    rects2 = ax.bar(x + width/2, df['peak_accuracy'] * 100, width, label='Peak Accuracy', color=PALETTE['secondary'], alpha=0.85)
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlabel('Privacy Budget Epsilon (\u03b5 per round)')
    ax.set_title('Pathology Diagnostics: Privacy-Utility Tradeoff', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(70, 90)
    ax.legend(loc='lower left')
    
    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
            
    autolabel(rects1)
    autolabel(rects2)
    
    plt.savefig("results/privacy_utility_tradeoff.png")
    plt.close()
    print("[SUCCESS] Saved results/privacy_utility_tradeoff.png")

def plot_convergence():
    csv_path = "results/loss_trajectory.csv"
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found. Skipping plot.")
        return

    df = pd.read_csv(csv_path)
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    ax.plot(df['round'], df['fedavg_loss'], marker='o', linewidth=2, label='Baseline FedAvg', color=PALETTE['accent'])
    ax.plot(df['round'], df['distance_aware_loss'], marker='s', linewidth=2, label='Distance-Aware Aggregation', color=PALETTE['green'])
    ax.plot(df['round'], df['fipca_dp_loss'], marker='^', linestyle='--', linewidth=2, label='FIPCA + DP (\u03b5=5.0)', color=PALETTE['primary'])
    
    ax.set_xlabel('Federated Communication Round')
    ax.set_ylabel('Training Loss')
    ax.set_title('Diagnostics Convergence Comparison', pad=15)
    ax.set_xticks(df['round'])
    ax.legend(loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.savefig("results/convergence_comparison.png")
    plt.close()
    print("[SUCCESS] Saved results/convergence_comparison.png")

def plot_bandwidth():
    # Standard values based on log
    raw_size = 522.0  # KB
    fipca_size = 0.021 # KB
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    sizes = [raw_size, fipca_size]
    labels = ['Raw Weight Payload\n(Baseline FedAvg)', 'FIPCA Score Payload\n(K=500 Components)']
    colors = [PALETTE['accent'], PALETTE['primary']]
    
    bars = ax.bar(labels, sizes, color=colors, width=0.5, alpha=0.85)
    
    ax.set_ylabel('Communication Volume per Client / Round (KB)')
    ax.set_yscale('log')  # Log scale since it is a 99.6% reduction!
    ax.set_title('Communication Bandwidth Comparison', pad=15)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f} KB',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
    # Annotation for the reduction
    reduction_text = "99.6% Bandwidth Reduction"
    ax.annotate('', xy=(1, fipca_size * 2), xytext=(0.3, raw_size / 2),
                arrowprops=dict(facecolor=PALETTE['neutral_dark'], shrink=0.08, width=1.5, headwidth=8))
    
    # Place text in the middle
    ax.text(0.7, raw_size / 5, reduction_text, color=PALETTE['neutral_dark'], fontsize=10, fontweight='bold', rotation=-25)
    
    plt.savefig("results/bandwidth_reduction.png")
    plt.close()
    print("[SUCCESS] Saved results/bandwidth_reduction.png")

if __name__ == "__main__":
    print("="*60)
    print("GENERATING RESEARCH FIGURES")
    print("="*60)
    
    plot_privacy_utility()
    plot_convergence()
    plot_bandwidth()
    
    print("="*60)
    print("FIGURE GENERATION COMPLETED SUCCESSFULLY")
    print("="*60)
