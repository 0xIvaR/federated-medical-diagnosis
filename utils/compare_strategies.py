import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'

STRATEGIES = [
    {
        "name": "FedAvg-IID",
        "final_acc": 82.04,
        "peak_acc": 82.04,
        "loss_start": 0.81,
        "loss_end": 0.57,
        "runtime_s": 400,
        "per_round_acc": [77.38, 79.38, 80.53, 80.74, 81.24, 81.44, 81.77, 81.90, 81.90, 82.04],
    },
    {
        "name": "FedAvg-NonIID",
        "final_acc": 78.26,
        "peak_acc": 78.83,
        "loss_start": 0.81,
        "loss_end": 0.57,
        "runtime_s": 406,
        "per_round_acc": [69.93, 74.70, 74.47, 77.36, 77.91, 77.60, 78.83, 78.49, 78.76, 78.26],
    },
    {
        "name": "DistanceAware-NonIID",
        "final_acc": 79.69,
        "peak_acc": 85.90,
        "loss_start": 0.74,
        "loss_end": 0.66,
        "runtime_s": 3286,
        "per_round_acc": [77.40, 82.92, 85.68, 85.90, 85.83, 85.89, 84.98, 84.07, 81.07, 79.69],
    },
]

OUTPUT_DIR = Path("experiments/m5_results")

def generate_comparison_table():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "comparison_table.csv"

    headers = ["Strategy", "Final_Acc", "Peak_Acc", "Loss_Start", "Loss_End", "Runtime_s"]

    print("\n" + "=" * 80)
    print(f"{'Strategy':<25} {'Final_Acc':>10} {'Peak_Acc':>10} {'Loss_Start':>12} {'Loss_End':>10} {'Runtime_s':>10}")
    print("-" * 80)

    rows = []
    for s in STRATEGIES:
        row = [
            s["name"],
            f"{s['final_acc']:.2f}%",
            f"{s['peak_acc']:.2f}%",
            f"{s['loss_start']:.2f}",
            f"{s['loss_end']:.2f}",
            str(s["runtime_s"]),
        ]
        rows.append(row)
        print(f"{row[0]:<25} {row[1]:>10} {row[2]:>10} {row[3]:>12} {row[4]:>10} {row[5]:>10}")

    print("=" * 80)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"\nCSV saved to {csv_path}")

def generate_convergence_plot():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = OUTPUT_DIR / "convergence_plot.png"

    rounds = list(range(1, 11))

    m3_colors = {
        "FedAvg-IID": "#6750A4",
        "FedAvg-NonIID": "#B3261E",
        "DistanceAware-NonIID": "#006D40",
    }

    m3_markers = {
        "FedAvg-IID": "o",
        "FedAvg-NonIID": "s",
        "DistanceAware-NonIID": "D",
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#FFFBFE")
    ax.set_facecolor("#FFFBFE")

    for s in STRATEGIES:
        ax.plot(
            rounds,
            s["per_round_acc"],
            color=m3_colors[s["name"]],
            marker=m3_markers[s["name"]],
            linewidth=2.5,
            markersize=7,
            label=f"{s['name']} (Peak: {s['peak_acc']:.1f}%)",
            zorder=3,
        )

    ax.set_xlabel("Federated Round", fontsize=12, fontweight="500", color="#1C1B1F")
    ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="500", color="#1C1B1F")
    ax.set_title(
        "Convergence Comparison: FedAvg vs Distance-Aware Aggregation",
        fontsize=14,
        fontweight="600",
        color="#1C1B1F",
        pad=16,
    )

    ax.set_xticks(rounds)
    ax.set_ylim(65, 90)
    ax.grid(True, alpha=0.15, color="#1C1B1F", linewidth=0.5)
    ax.tick_params(colors="#49454F", labelsize=10)

    for spine in ax.spines.values():
        spine.set_color("#CAC4D0")
        spine.set_linewidth(0.8)

    legend = ax.legend(
        loc="lower right",
        fontsize=10,
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        edgecolor="#CAC4D0",
    )
    legend.get_frame().set_facecolor("#FFFBFE")

    ax.annotate(
        f"Peak: {85.90}%",
        xy=(4, 85.90),
        xytext=(5.5, 88),
        fontsize=9,
        color="#006D40",
        fontweight="600",
        arrowprops=dict(arrowstyle="->", color="#006D40", lw=1.2),
    )

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor="#FFFBFE")
    print(f"Convergence plot saved to {plot_path}")
    plt.close()

if __name__ == "__main__":
    generate_comparison_table()
    generate_convergence_plot()
