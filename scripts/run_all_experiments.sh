#!/usr/bin/env bash
# Automated Experiment Runner Sweep
# Replicates experimental outcomes for epsilon = 0 (No DP), 1.0, 5.0 (Target), and 10.0

set -euo pipefail

echo "============================================================"
echo "LAUNCHING FEDERATED MEDICAL DIAGNOSTICS EXPERIMENT SWEEP"
echo "============================================================"

# Ensure results and logs directories exist
mkdir -p results logs

# Run Epsilon = 0.0 (Baseline, No DP, Infinity privacy budget)
echo -e "\n[SWEEP 1/4] Running Baseline Experiment (No DP, Epsilon = Infinity)..."
export DP_EPSILON="0.0"
export DP_CLIP_NORM="1.0"
python experiments/run_federated_noniid.py > logs/experiment_eps_0.log 2>&1

# Run Epsilon = 10.0 (Low Privacy, High Utility)
echo -e "\n[SWEEP 2/4] Running Low Privacy Experiment (Epsilon = 10.0)..."
export DP_EPSILON="10.0"
export DP_CLIP_NORM="1.0"
python experiments/run_federated_noniid.py > logs/experiment_eps_10.log 2>&1

# Run Epsilon = 5.0 (Target Privacy and Utility)
echo -e "\n[SWEEP 3/4] Running Target Privacy Experiment (Epsilon = 5.0)..."
export DP_EPSILON="5.0"
export DP_CLIP_NORM="1.0"
python experiments/run_federated_noniid.py > logs/experiment_eps_5.log 2>&1

# Run Epsilon = 1.0 (High Privacy, Low Utility)
echo -e "\n[SWEEP 4/4] Running High Privacy Experiment (Epsilon = 1.0)..."
export DP_EPSILON="1.0"
export DP_CLIP_NORM="1.0"
python experiments/run_federated_noniid.py > logs/experiment_eps_1.log 2>&1

echo -e "\n============================================================"
echo "EXPERIMENT SWEEP COMPLETED SUCCESSFULLY"
echo "Check detailed logs in logs/ folder."
echo "Generating publication-grade charts..."
echo "============================================================"
python scripts/plot_results.py
