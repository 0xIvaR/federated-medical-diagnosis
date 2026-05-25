# PowerShell Automated Experiment Runner Sweep
# Replicates experimental outcomes for epsilon = 0 (No DP), 1.0, 5.0 (Target), and 10.0

Write-Output "============================================================"
Write-Output "LAUNCHING FEDERATED MEDICAL DIAGNOSTICS EXPERIMENT SWEEP (PS)"
Write-Output "============================================================"

# Ensure directories exist
New-Item -ItemType Directory -Force -Path "results", "logs" | Out-Null

# 1. Epsilon = 0.0
Write-Output "`n[SWEEP 1/4] Running Baseline Experiment (No DP, Epsilon = Infinity)..."
$env:DP_EPSILON = "0.0"
$env:DP_CLIP_NORM = "1.0"
Start-Process -FilePath "python" -ArgumentList "experiments/run_federated_noniid.py" -NoNewWindow -Wait -RedirectStandardOutput "logs/experiment_eps_0.log" -RedirectStandardError "logs/experiment_eps_0.err"

# 2. Epsilon = 10.0
Write-Output "`n[SWEEP 2/4] Running Low Privacy Experiment (Epsilon = 10.0)..."
$env:DP_EPSILON = "10.0"
$env:DP_CLIP_NORM = "1.0"
Start-Process -FilePath "python" -ArgumentList "experiments/run_federated_noniid.py" -NoNewWindow -Wait -RedirectStandardOutput "logs/experiment_eps_10.log" -RedirectStandardError "logs/experiment_eps_10.err"

# 3. Epsilon = 5.0
Write-Output "`n[SWEEP 3/4] Running Target Privacy Experiment (Epsilon = 5.0)..."
$env:DP_EPSILON = "5.0"
$env:DP_CLIP_NORM = "1.0"
Start-Process -FilePath "python" -ArgumentList "experiments/run_federated_noniid.py" -NoNewWindow -Wait -RedirectStandardOutput "logs/experiment_eps_5.log" -RedirectStandardError "logs/experiment_eps_5.err"

# 4. Epsilon = 1.0
Write-Output "`n[SWEEP 4/4] Running High Privacy Experiment (Epsilon = 1.0)..."
$env:DP_EPSILON = "1.0"
$env:DP_CLIP_NORM = "1.0"
Start-Process -FilePath "python" -ArgumentList "experiments/run_federated_noniid.py" -NoNewWindow -Wait -RedirectStandardOutput "logs/experiment_eps_1.log" -RedirectStandardError "logs/experiment_eps_1.err"

Write-Output "`n============================================================"
Write-Output "EXPERIMENT SWEEP COMPLETED SUCCESSFULLY"
Write-Output "Generating publication-grade charts..."
Write-Output "============================================================"
& python scripts/plot_results.py
