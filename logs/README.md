# Execution and Experiment Logs

This directory stores runtime logs, stdout/stderr streams, and execution traces from federated simulation runs and parameter sweeps.

### Standard Output Files:
- `experiment_eps_0.log` / `.err` — Baseline experiment (No DP, $\varepsilon = \infty$)
- `experiment_eps_1.log` / `.err` — High privacy experiment ($\varepsilon = 1.0$)
- `experiment_eps_5.log` / `.err` — Target privacy experiment ($\varepsilon = 5.0$)
- `experiment_eps_10.log` / `.err` — Low privacy experiment ($\varepsilon = 10.0$)

Logs are generated via `scripts/run_all_experiments.ps1` or `scripts/run_all_experiments.sh`.
