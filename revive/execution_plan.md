# 🧭 Development Roadmap & Execution Time Blueprint (Plan)

**Project Title:** Resource-Efficient Federated Learning for Privacy-Preserving Medical Diagnostics  
**Author:** Soham Kundu  
**Hardware Baseline:** NVIDIA GeForce RTX 2050 (4GB VRAM) / 11th Gen Intel Core i5-11260H / 16GB RAM  
**Target Accuracy:** $\ge 82.35\% – 83.26\%$ (within $0.89\%$ of centralized ceiling under $\varepsilon=5.0$)  
**Target Completion:** 100% (Finalizing Milestones M7 & M8)  
**Creation Date:** October 13, 2026  

---

## ⏱️ Executive Runtime & Timing Guide

To ensure you never get trapped in an unexpectedly long training run, all remaining execution tasks are partitioned into strict **Time Windows**. You can develop the code immediately (0 seconds GPU time) and run each experiment when you have a matching block of free time.

| Phase / Task | Scope / Goal | Estimated Duration | Device Used | When to Run |
| :--- | :--- | :---: | :---: | :--- |
| **0. Test Suite Sanity** | Run full 13-test pytest suite | **~5 seconds** | CPU | Anytime (instant) |
| **1. Development & Wiring** | Implement `run_docker_server.py`, 5-client compose, energy plotter | **0 seconds** (No training) | N/A | Immediate |
| **2. Rapid Smoke Test** | 1-round federated run ($T=1$, 3 clients) to verify pipeline health | **~40 seconds** | RTX 2050 GPU | Whenever you have 1 minute |
| **3. Target FL Run ($\varepsilon=5.0$)** | 10 rounds federated non-IID + FIPCA + DP + CodeCarbon energy logging | **~10 – 12 minutes** | RTX 2050 GPU | 15-minute free window |
| **4. Centralized vs FL Energy Plot** | Plot `energy_comparison.png` comparing CO2 and kWh | **~3 seconds** | CPU | Immediately after Run #3 |
| **5. 5-Node Docker Simulation** | Spin up server + 5 real hospital containers via `docker-compose up` | **~12 – 15 minutes** | CPU / Container | 20-minute free window |
| **6. Full 4-Budget Sweep** | Run all 4 privacy budgets ($\varepsilon \in \{0, 1, 5, 10\}$) via `run_all_experiments.ps1` | **~45 – 50 minutes** | RTX 2050 GPU | Overnight / lunch break |

---

## 🎯 Hyperparameter Blueprint for Target Accuracy ($\ge 82.35\%$)

To prevent the late-round accuracy drop identified in `experiments/m5_results/insights.md` and achieve consistent $\ge 82.35\%$ test accuracy under non-IID Dirichlet skew ($\alpha=0.3$):

```yaml
# Optimal Configuration Blueprint (experiments/config.yaml)
ALPHA: 0.3                   # Heavy statistical non-IID Dirichlet hospital skew
NUM_CLIENTS: 3               # 3 hospitals (simulation) or 5 (docker scale)
NUM_ROUNDS: 10               # 10 communication rounds (optimal convergence plateau)
DP_EPSILON: 5.0              # Target Local Differential Privacy budget per round
DP_CLIP_NORM: 1.0            # L2 clipping sensitivity threshold
TARGET_COMPONENTS: 20        # FIPCA subspace rank (preserves 99.6% compression)
FIPCA_WARMUP_ROUNDS: 6       # Stable PCA subspace initialization rounds
FIPCA_MIN_COMPONENTS: 8      # Minimum PCA rank required before client compression
FIPCA_MAX_RECON_ERROR: 0.15  # Compression threshold (prevents excessive fallback)
DAA_DISTANCE_STRENGTH: 0.35  # Blending factor between uniform and distance weighting
DAA_MIN_WEIGHT: 0.20         # Lower client weight bound (prevents hospital starvation)
DAA_MAX_WEIGHT: 0.45         # Upper client weight bound (prevents dominant client drift)
SERVER_LR_COMPRESSED: 0.50   # Damped server learning rate for compressed deltas
LOCAL_EPOCHS: 1              # 1 local epoch per round (prevents head overfitting)
BATCH_SIZE: 32               # Fast GPU memory footprint (<80 MB VRAM)
CLIENT_LR: 0.0005            # Head learning rate
FEDPROX_MU: 0.01             # Proximal regularization penalty (stabilizes non-IID drift)
```

---

## 🛠️ Step-by-Step Implementation & Execution Plan

### Step 1: Implement Standalone Network Server for Docker (Milestone M7)
- **File to Create:** [`experiments/run_docker_server.py`](file:///c:/Users/soham/coding/Projects_2.0/DAAD/Project_1/experiments/run_docker_server.py)
- **Objective:** Enable Flower to bind a gRPC socket to port `0.0.0.0:8080` so independent Docker containers can connect over the container bridge network.
- **Implementation Detail:**
  ```python
  import flwr as fl
  from federated.strategies.distance_aware_agg import DistanceAwareAggregation
  from federated.strategies.fed_avg import weighted_average

  strategy = DistanceAwareAggregation(
      fraction_fit=1.0,
      fraction_evaluate=1.0,
      min_fit_clients=5,
      min_evaluate_clients=5,
      min_available_clients=5,
      evaluate_metrics_aggregation_fn=weighted_average,
      target_components=20,
      dp_epsilon=5.0,
      dp_clip_norm=1.0,
      compression_warmup_rounds=6,
      min_pca_components=8,
      max_recon_error=0.15,
  )

  fl.server.start_server(
      server_address="0.0.0.0:8080",
      config=fl.server.ServerConfig(num_rounds=10),
      strategy=strategy
  )
  ```
- **Execution Time:** 0 minutes (Development only).

---

### Step 2: Scale Docker Compose to 5-Node Hospital Network (Milestone M7)
- **File to Update:** [`docker-compose.yml`](file:///c:/Users/soham/coding/Projects_2.0/DAAD/Project_1/docker-compose.yml)
- **Objective:** Update the server command to use `run_docker_server.py` and expand the client definition from 3 to 5 real hospital containers (`client1` through `client5`), satisfying Milestone M7's "5-node Docker simulation" requirement.
- **Execution Time:** 0 minutes (Development only).

---

### Step 3: Embed Automated Energy Tracking in Federated Experiments (Milestone M7)
- **Files to Update:**
  1. [`experiments/run_federated_noniid.py`](file:///c:/Users/soham/coding/Projects_2.0/DAAD/Project_1/experiments/run_federated_noniid.py):
     - Wrap `fl.simulation.start_simulation` with `EnergyTracker(project_name="federated_m6b_target", output_dir="results")`.
     - Automatically appends the federated duration, power, and CO2 footprint into [`results/emissions.csv`](file:///c:/Users/soham/coding/Projects_2.0/DAAD/Project_1/results/emissions.csv) alongside the existing `centralized_baseline` row.
  2. [`scripts/plot_results.py`](file:///c:/Users/soham/coding/Projects_2.0/DAAD/Project_1/scripts/plot_results.py):
     - Add `plot_energy_comparison()` to parse both rows of `results/emissions.csv` and generate `results/energy_comparison.png`.
- **Execution Time:** 0 minutes (Development only).

---

### Step 4: Verification & Execution Commands (Hold For Your Free Time)

When you are ready to run, choose the command that fits your available time block:

#### Option A: Quick 1-Minute Health Check (Time: ~40s)
Test that everything compiles and runs for a single round:
```powershell
.venv\Scripts\activate
$env:NUM_ROUNDS="1"
python experiments/run_federated_noniid.py
```

#### Option B: Target Privacy & Energy Benchmark (Time: ~10 – 12 mins)
Run the full 10-round target experiment ($\varepsilon=5.0$) and record federated energy into `emissions.csv`:
```powershell
.venv\Scripts\activate
$env:NUM_ROUNDS="10"
$env:DP_EPSILON="5.0"
python experiments/run_federated_noniid.py

# Immediately generate all plots including energy comparison:
python scripts/plot_results.py
```
- **Expected Outcome:** Final test accuracy: **$82.35\%$**; energy consumption showing **$\ge 80\%$ reduction** vs centralized baseline.

#### Option C: 5-Node Docker Simulation (Time: ~12 – 15 mins)
Run the complete multi-container hospital network in Docker:
```powershell
docker-compose up --build --abort-on-container-exit
```

#### Option D: Full Overnight / Free-Time Experiment Sweep (Time: ~45 – 50 mins)
Run all 4 privacy levels ($\varepsilon \in \{0, 1, 5, 10\}$) sequentially:
```powershell
.\scripts\run_all_experiments.ps1
```

---

## 📌 Development vs. Execution Status Checklist

- [ ] **Task 1 (Dev, 0 mins):** Create `experiments/run_docker_server.py`.
- [ ] **Task 2 (Dev, 0 mins):** Update `docker-compose.yml` to 5 hospital clients with standalone server.
- [ ] **Task 3 (Dev, 0 mins):** Wrap federated runner with `EnergyTracker` and add `plot_energy_comparison()`.
- [ ] **Task 4 (Dev, 0 mins):** Verify test suite (`pytest tests/ -v`).
- [ ] **Task 5 (Run, 40s):** Execute 1-round smoke test.
- [ ] **Task 6 (Run, 10-12 mins):** Execute target experiment ($\varepsilon=5.0$) + energy logging.
- [ ] **Task 7 (Run, 3s):** Generate final plots (`scripts/plot_results.py`).
- [ ] **Task 8 (Run, 12-15 mins):** Execute 5-node Docker simulation test.
