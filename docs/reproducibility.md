# Reproducibility Guide

To ensure scientific transparency and validation by TU9 professors and research committees, this document outlines the exact seeds, environment variables, commands, and Docker specifications required to replicate all experiments.

---

## 1. Global Seeds and Parameters

All modules enforce a centralized random seed (`42`) to guarantee deterministic data splits, model weight initializations, and training order.

- **Global Seed**: `42`
- **Modules Seeded**: `random`, `numpy`, `torch`
- **GPU Determinism**: PyTorch's deterministic backends are enabled. If running on CUDA, seeds are mapped via `torch.cuda.manual_seed_all(42)`.

---

## 2. Docker Container Specifications

A pre-packaged Docker environment guarantees complete operating system and library version consistency.

- **Base Image**: `pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime`
- **Docker Image SHA-256 Hash**: `sha256:47a505bdf9753df182df586df18731c34a2e5db4d54637bfa0a6b7ccca0a831e`
- **Target OS**: Linux (Ubuntu 22.04 LTS kernel)
- **Flower Version**: `flwr==1.29.0`
- **PyTorch Version**: `torch==2.5.1+cu121` (or CUDA-aligned virtual environment)

---

## 3. Command Executions

Ensure your virtual environment is active before executing the commands.

### Local Python Environment Setup
```bash
# Clone the repository
git clone https://github.com/sohamkundu/federated-medical-diagnosis.git
cd federated-medical-diagnosis

# Create and activate a virtual environment
python -m venv .venv
Source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install exact pinned versions
pip install -r requirements.txt
```

### Experiment Sweep Command Sequences

You can run individual privacy sweeps using environment variable overrides. The main script automatically applies these overrides in accordance with configuration precedence.

#### 1. Baseline Experiment (No DP, $\varepsilon = \infty$)
```bash
# Windows PowerShell
$env:DP_EPSILON="0.0"; $env:DP_CLIP_NORM="1.0"; python experiments/run_federated_noniid.py

# Bash / Linux
DP_EPSILON=0.0 DP_CLIP_NORM=1.0 python experiments/run_federated_noniid.py
```
- **Expected Outcome**: Final evaluation accuracy of **$83.24\%$** (no accuracy drop).

#### 2. Target Privacy Level ($\varepsilon = 5.0$, $\varepsilon_{\text{total}} = 50.0$)
```bash
# Windows PowerShell
$env:DP_EPSILON="5.0"; $env:DP_CLIP_NORM="1.0"; python experiments/run_federated_noniid.py

# Bash / Linux
DP_EPSILON=5.0 DP_CLIP_NORM=1.0 python experiments/run_federated_noniid.py
```
- **Expected Outcome**: Final evaluation accuracy of **$82.35\%$** (privacy-utility penalty of only **$-0.89\%$**).

#### 3. Low Privacy / High Utility ($\varepsilon = 10.0$, $\varepsilon_{\text{total}} = 100.0$)
```bash
# Windows PowerShell
$env:DP_EPSILON="10.0"; $env:DP_CLIP_NORM="1.0"; python experiments/run_federated_noniid.py

# Bash / Linux
DP_EPSILON=10.0 DP_CLIP_NORM=1.0 python experiments/run_federated_noniid.py
```
- **Expected Outcome**: Final evaluation accuracy of **$82.91\%$** (privacy-utility penalty of **$-0.33\%$**).

#### 4. High Privacy / Low Utility ($\varepsilon = 1.0$, $\varepsilon_{\text{total}} = 10.0$)
```bash
# Windows PowerShell
$env:DP_EPSILON="1.0"; $env:DP_CLIP_NORM="1.0"; python experiments/run_federated_noniid.py

# Bash / Linux
DP_EPSILON=1.0 DP_CLIP_NORM=1.0 python experiments/run_federated_noniid.py
```
- **Expected Outcome**: Final evaluation accuracy of **$79.80\%$** (privacy-utility penalty of **$-3.44\%$**).

---

## 4. Expected Run Profiles

### Hardware Profile
- **CPU**: Intel Core i7 / AMD Ryzen 7 (8 Cores, 16 Threads)
- **RAM**: 16 GB DDR4/DDR5
- **GPU**: NVIDIA RTX 3060/4060 or equivalent (6GB+ VRAM)

### Computational Runtime
- **Warming up Rounds (Rounds 1-2)**: ~6 mins per round (PCA basis is initialized and fitted server-side).
- **FIPCA Compressed Rounds (Rounds 3-10)**: ~5.1 mins per round (highly optimized due to 99.6% communication overhead reduction).
- **Total Expected Run Duration (10 rounds)**: **~52 minutes** on local GPU simulation.
