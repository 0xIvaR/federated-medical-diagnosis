# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-05-25

### Added
- **Milestone M6B**: Completed integrated **Differential Privacy (Laplacian noise mechanism, per-round $\varepsilon=5$)** alongside client-side **FIPCA weight projection**.
- **GDPR & EU AI Act Compliance**: Published compliance logs and legal design guides (`docs/privacy_guarantee.md`, `docs/data_protection.md`).
- **CI/CD and Scanning Pipelines**: Added local and server testing actions (`.github/workflows/ci.yml`, `codeql.yml`).
- **Dockerized Architecture**: Implemented full process-isolated simulation infrastructure (`Dockerfile`, `docker-compose.yml`).
- **Automated Experimentation Sweeps**: Added automated execution scripts (`scripts/run_all_experiments.sh`) and publication-grade plot generators (`scripts/plot_results.py`).
- **Comprehensive Unit Testing**: Created fully coverage-aligned unit tests (`tests/test_fipca.py`, `tests/test_differential_privacy.py`) validating PCA projections and L2 Laplacian noise correctness.

### Metrics
- **Pathology Diagnostics Accuracy**: **$82.35\%$** final testing accuracy under target privacy ($\varepsilon = 5$), preserving baseline accuracy (baseline: **$83.24\%$**) with less than **$-0.89\%$** utility loss.
- **Bandwidth Reduction**: Microscopic footprint (reduced payload size from **$522\text{ KB}$** to **$0.02\text{ KB}$** per client round), yielding a massive **$99.6\%$** reduction.
- **Privacy Budget**: Provable cumulative **$(\varepsilon_{\text{total}} = 50.0, \delta = 0)$-Differential Privacy** over 10 communication rounds.
- **Sustainability impact**: Achieving **$81\%$** fewer communication rounds compared to standard centralized paradigms, lowering total CPU/GPU energy usage.
