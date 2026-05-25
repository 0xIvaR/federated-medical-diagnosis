# Contributing

This repository follows academic open-science practices to support reproducible and transparent collaborative research. We welcome contributions that extend, refine, or validate this privacy-preserving federated diagnostics framework.

---

## Technical Workflow

To propose a research extension or implementation change:

1. **Fork the Repository** and create a specialized feature or branch:
   ```bash
   git checkout -b feat/your-change
   ```
2. **Add Unit Tests** for all new functionality or algorithms in the `tests/` directory.
3. **Update Methodology Documentation** (`docs/methodology.md`) if mathematical formulations, clipping profiles, or noise mechanisms are modified.
4. **Pre-commit Verification**:
   - Run the full unit testing suite:
     ```bash
     pytest tests/ -v
     ```
   - Regenerate the high-fidelity validation plots:
     ```bash
     python scripts/plot_results.py
     ```
5. **Submit a Pull Request (PR)** with:
   - A descriptive title and summary of key mathematical and algorithmic adjustments.
   - An update to the `CHANGELOG.md` highlighting the contributions under `[Unreleased]` or a new version tag.

---

## Reporting Issues & Bugs

If you find a bug or have questions about the experimental findings, please open a GitHub Issue containing:
- **Reproduction Steps**: Exact OS, Python/CUDA version, and CLI commands executed.
- **Observed Behavior vs. Expected Behavior**: Quantitative accuracy scores or logs.
- **Relevant Logs**: Snippets from the `logs/*.log` files.
