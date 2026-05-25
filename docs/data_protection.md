# Data Protection and Dataset Handling

To preserve patient confidentiality, this project does not store or distribute raw medical imaging datasets. Instead, it provides automated utility scripts to fetch, parse, and partition benchmarks in a secure local environment.

---

## 1. Fed-KITS2019 / PathMNIST Dataset

For local validation and simulation, we use the standardized medical imaging benchmark **PathMNIST** (from the **MedMNIST v2** suite), which represents a high-density clinical histology dataset for medical diagnostics.

- **Source**: [MedMNIST v2 Repository & Citation](https://medmnist.com/)
- **Image Format**: $28 \times 28 \times 3$ (Histological slices)
- **Task**: 9-class pathology classification
- **License**: CC BY 4.0 (Permissive for scientific, open-source, and academic reuse)
- **Data Minimization Guarantee**: Raw clinical data never leaves the client's local node. Eavesdroppers or servers only interact with highly compressed, mathematically perturbed FIPCA scores.

---

## 2. Secure Local Dataset Setup

To replicate this simulation locally without committing any raw data to Git, run the secure downloader:

```bash
# Downloads raw dataset files to data/raw/ (automatically ignored by git)
python scripts/download_dataset.py
```

This script will verify the local folder structures and download the MedMNIST archive (`pathmnist.npz`) directly.

---

## 3. Dirichlet Non-IID Partitioning

In real-world medical networks, hospitals observe highly imbalanced patient distributions (e.g., specialized oncology clinics vs. general clinics). We simulate this severe statistical skew using a Dirichlet distribution over the training dataset.

The partitioning generator (`data/partition_generator.py`) assigns sample indices to clients according to:

$$\mathbf{p}_c \sim \text{Dirichlet}(\alpha \cdot \mathbf{1}_N)$$

where:
- $\alpha = 0.3$ is the skew coefficient (lower $\alpha$ creates higher statistical imbalance).
- $N = 3$ represents the number of simulated hospital nodes.
- $\mathbf{p}_c$ is the vector of proportions of class $c$ allocated to each hospital node.

### Local Node Data Sovereignty
Once partitioning is generated, each hospital client load only maps to a specific local PyTorch subset (`Subset(train_dataset, client_indices[client_id])`). The other clients and the central server have absolutely zero visibility into these localized dataset assignments, ensuring complete clinical data sovereignty.
