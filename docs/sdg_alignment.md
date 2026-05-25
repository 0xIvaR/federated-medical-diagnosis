# UN Sustainable Development Goals (SDGs) Alignment

This research project is designed to directly support the United Nations Sustainable Development Goals (SDGs), focusing on **SDG 3 (Good Health and Well-Being)** and **SDG 10 (Reduced Inequalities)**. This alignment highlights the project's real-world developmental and social impact, making it ideal for evaluation by DAAD scholarship committees and international research panels.

---

## 🎯 SDG 3: Good Health and Well-Being

### Target 3.d: Improve Early Warning, Risk Reduction, and Management of National and Global Health Risks
Traditional clinical machine learning requires medical institutions to upload sensitive, raw patient images (such as prostate MRI scans) to a centralized cloud server. In practice, strict medical confidentiality laws (such as GDPR in Europe and HIPAA in the US) restrict such data transfers, leaving hospitals in isolated siloes.

Our project addresses this bottleneck by implementing a **privacy-preserving federated diagnostic architecture**:
1. **Local Model Training**: Pathology models are trained directly on-site at hospital nodes. Raw medical images are never transmitted or exposed to external entities.
2. **Differential Privacy Security**: Local model updates are heavily protected via Laplacian noise injection, mathematically guaranteeing that patient records cannot be extracted from shared weights.
3. **Clinical Application**: Enabling collaborative, secure training across clinical boundaries yields highly generalizable pathology classifiers (achieving **$82.35\%$** final accuracy), directly improving diagnostic accuracy and early detection of prostate and colorectal diseases globally.

---

## ⚖️ SDG 10: Reduced Inequalities

### Target 10.2: Promote Universal Social, Economic, and Political Inclusion
AI-driven medical diagnostic systems are typically built using datasets from wealthy, urban medical hubs. As a result, these models struggle to generalize to smaller, rural, or under-resourced community hospitals due to severe statistical variations (Non-IID data distribution).

This project democratizes AI access by addressing the fundamental computational and network barriers in remote clinics:
1. **99.6% Bandwidth Reduction**: Our FIPCA (Federated Incremental Principal Component Analysis) compression reduces the per-round client transmission payload from **$522\text{ KB}$** down to a microscopic **$0.02\text{ KB}$**.
2. **Support for Low-Bandwidth Infrastructure**: Under-resourced clinics with slow, unstable internet connections can participate fully in state-of-the-art collaborative AI training.
3. **Decoupled Architecture**: By freezing the heavy convolutional backbones and updating only lightweight heads, local training demands minimal GPU resource capabilities (e.g., lightweight CUDA cards or standard multi-core CPUs).
4. **Dirichlet Non-IID Splitting**: We explicitly model severe hospital data imbalances (using Dirichlet $\alpha=0.3$ splits), proving that our distance-aware aggregation strategy stabilizes convergence even when hospital class distributions are highly skewed and unequal.
