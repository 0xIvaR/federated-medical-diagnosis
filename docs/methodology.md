# Methodology and Mathematical Formulations

This document provides the formal mathematical formulations for the privacy and communication-efficiency stack implemented in this repository.

---

## 1. Federated Incremental PCA (FIPCA)

To address the extreme bandwidth bottlenecks in deep federated learning, we employ a Federated Incremental Principal Component Analysis (FIPCA) projection. Instead of transmitting raw pathology classifier parameters (which represents a massive payload), clients compute their local update vectors, project them onto a dynamically updated server-side PCA basis, and transmit only the compressed low-dimensional projection scores.

Let $\mathbf{w}_t$ be the global server weights at round $t$, and let $\mathbf{w}_{i, t}$ be the local weights of client $i$ after local training in round $t$. The local model parameter update (delta) for client $i$ is defined as:

$$\Delta \mathbf{w}_{i, t} = \mathbf{w}_{i, t} - \mathbf{w}_t \in \mathbb{R}^d$$

where $d$ is the number of trainable weights in the network head.

### Server-Side Basis Maintenance

The central server maintains a low-rank basis matrix $\mathbf{W}_t \in \mathbb{R}^{K \times d}$ representing the $K$ principal directions of global updates, along with a historical update mean vector $\boldsymbol{\mu}_t \in \mathbb{R}^d$. This basis is iteratively trained on the global updates using Incremental PCA (IPCA) to accommodate incoming updates online without storing past weights:

$$\mathbf{W}_{t+1},\ \boldsymbol{\mu}_{t+1} = \mathrm{IPCA}(\mathbf{W}_t,\ \boldsymbol{\mu}_t,\ \Delta \mathbf{w}_t)$$

### Client-Side Dynamic Projection

For a target dimensionality $K \ll d$, the client projects the parameter delta onto the server's orthogonal basis $\mathbf{W}_t$:

$$\mathbf{z}_{i, t} = (\Delta \mathbf{w}_{i, t} - \boldsymbol{\mu}_t)\, \mathbf{W}_t^T \in \mathbb{R}^K$$

The projected vector $\mathbf{z}_{i, t}$ is a dense vector of scores representing the projected coordinates.

### Reconstruction

Upon receiving the low-dimensional projected scores $\mathbf{z}_{i, t}$ from the clients, the server reconstructs the high-dimensional parameter update vector via back-projection:

$$\Delta \hat{\mathbf{w}}_{i, t} = \mathbf{z}_{i, t}\, \mathbf{W}_t + \boldsymbol{\mu}_t \in \mathbb{R}^d$$

By setting $K = 500$ components relative to a parameter flat dimensionality of $d = 115{,}209$, we achieve a **99.6% bandwidth reduction** per round (reducing client payload from $522\ \mathrm{KB}$ down to $0.02\ \mathrm{KB}$).

---

## 2. Differential Privacy Stack

To formally defend against reconstruction, model inversion, and membership inference attacks on the hospital datasets, we implement node-level **Differential Privacy (DP)** by perturbing the FIPCA projection scores before transmission.

A randomized algorithm $\mathcal{M}$ provides $(\varepsilon, \delta)$-differential privacy if, for any two neighboring datasets $D$ and $D'$ differing by at most one individual patient record, and for any set of query outputs $\mathcal{S} \subseteq \mathrm{Range}(\mathcal{M})$:

$$\mathbb{P}[\mathcal{M}(D) \in \mathcal{S}] \leq e^{\varepsilon} \times \mathbb{P}[\mathcal{M}(D') \in \mathcal{S}] + \delta$$

Our stack uses a pure $\delta = 0$ Laplacian mechanism applied to the low-dimensional projection scores.

### L2 Sensitivity Bounding (Clipping)

To bound the maximum impact any single patient record can have on the transmitted client scores, we apply strict L2 norm clipping to the FIPCA scores. The clipped score vector $\bar{\mathbf{z}}_{i, t}$ is obtained by:

$$\bar{\mathbf{z}}_{i, t} = \mathrm{clip}_{L2}(\mathbf{z}_{i, t},\ C) = \mathbf{z}_{i, t} \times \min\!\left(1,\ \frac{C}{\|\mathbf{z}_{i, t}\|_2}\right)$$

where $C$ is the clipping norm threshold (configured as `DP_CLIP_NORM = 1.0`). This guarantees that the L2 sensitivity of the query function $\Delta f$ is strictly bounded by:

$$\Delta f = \max_{D,\, D'} \left\|\mathrm{clip}_{L2}(D, C) - \mathrm{clip}_{L2}(D', C)\right\|_2 \leq C$$

### Laplacian Noise Injection

The Laplacian mechanism injects zero-mean noise drawn from a Laplacian distribution scaled by the L2 sensitivity $C$ and the targeted per-round privacy parameter $\varepsilon$:

$$\mathbf{z}_{i, t}^{\mathrm{priv}} = \bar{\mathbf{z}}_{i, t} + \boldsymbol{\eta}$$

where each element of the noise vector $\boldsymbol{\eta}$ is sampled independently:

$$\eta_j \sim \mathrm{Lap}\!\left(0,\ \frac{C}{\varepsilon}\right)$$

The probability density function of the Laplacian noise $\mathrm{Lap}(0, b)$ is:

$$f(x) = \frac{1}{2b} \exp\!\left(-\frac{|x|}{b}\right)$$

For our target configuration, we specify a local, per-round privacy budget of $\varepsilon = 5.0$.

---

## 3. Privacy Budget Composition

To compute the total cumulative privacy guarantee over a multi-round federated training run, we apply the standard, robust **Naive Composition Theorem**.

### Naive Composition (Strict Upper Bound)

If an algorithm consists of $T$ sequential mechanisms $\mathcal{M}_1, \dots, \mathcal{M}_T$, where each mechanism $\mathcal{M}_t$ satisfies $(\varepsilon_t, \delta_t)$-differential privacy, then the entire composite mechanism satisfies $(\varepsilon_{\mathrm{total}}, \delta_{\mathrm{total}})$-differential privacy, where:

$$\varepsilon_{\mathrm{total}} = \sum_{t=1}^T \varepsilon_t \qquad \text{and} \qquad \delta_{\mathrm{total}} = \sum_{t=1}^T \delta_t$$

Under our simulation parameters, for $T = 10$ communication rounds with a per-round privacy budget of $\varepsilon = 5.0$, the formal total privacy guarantee is:

$$\varepsilon_{\mathrm{total}} = 10 \times 5.0 = 50.0 \qquad \text{with} \qquad \delta_{\mathrm{total}} = 0$$

This represents a mathematically rigorous, verifiable upper bound on the total cumulative privacy budget.
