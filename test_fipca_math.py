import numpy as np
from utils.fipca import ServerFIPCA, project_delta, reconstruct_delta, flatten_weights

rng = np.random.default_rng(42)
global_weights = [rng.standard_normal((64, 32)).astype(np.float32)]
local_weights = [global_weights[0] + rng.standard_normal((64, 32)).astype(np.float32) * 0.01]

fipca = ServerFIPCA(target_components=10)

for i in range(25):
    noise = rng.standard_normal(global_weights[0].shape).astype(np.float32) * 0.005 * (i + 1)
    fipca.update(flatten_weights([global_weights[0] + noise]))

local_flat = flatten_weights(local_weights)
components, mean_ = fipca.pca.components_, fipca.pca.mean_

scores = project_delta(local_flat, components, mean_)
reconstructed_flat = reconstruct_delta(scores, components, mean_)

error = np.linalg.norm(local_flat - reconstructed_flat) / np.linalg.norm(local_flat)
print(f"Absolute Weight Reconstruction Error: {error:.6f}")
assert error < 0.2, f"Absolute weight reconstruction failed: {error:.4f}"
print("✅ New absolute-weight pipeline verified successfully.")