import base64
import zlib
import numpy as np
from collections import deque
from sklearn.decomposition import IncrementalPCA


class ServerFIPCA:
    def __init__(self, target_components=50, buffer_capacity=20):
        self.target_components = target_components
        self.buffer_capacity = buffer_capacity
        self.history = deque(maxlen=buffer_capacity)
        self.pca = None
        self.n_components_current = 0
        self._fitted = False
        self._basis_changed = False

    def update(self, global_flat: np.ndarray) -> None:
        self._basis_changed = False
        if len(self.history) > 0 and global_flat.shape[0] != self.history[0].shape[0]:
            self.history.clear()
            self.pca = None
            self._fitted = False
            self.n_components_current = 0
        self.history.append(global_flat.astype(np.float32))
        if len(self.history) < 2:
            return
        k = min(self.target_components, len(self.history) - 1)
        if k < 1:
            return
        if k != self.n_components_current or self.pca is None:
            self.pca = IncrementalPCA(n_components=k)
            history_matrix = np.stack(list(self.history))
            self.pca.fit(history_matrix)
            self.n_components_current = self.pca.n_components_
            self._fitted = True
            self._basis_changed = True
        else:
            try:
                self.pca.partial_fit(global_flat.reshape(1, -1))
                self._fitted = True
                self._basis_changed = True
            except Exception:
                pass

    def is_fitted(self) -> bool:
        return self._fitted

    def basis_changed(self) -> bool:
        return self._basis_changed

    def get_basis_b64(self) -> tuple:
        components = self.pca.components_.astype(np.float32)
        mean_ = self.pca.mean_.astype(np.float32)
        actual_n_components = components.shape[0]
        actual_dim = components.shape[1]
        print(f"[FIPCA-DEBUG] get_basis_b64: n_components_current={self.n_components_current}, "
              f"actual_pca_n_components={actual_n_components}, actual_dim={actual_dim}, "
              f"mean_shape={mean_.shape}, history_len={len(self.history)}")
        if self.n_components_current != actual_n_components:
            print(f"[FIPCA-WARNING] n_components_current ({self.n_components_current}) != "
                  f"actual PCA n_components ({actual_n_components}). Using actual value.")
        raw = components.tobytes() + mean_.tobytes()
        compressed = zlib.compress(raw, level=1)
        b64 = base64.b64encode(compressed).decode("utf-8")
        return b64, str(actual_n_components), str(actual_dim)


def decode_basis(b64_str: str, n_k: int, d: int) -> tuple:
    raw = zlib.decompress(base64.b64decode(b64_str))
    expected_size = n_k * d * 4
    components = np.frombuffer(raw[:expected_size], dtype=np.float32).reshape(n_k, d).copy()
    mean_ = np.frombuffer(raw[expected_size:], dtype=np.float32).copy()
    return components, mean_


def project_delta(delta_flat: np.ndarray, components: np.ndarray, mean: np.ndarray = None) -> np.ndarray:
    if mean is not None:
        delta_flat = delta_flat - mean
    return (delta_flat.astype(np.float32) @ components.T).astype(np.float32)


def reconstruct_delta(scores: np.ndarray, components: np.ndarray, mean: np.ndarray = None) -> np.ndarray:
    rec = (scores.astype(np.float32) @ components).astype(np.float32)
    if mean is not None:
        rec = rec + mean
    return rec


def flatten_weights(ndarrays: list) -> np.ndarray:
    return np.concatenate([a.astype(np.float32).flatten() for a in ndarrays])


def unflatten_weights(flat: np.ndarray, shapes: list) -> list:
    total_expected = sum(int(np.prod(s)) for s in shapes)
    if total_expected != flat.size:
        raise ValueError(f"unflatten mismatch: flat size {flat.size} vs shapes total {total_expected}")
    arrays = []
    offset = 0
    for shape in shapes:
        n = int(np.prod(shape))
        arrays.append(flat[offset:offset + n].reshape(shape))
        offset += n
    return arrays
