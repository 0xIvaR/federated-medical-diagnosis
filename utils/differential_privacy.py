import numpy as np


def clip_l2(scores: np.ndarray, clip_norm: float) -> np.ndarray:
    norm = np.linalg.norm(scores)
    if norm > clip_norm and norm > 0:
        return (scores * (clip_norm / norm)).astype(np.float32)
    return scores.astype(np.float32)


def laplacian_noise(scores: np.ndarray, epsilon: float, clip_norm: float,
                    rng: np.random.Generator = None) -> np.ndarray:
    if epsilon <= 0:
        return scores.astype(np.float32)
    clipped = clip_l2(scores, clip_norm)
    scale = clip_norm / epsilon
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.laplace(0.0, scale, size=clipped.shape).astype(np.float32)
    result = clipped + noise
    if np.isnan(result).any():
        noise = rng.laplace(0.0, scale, size=clipped.shape).astype(np.float32)
        result = clipped + noise
    return result.astype(np.float32)
