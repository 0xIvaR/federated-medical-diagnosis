import numpy as np
import pytest
from utils.fipca import (
    ServerFIPCA,
    project_delta,
    reconstruct_delta,
    flatten_weights,
    unflatten_weights
)

def test_flatten_unflatten_weights():
    # Arrange: Create typical model parameter layer matrices
    weights = [
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        np.array([5.0, 6.0], dtype=np.float32),
        np.array([[[7.0]]], dtype=np.float32)
    ]
    shapes = [w.shape for w in weights]
    
    # Act: Flatten and then reconstruct
    flat = flatten_weights(weights)
    reconstructed = unflatten_weights(flat, shapes)
    
    # Assert
    assert len(reconstructed) == len(weights)
    for w, r in zip(weights, reconstructed):
        assert w.shape == r.shape
        np.testing.assert_array_equal(w, r)

def test_flatten_unflatten_mismatch():
    flat = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    shapes = [(2, 2)]  # total 4 elements, but flat has 3
    
    with pytest.raises(ValueError, match="unflatten mismatch"):
        unflatten_weights(flat, shapes)

def test_server_fipca_lifecycle():
    # Arrange: Set up ServerFIPCA and synthetic global updates
    target_components = 5
    d = 100
    fipca = ServerFIPCA(target_components=target_components, buffer_capacity=10)
    
    assert not fipca.is_fitted()
    assert not fipca.basis_changed()
    
    # Act & Assert: Feeding updates to history buffer
    # PCA requires at least 2 updates before it fits.
    update1 = np.random.randn(d).astype(np.float32)
    fipca.update(update1)
    assert not fipca.is_fitted()
    
    update2 = np.random.randn(d).astype(np.float32)
    fipca.update(update2)
    assert fipca.is_fitted()
    assert fipca.basis_changed()
    
    # Get basis and verify dimensions
    b64, n_components, original_dim = fipca.get_basis_b64()
    assert n_components == "1"  # k = min(target, len(history)-1) = min(5, 1) = 1
    assert original_dim == str(d)
    
    # Feed more updates to reach target components
    for _ in range(5):
        fipca.update(np.random.randn(d).astype(np.float32))
        
    assert fipca.n_components_current == min(target_components, len(fipca.history) - 1)
    
def test_project_reconstruct():
    # Arrange: Setup synthetic basis and updates
    k = 5
    d = 50
    components = np.random.randn(k, d).astype(np.float32)
    # orthonormalize basis components
    q, r = np.linalg.qr(components.T)
    components = q.T
    
    delta = np.random.randn(d).astype(np.float32)
    
    # Act: Project to lower-dim and reconstruct back
    scores = project_delta(delta, components)
    assert scores.shape == (k,)
    
    reconstructed = reconstruct_delta(scores, components)
    assert reconstructed.shape == (d,)
    
    # Projection onto orthonormal basis loses residual information, 
    # but projecting and reconstructing again in score space is exact.
    scores_re = project_delta(reconstructed, components)
    np.testing.assert_allclose(scores, scores_re, rtol=1e-5, atol=1e-5)
