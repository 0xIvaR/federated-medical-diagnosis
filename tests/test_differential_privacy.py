import numpy as np
import pytest
from utils.differential_privacy import clip_l2, laplacian_noise

def test_clip_l2_below_threshold():
    # Arrange
    scores = np.array([0.5, 0.2, -0.4], dtype=np.float32)
    clip_norm = 1.0
    
    # Act
    clipped = clip_l2(scores, clip_norm)
    
    # Assert: Should be completely unchanged
    np.testing.assert_array_equal(scores, clipped)
    assert clipped.dtype == np.float32

def test_clip_l2_above_threshold():
    # Arrange
    scores = np.array([3.0, 4.0], dtype=np.float32)  # Norm = 5.0
    clip_norm = 2.0
    
    # Act
    clipped = clip_l2(scores, clip_norm)
    
    # Assert: Should be scaled to norm of exactly 2.0
    expected = np.array([1.2, 1.6], dtype=np.float32)  # [3 * 2/5, 4 * 2/5]
    np.testing.assert_allclose(clipped, expected, rtol=1e-6)
    assert np.linalg.norm(clipped) == pytest.approx(2.0)

def test_clip_l2_zero_vector():
    # Arrange
    scores = np.zeros(5, dtype=np.float32)
    clip_norm = 1.0
    
    # Act
    clipped = clip_l2(scores, clip_norm)
    
    # Assert: Should be a clean zero vector, no division by zero!
    np.testing.assert_array_equal(scores, clipped)

def test_laplacian_noise_no_epsilon():
    # Arrange
    scores = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    
    # Act: Epsilon <= 0 maps directly to original values
    result_zero = laplacian_noise(scores, epsilon=0.0, clip_norm=1.0)
    result_negative = laplacian_noise(scores, epsilon=-1.0, clip_norm=1.0)
    
    # Assert
    np.testing.assert_array_equal(scores, result_zero)
    np.testing.assert_array_equal(scores, result_negative)

def test_laplacian_noise_injection():
    # Arrange
    scores = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    epsilon = 5.0
    clip_norm = 1.0
    rng = np.random.default_rng(42)
    
    # Act
    result = laplacian_noise(scores, epsilon=epsilon, clip_norm=clip_norm, rng=rng)
    
    # Assert
    assert result.shape == scores.shape
    assert result.dtype == np.float32
    # Result must not be identical to original due to noise addition
    assert not np.array_equal(result, scores)

def test_laplacian_noise_statistical_scale():
    # Arrange: Use a large sample size to verify the empirical variance matches analytical expectation.
    # Analytical scale b = clip_norm / epsilon = 2.0 / 0.5 = 4.0
    # Variance of Lap(0, b) = 2 * b^2 = 2 * 16 = 32
    size = 20000
    scores = np.zeros(size, dtype=np.float32)
    epsilon = 0.5
    clip_norm = 2.0
    rng = np.random.default_rng(42)
    
    # Act
    result = laplacian_noise(scores, epsilon=epsilon, clip_norm=clip_norm, rng=rng)
    
    # Assert: Zero-mean input clipped is zero. Output matches Laplacian variance.
    empirical_variance = np.var(result)
    expected_variance = 2 * ((clip_norm / epsilon) ** 2)
    
    assert empirical_variance == pytest.approx(expected_variance, rel=0.05)
