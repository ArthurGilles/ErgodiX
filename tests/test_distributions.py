import jax
import jax.numpy as jnp
import pytest
from ergodix.distributions import (
    Banana, Funnel, IsotropicGaussian, IsotropicGMM, FullCovGMM,
    Rings, Rosenbrock, BayesianLogisticRegression
)

def assert_scalar_and_finite_grad(dist, x):
    """Helper to ensure the distribution evaluates cleanly and has valid gradients."""
    val_and_grad_fn = jax.value_and_grad(dist)
    val, grad = val_and_grad_fn(x)
    
    assert val.shape == (), f"Expected scalar log-prob, got shape {val.shape}"
    assert jnp.isfinite(val), "Log-prob evaluated to NaN or Inf"
    assert grad.shape == x.shape, "Gradient shape mismatch"
    assert jnp.all(jnp.isfinite(grad)), "Gradients contain NaN or Inf"

def test_banana(dummy_2d_x):
    dist = Banana()
    assert_scalar_and_finite_grad(dist, dummy_2d_x)

def test_funnel(dummy_5d_x):
    dist = Funnel(sigma=3.0)
    assert_scalar_and_finite_grad(dist, dummy_5d_x)

def test_isotropic_gaussian(dummy_5d_x):
    dist = IsotropicGaussian(mean=jnp.zeros(5), std=jnp.ones(5))
    assert_scalar_and_finite_grad(dist, dummy_5d_x)

def test_isotropic_gmm(dummy_2d_x):
    weights = jnp.array([0.3, 0.7])
    means = jnp.array([[0.0, 0.0], [2.0, 2.0]])
    variances = jnp.array([1.0, 0.5])
    dist = IsotropicGMM(weights, means, variances)
    assert_scalar_and_finite_grad(dist, dummy_2d_x)

def test_full_cov_gmm(dummy_2d_x):
    weights = jnp.array([0.5, 0.5])
    means = jnp.array([[0.0, 0.0], [1.0, -1.0]])
    covs = jnp.array([
        [[1.0, 0.2], [0.2, 1.0]],
        [[0.5, -0.1], [-0.1, 0.5]]
    ])
    dist = FullCovGMM(weights, means, covs)
    assert_scalar_and_finite_grad(dist, dummy_2d_x)

def test_rings(dummy_2d_x):
    dist = Rings()
    assert_scalar_and_finite_grad(dist, dummy_2d_x)

def test_rosenbrock(dummy_2d_x):
    dist = Rosenbrock()
    assert_scalar_and_finite_grad(dist, dummy_2d_x)

def test_bayesian_logistic_regression():
    # Construct a dummy dataset for a 3-feature problem (plus 1 bias term)
    X = jnp.array([[1.0, 2.0, 3.0], [-1.0, -2.0, 0.0], [0.5, 0.5, 0.5]])
    y = jnp.array([1.0, 0.0, 1.0])
    dist = BayesianLogisticRegression(X=X, y=y)
    
    # theta must be shape (d + 1) = 4
    theta = jnp.array([0.1, -0.2, 0.3, 0.0])
    assert_scalar_and_finite_grad(dist, theta)