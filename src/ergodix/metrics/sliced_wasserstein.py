from typing import Tuple
from functools import partial
import jax
import jax.numpy as jnp

@partial(jax.jit, static_argnames=["num_projections", "p"])
def sliced_wasserstein_mc(
    key: jax.Array,
    x: jax.Array,
    y: jax.Array,
    num_projections: int = 4096,
    p: int = 2,
) -> Tuple[jax.Array, jax.Array]:
    """
    Computes the p-th order Sliced Wasserstein Distance (SWD) between two point clouds.
    
    This function is pure, JIT-compatible, and fully vectorized. It computes the 
    Monte Carlo approximation of the SWD by projecting the empirical distributions 
    onto random 1D lines.

    Args:
        key: A JAX PRNG key used to generate random projection directions.
        x: A JAX array of shape (N, d) representing the first empirical distribution.
        y: A JAX array of shape (N, d) representing the second empirical distribution.
           Must have the exact same shape as `x`.
        num_projections: The number of random 1D projections to compute. 
                         Passed as a static argument to allow JIT compilation.
        p: The order of the Wasserstein distance (e.g., 1 for Manhattan, 2 for Euclidean).
           Passed as a static argument.

    Returns:
        A tuple containing:
            - swd: A scalar JAX array representing the Sliced Wasserstein Distance.
            - variance: A scalar JAX array representing the variance of the distance 
                        across the Monte Carlo projections.
    """
    d = x.shape[1]
    
    # Generate random projections and normalize
    dirs = jax.random.normal(key, shape=(num_projections, d)) # Shape: (num_projections, d)
    dirs = dirs / jnp.linalg.norm(dirs, axis=1, keepdims=True) # Shape: (num_projections, d)
    
    # Project the samples 
    x_proj = jnp.matmul(x, dirs.T) # Shape: (N, num_projections)
    y_proj = jnp.matmul(y, dirs.T) # Shape: (N, num_projections)
    
    # Sort projections
    x_proj_sorted = jnp.sort(x_proj, axis=0) # Shape: (N, num_projections)
    y_proj_sorted = jnp.sort(y_proj, axis=0) # Shape: (N, num_projections)
    
    # 1D Wasserstein-p distance to the p for each projection
    wp_p_1d = jnp.mean(jnp.abs(x_proj_sorted - y_proj_sorted) ** p, axis=0) # Shape: (num_projections,)
    
    # Calculate the EV and variance over projections
    expected_wp_p = jnp.mean(wp_p_1d)
    variance_wp_p = jnp.var(wp_p_1d)
    
    return expected_wp_p ** (1.0 / p), variance_wp_p