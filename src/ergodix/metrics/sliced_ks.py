from functools import partial
import jax
import jax.numpy as jnp


@jax.jit
def _ks_1d(x: jax.Array, y: jax.Array) -> jax.Array:
    """
    Computes the standard Kolmogorov-Smirnov statistic between two 1D arrays.
    """
    n, m = x.shape[0], y.shape[0]
    
    x_sorted = jnp.sort(x)
    y_sorted = jnp.sort(y)
    z = jnp.concatenate([x, y])
    
    # Evaluate empirical CDFs at all combined data points
    cdf_x = jnp.searchsorted(x_sorted, z, side="right") / n
    cdf_y = jnp.searchsorted(y_sorted, z, side="right") / m
    
    return jnp.max(jnp.abs(cdf_x - cdf_y))


# Vectorize the 1D KS computation over the projection axis (columns)
_vmap_ks_1d = jax.vmap(_ks_1d, in_axes=(1, 1))


@partial(jax.jit, static_argnames=["num_projections", "p", "max_sliced"])
def sliced_kolmogorov_smirnov(
    X: jax.Array,
    Y: jax.Array,
    key: jax.Array,
    num_projections: int = 50,
    p: float = 1.0,
    max_sliced: bool = False,
) -> jax.Array:
    """
    Computes the Sliced Kolmogorov-Smirnov distance between two point clouds.

    Args:
        X: Array of shape (N, d) representing the first empirical distribution.
        Y: Array of shape (M, d) representing the second empirical distribution.
        key: PRNGKey for sampling projection directions.
        num_projections: Number of random projections to draw from the unit sphere.
        p: Order of the integration norm (p >= 1).
        max_sliced: If True, computes the Max-Sliced KS distance (supremum over 
            projections) instead of the L^p expectation.

    Returns:
        A scalar JAX array representing the computed SKS distance.
    """
    d = X.shape[1]
    
    # Sample uniform directions
    theta = jax.random.normal(key, (d, num_projections))
    theta = theta / jnp.linalg.norm(theta, axis=0, keepdims=True)
    
    # Project data
    x_proj = X @ theta # shape: (N, num_projections)
    y_proj = Y @ theta # shape: (M, num_projections)
    
    # Compute 1D KS distances for all projections simultaneously
    ks_distances = _vmap_ks_1d(x_proj, y_proj)
    
    # Aggregate across projections
    if max_sliced:
        return jnp.max(ks_distances)
    
    return jnp.mean(ks_distances ** p) ** (1.0 / p)