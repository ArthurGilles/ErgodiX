import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .base import TargetDistribution

class Rings(TargetDistribution):
    """
    The Rings distribution from the SLIPS paper.
    Inverse polar reparameterization of a K-component Gaussian mixture for the radius
    and a uniform distribution for the angle.
    """
    radii: Float[Array, "K"]
    inv_var: float
    log_constant: float

    def __init__(self, radii: Array | None = None, sigma: float = 0.15):
        _radii = jnp.array([1.0, 2.0, 3.0, 4.0]) if radii is None else jnp.array(radii)
        self.radii = _radii
        
        # Precompute constants
        K = _radii.shape[0]
        self.inv_var = 1.0 / (sigma**2)
        
        log_weight = -jnp.log(K)
        log_angular = -jnp.log(2 * jnp.pi)
        log_gaussian_norm = -0.5 * jnp.log(2 * jnp.pi * sigma**2)
        
        self.log_constant = log_weight + log_angular + log_gaussian_norm

    def __call__(self, x: Float[Array, "2"]) -> Float[Array, ""]:
        sq_norm = jnp.sum(x**2)
        # safe differentiation of the sqrt near zero
        r = jnp.sqrt(jnp.maximum(sq_norm, 1e-16)) 
        
        comp_exps = -0.5 * self.inv_var * (r - self.radii)**2
        
        return self.log_constant + jax.scipy.special.logsumexp(comp_exps) - jnp.log(r)