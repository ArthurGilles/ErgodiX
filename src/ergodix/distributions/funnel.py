import jax.numpy as jnp
from jaxtyping import Array, Float

from .base import TargetDistribution



class Funnel(TargetDistribution):
    """
    d-dimensional Funnel distribution (Neal, 2003).
    """
    sigma: float = 3.0  
    
    def __call__(self, x: Float[Array, "dim"]) -> Float[Array, ""]:
        x0 = x[0]
        x_rest = x[1:]
        d = x.shape[0]
        
        log_p_x0 = -0.5 * jnp.log(2 * jnp.pi * self.sigma**2) - 0.5 * (x0**2) / (self.sigma**2)
        
        log_p_rest = -0.5 * (d - 1) * jnp.log(2 * jnp.pi) \
                     - 0.5 * (d - 1) * x0 \
                     - 0.5 * jnp.exp(-x0) * jnp.sum(x_rest**2)
                     
        return log_p_x0 + log_p_rest
