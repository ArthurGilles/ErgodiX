import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .base import TargetDistribution

class IsotropicGaussian(TargetDistribution):
    """
    A simple baseline distribution for testing purposes.
    Unnormalized
    """
    mean: Float[Array, "dim"]
    std: Float[Array, "dim"]
    
    def __call__(self, x: Float[Array, "dim"]) -> Float[Array, ""]:
        return -0.5 * jnp.sum(((x - self.mean) / self.std)**2)



class IsotropicGMM(TargetDistribution):
    """
    GMM for isotropic covariances.
    """
    log_weights: Float[Array, " K"]
    means: Float[Array, " K D"]
    inv_vars: Float[Array, " K"]
    log_norm_consts: Float[Array, " K"]

    def __init__(self, weights: Array, means: Array, variances: Array):
        D = means.shape[-1]
        self.log_weights = jnp.log(weights / jnp.sum(weights))
        self.means = means
        self.inv_vars = 1.0 / variances
        self.log_norm_consts = -0.5 * D * jnp.log(2 * jnp.pi * variances)

    def __call__(self, x: Float[Array, " D"]) -> Float[Array, ""]:
        def component_log_prob(mean, inv_var, log_norm_const):
            return log_norm_const - 0.5 * inv_var * jnp.sum((x - mean) ** 2)
            
        comp_log_pdfs = jax.vmap(component_log_prob)(self.means, self.inv_vars, self.log_norm_consts)
        return jax.scipy.special.logsumexp(self.log_weights + comp_log_pdfs)


class FullCovGMM(TargetDistribution):
    """
    General GMM for full covariance matrices.
    """
    log_weights: Float[Array, " K"]
    means: Float[Array, " K D"]
    covs: Float[Array, " K D D"]

    def __init__(self, weights: Array, means: Array, covs: Array):
        self.log_weights = jnp.log(weights / jnp.sum(weights))
        self.means = means
        self.covs = covs

    def __call__(self, x: Float[Array, " D"]) -> Float[Array, ""]:
        comp_log_pdfs = jax.vmap(
            jax.scipy.stats.multivariate_normal.logpdf, 
            in_axes=(None, 0, 0)
        )(x, self.means, self.covs)
        
        return jax.scipy.special.logsumexp(self.log_weights + comp_log_pdfs)
    


