import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .base import TargetDistribution


class BayesianLogisticRegression(TargetDistribution):
    """
    Unnormalized posterior distribution for Bayesian Logistic Regression.
    Matches the specification in Appendix H.3 of the SLIPS paper.
    
    Expects a parameter vector theta of shape (d + 1,), 
    where theta[:-1] are the weights and theta[-1] is the bias.
    """
    X: Float[Array, "N d"]
    y: Float[Array, "N"]
    weight_std: float = 1.0
    bias_std: float = 2.5

    def __call__(self, theta: Float[Array, "dim"]) -> Float[Array, ""]:
        w, b = theta[:-1], theta[-1]
        d = w.shape[0]

        log_prior_w = -0.5 * jnp.sum((w / self.weight_std)**2) - (d / 2.0) * jnp.log(2 * jnp.pi * self.weight_std**2)
        log_prior_b = -0.5 * ((b / self.bias_std)**2) - 0.5 * jnp.log(2 * jnp.pi * self.bias_std**2)
        log_prior = log_prior_w + log_prior_b

        logits = self.X @ w + b
        
        log_lik = jnp.sum(
            self.y * jax.nn.log_sigmoid(logits) + 
            (1.0 - self.y) * jax.nn.log_sigmoid(-logits)
        )
    
        return log_prior + log_lik