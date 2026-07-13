import jax
import equinox as eqx
from jaxtyping import Array, Float

# DOCUMENT DENSITY OR LOG DENSITY ???


class TargetDistribution(eqx.Module):
    """Base class for all target distributions."""

    def __call__(self, x: Float[Array, "dim"]) -> Float[Array, ""]:
        raise NotImplementedError

    def score(self, x: Float[Array, "dim"]) -> Float[Array, "dim"]:
        """
        Score of the distribution, i.e. the gradient of the (unnormalized)
        log-density with respect to ``x``.

        The default implementation differentiates :meth:`__call__` with
        ``jax.grad``, so every distribution gets a correct score for free.
        Override it with a closed-form expression when one is available and
        cheaper than automatic differentiation.

        Parameters
        ----------
        x: Float[Array, "dim"]
            Point at which to evaluate the score, of shape ``(dim,)``.
        Returns
        -------
        Float[Array, "dim"]
            The score at ``x``, of shape ``(dim,)``.
        """
        return jax.grad(self.__call__)(x)

    def sample(self, key: jax.Array, shape: tuple[int, ...]) -> Float[Array, "*shape dim"]:
        """
        Draw i.i.d. samples from the distribution.

        Only reference distributions that a sampler starts from (e.g. the
        isotropic Gaussian base of a diffusion sampler) need to provide this;
        general targets are defined through their (unnormalized) log-density
        and score alone and leave it unimplemented.

        Parameters
        ----------
        key: jax.Array
            Random key for JAX random number generation.
        shape: tuple[int, ...]
            Leading batch shape of the draw. The event dimension ``(dim,)`` is
            appended, so the returned array has shape ``(*shape, dim)``.
        Returns
        -------
        Float[Array, "*shape dim"]
            Samples drawn from the distribution.
        """
        raise NotImplementedError