import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import equinox as eqx


class AnnealingSchedule(eqx.Module):
    """
    Base class for AIS annealing schedules.

    An annealing schedule defines the ladder of inverse temperatures
    ``0 = beta_0 < beta_1 < ... < beta_n = 1`` bridging the reference
    distribution ``pi_0`` and the target ``pi_n``. The k-th intermediate
    distribution is the geometric path

        pi_k(x)  proportional to  pi_0(x) ** (1 - beta_k)  *  pi_n(x) ** beta_k,

    so ``beta_0 = 0`` recovers the reference and ``beta_n = 1`` the target.

    Subclass this and override :meth:`beta`, a monotone map from the normalized
    progress ``s = k / n`` in ``[0, 1]`` to an inverse temperature in ``[0, 1]``
    with ``beta(0) = 0`` and ``beta(1) = 1``. :meth:`beta` must accept array
    inputs (use ``jnp`` operations) so :meth:`betas` can build the whole ladder
    in one call.

    (Distinct from ``ergodix.slips.NoiseSchedule`` and
    ``ergodix.dippax.MixingSchedule``, which parametrise different samplers.)
    """

    def beta(self, s: Float[Array, "..."]) -> Float[Array, "..."]:
        raise NotImplementedError

    def betas(self, n_steps: int) -> Float[Array, " n_steps_plus_1"]:
        """
        Build the full inverse-temperature ladder.

        Parameters
        ----------
        n_steps: int
            Number of annealing steps (i.e. MCMC transitions). The ladder has
            ``n_steps + 1`` entries, one per intermediate distribution.
        Returns
        -------
        Float[Array, "n_steps + 1"]
            Monotone array with ``betas[0] == 0`` and ``betas[-1] == 1``.
        """
        s = jnp.linspace(0.0, 1.0, n_steps + 1)
        b = self.beta(s)
        # Pin the endpoints exactly so pi_0 == reference and pi_n == target,
        # even if beta(0)/beta(1) are only approximately 0/1 numerically.
        return b.at[0].set(0.0).at[-1].set(1.0)


class LinearSchedule(AnnealingSchedule):
    """Evenly spaced ladder, ``beta(s) = s``. A sensible default."""

    def beta(self, s: Float[Array, "..."]) -> Float[Array, "..."]:
        return s


class PowerSchedule(AnnealingSchedule):
    """
    Power-law ladder, ``beta(s) = s ** power``.

    ``power > 1`` clusters the intermediate distributions near the target
    (fine steps close to ``pi_n``); ``power < 1`` clusters them near the
    reference; ``power == 1`` recovers :class:`LinearSchedule`. Concentrating
    steps near the target is a common, cheap way to improve AIS when the target
    is much more peaked than the reference.

    Parameters
    ----------
    power: float
        Exponent of the schedule (must be > 0).
    """
    power: float = 2.0

    def beta(self, s: Float[Array, "..."]) -> Float[Array, "..."]:
        return s ** self.power


class SigmoidSchedule(AnnealingSchedule):
    """
    Sigmoidal ladder refining both endpoints of the path.

    A logistic curve is evaluated on ``[-sharpness/2, sharpness/2]`` and
    rescaled affinely so that ``beta(0) = 0`` and ``beta(1) = 1``. Because the
    curve is steep in the middle and flat at the ends, consecutive inverse
    temperatures are packed closely near ``beta = 0`` and ``beta = 1`` and are
    spread out in between -- useful when reference and target are very different
    and most of the bridging difficulty sits right next to each endpoint.
    Larger ``sharpness`` makes this concentration more pronounced.

    Parameters
    ----------
    sharpness: float
        Steepness of the logistic curve (must be > 0).
    """
    sharpness: float = 6.0

    def beta(self, s: Float[Array, "..."]) -> Float[Array, "..."]:
        a = self.sharpness
        lo = jax.nn.sigmoid(-a / 2.0)
        hi = jax.nn.sigmoid(a / 2.0)
        return (jax.nn.sigmoid(a * (s - 0.5)) - lo) / (hi - lo)
