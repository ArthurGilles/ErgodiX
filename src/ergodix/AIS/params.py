import jax
import equinox as eqx

from .schedules import AnnealingSchedule, LinearSchedule
from .._utils import as_float_array


class AISParams(eqx.Module):
    """
    Parameters for the Annealed Importance Sampling (AIS) algorithm.
    Object passed to the ``ais`` function, bundling everything the sampler
    needs.

    The intermediate distributions are visited by a MALA kernel whose step
    size is (optionally) tuned online from the acceptance rate, exactly as in
    the SLIPS sampler.

    Parameters
    ----------
    step_size: float
        Initial MALA step size, shared by every particle at the first
        transition and then adapted per particle when ``adapt_step_size`` is
        ``True``.
    target_accept: float
        Target MALA acceptance rate the step size is tuned towards
        (``0.574`` is the asymptotically optimal MALA value).
    step_min: float
        Lower clip on the (adapted) MALA step size.
    step_max: float
        Upper clip on the (adapted) MALA step size.
    learning_rate: float
        Rate at which the step size is tuned from the acceptance rate.
    schedule: AnnealingSchedule
        Object defining the inverse-temperature ladder ``beta_k`` (see
        ``AnnealingSchedule`` and its subclasses). Defaults to a linear ladder.
    n_steps: int
        Number of annealing steps, i.e. MCMC transitions between the reference
        and the target. There are ``n_steps + 1`` intermediate distributions.
    n_mcmc_steps: int
        Number of MALA steps applied at each intermediate distribution.
    adapt_step_size: bool
        Whether to adapt the MALA step size online from the acceptance rate.
    return_history: bool
        Whether to also return the per-step trajectory and acceptance rates.
    """
    # Dynamic (traced): coerced to arrays so they are ALWAYS traced under
    # filter_jit, even if a caller passes a plain Python float, and can be
    # swept without recompilation.
    step_size:     jax.Array = eqx.field(default=0.1,   converter=as_float_array)
    target_accept: jax.Array = eqx.field(default=0.574, converter=as_float_array)
    step_min:      jax.Array = eqx.field(default=1e-4,  converter=as_float_array)
    step_max:      jax.Array = eqx.field(default=1e2,   converter=as_float_array)
    learning_rate: jax.Array = eqx.field(default=1.0,   converter=as_float_array)

    # Named schedule
    schedule: AnnealingSchedule = LinearSchedule()

    # Static named params
    n_steps:         int  = eqx.field(default=64, static=True)
    n_mcmc_steps:    int  = eqx.field(default=5,  static=True)
    adapt_step_size: bool = eqx.field(default=True,  static=True)
    return_history:  bool = eqx.field(default=False, static=True)
