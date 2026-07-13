import jax
import equinox as eqx

from .._utils import as_float_array


class RDMCParams(eqx.Module):
    # Dynamic (traced): coerced to arrays so they are ALWAYS traced under
    # filter_jit, even if a caller passes a plain Python float.
    T:             jax.Array = eqx.field(converter=as_float_array)
    ula_step_size: jax.Array = eqx.field(converter=as_float_array)

    # Static: these determine array shapes / arange sizes, so they must be
    # concrete at trace time. Changing one recompiles (unavoidable for a shape).
    n_steps:      int    = eqx.field(default=32, static=True)
    n_mc_samples: int    = eqx.field(default=32, static=True)
    n_particles:  int    = eqx.field(default=32, static=True)
    n_ula_steps:  int    = eqx.field(default=32, static=True)
    return_history: bool = eqx.field(default=False, static=True)
