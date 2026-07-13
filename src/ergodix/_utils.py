"""Internal helpers shared across ergodix subpackages."""
import jax
import jax.numpy as jnp


def as_float_array(x) -> jax.Array:
    """Coerce ``x`` to a floating JAX array.

    Used as an ``eqx.field(converter=...)`` on dynamic scalar parameters so
    they are ALWAYS traced under ``filter_jit`` (arrays are traced, Python
    scalars are held static), even when a caller passes a bare Python
    ``float``/``int``. This lets those scalars be tuned without triggering a
    recompilation.
    """
    return jnp.asarray(x, dtype=float)


def as_optional_float_array(x):
    """Like :func:`as_float_array` but lets ``None`` pass through.

    For optional scalar fields whose ``None`` sentinel is resolved to a
    concrete value later (e.g. in ``__post_init__``).
    """
    return None if x is None else as_float_array(x)
