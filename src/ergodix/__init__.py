"""
ergodix: JAX samplers with target distributions and visualisation tools.

Subpackages
-----------
- ``ergodix.slips``: SLIPS (stochastic localization) sampler.
- ``ergodix.RDMC``: RDMC (reverse diffusion Monte Carlo) sampler.
- ``ergodix.AIS``: AIS (annealed importance sampling) sampler.
- ``ergodix.distributions``: target distributions (all ``TargetDistribution``).
- ``ergodix.metrics``: sliced distances to compare two sets of samples.
- ``ergodix.visuals``: animation / visualisation helpers (needs matplotlib).

The subpackages are imported lazily on first access, so ``import ergodix`` stays
fast and does not eagerly pull in heavy optional dependencies (matplotlib,
pandas). Import what you need, e.g.::

    from ergodix.slips import slips, SLIPSParams
    from ergodix.AIS import ais, AISParams
"""
import importlib

__version__ = "0.1.0"

_SUBPACKAGES = ("distributions", "slips", "RDMC", "AIS", "metrics", "visuals")


def __getattr__(name: str):
    if name in _SUBPACKAGES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted([*globals(), *_SUBPACKAGES])


__all__ = [*_SUBPACKAGES, "__version__"]
