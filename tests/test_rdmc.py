import jax
import jax.numpy as jnp
import pytest

from ergodix.distributions import IsotropicGaussian, Banana
from ergodix.RDMC import RDMC, RDMCParams


# Tests on the RDMC parameters
def test_rdmc_params_defaults():
    p = RDMCParams(T=2.0, ula_step_size=0.01)
    # Static shape-determining fields keep their documented defaults.
    assert p.n_steps == 32
    assert p.n_mc_samples == 32
    assert p.n_particles == 32
    assert p.n_ula_steps == 32
    assert p.return_history is False


def test_rdmc_params_scalars_are_traced_arrays():
    # Dynamic scalars are coerced to arrays so they are traced under filter_jit
    # (and can be swept without recompilation), even when passed as Python floats.
    p = RDMCParams(T=2.0, ula_step_size=0.01)
    assert isinstance(p.T, jax.Array)
    assert isinstance(p.ula_step_size, jax.Array)
    assert jnp.isclose(p.T, 2.0)
    assert jnp.isclose(p.ula_step_size, 0.01)


# Test on the whole RDMC algorithm
def _small_params(**overrides):
    base = dict(
        T=2.0,
        ula_step_size=0.01,
        n_steps=4,
        n_mc_samples=8,
        n_particles=8,
        n_ula_steps=3,
    )
    base.update(overrides)
    return RDMCParams(**base)


def test_rdmc_1d_isotropic_gaussian(prng_key):
    target = IsotropicGaussian(mean=jnp.zeros(1), std=jnp.ones(1))
    samples = RDMC(prng_key, target, batch_size=6, dim=1, params=_small_params())
    assert samples.shape == (6, 1)
    assert jnp.all(jnp.isfinite(samples))


def test_rdmc_2d_isotropic_gaussian(prng_key):
    target = IsotropicGaussian(mean=jnp.zeros(2), std=jnp.ones(2))
    samples = RDMC(prng_key, target, batch_size=8, dim=2, params=_small_params())
    assert samples.shape == (8, 2)
    assert jnp.all(jnp.isfinite(samples))


def test_rdmc_score_only_target(prng_key):
    # RDMC is driven purely by target.score, so any TargetDistribution works,
    # including non-Gaussian ones without a closed-form score.
    target = Banana()
    samples = RDMC(prng_key, target, batch_size=6, dim=2, params=_small_params())
    assert samples.shape == (6, 2)
    assert jnp.all(jnp.isfinite(samples))


def test_rdmc_with_history(prng_key):
    dim = 2
    batch_size = 5
    n_steps = 4
    n_mc_samples = 8
    target = IsotropicGaussian(mean=jnp.zeros(dim), std=jnp.ones(dim))
    params = _small_params(
        n_steps=n_steps, n_mc_samples=n_mc_samples, return_history=True
    )

    out = RDMC(prng_key, target, batch_size=batch_size, dim=dim, params=params)

    # When return_history is True, RDMC returns (X, X_hist, samples_hist).
    assert isinstance(out, tuple)
    assert len(out) == 3

    X, x_hist, samples_hist = out
    assert X.shape == (batch_size, dim)
    assert x_hist.shape == (batch_size, n_steps, dim)
    assert samples_hist.shape == (batch_size, n_steps, n_mc_samples, dim)
    assert jnp.all(jnp.isfinite(X))
    # The returned sample is the last frame of the recorded trajectory.
    assert jnp.allclose(X, x_hist[:, -1, :])


def test_rdmc_recovers_gaussian_mean(prng_key):
    # Loose statistical guardrail: with a longer run RDMC should centre its
    # samples near the target mean. Tolerances are generous to stay robust.
    mean = jnp.array([1.0, -1.0])
    target = IsotropicGaussian(mean=mean, std=jnp.ones(2))
    params = RDMCParams(
        T=3.0,
        ula_step_size=0.02,
        n_steps=16,
        n_mc_samples=32,
        n_particles=32,
        n_ula_steps=8,
    )
    samples = RDMC(prng_key, target, batch_size=512, dim=2, params=params)
    assert jnp.all(jnp.isfinite(samples))
    assert jnp.allclose(jnp.mean(samples, axis=0), mean, atol=0.5)
