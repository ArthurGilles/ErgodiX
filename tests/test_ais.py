import jax
import jax.numpy as jnp
import pytest

from ergodix.distributions import IsotropicGaussian, IsotropicGMM
from ergodix.AIS import (
    ais,
    AISParams,
    log_normalizing_constant,
    effective_sample_size,
    resample,
    AnnealingSchedule,
    LinearSchedule,
    PowerSchedule,
    SigmoidSchedule,
)

# Test the schedules of AIS
@pytest.mark.parametrize(
    "schedule",
    [LinearSchedule(), PowerSchedule(), PowerSchedule(power=3.0), SigmoidSchedule(), SigmoidSchedule(sharpness=10.0)],
)
def test_schedule_betas_are_valid_ladder(schedule):
    n_steps = 16
    betas = schedule.betas(n_steps)
    # There are n_steps + 1 intermediate distributions.
    assert betas.shape == (n_steps + 1,)
    # Pinned endpoints: beta_0 == 0 (reference) and beta_n == 1 (target).
    assert jnp.isclose(betas[0], 0.0)
    assert jnp.isclose(betas[-1], 1.0)
    # Monotone non-decreasing ladder within [0, 1].
    assert jnp.all(betas >= -1e-6) and jnp.all(betas <= 1.0 + 1e-6)
    assert jnp.all(jnp.diff(betas) >= -1e-6)


def test_linear_schedule_is_evenly_spaced():
    betas = LinearSchedule().betas(8)
    assert jnp.allclose(betas, jnp.linspace(0.0, 1.0, 9))


def test_power_schedule_clusters_near_target():
    # power > 1 pushes intermediate betas below the linear ladder (steps packed
    # near the target).
    s = jnp.linspace(0.0, 1.0, 20)
    assert jnp.all(PowerSchedule(power=2.0).beta(s) <= s + 1e-6)


def test_schedule_base_is_abstract():
    with pytest.raises(NotImplementedError):
        AnnealingSchedule().beta(jnp.array(0.5))


# Test the parameters of AIS

def test_ais_params_defaults():
    p = AISParams()
    assert p.n_steps == 64
    assert p.n_mcmc_steps == 5
    assert p.adapt_step_size is True
    assert p.return_history is False
    assert isinstance(p.schedule, LinearSchedule)
    # Dynamic scalars coerced to traced arrays.
    assert isinstance(p.step_size, jax.Array)
    assert jnp.isclose(p.target_accept, 0.574)


# Test AIS in itself
def _target_and_reference():
    # A separated 2D bimodal target bridged from a broad Gaussian reference.
    target = IsotropicGMM(
        weights=jnp.ones(2),
        means=jnp.array([[-3.0, 0.0], [3.0, 0.0]]),
        variances=jnp.ones(2),
    )
    reference = IsotropicGaussian(mean=jnp.zeros(2), std=jnp.full((2,), 4.0))
    return target, reference


def test_ais_end_to_end_no_history(prng_key):
    dim = 2
    batch_size = 16
    target, reference = _target_and_reference()
    params = AISParams(n_steps=8, n_mcmc_steps=3)

    samples, log_w = ais(prng_key, target, reference, batch_size, dim, params)

    assert isinstance(samples, jax.Array)
    assert samples.shape == (batch_size, dim)
    assert log_w.shape == (batch_size,)
    assert jnp.all(jnp.isfinite(samples))
    assert jnp.all(jnp.isfinite(log_w))


def test_ais_end_to_end_with_history(prng_key):
    dim = 2
    batch_size = 12
    n_steps = 8
    target, reference = _target_and_reference()
    params = AISParams(n_steps=n_steps, n_mcmc_steps=3, return_history=True)

    out = ais(prng_key, target, reference, batch_size, dim, params)

    # With return_history: (samples, log_weights, x_hist, accept_hist).
    assert isinstance(out, tuple)
    assert len(out) == 4
    samples, log_w, x_hist, accept_hist = out

    assert samples.shape == (batch_size, dim)
    assert log_w.shape == (batch_size,)
    assert x_hist.shape == (batch_size, n_steps, dim)
    assert accept_hist.shape == (batch_size, n_steps)
    # Acceptance rates are valid probabilities.
    assert jnp.all(accept_hist >= 0.0) and jnp.all(accept_hist <= 1.0)
    # The returned sample is the final frame of the trajectory.
    assert jnp.allclose(samples, x_hist[:, -1, :])


def test_ais_accepts_plain_callable_target(prng_key):
    # target may be a bare callable log-density, not only a TargetDistribution.
    dim = 2
    reference = IsotropicGaussian(mean=jnp.zeros(dim), std=jnp.full((dim,), 4.0))
    log_target = lambda x: -0.5 * jnp.sum(x ** 2)
    params = AISParams(n_steps=6, n_mcmc_steps=2)

    samples, log_w = ais(prng_key, log_target, reference, 8, dim, params)
    assert samples.shape == (8, dim)
    assert jnp.all(jnp.isfinite(log_w))


def test_ais_without_step_size_adaptation(prng_key):
    dim = 2
    target, reference = _target_and_reference()
    params = AISParams(n_steps=6, n_mcmc_steps=2, adapt_step_size=False)

    samples, log_w = ais(prng_key, target, reference, 8, dim, params)
    assert samples.shape == (8, dim)
    assert jnp.all(jnp.isfinite(samples))


def test_ais_reference_dim_mismatch_raises(prng_key):
    # dim must match the reference's event dimension.
    reference = IsotropicGaussian(mean=jnp.zeros(3), std=jnp.ones(3))
    target = IsotropicGaussian(mean=jnp.zeros(3), std=jnp.ones(3))
    params = AISParams(n_steps=4, n_mcmc_steps=2)
    with pytest.raises(ValueError, match="dimension"):
        ais(prng_key, target, reference, batch_size=4, dim=2, params=params)



# Weights/ resampling
def test_effective_sample_size_bounds(prng_key):
    target, reference = _target_and_reference()
    params = AISParams(n_steps=16, n_mcmc_steps=3)
    _, log_w = ais(prng_key, target, reference, batch_size=64, dim=2, params=params)

    ess = effective_sample_size(log_w)
    assert ess.shape == ()
    assert 1.0 - 1e-4 <= float(ess) <= 64.0 + 1e-4


def test_effective_sample_size_equal_weights_is_n():
    # Uniform weights => ESS == N.
    log_w = jnp.zeros(32)
    assert jnp.isclose(effective_sample_size(log_w), 32.0)


def test_effective_sample_size_degenerate_weight_is_one():
    # All mass on a single sample => ESS == 1.
    log_w = jnp.array([0.0] + [-1e30] * 31)
    assert jnp.isclose(effective_sample_size(log_w), 1.0)


def test_log_normalizing_constant_matches_mean_weight():
    log_w = jnp.array([0.0, jnp.log(3.0), jnp.log(5.0), 0.0])
    # logsumexp(log_w) - log(N) == log(mean of exp(log_w)).
    expected = jnp.log(jnp.mean(jnp.exp(log_w)))
    assert jnp.isclose(log_normalizing_constant(log_w), expected)


def test_log_normalizing_constant_recovers_gaussian_ratio(prng_key):
    # log_normalizing_constant estimates log(Z_n / Z_0), where Z is the mass of
    # each unnormalized density (IsotropicGaussian.__call__ is unnormalized). 
    # For an isotropic Gaussian the mass integral is
    # Z = (2*pi)**(d/2) * std**d, so the ratio has the closed form
    # log(Z_n / Z_0) = d * (log std_n - log std_0). AIS should recover it.
    dim = 2
    std_ref, std_target = 2.0, 1.0
    reference = IsotropicGaussian(mean=jnp.zeros(dim), std=jnp.full((dim,), std_ref))
    target = IsotropicGaussian(mean=jnp.zeros(dim), std=jnp.full((dim,), std_target))
    params = AISParams(n_steps=32, n_mcmc_steps=5)
    _, log_w = ais(prng_key, target, reference, batch_size=512, dim=dim, params=params)

    log_ratio = log_normalizing_constant(log_w)
    analytic = dim * (jnp.log(std_target) - jnp.log(std_ref))
    assert jnp.isfinite(log_ratio)
    assert jnp.allclose(log_ratio, analytic, atol=0.3)


def test_resample_shape_and_membership(prng_key):
    dim = 2
    target, reference = _target_and_reference()
    params = AISParams(n_steps=8, n_mcmc_steps=3)
    k_ais, k_res = jax.random.split(prng_key)
    samples, log_w = ais(k_ais, target, reference, batch_size=32, dim=dim, params=params)

    resampled = resample(k_res, samples, log_w)
    assert resampled.shape == samples.shape
    # Every resampled point is one of the original weighted samples.
    matches = jnp.any(jnp.all(resampled[:, None, :] == samples[None, :, :], axis=-1), axis=1)
    assert jnp.all(matches)


def test_ais_weighted_mean_recovers_target(prng_key):
    # Self-normalized weighted mean should track a shifted-Gaussian target.
    dim = 2
    mean = jnp.array([2.0, -2.0])
    target = IsotropicGaussian(mean=mean, std=jnp.ones(dim))
    reference = IsotropicGaussian(mean=jnp.zeros(dim), std=jnp.full((dim,), 4.0))
    params = AISParams(n_steps=32, n_mcmc_steps=5)

    samples, log_w = ais(prng_key, target, reference, batch_size=512, dim=dim, params=params)
    weights = jax.nn.softmax(log_w)
    weighted_mean = jnp.sum(weights[:, None] * samples, axis=0)
    assert jnp.allclose(weighted_mean, mean, atol=0.5)
