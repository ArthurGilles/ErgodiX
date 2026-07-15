import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats

from ergodix.metrics import sliced_kolmogorov_smirnov, sliced_wasserstein_mc


def _two_gaussian_clouds(key, n=512, d=3, shift=0.0):
    """Two independent standard-normal clouds, the second shifted by `shift`."""
    key_x, key_y = jax.random.split(key)
    x = jax.random.normal(key_x, (n, d))
    y = jax.random.normal(key_y, (n, d)) + shift
    return x, y


# Tests on the Sliced Wasserstein distance
def test_swd_returns_scalars(prng_key):
    key_data, key_proj = jax.random.split(prng_key)
    x, y = _two_gaussian_clouds(key_data)
    swd, var = sliced_wasserstein_mc(key_proj, x, y)
    assert swd.shape == ()
    assert var.shape == ()
    assert jnp.isfinite(swd)
    assert swd >= 0.0
    assert var >= 0.0


def test_swd_identical_clouds_is_zero(prng_key):
    key_data, key_proj = jax.random.split(prng_key)
    x = jax.random.normal(key_data, (256, 4))
    swd, var = sliced_wasserstein_mc(key_proj, x, x)
    # Every projection gives a 1D distance of exactly 0, so the variance
    # across projections vanishes too.
    assert jnp.isclose(swd, 0.0, atol=1e-6)
    assert jnp.isclose(var, 0.0, atol=1e-12)


def test_swd_symmetric(prng_key):
    key_data, key_proj = jax.random.split(prng_key)
    x, y = _two_gaussian_clouds(key_data, shift=1.0)
    swd_xy, _ = sliced_wasserstein_mc(key_proj, x, y)
    swd_yx, _ = sliced_wasserstein_mc(key_proj, y, x)
    assert jnp.isclose(swd_xy, swd_yx, atol=1e-6)


def test_swd_translation_invariance(prng_key):
    # Shifting both clouds by the same vector shifts every projection by the
    # same amount and leaves the sorted differences unchanged.
    key_data, key_proj = jax.random.split(prng_key)
    x, y = _two_gaussian_clouds(key_data, shift=0.5)
    c = jnp.array([3.0, -2.0, 7.5])
    swd, _ = sliced_wasserstein_mc(key_proj, x, y)
    swd_shifted, _ = sliced_wasserstein_mc(key_proj, x + c, y + c)
    assert jnp.isclose(swd, swd_shifted, atol=1e-5)


def test_swd_positive_homogeneity(prng_key):
    # W_p is positively homogeneous: scaling both clouds by c scales the
    # distance by c (same key, hence same projection directions).
    key_data, key_proj = jax.random.split(prng_key)
    x, y = _two_gaussian_clouds(key_data, shift=1.0)
    swd, _ = sliced_wasserstein_mc(key_proj, x, y)
    swd_scaled, _ = sliced_wasserstein_mc(key_proj, 2.0 * x, 2.0 * y)
    assert jnp.isclose(swd_scaled, 2.0 * swd, rtol=1e-5)


def test_swd_1d_matches_exact_wasserstein(prng_key):
    # In 1D every unit direction is +/-1, so the sliced distance reduces to
    # the exact order-p Wasserstein distance between the empirical measures.
    key_x, key_y, key_proj = jax.random.split(prng_key, 3)
    x = jax.random.normal(key_x, (200, 1))
    y = 1.5 * jax.random.normal(key_y, (200, 1)) + 0.7

    # p = 1: cross-check against scipy's exact 1D Wasserstein distance.
    swd1, _ = sliced_wasserstein_mc(key_proj, x, y, num_projections=16, p=1)
    w1_scipy = stats.wasserstein_distance(np.asarray(x[:, 0]), np.asarray(y[:, 0]))
    assert jnp.isclose(swd1, w1_scipy, rtol=1e-5)

    # p = 2: closed form for equal-size empirical measures (sorted matching).
    swd2, _ = sliced_wasserstein_mc(key_proj, x, y, num_projections=16, p=2)
    w2_exact = jnp.sqrt(jnp.mean((jnp.sort(x[:, 0]) - jnp.sort(y[:, 0])) ** 2))
    assert jnp.isclose(swd2, w2_exact, rtol=1e-5)


def test_swd_recovers_gaussian_mean_shift(prng_key):
    # For N(0, I) vs N(m, I) the projected W2 along theta is |<m, theta>|, so
    # SW2^2 = E[<m, theta>^2] = ||m||^2 / d. Loose statistical guardrail.
    key_data, key_proj = jax.random.split(prng_key)
    m = jnp.array([2.0, 0.0])
    x, _ = _two_gaussian_clouds(key_data, n=2048, d=2)
    _, y = _two_gaussian_clouds(key_data, n=2048, d=2)
    swd, _ = sliced_wasserstein_mc(key_proj, x, y + m, p=2)
    expected = jnp.linalg.norm(m) / jnp.sqrt(2.0)
    assert jnp.isclose(swd, expected, atol=0.25)


def test_swd_detects_mean_shift(prng_key):
    # Samples from the same distribution should score well below samples
    # from a clearly shifted one.
    key_data, key_proj = jax.random.split(prng_key)
    x, y_same = _two_gaussian_clouds(key_data, n=1024, shift=0.0)
    swd_same, _ = sliced_wasserstein_mc(key_proj, x, y_same)
    swd_far, _ = sliced_wasserstein_mc(key_proj, x, y_same + 3.0)
    assert swd_far > 5.0 * swd_same


# Tests on the Sliced Kolmogorov-Smirnov distance
def test_sks_bounded_in_unit_interval(prng_key):
    key_data, key_proj = jax.random.split(prng_key)
    x, y = _two_gaussian_clouds(key_data, shift=1.0)
    sks = sliced_kolmogorov_smirnov(x, y, key_proj)
    assert sks.shape == ()
    assert 0.0 <= sks <= 1.0


def test_sks_identical_clouds_is_zero(prng_key):
    key_data, key_proj = jax.random.split(prng_key)
    x = jax.random.normal(key_data, (256, 4))
    sks = sliced_kolmogorov_smirnov(x, x, key_proj)
    assert jnp.isclose(sks, 0.0, atol=1e-6)


def test_sks_symmetric(prng_key):
    key_data, key_proj = jax.random.split(prng_key)
    x, y = _two_gaussian_clouds(key_data, shift=1.0)
    sks_xy = sliced_kolmogorov_smirnov(x, y, key_proj)
    sks_yx = sliced_kolmogorov_smirnov(y, x, key_proj)
    assert jnp.isclose(sks_xy, sks_yx, atol=1e-6)


def test_sks_supports_different_sample_sizes(prng_key):
    # Unlike the SWD implementation, the KS statistic does not require the
    # two clouds to have the same number of points.
    key_x, key_y, key_proj = jax.random.split(prng_key, 3)
    x = jax.random.normal(key_x, (300, 3))
    y = jax.random.normal(key_y, (150, 3))
    sks = sliced_kolmogorov_smirnov(x, y, key_proj)
    assert jnp.isfinite(sks)
    assert 0.0 <= sks <= 1.0


def test_sks_disjoint_atoms_is_one(prng_key):
    # Two point masses at distinct locations: almost surely no projection
    # maps them to the same point, so every 1D KS statistic is 1.
    x = jnp.zeros((32, 2))
    y = jnp.ones((32, 2))
    sks = sliced_kolmogorov_smirnov(x, y, prng_key)
    assert jnp.isclose(sks, 1.0, atol=1e-6)


def test_sks_max_sliced_dominates_mean(prng_key):
    key_data, key_proj = jax.random.split(prng_key)
    x, y = _two_gaussian_clouds(key_data, shift=0.5)
    sks_mean = sliced_kolmogorov_smirnov(x, y, key_proj, p=1.0)
    sks_max = sliced_kolmogorov_smirnov(x, y, key_proj, max_sliced=True)
    # Same key means the same projections, and a max dominates a mean.
    assert sks_max >= sks_mean - 1e-6


def test_sks_1d_matches_scipy_ks_2samp(prng_key):
    # The KS statistic is invariant under sign flips, so in 1D every
    # projection yields the standard two-sample KS statistic.
    key_x, key_y, key_proj = jax.random.split(prng_key, 3)
    x = jax.random.normal(key_x, (200, 1))
    y = 0.5 * jax.random.normal(key_y, (150, 1)) + 0.3
    sks = sliced_kolmogorov_smirnov(x, y, key_proj, num_projections=8)
    ks_scipy = stats.ks_2samp(np.asarray(x[:, 0]), np.asarray(y[:, 0])).statistic
    assert jnp.isclose(sks, ks_scipy, atol=1e-6)


def test_sks_detects_mean_shift(prng_key):
    key_data, key_proj = jax.random.split(prng_key)
    x, y_same = _two_gaussian_clouds(key_data, n=1024, shift=0.0)
    sks_same = sliced_kolmogorov_smirnov(x, y_same, key_proj)
    sks_far = sliced_kolmogorov_smirnov(x, y_same + 3.0, key_proj)
    assert sks_far > 2.0 * sks_same
    assert sks_far > 0.5
