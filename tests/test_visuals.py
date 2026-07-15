import matplotlib

matplotlib.use("Agg")  # headless backend for tests

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from matplotlib import animation
import matplotlib.pyplot as plt

from ergodix.distributions import IsotropicGaussian
from ergodix.slips import slips, SLIPSParams, GeomSchedule
from ergodix.visuals import animate_samples, animate_projection, compare_samples
from ergodix.visuals._core import _density_1d, _density_2d

# Several tests build an animation only to inspect it (without saving/showing),
# which makes matplotlib warn at garbage-collection time. That is expected here.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Animation was deleted without rendering:UserWarning"
)


# Helpers and fixtures
@pytest.fixture(autouse=True)
def _close_figures():
    """Close every figure after each test to avoid leaking state/memory."""
    yield
    plt.close("all")


def _make(key, shape):
    return jax.random.normal(key, shape)


def _n_frames(anim):
    return len(list(anim.new_frame_seq()))


def _axes(anim):
    return anim._fig.axes


# animate_samples
def test_animate_samples_1d(prng_key):
    samples = _make(prng_key, (40, 6, 1))
    anim = animate_samples(samples)
    assert isinstance(anim, animation.FuncAnimation)
    assert _n_frames(anim) == 6
    assert len(_axes(anim)) == 1


def test_animate_samples_2d_with_pdf(prng_key):
    samples = _make(prng_key, (60, 5, 2))
    pdf = IsotropicGaussian(mean=jnp.zeros(2), std=jnp.ones(2))
    anim = animate_samples(samples, pdf=pdf, grid_size=40)
    assert isinstance(anim, animation.FuncAnimation)
    assert _n_frames(anim) == 5
    assert len(_axes(anim)) == 1


def test_animate_samples_with_particles_two_panels(prng_key):
    k1, k2 = jax.random.split(prng_key)
    samples = _make(k1, (50, 6, 2))           # Y_hist layout
    particles = _make(k2, (50, 6, 3, 2))      # X_hist layout (n, T, chains, dim)
    pdf = IsotropicGaussian(mean=jnp.zeros(2), std=jnp.ones(2))
    anim = animate_samples(
        samples, pdf=pdf, particles=particles, particles_pdf=pdf, grid_size=40
    )
    assert isinstance(anim, animation.FuncAnimation)
    assert len(_axes(anim)) == 2  # samples + particles panels


def test_animate_samples_1d_linear_pdf(prng_key):
    samples = _make(prng_key, (40, 4, 1))
    linear_pdf = lambda x: jnp.exp(-0.5 * jnp.sum(x ** 2))
    anim = animate_samples(samples, pdf=linear_pdf, pdf_is_log=False, grid_size=80)
    assert isinstance(anim, animation.FuncAnimation)


def test_animate_samples_rejects_high_dim(prng_key):
    samples = _make(prng_key, (10, 4, 3))
    with pytest.raises(ValueError, match="animate_projection"):
        animate_samples(samples)


def test_animate_samples_rejects_bad_shape(prng_key):
    samples = _make(prng_key, (10, 4))  # missing the dim axis
    with pytest.raises(ValueError, match="n_samples, n_steps, dim"):
        animate_samples(samples)


# animate_projection
def test_animate_projection_to_1d(prng_key):
    samples = _make(prng_key, (50, 5, 5))
    direction = jnp.ones(5)
    anim = animate_projection(samples, direction)
    assert isinstance(anim, animation.FuncAnimation)
    assert _n_frames(anim) == 5
    assert len(_axes(anim)) == 1


def test_animate_projection_to_2d_with_pdf(prng_key):
    samples = _make(prng_key, (50, 5, 5))
    directions = jnp.stack([jnp.eye(5)[0], jnp.eye(5)[1]])
    pdf = IsotropicGaussian(mean=jnp.zeros(5), std=jnp.ones(5))
    anim = animate_projection(samples, directions, pdf=pdf, grid_size=40)
    assert isinstance(anim, animation.FuncAnimation)
    assert _n_frames(anim) == 5


def test_animate_projection_with_particles(prng_key):
    k1, k2 = jax.random.split(prng_key)
    samples = _make(k1, (30, 4, 5))
    particles = _make(k2, (30, 4, 2, 5))
    directions = jnp.stack([jnp.eye(5)[0], jnp.eye(5)[2]])
    anim = animate_projection(samples, directions, particles=particles)
    assert len(_axes(anim)) == 2


def test_animate_projection_collinear_directions_raise(prng_key):
    samples = _make(prng_key, (10, 4, 5))
    directions = jnp.stack([jnp.ones(5), 2.0 * jnp.ones(5)])  # collinear
    with pytest.raises(ValueError, match="collinear"):
        animate_projection(samples, directions)


def test_animate_projection_dim_mismatch_raises(prng_key):
    samples = _make(prng_key, (10, 4, 5))
    with pytest.raises(ValueError, match="must match"):
        animate_projection(samples, jnp.ones(3))  # 3 != 5


# compare_samples
def test_compare_samples_with_particles(prng_key):
    keys = jax.random.split(prng_key, 4)
    set_a = (_make(keys[0], (30, 5, 2)), _make(keys[1], (30, 5, 3, 2)))
    set_b = (_make(keys[2], (30, 5, 2)), _make(keys[3], (30, 5, 3, 2)))
    pdf = IsotropicGaussian(mean=jnp.zeros(2), std=jnp.ones(2))

    samples_anim, particles_anim = compare_samples(
        [set_a, set_b], pdfs=pdf, labels=["A", "B"], grid_size=40
    )
    assert isinstance(samples_anim, animation.FuncAnimation)
    assert isinstance(particles_anim, animation.FuncAnimation)
    assert len(_axes(samples_anim)) == 2
    assert len(_axes(particles_anim)) == 2


def test_compare_samples_without_particles_returns_none(prng_key):
    k1, k2 = jax.random.split(prng_key)
    samples_anim, particles_anim = compare_samples(
        [_make(k1, (20, 4, 1)), _make(k2, (20, 4, 1))]
    )
    assert isinstance(samples_anim, animation.FuncAnimation)
    assert particles_anim is None


def test_compare_samples_mismatched_steps_raise(prng_key):
    k1, k2 = jax.random.split(prng_key)
    with pytest.raises(ValueError, match="time steps"):
        compare_samples([_make(k1, (20, 5, 2)), _make(k2, (20, 6, 2))])


def test_compare_samples_per_set_pdfs_length_checked(prng_key):
    k1, k2 = jax.random.split(prng_key)
    with pytest.raises(ValueError, match="one entry per sample set"):
        compare_samples(
            [_make(k1, (20, 4, 1)), _make(k2, (20, 4, 1))],
            pdfs=[None],  # wrong length
        )


# pdf normalisation helpers
def test_density_1d_integrates_to_one():
    log_gauss = lambda x: -0.5 * jnp.sum(x ** 2)
    xs, dens = _density_1d(log_gauss, (-6.0, 6.0), grid_size=400, is_log=True)
    assert np.isclose(np.trapezoid(dens, xs), 1.0, atol=1e-2)


def test_density_1d_linear_input_integrates_to_one():
    gauss = lambda x: jnp.exp(-0.5 * jnp.sum(x ** 2))
    xs, dens = _density_1d(gauss, (-6.0, 6.0), grid_size=400, is_log=False)
    assert np.isclose(np.trapezoid(dens, xs), 1.0, atol=1e-2)


def test_density_2d_integrates_to_one():
    log_gauss = lambda x: -0.5 * jnp.sum(x ** 2)
    xx, yy, zz = _density_2d(log_gauss, (-6.0, 6.0), (-6.0, 6.0), grid_size=120, is_log=True)
    cell = (xx[0, 1] - xx[0, 0]) * (yy[1, 0] - yy[0, 0])
    assert np.isclose(zz.sum() * cell, 1.0, atol=1e-2)


# GIF export
def test_animate_samples_saves_gif(prng_key, tmp_path):
    samples = _make(prng_key, (30, 4, 2))
    out = tmp_path / "samples.gif"
    animate_samples(samples, fps=4, save_path=str(out), grid_size=30)
    assert out.exists() and out.stat().st_size > 0


def test_compare_samples_saves_two_gifs(prng_key, tmp_path):
    keys = jax.random.split(prng_key, 2)
    set_a = (_make(keys[0], (20, 3, 1)), _make(keys[1], (20, 3, 2, 1)))
    out = tmp_path / "cmp.gif"
    compare_samples([set_a], fps=4, save_path=str(out), bins=20)
    assert (tmp_path / "cmp.gif").exists()
    assert (tmp_path / "cmp_particles.gif").exists()


def test_unsupported_save_format_raises(prng_key, tmp_path):
    samples = _make(prng_key, (10, 3, 1))
    with pytest.raises(ValueError, match="Unsupported save format"):
        animate_samples(samples, save_path=str(tmp_path / "out.xyz"))


# End-to-end with real SLIPS output
def test_animate_samples_with_real_slips_output(prng_key):
    dim = 2
    batch_size = 8
    grid_steps = 5
    target = IsotropicGaussian(mean=jnp.zeros(dim), std=jnp.ones(dim))
    schedule = GeomSchedule()
    time_grid = schedule.get_snr_grid(t_0=0.1, t_end=0.9, steps=grid_steps)
    params = SLIPSParams(
        sigma=1.0,
        schedule=schedule,
        n_mcmc_steps=2,
        n_chains=2,
        n_init_steps=2,
        return_history=True,
    )

    _, y_hist, x_hist = slips(prng_key, target, time_grid, batch_size, dim, params)

    anim = animate_samples(
        y_hist, pdf=target, particles=x_hist, time_grid=time_grid, grid_size=30
    )
    assert isinstance(anim, animation.FuncAnimation)
    assert _n_frames(anim) == grid_steps - 1
    assert len(_axes(anim)) == 2
