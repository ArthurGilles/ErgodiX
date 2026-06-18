import jax
import jax.numpy as jnp
from ergodix.slips import slips, SLIPSParams, GeomSchedule
from ergodix.distributions import IsotropicGaussian

def test_slips_params_initialization():
    params = SLIPSParams(sigma=5.0)
    assert params.sigma == 5.0
    assert params.target_accept == 0.75
    assert params.n_mcmc_steps == 32

def test_slips_end_to_end_no_history(prng_key):
    dim = 2
    batch_size = 3
    target = IsotropicGaussian(mean=jnp.zeros(dim), std=jnp.ones(dim))
    
    schedule = GeomSchedule(alpha_1=1.0, alpha_2=1.0)
    time_grid = schedule.get_snr_grid(t_0=0.1, t_end=0.9, steps=5)
    
    params = SLIPSParams(
        sigma=1.0,
        schedule=schedule,
        n_mcmc_steps=4,
        n_chains=2,
        n_init_steps=4,
        return_history=False
    )
    
    samples = slips(prng_key, target, time_grid, batch_size, dim, params)
    
    # Assert return types and shapes
    assert isinstance(samples, jax.Array)
    assert samples.shape == (batch_size, dim)
    assert jnp.all(jnp.isfinite(samples))

def test_slips_end_to_end_with_history(prng_key):
    dim = 2
    batch_size = 2
    grid_steps = 4
    
    target = IsotropicGaussian(mean=jnp.zeros(dim), std=jnp.ones(dim))
    schedule = GeomSchedule()
    time_grid = schedule.get_snr_grid(t_0=0.1, t_end=0.9, steps=grid_steps)
    
    params = SLIPSParams(
        sigma=1.0,
        schedule=schedule,
        n_mcmc_steps=2,
        n_chains=2,
        n_init_steps=2,
        return_history=True
    )
    
    # When return_history is True, it returns a Tuple: (Samples, Y_hist, X_hist)
    out = slips(prng_key, target, time_grid, batch_size, dim, params)
    
    assert isinstance(out, tuple)
    assert len(out) == 3
    
    samples, y_hist, x_hist = out
    
    assert samples.shape == (batch_size, dim)
    assert y_hist.shape == (batch_size, grid_steps-1, dim)
    assert x_hist.shape == (batch_size, grid_steps-1, params.n_chains, dim)