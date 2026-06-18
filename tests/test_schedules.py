import pytest
import jax.numpy as jnp
from ergodix.slips import StandardSchedule, GeomSchedule

def test_standard_schedule():
    sched = StandardSchedule(alpha_1=1.0)
    t = jnp.array(0.5)
    
    # Base computations
    assert jnp.isfinite(sched.g(t))
    assert jnp.isfinite(sched.alpha(t))
    assert jnp.isfinite(sched.log_snr(t))
    
    # SNR grid generation
    grid = sched.get_snr_grid(t_0=0.1, t_end=0.9, steps=10)
    assert grid.shape == (10,)
    assert jnp.all(jnp.diff(grid) > 0), "Grid must be strictly monotonically increasing"
    
    # Grid validations (StandardSchedule expects t_0 > 0)
    valid_grid = sched.validate_grid(grid)
    assert valid_grid.shape == grid.shape
    
    with pytest.raises(Exception):
        invalid_grid = jnp.array([0.0, 0.5, 0.9])
        sched.validate_grid(invalid_grid)

def test_geom_schedule():
    sched = GeomSchedule(alpha_1=2.0, alpha_2=1.0)
    t = jnp.array(0.5)
    
    assert jnp.isfinite(sched.g(t))
    
    # SNR grid generation
    grid = sched.get_snr_grid(t_0=0.1, t_end=0.9, steps=15)
    assert grid.shape == (15,)
    assert jnp.all(jnp.diff(grid) > 0)
    
    # GeomSchedule specific validation: t_end must be < 1.0
    with pytest.raises(Exception):
        invalid_geom_grid = jnp.array([0.1, 0.5, 1.0])
        sched.validate_grid(invalid_geom_grid)