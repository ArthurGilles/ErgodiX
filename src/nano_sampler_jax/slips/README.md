# slips_jax subpackage

This subpackage contains the implementation in JAX of the paper:

### *"Stochastic Localization via Iterative Posterior Sampling"*

available at:

### https://arxiv.org/pdf/2402.10758

It provides the user with 5 functions/classes:

- `slips`: function which runs the SLIPS algorithm
- `SLIPSParams`: object which allows to define the parameters for the slips algorithm
- `GeomSchedule`: object representing the schedule (functions $\alpha(t)$ and $g(t)$ in the SLIPS paper) for a non-asymptotic geometric schedule, as defined in section **3.2.b**
- `StandardSchedule`: object representing the schedule (functions $\alpha(t)$ and $g(t)$ in the SLIPS paper) for an asymptotic geometric schedule, as defined in section **3.2.a**
- `Schedule`: object representing the schedule in general. It defines (using the notations from the paper) an `alpha` function, based on a `g` function. The `g`function can be defined by making a subclass which inherits from `Schedule` and  redefines a `g` function.


# Example usage of functions from this subpackage

```python

import jax
import jax.numpy as jnp
from slips_jax import slips, SLIPSParams, GeomSchedule


# Define JAX random key
key = jax.random.PRNGKey(42)

# Distribution to sample from
log_target = lambda x: -jnp.sum(((x-6)/2)**2)/2

# Several samplers can be ran in parallel with the batch_size variable (faster on GPU)
batch_size = 10

# Dimension of the distribution to sample from
dim = 2

# object to define the alpha and g functions
schedule = GeomSchedule(alpha_1=1.0, alpha_2=2.0)

# time discretization on which to solve the SDE defined in the SLIPS paper
time_grid = schedule.get_snr_grid(t_0=0.1, t_end=0.98, steps=20)

# Bundle the parameters in a single object
params = SLIPSParams(sigma=10.0, 
                     schedule=schedule,
                     n_mcmc_steps=64,
                     n_chains=8,
                     n_init_steps=64,
                     return_history=True)

# Run the algorithm
samples, Y_hist, X_hist = slips(key, log_target, time_grid, batch_size, dim, params)

print(samples)
```


