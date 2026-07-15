# AIS subpackage

This subpackage contains a JAX implementation of the algorithm from the paper:

### *"Annealed Importance Sampling"* (R. M. Neal, 2001)

available at:

### https://arxiv.org/abs/physics/9803008

AIS produces *weighted* samples from a target $\pi_n$ by bridging it with an
easy-to-sample reference $\pi_0$ through a ladder of intermediate distributions
along the **geometric path**

$$\pi_k(x) \propto \pi_0(x)^{1-\beta_k}\,\pi_n(x)^{\beta_k}, \qquad 0 = \beta_0 < \beta_1 < \dots < \beta_n = 1 .$$

Each particle is drawn from $\pi_0$, carried through the ladder by MCMC
transitions (here MALA) that leave each $\pi_k$ invariant, and accumulates a log
importance weight

$$\log w = \sum_{k=1}^{n} (\beta_k - \beta_{k-1})\,\big(\log \pi_n(x_{k-1}) - \log \pi_0(x_{k-1})\big).$$

The resulting `(samples, log_weights)` are *properly weighted* for $\pi_n$:
expectations are estimated by the self-normalized average
$\sum_i w_i f(x_i) / \sum_i w_i$, and the mean weight is an unbiased estimate of
the normalizing-constant ratio $Z_n / Z_0$.

It provides the following functions/classes:

- `ais`: function which runs the AIS algorithm and returns weighted samples
- `AISParams`: object which bundles the parameters for the algorithm
- `LinearSchedule`, `PowerSchedule`, `SigmoidSchedule`: inverse-temperature
  ladders $\beta_k$
- `AnnealingSchedule`: base class for the ladder; subclass it and override
  `beta(s)` (mapping normalized progress $s = k/n \in [0,1]$ to $\beta \in [0,1]$)
  to define a custom schedule
- `log_normalizing_constant`: estimate of $\log(Z_n/Z_0)$ from the log-weights
- `effective_sample_size`: ESS diagnostic of the weights
- `resample`: turn weighted samples into equally weighted ones targeting $\pi_n$

The MALA step size is tuned online from the acceptance rate, exactly as in the
SLIPS sampler (see `AISParams`).

# Example usage of functions from this subpackage

```python
import jax
import jax.numpy as jnp
from ergodix.AIS import ais, AISParams, PowerSchedule, effective_sample_size, resample
from ergodix.distributions import IsotropicGaussian, IsotropicGMM

# Define JAX random key
key = jax.random.PRNGKey(0)

# Target to sample from: any TargetDistribution (only its log-density
# __call__ is used). A plain callable returning the log-density also works.
target = IsotropicGMM(weights=jnp.ones(2),
                      means=jnp.array([[-4.0, 0.0], [4.0, 0.0]]),
                      variances=jnp.ones(2))

# Reference (proposal) pi_0: a TargetDistribution that can be sampled from
# (AIS is initialised by an exact draw from it). A broad Gaussian is typical.
reference = IsotropicGaussian(mean=jnp.zeros(2), std=jnp.full((2,), 5.0))

# Number of independent weighted samples (particles), run in parallel.
batch_size = 1024

# Dimension of the distributions.
dim = 2

# Bundle the parameters. The schedule sets the inverse-temperature ladder;
# PowerSchedule(power=2) packs more steps near the (peaked) target.
params = AISParams(step_size=0.1,
                   schedule=PowerSchedule(power=2.0),
                   n_steps=128,
                   n_mcmc_steps=5)

# Run the algorithm: weighted samples for the target.
samples, log_weights = ais(key, target, reference, batch_size, dim, params)

# Diagnostics and (optional) resampling into equally weighted samples.
print("ESS:", effective_sample_size(log_weights))
equal_weight_samples = resample(key, samples, log_weights)
```
