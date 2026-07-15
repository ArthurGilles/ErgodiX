# metrics subpackage

This subpackage contains JAX implementations of *sliced* statistical distances,
used to compare two sets of samples (e.g. the output of one of the samplers
against ground-truth samples of the target).

Both metrics project the point clouds onto random 1D directions $\theta$ drawn
uniformly on the unit sphere, compute a cheap 1D distance along each direction,
and aggregate over the projections. This keeps them tractable in high dimension,
where the exact multivariate distances are expensive (Wasserstein) or hard to
define (Kolmogorov-Smirnov).

It provides the user with 2 functions:

- `sliced_wasserstein_mc`: Monte Carlo estimate of the order-$p$ Sliced
  Wasserstein distance between two point clouds of the same shape $(N, d)$.
  Returns the distance together with the variance of the 1D distances across
  projections (a diagnostic of the Monte Carlo error).
- `sliced_kolmogorov_smirnov`: Sliced Kolmogorov-Smirnov distance between two
  point clouds of shapes $(N, d)$ and $(M, d)$ (sample sizes may differ). The
  result lies in $[0, 1]$. The 1D KS statistics are aggregated with an $L^p$
  mean, or with a supremum if `max_sliced=True` (Max-Sliced KS distance).

Both functions are pure and JIT-compiled; `num_projections` and `p` are static
arguments.

# Example usage of functions from this subpackage

```python
import jax
from ergodix.metrics import sliced_wasserstein_mc, sliced_kolmogorov_smirnov

# Define JAX random keys (data, and one per metric for the projections)
key = jax.random.PRNGKey(42)
key_x, key_y, key_swd, key_sks = jax.random.split(key, 4)

# Two point clouds to compare, e.g. sampler output vs ground-truth samples
x = jax.random.normal(key_x, (1024, 2))
y = jax.random.normal(key_y, (1024, 2)) + 1.0

# Sliced Wasserstein distance (order p), with its Monte Carlo variance
swd, swd_var = sliced_wasserstein_mc(key_swd, x, y, num_projections=4096, p=2)

# Sliced Kolmogorov-Smirnov distance, in [0, 1]
sks = sliced_kolmogorov_smirnov(x, y, key_sks, num_projections=50)

# Max-Sliced variant (supremum over the sampled projections)
max_sks = sliced_kolmogorov_smirnov(x, y, key_sks, max_sliced=True)

print(swd, sks, max_sks)
```
