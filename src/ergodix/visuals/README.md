# visuals subpackage

This subpackage provides animation utilities to **visualize the output of a sampler over time** —
typically the trajectory returned by `slips(..., return_history=True)`. Samples are drawn as a
density histogram in 1D and as a scatter cloud in 2D, with an optional probability density function
(pdf) overlaid and normalized so that its **total mass sums to 1**. Every function can save its
animation to a GIF.

It provides the user with 3 functions:

- `animate_samples`: animate 1D or 2D samples over time, with an optional pdf overlay and an optional second panel for the particles (the `X_hist` of SLIPS), drawn next to the samples in the same style.
- `animate_projection`: project N-dimensional samples onto a line (one vector) or a plane (two vectors), then animate the projection in 1D or 2D.
- `compare_samples`: compare several sets of samples — or `(samples, particles)` pairs — side by side, with time advancing at the same speed for all of them. The particles are shown in a separate window in the same style.

A few conventions shared by all three functions:

- **Input layout matches SLIPS.** Pass `Y_hist` of shape `(n_samples, n_steps, dim)` as `samples`, and `X_hist` of shape `(n_samples, n_steps, n_chains, dim)` as `particles` — straight from `slips(..., return_history=True)`.
- **A "pdf" is a log-density**, exactly like the log-density `target` you pass to `slips` (any `TargetDistribution` works). It is exponentiated and normalized internally. Pass `pdf_is_log=False` to supply an already-linear density instead.
- Each function returns a `matplotlib.animation.FuncAnimation`. Pass `save_path="....gif"` to write a GIF (via Pillow), or `"....mp4"` for a video (requires ffmpeg). To display interactively instead, call `matplotlib.pyplot.show()`.


# Example usage of functions from this subpackage

## Producing some data to visualize

```python
import jax
import jax.numpy as jnp
from ergodix.slips import slips, SLIPSParams, GeomSchedule
from ergodix.distributions import IsotropicGMM
from ergodix.visuals import animate_samples, animate_projection, compare_samples

key = jax.random.PRNGKey(0)
dim = 2

# A bimodal target distribution
target = IsotropicGMM(
    weights=jnp.array([0.5, 0.5]),
    means=jnp.array([[-2.5, 0.0], [2.5, 0.0]]),
    variances=jnp.array([0.25, 0.25]),
)

schedule = GeomSchedule()
time_grid = schedule.get_snr_grid(t_0=0.1, t_end=0.9, steps=40)

# return_history=True is what gives us the full trajectory to animate
params = SLIPSParams(sigma=1.0, schedule=schedule, return_history=True)
_, y_hist, x_hist = slips(key, target, time_grid, 300, dim, params)
# y_hist: (300, n_steps, 2)            -> samples over time
# x_hist: (300, n_steps, n_chains, 2)  -> particles over time
```

## 1) `animate_samples`: samples (and particles) over time

```python
# Samples on the left, the auxiliary MALA particles on the right, the target pdf
# drawn as contour level lines, saved as a GIF.
anim = animate_samples(
    y_hist,
    pdf=target,
    particles=x_hist,
    time_grid=time_grid,   # used to label the frames with the SLIPS time t
    save_path="slips.gif",
)
```

In 1D, the samples are shown as a density histogram beneath the pdf curve:

```python
# y_hist_1d has shape (n_samples, n_steps, 1); target_1d is a 1D distribution
anim = animate_samples(y_hist_1d, pdf=target_1d)
```

## 2) `animate_projection`: looking at high-dimensional samples

```python
# y_hist_hd has shape (n_samples, n_steps, 10).

# Project onto a plane (the first two canonical axes) and overlay a density slice:
directions = jnp.stack([jnp.eye(10)[0], jnp.eye(10)[1]])
anim = animate_projection(y_hist_hd, directions, pdf=target_hd, save_path="proj.gif")

# Or project onto a single direction (a line) for a 1D view:
anim = animate_projection(y_hist_hd, jnp.ones(10))
```

## 3) `compare_samples`: comparing several runs

```python
# Compare two SLIPS runs (e.g. different sigma) against the same target.
params_a = SLIPSParams(sigma=1.0, schedule=schedule, return_history=True)
params_b = SLIPSParams(sigma=5.0, schedule=schedule, return_history=True)
_, y_a, x_a = slips(key, target, time_grid, 300, dim, params_a)
_, y_b, x_b = slips(key, target, time_grid, 300, dim, params_b)

# Samples are compared in one window, particles in a second window, on a shared clock.
samples_anim, particles_anim = compare_samples(
    [(y_a, x_a), (y_b, x_b)],
    pdfs=target,                     # one pdf shared by every panel (or a list, one per set)
    labels=["sigma = 1", "sigma = 5"],
    save_path="compare.gif",         # also writes compare_particles.gif
)
```
