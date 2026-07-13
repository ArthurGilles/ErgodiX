"""1D / 2D animation of samples (and optionally particles) over time."""

from typing import Callable, Optional, Tuple, Union

from matplotlib import animation

from ..distributions import TargetDistribution
from ._core import _Panel, _particles_to_time_major, _render, _to_time_major

PDF = Union[TargetDistribution, Callable]


def animate_samples(
    samples,
    *,
    pdf: Optional[PDF] = None,
    particles=None,
    particles_pdf: Optional[PDF] = None,
    pdf_is_log: bool = True,
    time_grid=None,
    bins: int = 60,
    levels: int = 8,
    grid_size: int = 200,
    fps: int = 10,
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    titles: Tuple[str, str] = ("Samples", "Particles"),
) -> animation.FuncAnimation:
    """
    Animate 1D or 2D samples evolving over time, with an optional pdf overlay.

    The expected input is exactly the ``Y_hist`` returned by
    :func:`ergodix.slips.slips` when ``return_history=True``, so the SLIPS output
    can be passed straight through. In 1D the samples are shown as a
    density-normalised histogram; in 2D as a scatter cloud. If ``particles`` is
    given (the SLIPS ``X_hist``), a second, time-synchronised panel is drawn next
    to the samples in the same style.

    Parameters
    ----------
        samples: jax.Array
            Samples over time of shape ``(n_samples, n_steps, dim)`` with
            ``dim`` in ``{1, 2}``. For higher dimensions use
            :func:`ergodix.visuals.animate_projection`.
        pdf: TargetDistribution or Callable, optional
            Density overlaid on the samples. By default it is interpreted as a
            **log**-density (like the log-density ``target`` of `slips`, e.g. any
            `TargetDistribution`); it is exponentiated and normalised so that the
            displayed mass integrates to 1. Pass ``pdf_is_log=False`` to supply a
            linear density instead. It must accept a point of shape ``(dim,)`` and
            be ``jax.vmap``-compatible.
        particles: jax.Array, optional
            Particle history drawn in a second panel. Accepts the SLIPS
            ``X_hist`` of shape ``(n_samples, n_steps, n_chains, dim)`` or a flat
            ``(n_samples, n_steps, dim)`` array.
        particles_pdf: TargetDistribution or Callable, optional
            Optional density overlaid on the particle panel.
        pdf_is_log: bool, default True
            Whether ``pdf`` and ``particles_pdf`` return log-densities.
        time_grid: jax.Array, optional
            Time points used to label the frames; if omitted a step index is
            shown instead.
        bins: int, default 60
            Number of histogram bins (1D only).
        levels: int, default 8
            Number of pdf contour lines (2D only).
        grid_size: int, default 200
            Resolution of the grid on which the pdf is evaluated.
        fps: int, default 10
            Frames per second of the animation / saved GIF.
        save_path: str, optional
            If given, the animation is written here. ``.gif`` is saved with
            Pillow; video formats such as ``.mp4`` require ffmpeg.
        figsize: tuple, optional
            Matplotlib figure size; a sensible default scales with the number of
            panels.
        titles: tuple of str, default ("Samples", "Particles")
            Titles for the samples panel and (if any) the particles panel.
    Returns
    -------
    anim: matplotlib.animation.FuncAnimation
        The animation. Call `matplotlib.pyplot.show()` to display it, or pass
        ``save_path`` to write it to disk.

    Example
    -------
    >>> import jax.numpy as jnp
    >>> from ergodix.slips import slips, SLIPSParams, GeomSchedule
    >>> from ergodix.distributions import IsotropicGMM
    >>> from ergodix.visuals import animate_samples
    >>> target = IsotropicGMM(weights=jnp.array([0.5, 0.5]),
    ...                       means=jnp.array([[-2.0, 0.0], [2.0, 0.0]]),
    ...                       variances=jnp.array([0.3, 0.3]))
    >>> schedule = GeomSchedule()
    >>> grid = schedule.get_snr_grid(t_0=0.1, t_end=0.9, steps=40)
    >>> params = SLIPSParams(sigma=1.0, schedule=schedule, return_history=True)
    >>> _, y_hist, x_hist = slips(jax.random.PRNGKey(0), target, grid, 256, 2, params)
    >>> anim = animate_samples(y_hist, pdf=target, particles=x_hist,
    ...                        time_grid=grid, save_path="slips.gif")
    """
    sample_frames = _to_time_major(samples, name="samples")
    dim = sample_frames.shape[-1]
    if dim not in (1, 2):
        raise ValueError(
            f"`animate_samples` handles 1D or 2D samples (got dim={dim}). Use "
            f"`animate_projection` to project higher-dimensional samples onto a "
            f"line or plane."
        )

    panels = [_Panel(sample_frames, titles[0], pdf, pdf_is_log)]

    if particles is not None:
        particle_frames = _particles_to_time_major(particles)
        if particle_frames.shape[-1] != dim:
            raise ValueError(
                f"`particles` dimension ({particle_frames.shape[-1]}) must match "
                f"`samples` dimension ({dim})."
            )
        if particle_frames.shape[0] != sample_frames.shape[0]:
            raise ValueError(
                "`particles` and `samples` must have the same number of time "
                f"steps; got {particle_frames.shape[0]} and {sample_frames.shape[0]}."
            )
        panels.append(_Panel(particle_frames, titles[1], particles_pdf, pdf_is_log))

    _, anim = _render(
        panels,
        fps=fps,
        save_path=save_path,
        figsize=figsize,
        time_grid=time_grid,
        bins=bins,
        levels=levels,
        grid_size=grid_size,
    )
    return anim
