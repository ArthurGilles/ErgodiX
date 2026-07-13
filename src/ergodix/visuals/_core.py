"""
Internal rendering engine shared by every public function in ``ergodix.visuals``.

This module is **private**: nothing here is part of the public API. The three
user-facing functions (:func:`~ergodix.visuals.animate_samples`,
:func:`~ergodix.visuals.animate_projection` and
:func:`~ergodix.visuals.compare_samples`) all reduce their input to a list of
:class:`_Panel` objects and hand them to :func:`_render`, so that limit
computation, pdf normalisation, frame synchronisation and GIF export behave
identically everywhere.

Conventions
-----------
- Samples are expected in the **SLIPS layout** ``(n_samples, n_steps, dim)`` so
  that the ``Y_hist`` returned by :func:`ergodix.slips.slips` can be passed
  straight through. Internally everything is converted to the **time-major**
  layout ``(n_steps, n_points, dim)`` so that frame ``t`` is a single slice.
- A "pdf" is a callable mapping a point of shape ``(dim,)`` to a scalar. By
  default it is interpreted as a **log**-density (exactly like the log-density
  ``target`` passed to :func:`ergodix.slips.slips`); it is exponentiated and
  **numerically normalised to total mass 1** on the plotting grid.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

try:  # numpy >= 2.0 (a declared dependency) exposes ``trapezoid``.
    from numpy import trapezoid as _trapezoid
except ImportError:  # pragma: no cover - fallback for older numpy.
    from numpy import trapz as _trapezoid


# --------------------------------------------------------------------------- #
# Panel description
# --------------------------------------------------------------------------- #
@dataclass
class _Panel:
    """A single animated subplot.

    Parameters
    ----------
    data: np.ndarray
        Time-major array of shape ``(n_steps, n_points, dim)`` with ``dim`` in
        ``{1, 2}``.
    title: str
        Title displayed above the subplot.
    pdf: Callable | None
        Optional density callable overlaid on the samples (see module docstring).
    pdf_is_log: bool
        Whether ``pdf`` returns a log-density (default) or a linear density.
    """

    data: np.ndarray
    title: str
    pdf: Optional[Callable] = None
    pdf_is_log: bool = True


# --------------------------------------------------------------------------- #
# Shape handling
# --------------------------------------------------------------------------- #
def _to_time_major(samples, name: str = "samples") -> np.ndarray:
    """Validate a ``(n_samples, n_steps, dim)`` array and make it time-major.

    Returns an array of shape ``(n_steps, n_samples, dim)``.
    """
    arr = np.asarray(samples, dtype=float)
    if arr.ndim != 3:
        raise ValueError(
            f"`{name}` must have shape (n_samples, n_steps, dim) - i.e. the "
            f"`Y_hist` returned by `slips(..., return_history=True)`; got an "
            f"array with {arr.ndim} dimension(s) and shape {arr.shape}."
        )
    return np.transpose(arr, (1, 0, 2))


def _particles_to_time_major(particles) -> np.ndarray:
    """Make a particle history time-major, flattening the chain axis.

    Accepts the SLIPS ``X_hist`` of shape
    ``(n_samples, n_steps, n_chains, dim)`` - reshaped per frame into a
    ``(n_samples * n_chains, dim)`` cloud - or an already-flat
    ``(n_samples, n_steps, dim)`` array.
    """
    arr = np.asarray(particles, dtype=float)
    if arr.ndim == 4:
        n_samples, n_steps, n_chains, dim = arr.shape
        arr = np.transpose(arr, (1, 0, 2, 3))
        return arr.reshape(n_steps, n_samples * n_chains, dim)
    if arr.ndim == 3:
        return _to_time_major(arr, name="particles")
    raise ValueError(
        "`particles` must have shape (n_samples, n_steps, n_chains, dim) - i.e. "
        "the `X_hist` returned by `slips(..., return_history=True)` - or "
        f"(n_samples, n_steps, dim); got shape {arr.shape}."
    )


# --------------------------------------------------------------------------- #
# Axis limits
# --------------------------------------------------------------------------- #
def _pad(lo: float, hi: float, margin: float) -> Tuple[float, float]:
    span = hi - lo
    if not np.isfinite(span) or span == 0.0:
        span = 1.0
    pad = margin * span
    return (lo - pad, hi + pad)


def _finite(values: np.ndarray) -> np.ndarray:
    values = values.ravel()
    return values[np.isfinite(values)]


def _global_limits(data_list: Sequence[np.ndarray], dim: int, margin: float = 0.08):
    """Shared axis limits across every panel so the view never jumps."""
    xs = _finite(np.concatenate([d[..., 0].ravel() for d in data_list]))
    if xs.size == 0:
        xs = np.array([0.0, 1.0])
    xlim = _pad(float(xs.min()), float(xs.max()), margin)
    if dim == 1:
        return xlim, None
    ys = _finite(np.concatenate([d[..., 1].ravel() for d in data_list]))
    if ys.size == 0:
        ys = np.array([0.0, 1.0])
    ylim = _pad(float(ys.min()), float(ys.max()), margin)
    return xlim, ylim


# --------------------------------------------------------------------------- #
# PDF evaluation + normalisation
# --------------------------------------------------------------------------- #
def _weights_from_pdf(pdf: Callable, points: jnp.ndarray, is_log: bool) -> np.ndarray:
    """Evaluate ``pdf`` on a grid and return non-negative, unnormalised weights."""
    values = jnp.asarray(jax.vmap(pdf)(points)).reshape(-1)
    if is_log:
        weights = jnp.exp(values - jnp.max(values))  # subtract max for stability
    else:
        weights = jnp.maximum(values, 0.0)
    return np.asarray(weights)


def _density_1d(pdf: Callable, xlim, grid_size: int, is_log: bool):
    """Return ``(xs, density)`` with ``density`` integrating to 1 over ``xlim``."""
    xs = np.linspace(xlim[0], xlim[1], grid_size)
    weights = _weights_from_pdf(pdf, jnp.asarray(xs)[:, None], is_log)
    area = float(_trapezoid(weights, xs))
    if not np.isfinite(area) or area <= 0.0:
        return xs, np.zeros_like(weights)
    return xs, weights / area


def _density_2d(pdf: Callable, xlim, ylim, grid_size: int, is_log: bool):
    """Return ``(XX, YY, ZZ)`` with ``ZZ`` integrating to 1 over the grid."""
    xs = np.linspace(xlim[0], xlim[1], grid_size)
    ys = np.linspace(ylim[0], ylim[1], grid_size)
    XX, YY = np.meshgrid(xs, ys)
    points = jnp.stack([jnp.asarray(XX).ravel(), jnp.asarray(YY).ravel()], axis=-1)
    weights = _weights_from_pdf(pdf, points, is_log)
    cell = float(xs[1] - xs[0]) * float(ys[1] - ys[0])
    mass = float(weights.sum()) * cell
    if not np.isfinite(mass) or mass <= 0.0:
        return XX, YY, np.zeros((grid_size, grid_size))
    return XX, YY, (weights / mass).reshape(grid_size, grid_size)


def _contour_levels(density: np.ndarray, levels):
    """Choose contour level lines for a 2D density.

    When ``levels`` is an integer, the lines are placed at evenly spaced
    *fractions of the peak density* (excluding the near-zero floor). This keeps
    the contours hugging the modes instead of drawing a spurious outer ring where
    a sharply peaked density meets the wide, sample-driven axis range. A sequence
    of explicit levels is passed through unchanged.
    """
    if not isinstance(levels, int):
        return levels
    zmax = float(density.max())
    if not np.isfinite(zmax) or zmax <= 0.0:
        return None
    return np.linspace(zmax / (levels + 1), zmax, levels)


# --------------------------------------------------------------------------- #
# Saving
# --------------------------------------------------------------------------- #
_VIDEO_SUFFIXES = (".mp4", ".mov", ".m4v", ".webm", ".avi")


def _save_animation(anim: animation.FuncAnimation, save_path, fps: int) -> None:
    """Write ``anim`` to ``save_path``; the writer is chosen from the suffix."""
    path = Path(save_path)
    suffix = path.suffix.lower()
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".gif":
        writer = animation.PillowWriter(fps=fps)
    elif suffix in _VIDEO_SUFFIXES:
        writer = animation.FFMpegWriter(fps=fps)
    else:
        raise ValueError(
            f"Unsupported save format '{suffix}'. Use '.gif' (saved with Pillow) "
            f"or a video format such as '.mp4' (requires ffmpeg)."
        )
    anim.save(str(path), writer=writer)


# --------------------------------------------------------------------------- #
# Time labelling
# --------------------------------------------------------------------------- #
def _frame_label(time_grid, t: int, n_steps: int) -> str:
    if time_grid is not None:
        grid = np.asarray(time_grid).ravel()
        if grid.size >= n_steps:
            return f"t = {float(grid[t]):.3g}"
    return f"step {t + 1} / {n_steps}"


# --------------------------------------------------------------------------- #
# Core renderer
# --------------------------------------------------------------------------- #
def _render(
    panels: List[_Panel],
    *,
    fps: int = 10,
    save_path=None,
    figsize=None,
    time_grid=None,
    bins: int = 60,
    levels: int = 8,
    grid_size: int = 200,
) -> Tuple[plt.Figure, animation.FuncAnimation]:
    """Build a time-synchronised animation from a list of :class:`_Panel`.

    Every panel is drawn in its own subplot on a single row; one shared frame
    clock drives them all. Returns ``(figure, animation)``.
    """
    if not panels:
        raise ValueError("`_render` requires at least one panel.")

    dims = {p.data.shape[-1] for p in panels}
    if len(dims) != 1:
        raise ValueError(
            f"All panels must share the same dimension; got dimensions {sorted(dims)}."
        )
    dim = dims.pop()
    if dim not in (1, 2):
        raise ValueError(
            f"Can only animate 1D or 2D data (got dim={dim}). Project higher "
            f"dimensional samples first with `animate_projection`."
        )

    step_counts = {p.data.shape[0] for p in panels}
    if len(step_counts) != 1:
        raise ValueError(
            "All panels must share the same number of time steps; got "
            f"{sorted(step_counts)}."
        )
    n_steps = step_counts.pop()

    xlim, ylim = _global_limits([p.data for p in panels], dim)

    # Pre-compute the (static) normalised pdf grids on the shared limits.
    densities = []
    for p in panels:
        if p.pdf is None:
            densities.append(None)
        elif dim == 1:
            densities.append(_density_1d(p.pdf, xlim, grid_size, p.pdf_is_log))
        else:
            densities.append(_density_2d(p.pdf, xlim, ylim, grid_size, p.pdf_is_log))

    n = len(panels)
    if figsize is None:
        figsize = (5.4 * n, 5.0) if dim == 2 else (6.4 * n, 4.2)
    fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)
    axes = axes[0]

    states = []  # per-panel artists/data needed by ``update``
    if dim == 1:
        edges = np.linspace(xlim[0], xlim[1], bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        width = edges[1] - edges[0]

        # Pre-compute every histogram so the y-axis can be fixed up front.
        per_panel_counts = [
            np.stack(
                [np.histogram(p.data[t, :, 0], bins=edges, density=True)[0]
                 for t in range(n_steps)]
            )
            for p in panels
        ]
        ymax = max((c.max() for c in per_panel_counts if c.size), default=1.0)
        for d in densities:
            if d is not None and d[1].size:
                ymax = max(ymax, float(d[1].max()))
        ylim = (0.0, ymax * 1.15 if ymax > 0 else 1.0)

        for i, (p, ax) in enumerate(zip(panels, axes)):
            bars = ax.bar(
                centers, per_panel_counts[i][0], width=width, align="center",
                color="steelblue", alpha=0.65, edgecolor="white", linewidth=0.4,
            )
            if densities[i] is not None:
                xs, dens = densities[i]
                ax.plot(xs, dens, color="crimson", lw=2.0, label="pdf")
                ax.legend(loc="upper right", fontsize=9, frameon=False)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_xlabel("x")
            ax.set_ylabel("density")
            ax.set_title(p.title)
            ax.grid(True, alpha=0.15)
            states.append(("1d", bars, per_panel_counts[i]))
    else:
        for i, (p, ax) in enumerate(zip(panels, axes)):
            data0 = p.data[0]
            scat = ax.scatter(
                data0[:, 0], data0[:, 1], s=14, alpha=0.5,
                color="steelblue", edgecolors="none",
            )
            if densities[i] is not None:
                XX, YY, ZZ = densities[i]
                contour_levels = _contour_levels(ZZ, levels)
                if contour_levels is not None:
                    ax.contour(
                        XX, YY, ZZ, levels=contour_levels,
                        cmap="autumn", linewidths=1.2,
                    )
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")
            ax.set_title(p.title)
            ax.grid(True, alpha=0.15)
            states.append(("2d", scat, p.data))

    suptitle = fig.suptitle(_frame_label(time_grid, 0, n_steps))
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    def update(t: int):
        for kind, artist, payload in states:
            if kind == "1d":
                for bar, height in zip(artist, payload[t]):
                    bar.set_height(height)
            else:
                artist.set_offsets(payload[t])
        suptitle.set_text(_frame_label(time_grid, t, n_steps))
        return []

    anim = animation.FuncAnimation(
        fig, update, frames=n_steps, interval=1000.0 / fps, blit=False,
    )

    if save_path is not None:
        _save_animation(anim, save_path, fps)

    return fig, anim
