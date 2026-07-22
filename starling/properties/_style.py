"""Shared display conventions for starling maps.

One place for the colormap/masking boilerplate the notebooks used to
hand-roll per cell: off-grain pixels render light grey (not invisible
white-on-white), edge-clipped pixels are outlined so "scan range clipped the
peak" is visually distinct from "no grain here", and every map gets a
colorbar. All matplotlib imports are lazy so ``import starling`` stays light.

Default colormaps (darfix-parity where sensible)::

    DEFAULT_CMAPS = {
        "strain": "RdBu_r",   # diverging, centred
        "com": "RdBu_r",
        "fwhm": "viridis",
        "mosaicity": "magma",  # darfix parity (was 'plasma')
        "amplitude": "hot",
        "kam": "magma",
    }

Override per call (``imshow_map(..., cmap=...)``) or rebind the dict entry in
a notebook config cell.
"""

import numpy as np

from ._diagnostics import EDGE_CLIPPED, FAILED, NO_SIGNAL, OK

DEFAULT_CMAPS = {
    "strain": "RdBu_r",
    "com": "RdBu_r",
    "fwhm": "viridis",
    "mosaicity": "magma",
    "amplitude": "hot",
    "kam": "magma",
}

BAD_COLOR = "0.85"       # off-grain / NaN: light grey (visible on white)
FAILED_COLOR = "0.55"    # failed fits: darker grey
EDGE_OUTLINE = "#ff7f0e" # edge-clipped region outline: orange


def status_cmap(base_cmap="viridis", bad_color=BAD_COLOR):
    """A copy of ``base_cmap`` with NaN pixels rendered ``bad_color``.

    Args:
        base_cmap (str or Colormap): matplotlib colormap (name ok).
        bad_color: any matplotlib color for NaN/masked pixels.

    Returns:
        matplotlib.colors.Colormap
    """
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(base_cmap).copy()
    cmap.set_bad(bad_color)
    return cmap


def masked_for_display(arr, ok):
    """Float copy of ``arr`` with pixels outside ``ok`` set to NaN."""
    out = np.array(arr, dtype=float, copy=True)
    out[~np.asarray(ok, dtype=bool)] = np.nan
    return out


def robust_limits(arr, ok=None, n_sigma=3.0, symmetric=False):
    """(vmin, vmax) as median +/- n_sigma * std over the valid pixels.

    Args:
        arr: map array.
        ok: bool mask of valid pixels (default: finite values).
        n_sigma: half-width of the window in robust stds.
        symmetric: force a symmetric window about zero (diverging maps).

    Returns:
        tuple: (vmin, vmax); (0, 1) fallback when nothing is valid.
    """
    a = np.asarray(arr, dtype=float)
    sel = np.isfinite(a) if ok is None else (np.asarray(ok, dtype=bool) & np.isfinite(a))
    if not sel.any():
        return 0.0, 1.0
    v = a[sel]
    centre = float(np.median(v))
    spread = max(float(np.std(v)), 1e-12)
    if symmetric:
        half = abs(centre) + n_sigma * spread
        return -half, half
    return centre - n_sigma * spread, centre + n_sigma * spread


def imshow_map(ax, arr, fit_status=None, cmap="viridis", vmin=None, vmax=None,
               colorbar=True, cbar_label=None, show_clamped=None,
               edge_outline=True, extent=None, refit_mask=None, **kw):
    """Standard starling map display with fit-status-aware rendering.

    Rendering convention (when ``fit_status`` is given):

    * ``OK`` — normal colormap.
    * ``EDGE_CLIPPED`` — shown with the value in ``show_clamped`` (e.g. a
      constrained-refit or range-clamped estimate) when provided, else hidden
      like FAILED; the region is outlined in orange either way.
    * ``FAILED`` — flat darker grey, UNLESS the pixel is in ``refit_mask``
      (a replacement estimate exists): then it displays its ``show_clamped``
      value and joins the orange outline. Dark grey therefore means "no
      value at all", never "value hidden by classification".
    * ``NO_SIGNAL`` — light grey background.

    Without ``fit_status``, NaN pixels render light grey.

    Args:
        ax: matplotlib Axes.
        arr: (ny, nx) map.
        fit_status: optional (ny, nx) int map from ``classify_fit_status``.
        cmap: base colormap name (or a key of ``DEFAULT_CMAPS``).
        vmin, vmax: color limits (None = matplotlib auto).
        colorbar (bool): attach a colorbar.
        cbar_label (str): colorbar label.
        show_clamped: optional (ny, nx) values to display at EDGE_CLIPPED
            pixels (from ``clamp_edge_estimate``).
        edge_outline (bool): outline the EDGE_CLIPPED region.
        extent: passed to imshow (e.g. mm axes).
        **kw: forwarded to ``ax.imshow``.

    Returns:
        the AxesImage.
    """
    import matplotlib.pyplot as plt

    cmap = DEFAULT_CMAPS.get(cmap, cmap)
    cm = status_cmap(cmap)

    rmask = (np.zeros(np.asarray(arr).shape, bool) if refit_mask is None
             else np.asarray(refit_mask, dtype=bool))
    disp = np.array(arr, dtype=float, copy=True)
    if fit_status is not None:
        st = np.asarray(fit_status)
        disp[st == NO_SIGNAL] = np.nan
        disp[st == FAILED] = np.nan
        if show_clamped is not None:
            fill = (st == EDGE_CLIPPED) | rmask
            disp[fill] = np.asarray(show_clamped, dtype=float)[fill]
        else:
            disp[st == EDGE_CLIPPED] = np.nan

    im = ax.imshow(disp, cmap=cm, vmin=vmin, vmax=vmax, extent=extent, **kw)

    if fit_status is not None:
        st = np.asarray(fit_status)
        # dark grey ONLY where no replacement value exists
        failed = (st == FAILED) & ~rmask
        if failed.any():
            overlay = np.zeros((*st.shape, 4))
            overlay[failed] = plt.matplotlib.colors.to_rgba(FAILED_COLOR)
            ax.imshow(overlay, extent=extent, interpolation="nearest")
        outlined = (st == EDGE_CLIPPED) | rmask
        if edge_outline and outlined.any():
            ax.contour(
                outlined.astype(float),
                levels=[0.5],
                colors=[EDGE_OUTLINE],
                linewidths=0.8,
                extent=extent,
            )
    if colorbar:
        fig = ax.figure
        fig.colorbar(im, ax=ax, shrink=0.85, label=cbar_label)
    return im


def status_legend(ax, fit_status=None, loc="lower right"):
    """Small proxy legend explaining the status rendering on a map axes.

    Entries for FAILED/EDGE_CLIPPED are included only when present in
    ``fit_status`` (or always, when ``fit_status`` is None).
    """
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    st = None if fit_status is None else np.asarray(fit_status)
    handles = []
    if st is None or (st == NO_SIGNAL).any():
        handles.append(mpatches.Patch(color=BAD_COLOR, label="no grain signal"))
    if st is None or (st == EDGE_CLIPPED).any():
        handles.append(
            Line2D([], [], color=EDGE_OUTLINE, label="peak clipped by scan range")
        )
    if st is None or (st == FAILED).any():
        handles.append(mpatches.Patch(color=FAILED_COLOR, label="fit failed"))
    if handles:
        ax.legend(handles=handles, loc=loc, fontsize=7, framealpha=0.9)
