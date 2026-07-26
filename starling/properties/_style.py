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

Physical axes and the aspect ratio
----------------------------------
DFXM maps are NOT square-pixel images. ID03's vertical scattering geometry
foreshortens the along-beam direction (array axis 0) by ``sin(2theta)``, so a
map drawn on square pixels is compressed ~2.9x vertically at a typical
2theta of 20 deg. Pass the ``(dy, dx)`` pair from
:attr:`starling.DataSet.pixel_size_yx_um` (or
:func:`starling.io.sample_pixel_size`) as ``pixel_size=`` and
:func:`imshow_map` builds the matching ``extent``; matplotlib's default
``aspect='equal'`` then renders true sample proportions::

    imshow_map(ax, strain, pixel_size=dset.pixel_size_yx_um, unit='um')
    add_scalebar(ax)

A scalar ``pixel_size`` is accepted and treated as isotropic — correct only
when you have deliberately turned the aspect correction off.
"""

import numpy as np

from ._diagnostics import EDGE_CLIPPED, FAILED, NO_SIGNAL, OK

_UNIT_PER_UM = {"um": 1.0, "µm": 1.0, "mm": 1e-3, "nm": 1e3}

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


def pixel_pair_um(pixel_size):
    """Normalise ``pixel_size`` to a ``(dy, dx)`` float pair in micrometres.

    Accepts the ``(dy, dx)`` tuple from
    :attr:`starling.DataSet.pixel_size_yx_um`, or a scalar (treated as
    isotropic — no aspect correction).

    Args:
        pixel_size: scalar or length-2 sequence, micrometres.

    Returns:
        tuple: ``(dy, dx)`` floats.

    Raises:
        ValueError: on a non-positive/non-finite size or a wrong-length
            sequence — a silently bad scale is worse than a traceback.
    """
    a = np.asarray(pixel_size, dtype=float).ravel()
    if a.size == 1:
        dy = dx = float(a[0])
    elif a.size == 2:
        dy, dx = float(a[0]), float(a[1])
    else:
        raise ValueError(
            f"pixel_size must be a scalar or (dy, dx), got {pixel_size!r}"
        )
    if not (np.isfinite(dy) and np.isfinite(dx)) or dy <= 0 or dx <= 0:
        raise ValueError(f"pixel_size must be finite and positive, got {pixel_size!r}")
    return dy, dx


def map_extent(shape, pixel_size, unit="um", centre=False):
    """``extent=`` for ``imshow`` putting a map on physical sample axes.

    Args:
        shape: the map's ``(ny, nx)`` (extra trailing dims ignored, so an
            RGB map's shape works directly).
        pixel_size: ``(dy, dx)`` in micrometres, or an isotropic scalar. Pass
            :attr:`starling.DataSet.pixel_size_yx_um` to get the aspect right.
        unit (str): output unit for the axes — ``'um'``, ``'mm'`` or ``'nm'``.
        centre (bool): put the origin at the map centre rather than its
            top-left corner.

    Returns:
        tuple: ``(left, right, bottom, top)`` for ``origin='upper'`` (the
        matplotlib default), so row 0 stays at the top.
    """
    ny, nx = int(shape[0]), int(shape[1])
    dy, dx = pixel_pair_um(pixel_size)
    try:
        scale = _UNIT_PER_UM[unit]
    except KeyError:
        raise ValueError(
            f"unit must be one of {sorted(_UNIT_PER_UM)}, got {unit!r}"
        ) from None
    h, w = ny * dy * scale, nx * dx * scale
    if centre:
        return (-w / 2, w / 2, h / 2, -h / 2)
    return (0.0, w, h, 0.0)


def _nice_length(span):
    """A round number a bit under a quarter of ``span`` (1/2/5 x 10^n)."""
    target = span / 4.0
    if not np.isfinite(target) or target <= 0:
        return None
    decade = 10.0 ** np.floor(np.log10(target))
    for mult in (5.0, 2.0, 1.0):
        if mult * decade <= target:
            return mult * decade
    return decade


def add_scalebar(ax, length=None, unit="um", pixel_size=None, color="white",
                 pad=0.06, thickness=0.02, fontsize=8):
    """Draw a horizontal scale bar in the corner of a map axes.

    Works in two modes:

    * the axes already carries physical data coordinates (you passed
      ``pixel_size=`` to :func:`imshow_map`, or an explicit ``extent``) —
      leave ``pixel_size`` None here and the bar is measured off the x limits;
    * the axes is in raw pixel coordinates — pass ``pixel_size`` and the bar
      converts through it.

    Only the horizontal pixel size is ever used, since the bar is horizontal;
    on ID03 that is the axis the ``sin(2theta)`` correction leaves alone.

    Args:
        ax: matplotlib Axes holding an image.
        length (float): bar length in ``unit``; None picks a round value near
            a quarter of the field of view.
        unit (str): ``'um'``, ``'mm'`` or ``'nm'``.
        pixel_size: ``(dy, dx)`` or scalar in micrometres — only when the axes
            is in pixel coordinates.
        color: bar/label colour.
        pad (float): margin from the axes corner, in axes fractions.
        thickness (float): bar height, in axes fractions.
        fontsize (float): label size.

    Returns:
        the bar Rectangle, or None when it could not be drawn (degenerate
        scale, or a bar that would not fit).
    """
    from matplotlib.patches import Rectangle

    x0, x1 = ax.get_xlim()
    span = abs(float(x1) - float(x0))
    if pixel_size is not None:
        _, dx = pixel_pair_um(pixel_size)
        try:
            scale = _UNIT_PER_UM[unit]
        except KeyError:
            raise ValueError(
                f"unit must be one of {sorted(_UNIT_PER_UM)}, got {unit!r}"
            ) from None
        span *= dx * scale        # x limits were in pixels
    if not np.isfinite(span) or span <= 0:
        return None

    if length is None:
        length = _nice_length(span)
        if length is None:
            return None
    frac = length / span
    if not np.isfinite(frac) or frac <= 0 or frac > 1 - 2 * pad:
        return None

    label = f"{length:g} {unit}" if length >= 1 else f"{length:.3g} {unit}"
    bx, by = 1 - pad - frac, pad + thickness + 0.035
    bar = Rectangle((bx, by), frac, thickness, transform=ax.transAxes,
                    color=color, zorder=5, clip_on=False)
    ax.add_patch(bar)
    ax.text(bx + frac / 2, by - 0.015, label, transform=ax.transAxes,
            color=color, ha="center", va="top", fontsize=fontsize, zorder=5)
    return bar


def imshow_map(ax, arr, fit_status=None, cmap="viridis", vmin=None, vmax=None,
               colorbar=True, cbar_label=None, show_clamped=None,
               edge_outline=True, extent=None, refit_mask=None,
               pixel_size=None, unit="um", **kw):
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
        extent: passed to imshow verbatim; wins over ``pixel_size``.
        pixel_size: ``(dy, dx)`` in micrometres (e.g.
            ``dset.pixel_size_yx_um``) — builds ``extent`` via
            :func:`map_extent` so the map renders in true sample proportions
            instead of on square pixels. A scalar is treated as isotropic.
        unit (str): axis unit when ``pixel_size`` is given ('um'/'mm'/'nm').
        **kw: forwarded to ``ax.imshow``.

    Returns:
        the AxesImage.
    """
    import matplotlib.pyplot as plt

    if extent is None and pixel_size is not None:
        extent = map_extent(np.shape(arr), pixel_size, unit=unit)

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
