"""Orientation and mosaicity maps — the documented split.

darling/darfix conflate these two quantities and label the centre-of-mass image
"mosaicity". They are different things:

* **Orientation** is the per-pixel *mean* orientation (first moment / centre of
  mass) of the diffracted intensity. It says *which way* the lattice points.
* **Mosaicity** is the *spread* (second moment / width) of the orientation
  distribution within a pixel. It says *how disordered* the lattice is.

This module keeps them separate and names them honestly:
``orientation_map`` (first moment) and ``mosaicity`` (second moment).
"""

import numpy as np

# Gaussian FWHM = FWHM_FACTOR * sigma
FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))  # 2.35482...


def _hsv_to_rgb(h, s, v):
    """Vectorised HSV->RGB (h, s, v in [0, 1]); avoids a matplotlib dependency."""
    h = np.asarray(h, dtype=float)
    s = np.asarray(s, dtype=float)
    v = np.asarray(v, dtype=float)
    i = np.floor(h * 6.0).astype(int)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1)


def _colour_key(size=129):
    """A 2-D direction/magnitude colour wheel (hue=direction, sat=magnitude)."""
    yy, xx = np.mgrid[-1:1:size * 1j, -1:1:size * 1j]
    mag = np.sqrt(xx ** 2 + yy ** 2)
    ang = np.arctan2(yy, xx)
    hue = (ang / (2 * np.pi)) % 1.0
    sat = np.clip(mag, 0.0, 1.0)
    rgb = _hsv_to_rgb(hue, sat, np.ones_like(hue))
    rgb[mag > 1.0] = 1.0  # white outside the unit disk
    return rgb


def orientation_map(mean, axes=(0, 1), norm="dynamic", as_rgb=False, mask=None):
    """Per-pixel MEAN orientation (centre of mass) for the chosen motor axes.

    This is an **ORIENTATION** map, **NOT** mosaicity. It is the first moment of
    the intensity distribution — the mean lattice orientation per pixel — and
    carries no information about the orientation *spread* (that is
    :func:`mosaicity`).

    Optionally returns the classic darfix/darling-style HSV colour image
    (hue = orientation direction, saturation = deviation magnitude) together
    with a colour key, but it is documented explicitly as an orientation image.

    Args:
        mean (numpy.ndarray): (ny, nx) for a single motor, or (ny, nx, D) for D
            motors (e.g. the ``mean`` returned by :func:`moments` or
            ``GaussNDResult.mu``).
        axes (tuple): which orientation components to use when ``mean`` is
            (ny, nx, D). The first two selected axes drive the colour image.
        norm (str): "dynamic" scales the colour saturation by the per-map
            deviation range (robust 2-98 percentile); a float fixes the full
            scale magnitude in motor units.
        as_rgb (bool): if True, also build the colour image.
        mask (numpy.ndarray, optional): (ny, nx) bool grain mask — restricts
            the "dynamic" scale (and the deviation reference) to grain
            pixels, so off-grain outliers cannot inflate the scale and wash
            the grain out to a single colour block. Prefer
            :func:`orientation_stamp` for darfix-style fixed-range colour.

    Returns:
        If ``as_rgb`` is False: the orientation map — (ny, nx) for one axis or
        (ny, nx, k) for k selected axes.
        If ``as_rgb`` is True: ``(orientation, rgb, key)`` where ``rgb`` is the
        (ny, nx, 3) colour image and ``key`` an (S, S, 3) colour wheel.
    """
    mean = np.asarray(mean, dtype=float)
    if mean.ndim == 2:
        sel = mean
        comps = mean[..., None]
    else:
        idx = list(axes)
        comps = mean[..., idx]
        sel = comps if comps.shape[-1] > 1 else comps[..., 0]

    if not as_rgb:
        return sel

    # deviation about the map median (NaN-aware, grain-masked when given)
    k = comps.shape[-1]
    if mask is not None:
        sel_px = comps[np.asarray(mask, dtype=bool)]
        ref = np.nanmedian(sel_px.reshape(-1, k), axis=0)
    else:
        ref = np.nanmedian(comps.reshape(-1, k), axis=0)
    dev = comps - ref  # (ny, nx, k)
    if k == 1:
        dx = dev[..., 0]
        dy = np.zeros_like(dx)
    else:
        dx = dev[..., 0]
        dy = dev[..., 1]
    mag = np.sqrt(dx ** 2 + dy ** 2)
    if norm == "dynamic":
        finite = np.isfinite(mag)
        if mask is not None:
            finite &= np.asarray(mask, dtype=bool)
        scale = np.nanpercentile(mag[finite], 98) if finite.any() else 1.0
        scale = max(scale, 1e-12)
    else:
        scale = max(float(norm), 1e-12)
    ang = np.arctan2(dy, dx)
    hue = (ang / (2 * np.pi)) % 1.0
    sat = np.clip(mag / scale, 0.0, 1.0)
    val = np.ones_like(hue)
    rgb = _hsv_to_rgb(hue, sat, val)
    nan = ~np.isfinite(mag)
    rgb[nan] = 1.0  # white where there is no data
    if mask is not None:
        rgb[~np.asarray(mask, dtype=bool)] = 1.0
    return sel, rgb, _colour_key()


def orientation_stamp(mean, axes=(0, 1), vrange=None, colormap_name="hsv",
                      sat=40, mask=None, key_size=256):
    """darfix-style square 2-D colour stamp for two orientation axes.

    Unlike :func:`orientation_map` (round HSV wheel, dynamic
    percentile-of-deviation scale — which collapses to a near-uniform block
    when a few outlier pixels inflate the scale), this maps the two COM
    components through ``colorstamps.apply_stamp`` with **fixed per-axis
    vmin/vmax** — darfix's exact recipe — so the full colour square is spent
    on the grain's own orientation range.

    Display conventions (darfix parity): masked-off / non-finite pixels are
    black; in-range pixels get the stamp colour; out-of-range pixels (only
    possible with an explicit ``vrange``) are mid-grey.

    Args:
        mean (numpy.ndarray): (ny, nx, D) first-moment/COM array (or a
            (ny, nx, 2) slice), e.g. ``GaussNDResult.mu``.
        axes (tuple): the two components of ``mean`` to map (x-ish, y-ish).
        vrange (tuple, optional): ((lo0, hi0), (lo1, hi1)) fixed ranges per
            selected axis, in motor units. ``None`` uses each component's own
            finite min/max over ``mask`` (darfix default).
        colormap_name (str): colorstamps map — "hsv" (darfix default),
            "cut", "orangeBlue", "flat", ...
        sat (float): colorstamps saturation parameter (darfix default 40).
        mask (numpy.ndarray, optional): (ny, nx) bool; False pixels are
            rendered black and excluded from the auto ``vrange``.
        key_size (int): side length of the colour-key legend image.

    Returns:
        tuple: ``(rgb, color_key, vrange)`` — the (ny, nx, 3) image in
        [0, 1], the (key_size, key_size, 3) legend (plot with
        ``extent=(lo0, hi0, lo1, hi1)`` from the returned ``vrange``), and
        the ((lo0, hi0), (lo1, hi1)) ranges actually used.
    """
    import colorstamps

    mean = np.asarray(mean, dtype=float)
    if mean.ndim == 2:
        raise ValueError(
            "orientation_stamp needs two orientation components; got a "
            "single-motor (ny, nx) map"
        )
    c0 = mean[..., axes[0]].astype(float)
    c1 = mean[..., axes[1]].astype(float)

    valid = np.isfinite(c0) & np.isfinite(c1)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)

    if vrange is None:
        if not valid.any():
            raise ValueError("no valid pixels to derive the colour range from")
        vrange = (
            (float(np.min(c0[valid])), float(np.max(c0[valid]))),
            (float(np.min(c1[valid])), float(np.max(c1[valid]))),
        )
    (lo0, hi0), (lo1, hi1) = vrange
    if not (hi0 > lo0) or not (hi1 > lo1):
        raise ValueError(f"degenerate colour range {vrange}")

    # colorstamps mangles NaN input (invalid int cast) — feed range-midpoint
    # placeholders at invalid pixels and repaint them black afterwards
    f0 = np.where(valid, c0, 0.5 * (lo0 + hi0))
    f1 = np.where(valid, c1, 0.5 * (lo1 + hi1))

    rgb, stamp = colorstamps.apply_stamp(
        f0,
        f1,
        colormap_name,
        vmin_0=lo0,
        vmax_0=hi0,
        vmin_1=lo1,
        vmax_1=hi1,
        sat=sat,
        clip="none",
        l=key_size,
    )
    out_of_range = np.isnan(rgb).any(axis=-1)
    rgb = np.clip(np.nan_to_num(rgb, nan=0.2), 0.0, 1.0)
    rgb[out_of_range & valid] = 0.2  # grey: COM outside the fixed range
    rgb[~valid] = 0.0                # black: no data / off-mask
    return rgb, np.clip(np.asarray(stamp.cmap), 0.0, 1.0), vrange


def mosaicity(cov, mode="scalar", axes=None):
    """Per-pixel orientation SPREAD — the mosaicity of the grain, by definition.

    A **second-moment** quantity: the width of the per-pixel orientation
    distribution, distinct from the mean orientation returned by
    :func:`orientation_map`.

    Accepts either the covariance from :func:`moments` *or* the fitted
    covariance from ``fit_ND_gaussian`` / ``GaussNDResult.cov`` (preferred — the
    fit is less biased by the finite motor window and residual background; see
    the module/Appendix B note on window bias). Gate quantitative use on
    amplitude / SNR.

    Args:
        cov (numpy.ndarray): (ny, nx) scalar variance (1 motor), or
            (ny, nx, D, D) covariance.
        mode (str):
            ``"scalar"`` -> ``sqrt(trace(cov_block))``, the total RMS spread.
            It is rotation-invariant (independent of any chi-mu correlation),
            so it is a valid total mosaic spread regardless of cross-correlation.
            ``"ellipse"`` -> eigen-decomposition of the (2x2) orientation block
            into ``(major_fwhm, minor_fwhm, angle_deg)`` of the spread ellipse.
        axes (tuple): orientation sub-block to use when a scan mixes orientation
            and strain axes (e.g. a 3-D chi x mu x 2theta scan -> ``axes=(0, 1)``
            to take only the chi, mu block). ``None`` uses every axis.

    Returns:
        ``mode="scalar"``: (ny, nx) RMS spread, same angular units as the motor.
        ``mode="ellipse"``: tuple ``(major_fwhm, minor_fwhm, angle_deg)``, each
        (ny, nx); FWHMs in motor units, angle in degrees of the major axis.

    Note:
        For a single-motor scan ``cov`` is the scalar rocking-curve variance and
        ``mode="scalar"`` reduces to ``sqrt(var)`` (the rocking-curve sigma;
        multiply by 2.3548 for FWHM).
    """
    cov = np.asarray(cov, dtype=float)
    if cov.ndim == 2:  # (ny, nx) scalar variance for a 1-motor scan
        cov = cov[..., None, None]
    if cov.ndim < 2 or cov.shape[-1] != cov.shape[-2]:
        raise ValueError(
            f"cov must be (ny, nx) or (ny, nx, D, D) but got shape {cov.shape}"
        )
    D = cov.shape[-1]
    sel = tuple(range(D)) if axes is None else tuple(axes)
    block = cov[..., sel, :][..., :, sel]  # (ny, nx, k, k)

    if mode == "scalar":
        tr = np.einsum("...ii->...", block)
        return np.sqrt(np.clip(tr, 0.0, None))

    if mode == "ellipse":
        if block.shape[-1] != 2:
            raise ValueError(
                "mode='ellipse' needs a 2-D orientation block; pass axes=(i, j) "
                f"to select two orientation axes (got {block.shape[-1]})"
            )
        sym = 0.5 * (block + np.swapaxes(block, -1, -2))
        vals, vecs = np.linalg.eigh(sym)  # ascending eigenvalues
        vals = np.clip(vals, 0.0, None)
        minor_fwhm = FWHM_FACTOR * np.sqrt(vals[..., 0])
        major_fwhm = FWHM_FACTOR * np.sqrt(vals[..., 1])
        major_vec = vecs[..., :, 1]  # eigenvector of the largest eigenvalue
        angle_deg = np.degrees(np.arctan2(major_vec[..., 1], major_vec[..., 0]))
        return major_fwhm, minor_fwhm, angle_deg

    raise ValueError(f"mode must be 'scalar' or 'ellipse', got {mode!r}")
