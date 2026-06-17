"""Preprocessing: background estimation/subtraction, auto-ROI, hot pixels.

Background subtraction clamps to the background image before subtracting so
uint16 never wraps. These operations are memory-bandwidth bound, so they run
in numpy on the host where the data already lives; the GPU is reserved for
the fitting.
"""

import numpy as np


def estimate_background(data, n_lowest=5, mode="mean"):
    """Background image from the n lowest-total-intensity frames.

    Args:
        data (numpy.ndarray): shape (a, b, m[, n[, o]]).
        n_lowest (int): number of darkest frames to combine.
        mode (str): "mean" or "median" over the selected frames.

    Returns:
        numpy.ndarray: (a, b) background image, same dtype as data.
    """
    a, b = data.shape[:2]
    frames = data.reshape(a, b, -1)
    totals = frames.sum(axis=(0, 1), dtype=np.float64)
    order = np.argsort(totals)[: max(1, n_lowest)]
    sel = frames[:, :, order]
    bg = np.median(sel, axis=-1) if mode == "median" else sel.mean(axis=-1)
    return bg.astype(data.dtype)


def subtract(data, background):
    """In-place clamped background subtraction.

    Args:
        data (numpy.ndarray): shape (a, b, ...), unsigned integer dtype.
        background (int or numpy.ndarray): scalar or (a, b) image.

    Returns:
        numpy.ndarray: the same array, modified in place.
    """
    if isinstance(background, (int, np.integer)):
        bg = np.full(data.shape[:2], background, dtype=data.dtype)
    else:
        bg = np.asarray(background)
        if bg.squeeze().shape != data.shape[:2]:
            raise ValueError(
                f"background shape {bg.squeeze().shape} != detector shape {data.shape[:2]}"
            )
        bg = bg.squeeze().astype(data.dtype)
    bg = bg[(...,) + (None,) * (data.ndim - bg.ndim)]
    if np.issubdtype(data.dtype, np.unsignedinteger):
        data.clip(bg, None, out=data)
    data -= bg
    return data


def auto_roi(data, threshold_rel=0.05, pad=20):
    """Bounding-box ROI around the illuminated grain.

    Sums over all motor dimensions, thresholds at threshold_rel * max, and
    returns the padded bounding box.

    Args:
        data (numpy.ndarray): shape (a, b, ...).
        threshold_rel (float): threshold as a fraction of the z-sum maximum.
        pad (int): pixels of padding around the bounding box.

    Returns:
        tuple: (row_min, row_max, col_min, col_max).
    """
    a, b = data.shape[:2]
    zsum = data.reshape(a, b, -1).sum(axis=-1, dtype=np.float64)
    m = zsum > threshold_rel * zsum.max()
    if not m.any():
        return (0, a, 0, b)
    rows = np.flatnonzero(m.any(axis=1))
    cols = np.flatnonzero(m.any(axis=0))
    return (
        max(0, rows[0] - pad),
        min(a, rows[-1] + 1 + pad),
        max(0, cols[0] - pad),
        min(b, cols[-1] + 1 + pad),
    )


def remove_hot_pixels(data, n_sigma=5.0):
    """Replace hot pixels with the 3x3 spatial median, frame by frame.

    A pixel is hot in a frame when it deviates from the local median by more
    than n_sigma robust standard deviations (MAD-based) of that frame.

    Args:
        data (numpy.ndarray): shape (a, b, m[, n[, o]]), modified in place.
        n_sigma (float): detection threshold.

    Returns:
        numpy.ndarray: the same array, modified in place.
    """
    from scipy.ndimage import median_filter

    a, b = data.shape[:2]
    frames = data.reshape(a, b, -1)
    for k in range(frames.shape[-1]):
        frame = frames[:, :, k]
        med = median_filter(frame, size=3)
        diff = frame.astype(np.float64) - med
        mad = np.median(np.abs(diff))
        sigma = max(1.4826 * mad, 1e-6)
        hot = np.abs(diff) > n_sigma * sigma
        frame[hot] = med[hot]
    return data


# --------------------------------------------------------------------------- #
# non-destructive variants (Section 8: interactive preview)
# --------------------------------------------------------------------------- #


def subtracted(data, background):
    """Non-destructive :func:`subtract` — returns a new array, leaves data intact."""
    out = data.copy()
    subtract(out, background)
    return out


def hot_pixels_removed(data, n_sigma=5.0):
    """Non-destructive :func:`remove_hot_pixels` — returns a new array."""
    out = data.copy()
    remove_hot_pixels(out, n_sigma=n_sigma)
    return out


def preview(data, *, bg_n=5, bg_mode="mean", hot_sigma=5.0, roi_threshold=0.05):
    """Compute a before/after preview of the noise-reduction chain.

    **Non-destructive**: ``data`` is never modified. For responsiveness on large
    stacks the processed z-sum is computed in a single pass and hot-pixel
    filtering is applied only to the displayed z-sum and a representative frame
    (the full per-frame hot-pixel filter runs only when settings are committed).

    Args:
        data (numpy.ndarray): shape (a, b, m[, n[, o]]).
        bg_n (int): number of darkest frames for the background estimate.
        bg_mode (str): "mean" or "median" background.
        hot_sigma (float): hot-pixel detection threshold.
        roi_threshold (float): auto-ROI threshold (fraction of z-sum max).

    Returns:
        dict with keys: ``raw_zsum``, ``proc_zsum`` ((a, b) z-projections before
        and after), ``raw_frame``, ``proc_frame`` (the brightest single frame,
        before and after), ``background`` ((a, b)), ``roi`` ((r1, r2, c1, c2) on
        the processed data) and ``hist`` ((counts, edges) of processed signal).
    """
    from scipy.ndimage import median_filter

    a, b = data.shape[:2]
    frames = data.reshape(a, b, -1)
    n_frames = frames.shape[-1]

    raw_zsum = frames.sum(axis=-1, dtype=np.float64)
    bg = estimate_background(data, n_lowest=bg_n, mode=bg_mode)
    # processed z-sum = sum_k clip(frame_k - bg, 0); one pass, signed ints
    proc_zsum = np.clip(
        frames.astype(np.int64) - bg[:, :, None].astype(np.int64), 0, None
    ).sum(axis=-1)

    # representative frame: the brightest single frame
    bright_k = int(np.argmax(frames.sum(axis=(0, 1), dtype=np.float64)))
    raw_frame = frames[:, :, bright_k].astype(np.float64)
    proc_frame = np.clip(raw_frame - bg.astype(np.float64), 0.0, None)
    med = median_filter(proc_frame, size=3)
    diff = proc_frame - med
    mad = np.median(np.abs(diff))
    sigma = max(1.4826 * mad, 1e-6)
    hot = np.abs(diff) > hot_sigma * sigma
    proc_frame = proc_frame.copy()
    proc_frame[hot] = med[hot]

    roi = auto_roi(np.clip(frames.astype(np.int64) - bg[:, :, None].astype(np.int64),
                           0, None).reshape(a, b, n_frames),
                   threshold_rel=roi_threshold, pad=0)
    sig = proc_zsum[proc_zsum > 0]
    hist = np.histogram(sig, bins=64) if sig.size else (np.zeros(64), np.arange(65))

    return {
        "raw_zsum": raw_zsum,
        "proc_zsum": proc_zsum,
        "raw_frame": raw_frame,
        "proc_frame": proc_frame,
        "background": bg,
        "roi": roi,
        "hist": hist,
    }


# --------------------------------------------------------------------------- #
# grain masking (Section 9)
# --------------------------------------------------------------------------- #


def _otsu_threshold(values):
    """Otsu's threshold on a 1-D array of intensities."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    counts, edges = np.histogram(finite, bins=256)
    centres = 0.5 * (edges[:-1] + edges[1:])
    w = counts.astype(np.float64)
    total = w.sum()
    if total == 0:
        return 0.0
    omega = np.cumsum(w) / total
    mu = np.cumsum(w * centres) / total
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    denom[denom == 0] = np.nan
    sigma_b2 = (mu_t * omega - mu) ** 2 / denom
    k = int(np.nanargmax(sigma_b2))
    return float(centres[k])


def grain_mask(data, threshold_rel=0.05, method="fraction", close=2, fill=True):
    """Boolean grain mask from the detector z-sum.

    Pixels above the threshold are kept, then morphologically closed and
    hole-filled into a solid grain footprint. Restricting fits to this mask
    cuts fit time roughly in proportion to the grain's area fraction, and (per
    :func:`starling.properties.moments` / the fits) changes only *which* pixels
    are computed, never the values on grain pixels.

    Args:
        data (numpy.ndarray): shape (a, b, ...).
        threshold_rel (float): for ``method="fraction"``, threshold as a fraction
            of the z-sum maximum.
        method (str): "fraction" (threshold_rel * max) or "otsu" (automatic).
        close (int): binary-closing iterations (0 disables).
        fill (bool): fill interior holes.

    Returns:
        numpy.ndarray: (a, b) bool mask.
    """
    a, b = data.shape[:2]
    zsum = data.reshape(a, b, -1).sum(axis=-1, dtype=np.float64)
    if method == "otsu":
        thr = _otsu_threshold(zsum)
    elif method == "fraction":
        thr = threshold_rel * zsum.max()
    else:
        raise ValueError(f"method must be 'fraction' or 'otsu', got {method!r}")
    mask = zsum > thr
    if close and close > 0:
        from scipy.ndimage import binary_closing

        mask = binary_closing(mask, iterations=int(close))
    if fill:
        from scipy.ndimage import binary_fill_holes

        mask = binary_fill_holes(mask)
    return mask.astype(bool)


def polygon_mask(shape, vertices):
    """Rasterise a polygon outline to a boolean mask.

    Vertices are (x=column, y=row) pairs — the convention emitted by
    matplotlib's ``PolygonSelector`` — so a hand-drawn grain outline maps
    directly to a mask. Pair with ``PolygonSelector`` in a
    ``%matplotlib widget`` notebook for "draw the grain outline".

    Args:
        shape (tuple): (ny, nx) detector shape.
        vertices: sequence of (x, y) vertex coordinates.

    Returns:
        numpy.ndarray: (ny, nx) bool mask, True inside the polygon.
    """
    from matplotlib.path import Path

    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    inside = Path(np.asarray(vertices, dtype=float)).contains_points(pts)
    return inside.reshape(ny, nx)
