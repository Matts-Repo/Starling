"""Preprocessing: background estimation/subtraction, auto-ROI, hot pixels.

Background subtraction clamps to the background image before subtracting so
uint16 never wraps. These operations are memory-bandwidth bound, so they run
in numpy on the host where the data already lives; the GPU is reserved for
the fitting.
"""

import numpy as np


def _cast_bg(bg, dtype):
    """Cast a float background to ``dtype``, rounding (not flooring) integers.

    Flooring a fractional pedestal (e.g. 100.7 -> 100) under-subtracts by up to
    ~1 count at every pixel; summed over many frames that leaves a positive
    rectified pedestal in the clamped z-sum. Rounding centres the residual.
    """
    if np.issubdtype(dtype, np.integer):
        bg = np.rint(bg)
    return bg.astype(dtype)


def estimate_background(
    data,
    n_lowest=5,
    mode="mean",
    percentile=10.0,
    darks=None,
    chunk_rows=None,
):
    """Background image, grain-safe, with several estimators.

    The default (``mode="mean"``/``"median"``/``"lowest"``) pools only the
    globally-darkest *frames* (lowest total detector signal) and averages them
    per pixel. For a grain peaked at different motor steps across the grain these
    frames carry almost no grain signal, so the estimate is unbiased *at grain
    pixels* — it does not eat the grain (and empirically retains more grain than
    darfix's per-pixel median over the full stack). For a *broad* grain that is
    lit at every motor step (no clean dark frame), prefer a low per-pixel
    ``"percentile"`` or a dedicated ``darks`` stack (see Notes).

    Args:
        data (numpy.ndarray): shape (a, b, m[, n[, o]]).
        n_lowest (int): number of darkest frames to combine (modes
            "mean"/"median"/"lowest").
        mode (str): one of

            * ``"mean"`` / ``"lowest"`` — mean over the ``n_lowest`` darkest
              frames (current default behaviour; "lowest" is an alias).
            * ``"median"`` — median over the ``n_lowest`` darkest frames.
            * ``"pmedian"`` — per-pixel median over **all** frames (darfix
              parity; biased high where a pixel is lit in >50% of frames).
            * ``"percentile"`` — per-pixel ``percentile`` over all frames
              (grain-safe generalisation; only fails if a pixel is lit in
              >~(100-percentile)% of frames).
        percentile (float): percentile for ``mode="percentile"`` (e.g. 10-25).
        darks (numpy.ndarray, optional): a dedicated dark/blank stack
            (a, b, k) or precomputed (a, b) image. If given it overrides
            ``mode`` and yields a by-construction grain-free pedestal.
        chunk_rows (int, optional): for the per-pixel modes ("pmedian",
            "percentile"), process this many detector rows at a time to bound
            peak memory. ``None`` auto-sizes the blocks to ~512 MB.

    Returns:
        numpy.ndarray: (a, b) background image, same dtype as data.

    Notes:
        Integer dtypes are rounded (not floored) before the cast (see
        :func:`_cast_bg`). The per-pixel modes cost ~O(n_frames) memory per
        chunk and a sort per pixel; the darkest-frames modes are much cheaper.
    """
    a, b = data.shape[:2]

    if darks is not None:
        d = np.asarray(darks)
        if d.ndim == 3 and d.shape[:2] == (a, b):
            bg = d.reshape(a, b, -1).mean(axis=-1, dtype=np.float64)
        elif d.squeeze().shape == (a, b):
            bg = d.squeeze().astype(np.float64)
        else:
            raise ValueError(
                f"darks shape {d.shape} is not an (a, b) image or (a, b, k) stack"
            )
        return _cast_bg(bg, data.dtype)

    frames = data.reshape(a, b, -1)

    if mode in ("mean", "median", "lowest"):
        totals = frames.sum(axis=(0, 1), dtype=np.float64)
        order = np.argsort(totals)[: max(1, n_lowest)]
        sel = frames[:, :, order]
        bg = (
            np.median(sel, axis=-1)
            if mode == "median"
            else sel.mean(axis=-1, dtype=np.float64)
        )
        return _cast_bg(bg, data.dtype)

    if mode in ("pmedian", "percentile"):
        nf = frames.shape[-1]
        if chunk_rows is None:
            bytes_per_row = max(1, b * nf * 8)
            chunk_rows = max(1, min(a, (512 * 1024 ** 2) // bytes_per_row))
        bg = np.empty((a, b), dtype=np.float64)
        for r0 in range(0, a, chunk_rows):
            r1 = min(a, r0 + chunk_rows)
            blk = frames[r0:r1].astype(np.float64, copy=False)
            if mode == "pmedian":
                bg[r0:r1] = np.median(blk, axis=-1)
            else:
                bg[r0:r1] = np.percentile(blk, percentile, axis=-1)
        return _cast_bg(bg, data.dtype)

    raise ValueError(
        f"mode must be 'mean'/'lowest', 'median', 'pmedian' or 'percentile', "
        f"got {mode!r}"
    )


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


def signed_zsum(data, background):
    """Unclamped background-free z-sum: ``sum_k (frame_k - bg)`` (float).

    Unlike the clamped path (:func:`subtract` then sum), this does **not** floor
    each frame at zero, so symmetric read/Poisson noise cancels in the sum
    instead of rectifying into a positive floor. Use it to *quantify* true
    integrated grain counts with a clean off-grain baseline (~0).

    Do **not** use it as the grain footprint image or to feed :func:`auto_roi` /
    :func:`grain_mask`: where ``bg`` is biased high (a broad grain whose own
    signal contaminates an all-frames background) the signed sum under-retains
    the grain. For the displayed footprint keep the clamped z-sum; for honest
    total counts use this.

    Args:
        data (numpy.ndarray): shape (a, b, m[, n[, o]]). Not modified.
        background (int or numpy.ndarray): scalar or (a, b) image (e.g. from
            :func:`estimate_background`).

    Returns:
        numpy.ndarray: (a, b) float64 signed z-sum.
    """
    a, b = data.shape[:2]
    frames = data.reshape(a, b, -1)
    nf = frames.shape[-1]
    if isinstance(background, (int, np.integer)):
        bg = float(background)
    else:
        bg = np.asarray(background).squeeze().astype(np.float64)
        if bg.shape != (a, b):
            raise ValueError(
                f"background shape {bg.shape} != detector shape {(a, b)}"
            )
        bg = bg[:, :, None]
    # stream over frame-chunks so a full 2048^2 x n_frames stack never has to be
    # materialised as float64 at once
    out = np.zeros((a, b), dtype=np.float64)
    step = max(1, (512 * 1024 ** 2) // max(1, a * b * 8))
    for k0 in range(0, nf, step):
        blk = frames[:, :, k0:k0 + step].astype(np.float64)
        out += (blk - bg).sum(axis=-1)
    return out


def grain_signal_retained(data, background, mask=None, ref_percentile=10.0,
                          lit_nsigma=5.0):
    """Diagnostic: is ``background`` eating grain signal at the grain pixels?

    The honest question is whether ``background`` sits *above* each grain pixel's
    own pedestal — if so it subtracts real signal from every frame in which that
    pixel is lit. Two confounds must be avoided: the legitimate pedestal *under*
    the grain (which subtraction should remove), and the per-frame clamp's
    rectified-noise floor, which — summed over the many off-Bragg frames — dwarfs
    the grain term. Both are removed by measuring retention **only over the lit
    (near-Bragg) frames**, against a per-pixel low-percentile pedestal estimate:

    * ``floor`` = per-pixel ``ref_percentile`` background (best pedestal estimate).
    * a frame is *lit* at a pixel when ``frame > floor + lit_nsigma*sqrt(floor)``.
    * ``signal_floor`` = sum over lit frames of ``frame - floor`` (grain signal).
    * ``signal_kept``  = sum over lit frames of ``max(frame - background, 0)``.
    * ``retained = signal_kept / signal_floor``.

    ``retained ~ 1.0`` means ``background`` keeps essentially all the grain signal
    (it removed only the pedestal); ``retained < ~0.9`` means ``background`` is
    biased high at grain pixels and is eating the grain — prefer
    ``mode="percentile"`` or a dedicated dark stack.

    Args:
        data (numpy.ndarray): RAW (pre-subtraction) stack (a, b, ...).
        background (int or numpy.ndarray): the bg about to be subtracted.
        mask (numpy.ndarray, optional): (a, b) bool grain footprint. If None a
            footprint is built from the raw z-sum via :func:`grain_mask`.
        ref_percentile (float): percentile for the per-pixel pedestal floor.
        lit_nsigma (float): a frame is "lit" above ``floor + lit_nsigma*sqrt(floor)``.

    Returns:
        dict: ``retained`` (in-grain, lit-frame signal kept / true), ``signal_kept``,
        ``signal_floor`` (in-grain integrals over lit frames), ``overshoot_median``
        (median of ``background - floor`` over grain pixels, counts above the
        pedestal — ~0 is safe), ``floored_px`` (in-grain pixels zeroed by the
        clamped subtract), ``grain_px``, ``diff_map`` ((a, b) raw_zsum - proc_zsum
        = counts ``background`` removed, mostly pedestal).
    """
    a, b = data.shape[:2]
    if mask is None:
        mask = grain_mask(data)
    mask = np.asarray(mask, dtype=bool)
    frames = data.reshape(a, b, -1)
    nf = frames.shape[-1]

    floor = estimate_background(
        data, mode="percentile", percentile=ref_percentile
    ).astype(np.float64)
    if isinstance(background, (int, np.integer)):
        bg = np.full((a, b), float(background))
    else:
        bg = np.asarray(background).squeeze().astype(np.float64)
    thr = floor + lit_nsigma * np.sqrt(np.maximum(floor, 1.0))

    raw_zsum = frames.sum(axis=-1, dtype=np.float64)
    proc_zsum = np.zeros((a, b), dtype=np.float64)
    sig_floor = np.zeros((a, b), dtype=np.float64)
    sig_kept = np.zeros((a, b), dtype=np.float64)
    step = max(1, (512 * 1024 ** 2) // max(1, a * b * 8))
    for k0 in range(0, nf, step):
        blk = frames[:, :, k0:k0 + step].astype(np.float64)
        proc_zsum += np.clip(blk - bg[:, :, None], 0.0, None).sum(axis=-1)
        lit = blk > thr[:, :, None]
        sig_floor += np.where(lit, blk - floor[:, :, None], 0.0).sum(axis=-1)
        sig_kept += np.where(
            lit, np.clip(blk - bg[:, :, None], 0.0, None), 0.0
        ).sum(axis=-1)

    sf = float(sig_floor[mask].sum())
    sk = float(sig_kept[mask].sum())
    return {
        "retained": (sk / sf) if sf > 0 else float("nan"),
        "signal_kept": sk,
        "signal_floor": sf,
        "overshoot_median": float(np.median((bg - floor)[mask])) if mask.any() else 0.0,
        "floored_px": int((mask & (proc_zsum <= 0)).sum()),
        "grain_px": int(mask.sum()),
        "diff_map": raw_zsum - proc_zsum,
    }


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


def remove_hot_pixels(data, n_sigma=5.0, one_sided=False, protect=None, min_sigma=None):
    """Replace hot pixels with the 3x3 spatial median, frame by frame.

    A pixel is hot in a frame when it deviates from the local median by more
    than n_sigma robust standard deviations (MAD-based) of that frame.

    Grain-safe options (recommended when running *after* a clamped subtract, or
    on grains with genuine interior dark sub-features):

    * ``one_sided=True`` flags only **positive** outliers (``diff >``), so a
      genuine interior dark void / weaker sub-grain is never filled brighter.
    * ``protect`` (an (a, b) bool grain mask) is never altered — nothing inside
      the grain footprint is touched.
    * ``min_sigma`` floors the robust sigma at an absolute count, so a frame
      that is mostly zeros (MAD collapses to 0 after a clamped subtract) cannot
      drive the threshold to ~0 and flag genuine grain-interior pixels. Default
      ``None`` keeps the legacy 1e-6 floor; set e.g. ``1.0`` post-subtraction.

    Args:
        data (numpy.ndarray): shape (a, b, m[, n[, o]]), modified in place.
        n_sigma (float): detection threshold.
        one_sided (bool): flag only bright outliers (never fill dark features).
        protect (numpy.ndarray, optional): (a, b) bool mask of pixels to leave
            untouched (e.g. a grain footprint).
        min_sigma (float, optional): absolute floor on the robust sigma.

    Returns:
        numpy.ndarray: the same array, modified in place.
    """
    from scipy.ndimage import median_filter

    a, b = data.shape[:2]
    frames = data.reshape(a, b, -1)
    keep = None if protect is None else np.asarray(protect, dtype=bool)
    floor = 1e-6 if min_sigma is None else float(min_sigma)
    for k in range(frames.shape[-1]):
        frame = frames[:, :, k]
        med = median_filter(frame, size=3)
        diff = frame.astype(np.float64) - med
        mad = np.median(np.abs(diff))
        sigma = max(1.4826 * mad, floor)
        hot = diff > n_sigma * sigma if one_sided else np.abs(diff) > n_sigma * sigma
        if keep is not None:
            hot &= ~keep
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


def hot_pixels_removed(data, n_sigma=5.0, one_sided=False, protect=None, min_sigma=None):
    """Non-destructive :func:`remove_hot_pixels` — returns a new array."""
    out = data.copy()
    remove_hot_pixels(
        out, n_sigma=n_sigma, one_sided=one_sided, protect=protect, min_sigma=min_sigma
    )
    return out


def preview(data, *, bg_n=5, bg_mode="mean", bg_percentile=10.0, hot_sigma=5.0,
            roi_threshold=0.05):
    """Compute a before/after preview of the noise-reduction chain.

    **Non-destructive**: ``data`` is never modified. For responsiveness on large
    stacks the processed z-sum is computed in a single pass and hot-pixel
    filtering is applied only to the displayed z-sum and a representative frame
    (the full per-frame hot-pixel filter runs only when settings are committed).

    Args:
        data (numpy.ndarray): shape (a, b, m[, n[, o]]).
        bg_n (int): number of darkest frames for the background estimate.
        bg_mode (str): background estimator — see :func:`estimate_background`
            ("mean"/"median"/"lowest"/"pmedian"/"percentile").
        bg_percentile (float): percentile for ``bg_mode="percentile"``.
        hot_sigma (float): hot-pixel detection threshold.
        roi_threshold (float): auto-ROI threshold (fraction of z-sum max).

    Returns:
        dict with keys: ``raw_zsum``, ``proc_zsum`` ((a, b) z-projections before
        and after), ``removed_zsum`` ((a, b) raw_zsum - proc_zsum = counts taken
        out), ``retained`` (in-grain proc/raw integral fraction), ``grain``
        ((a, b) bool footprint), ``raw_frame``, ``proc_frame`` (the brightest
        single frame, before and after), ``background`` ((a, b)), ``roi``
        ((r1, r2, c1, c2) on the processed data) and ``hist`` ((counts, edges)
        of processed signal).
    """
    from scipy.ndimage import median_filter

    a, b = data.shape[:2]
    frames = data.reshape(a, b, -1)
    n_frames = frames.shape[-1]

    raw_zsum = frames.sum(axis=-1, dtype=np.float64)
    bg = estimate_background(
        data, n_lowest=bg_n, mode=bg_mode, percentile=bg_percentile
    )
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

    # how much in-grain signal the subtraction removed (the acceptance criterion):
    # lit-frame retention vs the grain-safe low-percentile pedestal floor.
    removed_zsum = raw_zsum - proc_zsum
    grain = grain_mask(data, threshold_rel=max(roi_threshold, 1e-3))
    retained = grain_signal_retained(data, bg, mask=grain)["retained"]

    return {
        "raw_zsum": raw_zsum,
        "proc_zsum": proc_zsum,
        "removed_zsum": removed_zsum,
        "retained": retained,
        "grain": grain,
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
