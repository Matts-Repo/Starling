"""Preprocessing: background estimation/subtraction, auto-ROI, hot pixels.

Background subtraction clamps to the background image before subtracting so
uint16 never wraps. The reductions (background, z-sums, ROI) are
memory-bandwidth bound, so they run in numpy on the host where the data
already lives. Hot-pixel removal is compute-bound (a 3x3 median per pixel per
frame) and runs batched on the torch device by default (cuda > mps > cpu);
a scipy reference path is kept behind ``backend="scipy"``.
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


# The 19-comparator median-of-9 sorting network (Paeth). Each pair (i, j)
# leaves min at i and max at j; after all passes index 4 holds the median.
_MEDIAN9_NET = (
    (1, 2), (4, 5), (7, 8), (0, 1), (3, 4), (6, 7), (1, 2), (4, 5), (7, 8),
    (0, 3), (5, 8), (4, 7), (3, 6), (1, 4), (2, 5), (4, 7), (4, 2), (6, 4),
    (4, 2),
)


def _median9(planes):
    """Median of 9 same-shape tensors via an explicit min/max sorting network.

    WHY a network and not ``torch.median``/``torch.sort`` over a dim:
    on MPS (torch 2.12) both are **silently wrong** for many-lane small sorts
    — sorting 9 elements per lane over ~4M pixel lanes returns wrong values in
    ~90% of lanes with no error raised (reproduced synthetically; see
    starling-review 07_baseline_benchmarks.md). Elementwise
    ``torch.minimum``/``torch.maximum`` are correct on every backend, so the
    network is used unconditionally — including on CUDA, where behaviour at
    these odd shapes is unverified — because it is portable, deterministic,
    and at 19 fused elementwise ops also faster than a general sort.
    """
    import torch

    p = list(planes)
    for i, j in _MEDIAN9_NET:
        lo = torch.minimum(p[i], p[j])
        p[j] = torch.maximum(p[i], p[j])
        p[i] = lo
    return p[4]


def _remove_hot_frames_scipy(frames, n_sigma, one_sided, keep, floor):
    """Reference implementation: serial per-frame scipy median filter (host)."""
    from scipy.ndimage import median_filter

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


def _remove_hot_frames_torch(frames, n_sigma, one_sided, keep, floor, device=None,
                             chunk_frames=None):
    """Batched hot-pixel replacement on a torch device. In-place on ``frames``.

    Per chunk of frames: 3x3 median via 9 shifted views of a replicate-padded
    stack + the min/max sorting network, per-frame robust sigma from the MAD of
    the residual, threshold, protect, and in-place replacement on the host
    array. Pixel-identical to the scipy path (validated on real 2048^2 data).
    """
    import torch
    import torch.nn.functional as F

    from .device import compute_dtype, get_device, plan_chunks

    dev = get_device(device)
    cdt = compute_dtype(dev)  # float32 on cuda/mps (no float64 on MPS), float64 on cpu
    np_cdt = np.float32 if cdt == torch.float32 else np.float64
    a, b, nf = frames.shape
    if chunk_frames is None:
        # peak working set per frame: padded copy, the 9 network planes plus
        # temporaries, |diff|, sort values+indices, masks -> ~20 frame buffers
        per_frame = a * b * np.dtype(np_cdt).itemsize * 20
        chunk_frames = plan_chunks(nf, per_frame, dev)
    # cap the chunk so the per-frame MAD sort below keeps a modest lane count
    # (the broken MPS regime is millions of lanes; production chunks are ~50)
    chunk_frames = max(1, min(int(chunk_frames), nf, 256))
    keep_t = None
    if keep is not None:
        keep_t = torch.from_numpy(np.ascontiguousarray(keep)).to(dev)
    out_dtype = frames.dtype
    for k0 in range(0, nf, chunk_frames):
        k1 = min(k0 + chunk_frames, nf)
        # gather the strided detector-first slab into contiguous (k, a, b)
        blk = frames[:, :, k0:k1].transpose(2, 0, 1).astype(np_cdt, order="C")
        x = torch.from_numpy(blk).to(dev)
        # scipy median_filter(size=3, mode="reflect") duplicates the edge
        # sample for a width-1 border == torch "replicate" (torch "reflect"
        # would differ on every border row/col)
        xp = F.pad(x.unsqueeze(1), (1, 1, 1, 1), mode="replicate").squeeze(1)
        med = _median9([xp[:, i:i + a, j:j + b] for i in range(3) for j in range(3)])
        diff = x - med
        # per-frame robust sigma: MAD = median(|diff|) over the whole frame.
        # torch.sort along ONE LARGE dim (few frame-lanes x a*b elements) is
        # the orientation validated correct on MPS — it is the many-lane
        # small-dim case that is silently wrong there (see _median9). This
        # exact formulation was verified pixel-identical to np.median/scipy.
        flat = diff.abs().reshape(diff.shape[0], -1)
        n = flat.shape[1]
        srt = torch.sort(flat, dim=1).values
        if n % 2:
            mad = srt[:, n // 2]
        else:
            mad = 0.5 * (srt[:, n // 2 - 1] + srt[:, n // 2])
        sigma = torch.clamp(1.4826 * mad, min=floor)
        thr = (n_sigma * sigma).view(-1, 1, 1)
        hot = diff > thr if one_sided else diff.abs() > thr
        if keep_t is not None:
            hot &= ~keep_t
        fixed = torch.where(hot, med, x)
        # uint16 values are exact in float32 (< 2^24), so the round-trip cast
        # is lossless for integer data
        frames[:, :, k0:k1] = (
            fixed.cpu().numpy().astype(out_dtype, copy=False).transpose(1, 2, 0)
        )
        del x, xp, med, diff, flat, srt, mad, sigma, thr, hot, fixed
        if dev.type == "mps":
            torch.mps.empty_cache()


def _static_hot_map(frames, n_sigma, one_sided, keep, floor):
    """(a, b) bool map of *persistently* hot pixels, from the temporal median.

    The per-pixel temporal median over all frames suppresses single-frame
    zingers, so anything still anomalous vs its 3x3 spatial neighbourhood in
    that image is a static detector defect. The temporal median is an N-frame
    sort over millions of pixel lanes — exactly the shape class where MPS sort
    kernels are silently wrong — so it stays on the host in numpy (chunked
    over detector rows); it is a single pass and not the bottleneck.
    """
    from scipy.ndimage import median_filter

    a, b, nf = frames.shape
    tmed = np.empty((a, b), np.float64)
    chunk_rows = max(1, min(a, (512 * 1024 ** 2) // max(1, b * nf * 8)))
    for r0 in range(0, a, chunk_rows):
        blk = frames[r0:r0 + chunk_rows].astype(np.float64, copy=False)
        tmed[r0:r0 + chunk_rows] = np.median(blk, axis=-1)
    med = median_filter(tmed, size=3)
    diff = tmed - med
    mad = np.median(np.abs(diff))
    sigma = max(1.4826 * mad, floor)
    hot = diff > n_sigma * sigma if one_sided else np.abs(diff) > n_sigma * sigma
    if keep is not None:
        hot &= ~keep
    return hot


def _fill_static(frames, hot_map):
    """Replace ``hot_map`` pixels in every frame by that frame's own 3x3 median.

    Only the flagged pixels are touched: their 3x3 neighbourhoods are gathered
    with clipped (edge-replicated, == scipy reflect for width 1) indices and
    the per-frame median taken on the host. Cost is O(n_hot * n_frames), not
    O(n_pixels * n_frames).
    """
    ys, xs = np.nonzero(hot_map)
    if ys.size == 0:
        return
    a, b, nf = frames.shape
    off = np.array([-1, 0, 1])
    yn = np.clip(ys[:, None, None] + off[None, :, None], 0, a - 1)
    xn = np.clip(xs[:, None, None] + off[None, None, :], 0, b - 1)
    neigh = frames[yn, xn, :].reshape(ys.size, 9, nf)  # (K, 9, nf)
    fill = np.empty((ys.size, nf), np.float64)
    step = max(1, (256 * 1024 ** 2) // max(1, ys.size * 9 * 8))
    for k0 in range(0, nf, step):
        fill[:, k0:k0 + step] = np.median(
            neigh[:, :, k0:k0 + step], axis=1
        )
    frames[ys, xs, :] = fill.astype(frames.dtype, copy=False)


def remove_hot_pixels(data, n_sigma=5.0, one_sided=False, protect=None, min_sigma=None,
                      method="frame", backend=None, device=None, chunk_frames=None):
    """Replace hot pixels with the 3x3 spatial median.

    A pixel is hot in a frame when it deviates from the local median by more
    than n_sigma robust standard deviations (MAD-based) of that frame.

    Detection modes (``method``):

    * ``"frame"`` (default) — per-frame spatial detection, as always: each
      frame gets its own 3x3-median comparison and robust sigma. Catches both
      static detector defects and single-frame zingers (cosmic hits).
    * ``"static"`` — one hot-pixel map for the whole stack: the per-pixel
      *temporal* median image (which suppresses single-frame events) is
      compared against its own 3x3 spatial median, and the resulting map of
      persistently-defective detector pixels is filled in **every** frame with
      that frame's local 3x3 median. This targets detector defects only — it
      deliberately **misses single-frame zingers** — but is one cheap pass.
    * ``"hybrid"`` — the static map first, then the per-frame pass on the
      cleaned data: detector defects *and* zingers.

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

    The library defaults above are the **legacy** (not grain-safe) settings for
    backward compatibility. The notebook (03), :func:`starling.viz.denoise_widget`
    and the batch runner all pass the grain-safe settings
    ``one_sided=True, min_sigma=1.0`` explicitly — pass them yourself when
    calling this directly on subtracted data.

    Backends: the default (``backend="torch"``) batches frames on the compute
    device (auto-select cuda > mps > cpu; ~8x faster than scipy per frame on
    MPS, measured) and is pixel-identical to ``backend="scipy"``, the original
    serial host loop kept for reference/parity. Integer data round-trips the
    GPU float32 path losslessly; float64 input is processed in float32 on
    cuda/mps (MPS has no float64 — use ``device="cpu"`` or ``backend="scipy"``
    if that matters).

    Args:
        data (numpy.ndarray): shape (a, b, m[, n[, o]]), modified in place.
        n_sigma (float): detection threshold.
        one_sided (bool): flag only bright outliers (never fill dark features).
        protect (numpy.ndarray, optional): (a, b) bool mask of pixels to leave
            untouched (e.g. a grain footprint).
        min_sigma (float, optional): absolute floor on the robust sigma.
        method (str): "frame" (default), "static" or "hybrid" — see above.
        backend (str, optional): "torch" (default) or "scipy".
        device (str or torch.device, optional): torch device for the batched
            backend; default auto-selects (cuda > mps > cpu).
        chunk_frames (int, optional): frames per device chunk for the torch
            backend; default auto-sizes to the device memory budget.

    Returns:
        numpy.ndarray: the same array, modified in place.
    """
    a, b = data.shape[:2]
    frames = data.reshape(a, b, -1)
    keep = None if protect is None else np.asarray(protect, dtype=bool)
    floor = 1e-6 if min_sigma is None else float(min_sigma)
    backend = "torch" if backend is None else backend
    if backend not in ("torch", "scipy"):
        raise ValueError(f"backend must be 'torch' or 'scipy', got {backend!r}")
    if method not in ("frame", "static", "hybrid"):
        raise ValueError(
            f"method must be 'frame', 'static' or 'hybrid', got {method!r}"
        )

    if method in ("static", "hybrid"):
        hot_map = _static_hot_map(frames, n_sigma, one_sided, keep, floor)
        _fill_static(frames, hot_map)
    if method in ("frame", "hybrid"):
        if backend == "scipy":
            _remove_hot_frames_scipy(frames, n_sigma, one_sided, keep, floor)
        else:
            _remove_hot_frames_torch(
                frames, n_sigma, one_sided, keep, floor,
                device=device, chunk_frames=chunk_frames,
            )
    return data


# --------------------------------------------------------------------------- #
# non-destructive variants (Section 8: interactive preview)
# --------------------------------------------------------------------------- #


def subtracted(data, background):
    """Non-destructive :func:`subtract` — returns a new array, leaves data intact."""
    out = data.copy()
    subtract(out, background)
    return out


def hot_pixels_removed(data, n_sigma=5.0, one_sided=False, protect=None, min_sigma=None,
                       method="frame", backend=None, device=None):
    """Non-destructive :func:`remove_hot_pixels` — returns a new array."""
    out = data.copy()
    remove_hot_pixels(
        out, n_sigma=n_sigma, one_sided=one_sided, protect=protect,
        min_sigma=min_sigma, method=method, backend=backend, device=device,
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


def support_count(data, motor_axis, threshold):
    """Per-pixel count of planes along one motor axis with any signal.

    For each detector pixel, counts in how many planes along ``motor_axis``
    (0-based among the motor dimensions) the pixel has ANY voxel strictly
    above ``threshold`` across the remaining motor dimensions.

    The intended use is gating 3-D strain-mosa fits on ccmth support: a
    grain-edge pixel lit at only 1-3 ccmth planes cannot constrain a 3-D
    Gaussian along ccmth (the fitted ccmth width/centre is degenerate), so
    fits are restricted to pixels with e.g. ``support_count(...) >= 4``.

    Args:
        data (numpy.ndarray): shape (ny, nx, *motor_dims).
        motor_axis (int): which motor dimension to count support along,
            0-based among the motor dims (i.e. axis ``2 + motor_axis`` of
            ``data``).
        threshold: a plane counts only where a voxel is strictly ``>`` this
            value (a voxel exactly at the threshold does not count).

    Returns:
        numpy.ndarray: (ny, nx) int16 — 0 for an all-dark pixel, up to
        ``data.shape[2 + motor_axis]`` for a pixel lit at every plane.
    """
    n_motor = data.ndim - 2
    if n_motor < 1:
        raise ValueError(f"data must be (ny, nx, *motor_dims), got {data.shape}")
    if not 0 <= motor_axis < n_motor:
        raise ValueError(
            f"motor_axis must be in [0, {n_motor - 1}] for {n_motor} motor "
            f"dims, got {motor_axis}"
        )
    lit = data > threshold
    other = tuple(2 + i for i in range(n_motor) if i != motor_axis)
    if other:
        lit = lit.any(axis=other)  # -> (ny, nx, n_planes)
    return lit.sum(axis=-1, dtype=np.int16)
