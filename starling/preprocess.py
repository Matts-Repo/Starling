"""Preprocessing: background estimation/subtraction, auto-ROI, hot pixels.

Background subtraction follows darling.DataSet.subtract semantics
(clamp-to-background before subtracting so uint16 never wraps). These
operations are memory-bandwidth bound, so they run in numpy on the host
where the data already lives; the GPU is reserved for the fitting.
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
    """In-place clamped background subtraction (darling semantics).

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
