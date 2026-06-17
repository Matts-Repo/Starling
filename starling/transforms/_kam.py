"""Kernel average misorientation (KAM) from an orientation (COM) field.

KAM is cheap, so this is a plain numpy implementation (no GPU needed).
"""

import numpy as np


def _shift(arr, di, dj):
    """arr shifted so out[i, j] = arr[i + di, j + dj]; out-of-bounds -> NaN."""
    out = np.full_like(arr, np.nan)
    ny, nx = arr.shape[:2]
    src_y = slice(max(0, di), min(ny, ny + di))
    dst_y = slice(max(0, -di), min(ny, ny - di))
    src_x = slice(max(0, dj), min(nx, nx + dj))
    dst_x = slice(max(0, -dj), min(nx, nx - dj))
    out[dst_y, dst_x] = arr[src_y, src_x]
    return out


def kam(orientation_com, size=(3, 3), min_neighbors=1):
    """Kernel average misorientation from a mean-orientation (COM) map.

    For each pixel, the misorientation to a kernel neighbour is the L2 distance
    between the pixel's COM vector and the neighbour's COM vector; KAM is the
    average over valid neighbours in the kernel window.

    Args:
        orientation_com (numpy.ndarray): (ny, nx) or (ny, nx, D) mean-orientation
            map (e.g. ``moments`` mean or ``GaussNDResult.mu``). NaN pixels are
            treated as missing and skipped.
        size (tuple): (ky, kx) odd kernel window. Defaults to (3, 3).
        min_neighbors (int): minimum number of valid neighbours required;
            pixels with fewer are set to NaN.

    Returns:
        numpy.ndarray: (ny, nx) KAM map in the same angular units as the input.

    Caveat:
        This is a **projected** misorientation between diffraction vectors, not a
        full SO(3) misorientation. For 1-D rocking scans it is further reduced
        because the rolling angle is unknown, so only one orientation component
        is observed.
    """
    arr = np.asarray(orientation_com, dtype=float)
    if arr.ndim == 2:
        arr = arr[..., None]
    elif arr.ndim != 3:
        raise ValueError(
            f"orientation_com must be (ny, nx) or (ny, nx, D), got shape {arr.shape}"
        )
    ny, nx, _ = arr.shape
    ky, kx = size
    ry, rx = ky // 2, kx // 2

    valid_center = np.isfinite(arr).all(-1)
    acc = np.zeros((ny, nx))
    cnt = np.zeros((ny, nx), dtype=int)
    for di in range(-ry, ry + 1):
        for dj in range(-rx, rx + 1):
            if di == 0 and dj == 0:
                continue
            nb = _shift(arr, di, dj)
            both = valid_center & np.isfinite(nb).all(-1)
            dist = np.sqrt(np.sum((arr - nb) ** 2, axis=-1))
            acc += np.where(both, dist, 0.0)
            cnt += both

    out = np.where(cnt >= max(1, min_neighbors), acc / np.maximum(cnt, 1), np.nan)
    out[~valid_center] = np.nan
    return out
