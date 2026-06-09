"""GPU weighted first/second moments over arbitrary motor dimensions.

API-compatible with darling.properties.moments / mean / covariance:
data shape (a, b, m[, n[, o]]), coordinates shape (ndim, m[, n[, o]]),
zero-intensity pixels map to 0, outputs are squeezed.
"""

import numpy as np
import torch

from ..device import compute_dtype, get_device, plan_chunks


def moments(data, coordinates, device=None):
    """Compute per-pixel intensity-weighted mean and covariance.

    Args:
        data (numpy.ndarray): shape (a, b, m[, n[, o]]).
        coordinates (numpy.ndarray): shape (ndim, m[, n[, o]]).
        device: torch device or name; None auto-detects.

    Returns:
        tuple of numpy.ndarray: mean (a, b[, ndim]) and covariance
        (a, b[, ndim, ndim]), squeezed like darling.
    """
    mu, cov = _moments_full(data, coordinates, device)
    return np.squeeze(mu), np.squeeze(cov)


def mean(data, coordinates, device=None):
    """Per-pixel intensity-weighted mean (see moments)."""
    mu, _ = _moments_full(data, coordinates, device, want_cov=False)
    return np.squeeze(mu)


def covariance(data, coordinates, first_moments=None, device=None):
    """Per-pixel intensity-weighted covariance (see moments)."""
    # first_moments accepted for darling API compatibility; recomputing on GPU
    # is cheaper than validating/broadcasting a user-supplied array.
    _, cov = _moments_full(data, coordinates, device)
    return np.squeeze(cov)


def _check_data(data, coordinates):
    if not isinstance(coordinates, np.ndarray):
        raise ValueError(
            f"coordinates must be a numpy array but got coordinates of type {type(coordinates)}"
        )
    if not coordinates.shape[1:] == data.shape[2:]:
        raise ValueError(
            f"trailing dimensions of coordinates shape {coordinates.shape[1:]} "
            f"do not match trailing dimensions of data shape {data.shape[2:]}"
        )
    if data.shape[0] == 1 or data.shape[1] == 1:
        raise ValueError(
            "First two detector row-column dimensions of data array must be greater than 1"
        )


def _moments_full(data, coordinates, device=None, want_cov=True):
    _check_data(data, coordinates)
    dev = get_device(device)
    dtype = compute_dtype(dev)

    a, b = data.shape[:2]
    ndim = len(coordinates)
    M = int(np.prod(data.shape[2:]))
    Y = data.reshape(a * b, M)
    c = torch.as_tensor(
        np.ascontiguousarray(coordinates.reshape(ndim, M)), dtype=dtype, device=dev
    )

    out_mu = np.zeros((a * b, ndim), dtype=coordinates.dtype)
    out_cov = np.zeros((a * b, ndim, ndim), dtype=coordinates.dtype) if want_cov else None

    bytes_per_pixel = 4 * M * (3 + ndim)
    chunk = plan_chunks(a * b, bytes_per_pixel, dev)
    for lo in range(0, a * b, chunk):
        hi = min(lo + chunk, a * b)
        w = torch.as_tensor(np.ascontiguousarray(Y[lo:hi]), device=dev).to(dtype)
        I = w.sum(-1)
        nz = I != 0
        Isafe = I.clamp_min(1e-37)
        mu = (w @ c.T) / Isafe[:, None]  # (C, ndim)
        mu = torch.where(nz[:, None], mu, torch.zeros_like(mu))
        out_mu[lo:hi] = mu.cpu().numpy()
        if want_cov:
            cov = torch.zeros(hi - lo, ndim, ndim, dtype=dtype, device=dev)
            for p in range(ndim):
                dp = (c[p][None, :] - mu[:, p : p + 1]) * w
                for q in range(p, ndim):
                    v = (dp * (c[q][None, :] - mu[:, q : q + 1])).sum(-1) / Isafe
                    v = torch.where(nz, v, torch.zeros_like(v))
                    cov[:, p, q] = v
                    cov[:, q, p] = v
            out_cov[lo:hi] = cov.cpu().numpy()

    mu_map = out_mu.reshape(a, b, ndim)
    cov_map = out_cov.reshape(a, b, ndim, ndim) if want_cov else None
    return mu_map, cov_map
