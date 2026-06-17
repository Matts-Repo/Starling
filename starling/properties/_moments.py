"""GPU weighted moments over arbitrary motor dimensions.

data shape (a, b, m[, n[, o]]), coordinates shape (ndim, m[, n[, o]]),
zero-intensity (and masked-out) pixels map to 0, outputs are squeezed.

``order=2`` (default) returns the intensity-weighted mean and covariance.
``order=4`` additionally returns per-axis skewness and excess kurtosis.
"""

import numpy as np
import torch

from ..device import compute_dtype, get_device, plan_chunks


def moments(data, coordinates, order=2, mask=None, device=None):
    """Compute per-pixel intensity-weighted moments.

    Args:
        data (numpy.ndarray): shape (a, b, m[, n[, o]]).
        coordinates (numpy.ndarray): shape (ndim, m[, n[, o]]).
        order (int): 2 (default) returns (mean, covariance); 4 additionally
            returns per-axis skewness and excess kurtosis.
        mask (numpy.ndarray): optional (a, b) bool; only True pixels are
            computed (the rest stay 0). On-pixel values are identical to the
            unmasked result, so masking only changes *which* pixels are
            processed, never the values — it is purely a speed-up.
        device: torch device or name; None auto-detects.

    Returns:
        order=2: tuple (mean, covariance), shapes (a, b[, ndim]) and
            (a, b[, ndim, ndim]), squeezed.
        order=4: tuple (mean, covariance, skew, kurtosis); skew and kurtosis are
            per-axis (a, b[, ndim]). Kurtosis is *excess* (Gaussian -> 0; the -3
            is already applied).
    """
    if order not in (2, 4):
        raise ValueError(f"order must be 2 or 4, got {order}")
    mu, cov, skew, kurt = _moments_full(
        data, coordinates, device, want_cov=True, order=order, mask=mask
    )
    if order == 2:
        return np.squeeze(mu), np.squeeze(cov)
    return np.squeeze(mu), np.squeeze(cov), np.squeeze(skew), np.squeeze(kurt)


def higher_moments(data, coordinates, mask=None, device=None):
    """Per-axis skewness and excess kurtosis (convenience for ``order=4``)."""
    _, _, skew, kurt = _moments_full(
        data, coordinates, device, want_cov=True, order=4, mask=mask
    )
    return np.squeeze(skew), np.squeeze(kurt)


def mean(data, coordinates, mask=None, device=None):
    """Per-pixel intensity-weighted mean (see moments)."""
    mu, _, _, _ = _moments_full(
        data, coordinates, device, want_cov=False, mask=mask
    )
    return np.squeeze(mu)


def covariance(data, coordinates, first_moments=None, mask=None, device=None):
    """Per-pixel intensity-weighted covariance (see moments)."""
    # first_moments arg accepted for API compatibility; recomputing on GPU
    # is cheaper than validating/broadcasting a user-supplied array.
    _, cov, _, _ = _moments_full(data, coordinates, device, mask=mask)
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


def _moments_full(data, coordinates, device=None, want_cov=True, order=2, mask=None):
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
    out_skew = np.zeros((a * b, ndim), dtype=coordinates.dtype) if order == 4 else None
    out_kurt = np.zeros((a * b, ndim), dtype=coordinates.dtype) if order == 4 else None

    idx = np.flatnonzero(mask.ravel()) if mask is not None else np.arange(a * b)
    bytes_per_pixel = 4 * M * (3 + ndim)
    chunk = plan_chunks(len(idx), bytes_per_pixel, dev)
    for lo in range(0, len(idx), chunk):
        sel = idx[lo : lo + chunk]
        # cast to compute dtype on host before device transfer (MPS has no float64)
        w = torch.as_tensor(np.ascontiguousarray(Y[sel]), dtype=dtype).to(dev)
        I = w.sum(-1)
        nz = I != 0
        Isafe = I.clamp_min(1e-37)
        mu = (w @ c.T) / Isafe[:, None]  # (C, ndim)
        mu = torch.where(nz[:, None], mu, torch.zeros_like(mu))
        out_mu[sel] = mu.cpu().numpy()
        if want_cov or order == 4:
            cov = torch.zeros(len(sel), ndim, ndim, dtype=dtype, device=dev)
            for p in range(ndim):
                dp = (c[p][None, :] - mu[:, p : p + 1]) * w
                for qd in range(p, ndim):
                    v = (dp * (c[qd][None, :] - mu[:, qd : qd + 1])).sum(-1) / Isafe
                    v = torch.where(nz, v, torch.zeros_like(v))
                    cov[:, p, qd] = v
                    cov[:, qd, p] = v
            if want_cov:
                out_cov[sel] = cov.cpu().numpy()
            if order == 4:
                for k in range(ndim):
                    dk = c[k][None, :] - mu[:, k : k + 1]
                    sig = torch.sqrt(cov[:, k, k].clamp_min(0.0))
                    good = nz & (sig > 1e-12)
                    m3 = (w * dk**3).sum(-1) / Isafe
                    m4 = (w * dk**4).sum(-1) / Isafe
                    sk = torch.where(good, m3 / sig**3, torch.zeros_like(m3))
                    ku = torch.where(good, m4 / sig**4 - 3.0, torch.zeros_like(m4))
                    out_skew[sel, k] = sk.cpu().numpy()
                    out_kurt[sel, k] = ku.cpu().numpy()

    mu_map = out_mu.reshape(a, b, ndim)
    cov_map = out_cov.reshape(a, b, ndim, ndim) if want_cov else None
    skew_map = out_skew.reshape(a, b, ndim) if order == 4 else None
    kurt_map = out_kurt.reshape(a, b, ndim) if order == 4 else None
    return mu_map, cov_map, skew_map, kurt_map
