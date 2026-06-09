"""Public curve-fitting API, darling-compatible.

fit_1D_gaussian matches darling.properties.fit_1D_gaussian: output shape
(ny, nx, 6) float64 with parameters ordered [A, sigma, mu, k, m, success]
(the order darling's code actually writes).

float32 GPU stability: motor coordinates are centred/scaled to O(1) and each
curve is normalised by its maximum before fitting; parameters are transformed
back afterwards. Without this, motor values like ccmth ~ 6.68 deg with
1e-3 deg steps condition the normal equations beyond float32.
"""

import numpy as np
import torch

from ..device import compute_dtype, get_device, plan_chunks
from ._gaussnewton import gauss_newton_batched
from ._init_guess import gaussian_seed, linear_trend
from ._models import gauss1d_lin


def fit_1D_gaussian(data, coordinates, n_iter_gauss_newton=7, mask=None, device=None):
    """Fit a Gaussian + linear background per pixel (batched on GPU).

    f(x) = A * exp(-(x - mu)**2 / (2 * sigma**2)) + k * x + m

    Args:
        data (numpy.ndarray): shape (ny, nx, m).
        coordinates: length-1 sequence holding the (m,) motor coordinate array.
        n_iter_gauss_newton (int): Gauss-Newton iterations. Defaults to 7.
        mask (numpy.ndarray): optional (ny, nx) bool; only True pixels are fitted.
        device: torch device or name; None auto-detects.

    Returns:
        numpy.ndarray: (ny, nx, 6) float64, [A, sigma, mu, k, m, success].
    """
    if len(coordinates) != 1:
        raise ValueError(
            f"coordinates must be a 1d tuple but got {len(coordinates)} dimensions"
        )
    if data.ndim != 3:
        raise ValueError(f"data must be a 3d numpy array but got {data.ndim} dimensions")

    x = np.asarray(coordinates[0], dtype=np.float64)
    ny, nx, N = data.shape
    Y = data.reshape(ny * nx, N)

    if mask is not None:
        idx = np.flatnonzero(mask.ravel())
    else:
        idx = np.arange(ny * nx)

    dev = get_device(device)
    dtype = compute_dtype(dev)

    # coordinate centring/scaling and per-curve intensity normalisation
    xc = float(x.mean())
    xs = float(max(np.ptp(x) / 2.0, 1e-12))
    xh = torch.as_tensor((x - xc) / xs, dtype=dtype, device=dev)

    out = np.zeros((ny * nx, 6), dtype=np.float64)

    p = 5
    bytes_per_pixel = dtype.itemsize * N * (3 + p) * 2
    chunk = plan_chunks(len(idx), bytes_per_pixel, dev)
    for lo in range(0, len(idx), chunk):
        sel = idx[lo : lo + chunk]
        block = Y[sel]
        # pad short final chunks to the uniform chunk shape: torch.compile
        # specialises on static shapes, so a ragged last chunk would trigger
        # a full recompile (~seconds) for a few leftover pixels
        n_pad = 0
        if len(sel) < chunk and lo > 0:
            n_pad = chunk - len(sel)
            block = np.concatenate([block, np.repeat(block[:1], n_pad, axis=0)])
        y = torch.as_tensor(np.ascontiguousarray(block), device=dev).to(dtype)
        ys = y.amax(-1).clamp_min(1.0)
        yn = y / ys[:, None]

        k0, m0 = linear_trend(yn, xh)
        A0, s0, mu0, degenerate = gaussian_seed(yn, xh, k0, m0)
        # degenerate rows get harmless dummy params and are fitted anyway
        # (keeping the batch shape uniform for torch.compile), then their
        # outputs are overwritten with darling's convention below
        s0 = torch.where(degenerate, torch.ones_like(s0), s0)
        params0 = torch.stack([A0, s0, mu0, k0, m0], dim=-1)

        params, ok = gauss_newton_batched(
            yn, xh, params0, gauss1d_lin, n_iter=n_iter_gauss_newton
        )
        success = ok & ~degenerate

        A, sigma, mu, k, m = params.unbind(-1)
        res = torch.stack(
            [
                A * ys,
                sigma.abs() * xs,
                xc + xs * mu,
                k * ys / xs,
                (m - k * xc / xs) * ys,
                success.to(params.dtype),
            ],
            dim=-1,
        )
        # degenerate pixels keep darling's convention:
        # A = sigma = mu = 0, background from the initial linear trend
        deg = degenerate.unsqueeze(-1)
        res = torch.where(
            deg,
            torch.stack(
                [
                    torch.zeros_like(A),
                    torch.zeros_like(A),
                    torch.zeros_like(A),
                    k0 * ys / xs,
                    (m0 - k0 * xc / xs) * ys,
                    torch.zeros_like(A),
                ],
                dim=-1,
            ),
            res,
        )
        res_np = res.cpu().double().numpy()
        out[sel] = res_np[: len(sel)]

    return out.reshape(ny, nx, 6)
