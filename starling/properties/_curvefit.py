"""Public curve-fitting API.

fit_1D_gaussian returns shape (ny, nx, 6) float64 with parameters ordered
[A, sigma, mu, k, m, success].

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
        # outputs are overwritten with zeros for degenerate pixels below
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
        # degenerate pixels: A = sigma = mu = 0, background from linear trend
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


def fit_two_gaussians_1D(
    data,
    coordinates,
    n_iter_gauss_newton=12,
    mask=None,
    delta_bic=10.0,
    device=None,
):
    """Fit 1-peak and 2-peak Gaussian models per pixel, with BIC selection.

    Both models share a linear background. A pixel is classified 2-peak when
    BIC2 + delta_bic < BIC1 *and* physical gates pass (peak separation above
    2 motor steps, both amplitudes positive, both widths within the scan
    span). Peaks in params2 are sorted by mu so peak 1 is the lower-angle one.

    Args:
        data (numpy.ndarray): shape (ny, nx, m).
        coordinates: length-1 sequence holding the (m,) motor array.
        n_iter_gauss_newton (int): Gauss-Newton iterations for both fits.
        mask (numpy.ndarray): optional (ny, nx) bool.
        delta_bic (float): margin BIC2 must win by to select the 2-peak model.
        device: torch device or name; None auto-detects.

    Returns:
        dict with:
            params1: (ny, nx, 6) [A, sigma, mu, k, m, success]
            params2: (ny, nx, 9) [A1, sigma1, mu1, A2, sigma2, mu2, k, m, success]
            n_peaks: (ny, nx) uint8 (0 = no valid fit, 1, or 2)
            bic1, bic2: (ny, nx) float64
    """
    from ._models import gauss1d_two_lin
    from ._init_guess import two_peak_seed

    if len(coordinates) != 1:
        raise ValueError(
            f"coordinates must be a 1d tuple but got {len(coordinates)} dimensions"
        )
    if data.ndim != 3:
        raise ValueError(f"data must be a 3d numpy array but got {data.ndim} dimensions")

    x = np.asarray(coordinates[0], dtype=np.float64)
    ny, nx, N = data.shape
    Y = data.reshape(ny * nx, N)
    idx = np.flatnonzero(mask.ravel()) if mask is not None else np.arange(ny * nx)

    dev = get_device(device)
    dtype = compute_dtype(dev)
    xc = float(x.mean())
    xs = float(max(np.ptp(x) / 2.0, 1e-12))
    xh = torch.as_tensor((x - xc) / xs, dtype=dtype, device=dev)
    step_scaled = float(np.abs(np.diff((x - xc) / xs)).mean())

    out1 = np.zeros((ny * nx, 6), dtype=np.float64)
    out2 = np.zeros((ny * nx, 9), dtype=np.float64)
    out_n = np.zeros(ny * nx, dtype=np.uint8)
    out_b1 = np.zeros(ny * nx, dtype=np.float64)
    out_b2 = np.zeros(ny * nx, dtype=np.float64)

    bytes_per_pixel = dtype.itemsize * N * (3 + 8) * 2
    chunk = plan_chunks(len(idx), bytes_per_pixel, dev)
    for lo in range(0, len(idx), chunk):
        sel = idx[lo : lo + chunk]
        block = Y[sel]
        if len(sel) < chunk and lo > 0:
            block = np.concatenate(
                [block, np.repeat(block[:1], chunk - len(sel), axis=0)]
            )
        y = torch.as_tensor(np.ascontiguousarray(block), device=dev).to(dtype)
        ys = y.amax(-1).clamp_min(1.0)
        yn = y / ys[:, None]

        # --- 1-peak fit ---
        k0, m0 = linear_trend(yn, xh)
        A0, s0, mu0, degenerate = gaussian_seed(yn, xh, k0, m0)
        s0 = torch.where(degenerate, torch.ones_like(s0), s0)
        p1_0 = torch.stack([A0, s0, mu0, k0, m0], dim=-1)
        p1, ok1 = gauss_newton_batched(yn, xh, p1_0, gauss1d_lin, n_iter=n_iter_gauss_newton)
        ok1 = ok1 & ~degenerate

        # --- 2-peak fit ---
        A1s, sg1, mu1s, A2s, sg2, mu2s, has2 = two_peak_seed(yn, xh, k0, m0)
        # when no second local maximum exists (peaks merged into one hump),
        # seed a moment split: two peaks at envelope mean +/- envelope sigma
        s_env = torch.where(degenerate, torch.full_like(s0, 0.1), s0)
        A1s = torch.where(has2, A1s, 0.6 * A0.clamp_min(1e-3))
        A2s = torch.where(has2, A2s, 0.6 * A0.clamp_min(1e-3))
        mu1s = torch.where(has2, mu1s, mu0 - s_env)
        mu2s = torch.where(has2, mu2s, mu0 + s_env)
        sg1 = torch.where(has2, sg1, 0.6 * s_env)
        sg2 = torch.where(has2, sg2, 0.6 * s_env)
        p2_0 = torch.stack([A1s, sg1, mu1s, A2s, sg2, mu2s, k0, m0], dim=-1)
        # heavier damping + box projection: the 8-parameter model is prone to
        # runaway widths/centres; bounds are physical in scaled coordinates
        # (curves normalised to <=1, motor span scaled to [-1, 1])
        lo = torch.tensor(
            [0.0, 1e-4, -1.5, 0.0, 1e-4, -1.5, -10.0, -10.0], dtype=dtype, device=dev
        )
        hi = torch.tensor(
            [4.0, 2.0, 1.5, 4.0, 2.0, 1.5, 10.0, 10.0], dtype=dtype, device=dev
        )
        p2, ok2 = gauss_newton_batched(
            yn, xh, p2_0, gauss1d_two_lin, n_iter=n_iter_gauss_newton, lam=1e-2,
            bounds=(lo, hi),
        )
        ok2 = ok2 & ~degenerate

        # --- BIC on the scaled data ---
        f1, _ = gauss1d_lin(p1, xh)
        f2, _ = gauss1d_two_lin(p2, xh)
        rss1 = ((yn - f1) ** 2).sum(-1).clamp_min(1e-30)
        rss2 = ((yn - f2) ** 2).sum(-1).clamp_min(1e-30)
        n_pts = float(N)
        bic1 = n_pts * torch.log(rss1 / n_pts) + 5 * np.log(n_pts)
        bic2 = n_pts * torch.log(rss2 / n_pts) + 8 * np.log(n_pts)
        # a diverged fit (NaN residuals) or failed solve must lose selection
        big = torch.full_like(bic1, 1e30)
        bic1 = torch.where(torch.isfinite(bic1) & ok1, bic1, big)
        bic2 = torch.where(torch.isfinite(bic2) & ok2, bic2, big)

        # --- physical gates + model selection ---
        a1, w1, c1, a2, w2, c2, kk, mm = p2.unbind(-1)
        sep_ok = (c1 - c2).abs() > 2 * step_scaled
        amp_ok = (a1 > 0) & (a2 > 0)
        width_ok = (w1.abs() < 2.0) & (w2.abs() < 2.0) & (w1.abs() > 1e-6) & (w2.abs() > 1e-6)
        in_span = (c1.abs() < 1.5) & (c2.abs() < 1.5)
        two = ok2 & sep_ok & amp_ok & width_ok & in_span & (bic2 + delta_bic < bic1)

        # --- untransform; sort the two peaks by mu ---
        first_lower = c1 <= c2
        a_lo = torch.where(first_lower, a1, a2)
        w_lo = torch.where(first_lower, w1, w2)
        c_lo = torch.where(first_lower, c1, c2)
        a_hi = torch.where(first_lower, a2, a1)
        w_hi = torch.where(first_lower, w2, w1)
        c_hi = torch.where(first_lower, c2, c1)

        A_, sg_, mu_, k_, m_ = p1.unbind(-1)
        res1 = torch.stack(
            [
                A_ * ys,
                sg_.abs() * xs,
                xc + xs * mu_,
                k_ * ys / xs,
                (m_ - k_ * xc / xs) * ys,
                ok1.to(p1.dtype),
            ],
            dim=-1,
        )
        res2 = torch.stack(
            [
                a_lo * ys,
                w_lo.abs() * xs,
                xc + xs * c_lo,
                a_hi * ys,
                w_hi.abs() * xs,
                xc + xs * c_hi,
                kk * ys / xs,
                (mm - kk * xc / xs) * ys,
                ok2.to(p2.dtype),
            ],
            dim=-1,
        )
        n_peaks = torch.where(
            two, torch.full_like(ok1, 2, dtype=torch.uint8),
            torch.where(ok1, torch.ones_like(ok1, dtype=torch.uint8),
                        torch.zeros_like(ok1, dtype=torch.uint8)),
        )

        out1[sel] = res1.cpu().double().numpy()[: len(sel)]
        out2[sel] = res2.cpu().double().numpy()[: len(sel)]
        out_n[sel] = n_peaks.cpu().numpy()[: len(sel)]
        out_b1[sel] = bic1.cpu().double().numpy()[: len(sel)]
        out_b2[sel] = bic2.cpu().double().numpy()[: len(sel)]

    return {
        "params1": out1.reshape(ny, nx, 6),
        "params2": out2.reshape(ny, nx, 9),
        "n_peaks": out_n.reshape(ny, nx),
        "bic1": out_b1.reshape(ny, nx),
        "bic2": out_b2.reshape(ny, nx),
    }


def fit_2D_gaussian(data, coordinates, n_iter_gauss_newton=10, mask=None, device=None):
    """Fit a 2D Gaussian + constant background per pixel over a mosa grid.

    The inverse covariance is Cholesky-parameterised during the fit (positive
    definite by construction); the output reports the covariance matrix in
    motor units.

    Args:
        data (numpy.ndarray): shape (ny, nx, m, n).
        coordinates (numpy.ndarray): shape (2, m, n) motor meshgrids.
        n_iter_gauss_newton (int): Gauss-Newton iterations.
        mask (numpy.ndarray): optional (ny, nx) bool.
        device: torch device or name; None auto-detects.

    Returns:
        numpy.ndarray: (ny, nx, 8) float64,
        [A, mu0, mu1, cov00, cov01, cov11, c, success].
    """
    from ._models import gauss2d_const

    if data.ndim != 4:
        raise ValueError(f"data must be a 4d numpy array but got {data.ndim} dimensions")
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.shape != (2, *data.shape[2:]):
        raise ValueError(
            f"coordinates shape {coordinates.shape} does not match (2, {data.shape[2]}, {data.shape[3]})"
        )

    ny, nx = data.shape[:2]
    N = data.shape[2] * data.shape[3]
    Y = data.reshape(ny * nx, N)
    idx = np.flatnonzero(mask.ravel()) if mask is not None else np.arange(ny * nx)

    dev = get_device(device)
    dtype = compute_dtype(dev)

    c_flat = coordinates.reshape(2, N)
    xc = c_flat.mean(axis=1)
    xs = np.maximum(np.ptp(c_flat, axis=1) / 2.0, 1e-12)
    ch = torch.as_tensor((c_flat - xc[:, None]) / xs[:, None], dtype=dtype, device=dev)

    out = np.zeros((ny * nx, 8), dtype=np.float64)

    bytes_per_pixel = dtype.itemsize * N * (3 + 7) * 2
    chunk = plan_chunks(len(idx), bytes_per_pixel, dev)
    for lo in range(0, len(idx), chunk):
        sel = idx[lo : lo + chunk]
        block = Y[sel]
        if len(sel) < chunk and lo > 0:
            block = np.concatenate(
                [block, np.repeat(block[:1], chunk - len(sel), axis=0)]
            )
        y = torch.as_tensor(np.ascontiguousarray(block), device=dev).to(dtype)
        ys = y.amax(-1).clamp_min(1.0)
        yn = y / ys[:, None]

        # seeds from weighted moments of the background-subtracted signal
        c0 = yn.amin(-1)
        w = (yn - c0[:, None]).clamp_min(0.0)
        I = w.sum(-1).clamp_min(1e-12)
        mu = (w @ ch.T) / I[:, None]  # (P, 2)
        d0 = ch[0][None, :] - mu[:, 0:1]
        d1 = ch[1][None, :] - mu[:, 1:2]
        v00 = (w * d0 * d0).sum(-1) / I
        v01 = (w * d0 * d1).sum(-1) / I
        v11 = (w * d1 * d1).sum(-1) / I
        det = (v00 * v11 - v01 * v01).clamp_min(1e-12)
        degenerate = (v00 <= 1e-10) | (v11 <= 1e-10) | (I <= 1e-6)
        # precision = inv(cov), then its Cholesky factor L (lower)
        p00 = (v11 / det).clamp_min(1e-12)
        p01 = -v01 / det
        p11 = (v00 / det).clamp_min(1e-12)
        L00 = torch.sqrt(p00)
        L10 = p01 / L00
        L11 = torch.sqrt((p11 - L10 * L10).clamp_min(1e-12))
        A0 = (yn.amax(-1) - c0).clamp_min(1e-3)
        L00 = torch.where(degenerate, torch.ones_like(L00), L00)
        L10 = torch.where(degenerate, torch.zeros_like(L10), L10)
        L11 = torch.where(degenerate, torch.ones_like(L11), L11)

        p0 = torch.stack([A0, mu[:, 0], mu[:, 1], L00, L10, L11, c0], dim=-1)
        params, ok = gauss_newton_batched(
            yn, ch, p0, gauss2d_const, n_iter=n_iter_gauss_newton
        )
        ok = ok & ~degenerate

        A_, m0_, m1_, l00, l10, l11, cc = params.unbind(-1)
        # covariance from the fitted precision LL^T, back in motor units
        q00 = l00 * l00
        q01 = l00 * l10
        q11 = l10 * l10 + l11 * l11
        detq = (q00 * q11 - q01 * q01).clamp_min(1e-30)
        cov00 = (q11 / detq) * xs[0] * xs[0]
        cov01 = (-q01 / detq) * xs[0] * xs[1]
        cov11 = (q00 / detq) * xs[1] * xs[1]

        res = torch.stack(
            [
                A_ * ys,
                xc[0] + xs[0] * m0_,
                xc[1] + xs[1] * m1_,
                cov00,
                cov01,
                cov11,
                cc * ys,
                (ok & ~degenerate).to(params.dtype),
            ],
            dim=-1,
        )
        out[sel] = res.cpu().double().numpy()[: len(sel)]

    return out.reshape(ny, nx, 8)
