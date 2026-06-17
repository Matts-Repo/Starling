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
from ._models import gauss1d_lin, gaussND_const, pseudovoigt1d_lin
from ._results import GaussNDResult, PseudoVoigtResult


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
        # cast to the compute dtype on the host *before* moving to the device:
        # float64 cannot live on MPS, so as_tensor(..., device=mps) on a float64
        # block would raise before the .to(dtype) downcast could run
        y = torch.as_tensor(np.ascontiguousarray(block), dtype=dtype).to(dev)
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


def fit_1D_pseudo_voigt(data, coordinates, n_iter_gauss_newton=10, mask=None, device=None):
    """Fit a pseudo-Voigt + linear background per pixel (batched on GPU).

    f(x) = A [ (1 - eta) exp(-(x-mu)^2 / (2 sigma^2))
               + eta / (1 + ((x-mu)/gamma)^2) ] + k x + m

    For Lorentzian-tailed rocking curves a pure Gaussian biases the width; the
    pseudo-Voigt mixes Gaussian and Lorentzian via ``eta`` in [0, 1] (0 = pure
    Gaussian, 1 = pure Lorentzian). ``eta`` is box-constrained to [0, 1] by the
    Gauss-Newton driver; ``gamma`` is seeded near ``sigma`` and ``eta`` at 0.5.

    Args:
        data (numpy.ndarray): shape (ny, nx, m).
        coordinates: length-1 sequence holding the (m,) motor coordinate array.
        n_iter_gauss_newton (int): Gauss-Newton iterations. Defaults to 10.
        mask (numpy.ndarray): optional (ny, nx) bool; only True pixels are fit.
        device: torch device or name; None auto-detects.

    Returns:
        PseudoVoigtResult: fields A, sigma, mu, gamma, eta, k, m, success
        (each (ny, nx)).
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
    idx = np.flatnonzero(mask.ravel()) if mask is not None else np.arange(ny * nx)

    dev = get_device(device)
    dtype = compute_dtype(dev)
    xc = float(x.mean())
    xs = float(max(np.ptp(x) / 2.0, 1e-12))
    xh = torch.as_tensor((x - xc) / xs, dtype=dtype, device=dev)

    out = np.zeros((ny * nx, 8), dtype=np.float64)

    p = 7
    bytes_per_pixel = dtype.itemsize * N * (3 + p) * 2
    chunk = plan_chunks(len(idx), bytes_per_pixel, dev)
    # eta is bounded to [0, 1]; the other parameters are left free (+/- inf)
    lo = torch.tensor(
        [-torch.inf, -torch.inf, -torch.inf, -torch.inf, 0.0, -torch.inf, -torch.inf],
        dtype=dtype, device=dev,
    )
    hi = torch.tensor(
        [torch.inf, torch.inf, torch.inf, torch.inf, 1.0, torch.inf, torch.inf],
        dtype=dtype, device=dev,
    )
    for lo_i in range(0, len(idx), chunk):
        sel = idx[lo_i : lo_i + chunk]
        block = Y[sel]
        n_pad = 0
        if len(sel) < chunk and lo_i > 0:
            n_pad = chunk - len(sel)
            block = np.concatenate([block, np.repeat(block[:1], n_pad, axis=0)])
        # cast to the compute dtype on the host *before* moving to the device:
        # float64 cannot live on MPS, so as_tensor(..., device=mps) on a float64
        # block would raise before the .to(dtype) downcast could run
        y = torch.as_tensor(np.ascontiguousarray(block), dtype=dtype).to(dev)
        ys = y.amax(-1).clamp_min(1.0)
        yn = y / ys[:, None]

        k0, m0 = linear_trend(yn, xh)
        A0, s0, mu0, degenerate = gaussian_seed(yn, xh, k0, m0)
        s0 = torch.where(degenerate, torch.ones_like(s0), s0)
        gamma0 = s0.clamp_min(1e-3)
        eta0 = torch.full_like(A0, 0.5)
        params0 = torch.stack([A0, s0, mu0, gamma0, eta0, k0, m0], dim=-1)

        params, ok = gauss_newton_batched(
            yn, xh, params0, pseudovoigt1d_lin,
            n_iter=n_iter_gauss_newton, bounds=(lo, hi),
        )
        success = ok & ~degenerate

        A, sigma, mu, gamma, eta, k, m = params.unbind(-1)
        res = torch.stack(
            [
                A * ys,
                sigma.abs() * xs,
                xc + xs * mu,
                gamma.abs() * xs,
                eta.clamp(0.0, 1.0),
                k * ys / xs,
                (m - k * xc / xs) * ys,
                success.to(params.dtype),
            ],
            dim=-1,
        )
        out[sel] = res.cpu().double().numpy()[: len(sel)]

    return PseudoVoigtResult.from_raw(out.reshape(ny, nx, 8))


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
        # cast to the compute dtype on the host *before* moving to the device:
        # float64 cannot live on MPS, so as_tensor(..., device=mps) on a float64
        # block would raise before the .to(dtype) downcast could run
        y = torch.as_tensor(np.ascontiguousarray(block), dtype=dtype).to(dev)
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


# per-dimension cached model closure: gaussND_const needs D as a Python
# constant, and gauss_newton_batched caches its compiled step by the callable's
# identity — so one stable closure per D compiles the step exactly once per
# (D, device) instead of recompiling every chunk.
_ND_MODEL_CACHE = {}


def _nd_model(D):
    if D not in _ND_MODEL_CACHE:
        _ND_MODEL_CACHE[D] = lambda params, x: gaussND_const(params, x, D)
    return _ND_MODEL_CACHE[D]


def _safe_chol_lower(M):
    """Batched lower Cholesky in numpy with clamped pivots (never raises).

    M: (P, D, D) symmetric; returns L (P, D, D) lower-triangular with
    ``L L^T ~= M``. Mirrors the MPS-safe unrolled solve in ``_solve``.
    """
    P, D, _ = M.shape
    L = np.zeros_like(M)
    for j in range(D):
        s = M[:, j, j] - (L[:, j, :j] ** 2).sum(-1)
        d = np.sqrt(np.clip(s, 1e-37, None))
        L[:, j, j] = d
        if j + 1 < D:
            L[:, j + 1 :, j] = (
                M[:, j + 1 :, j] - (L[:, j + 1 :, :j] * L[:, j : j + 1, :j]).sum(-1)
            ) / d[:, None]
    return L


def _fit_ND_engine(data, coordinates, n_iter_gauss_newton, mask, device):
    """Core N-D Gaussian + constant-background fit; returns a dict of maps.

    Generalises the 2D mosa fit to arbitrary D. The inverse covariance is
    Cholesky-parameterised during the fit (PSD by construction); the output
    reports the covariance matrix in motor units.
    """
    coordinates = np.asarray(coordinates, dtype=np.float64)
    D = coordinates.shape[0]
    if data.ndim != D + 2:
        raise ValueError(
            f"data must be a {D + 2}d numpy array for {D} motor dimensions but "
            f"got {data.ndim} dimensions"
        )
    if coordinates.shape != (D, *data.shape[2:]):
        raise ValueError(
            f"coordinates shape {coordinates.shape} does not match "
            f"(D={D}, {data.shape[2:]})"
        )

    ny, nx = data.shape[:2]
    N = int(np.prod(data.shape[2:]))
    Y = data.reshape(ny * nx, N)
    idx = np.flatnonzero(mask.ravel()) if mask is not None else np.arange(ny * nx)

    dev = get_device(device)
    dtype = compute_dtype(dev)

    c_flat = coordinates.reshape(D, N)
    xc = c_flat.mean(axis=1)  # (D,)
    xs = np.maximum(np.ptp(c_flat, axis=1) / 2.0, 1e-12)  # (D,)
    ch = torch.as_tensor((c_flat - xc[:, None]) / xs[:, None], dtype=dtype, device=dev)

    model = _nd_model(D)
    n_L = D * (D + 1) // 2
    n_p = 1 + D + n_L + 1
    tril = [(r, s) for r in range(D) for s in range(r + 1)]  # row-major lower-tri
    eye = np.eye(D)

    out_A = np.zeros(ny * nx)
    out_mu = np.zeros((ny * nx, D))
    out_cov = np.zeros((ny * nx, D, D))
    out_c = np.zeros(ny * nx)
    out_s = np.zeros(ny * nx)

    bytes_per_pixel = dtype.itemsize * N * (3 + n_p) * 2
    chunk = plan_chunks(len(idx), bytes_per_pixel, dev)
    for lo in range(0, len(idx), chunk):
        sel = idx[lo : lo + chunk]
        block = Y[sel]
        if len(sel) < chunk and lo > 0:
            block = np.concatenate(
                [block, np.repeat(block[:1], chunk - len(sel), axis=0)]
            )
        # cast to the compute dtype on the host *before* moving to the device:
        # float64 cannot live on MPS, so as_tensor(..., device=mps) on a float64
        # block would raise before the .to(dtype) downcast could run
        y = torch.as_tensor(np.ascontiguousarray(block), dtype=dtype).to(dev)
        ys = y.amax(-1).clamp_min(1.0)
        yn = y / ys[:, None]

        # --- seed from weighted moments of the background-subtracted signal ---
        c0 = yn.amin(-1)
        w = (yn - c0[:, None]).clamp_min(0.0)
        I = w.sum(-1).clamp_min(1e-12)
        mu = (w @ ch.T) / I[:, None]  # (P, D)
        d = ch[None, :, :] - mu[:, :, None]  # (P, D, N)
        cov = torch.einsum("pin,pjn->pij", d * w[:, None, :], d) / I[:, None, None]
        diagcov = torch.einsum("...ii->...i", cov)
        degenerate = (diagcov <= 1e-10).any(-1) | (I <= 1e-6)
        A0 = (yn.amax(-1) - c0).clamp_min(1e-3)

        # precision = inv(cov), L = chol(precision) — done in numpy (robust,
        # MPS has no batched linalg) on degenerate-safe copies
        cov_np = cov.detach().cpu().double().numpy()
        deg_np = degenerate.cpu().numpy()
        mu_np = mu.detach().cpu().double().numpy()
        cov_safe = cov_np.copy()
        cov_safe[deg_np] = eye
        cov_safe = 0.5 * (cov_safe + np.swapaxes(cov_safe, -1, -2))
        precision = np.linalg.inv(cov_safe)
        precision = 0.5 * (precision + np.swapaxes(precision, -1, -2)) + 1e-9 * eye
        L_seed = _safe_chol_lower(precision)
        L_seed[deg_np] = eye
        mu_np[deg_np] = 0.0

        L_flat = np.stack([L_seed[:, r, s] for (r, s) in tril], axis=-1)  # (P, n_L)
        seed = np.concatenate(
            [
                A0.detach().cpu().double().numpy()[:, None],
                mu_np,
                L_flat,
                c0.detach().cpu().double().numpy()[:, None],
            ],
            axis=1,
        )
        p0 = torch.as_tensor(seed, dtype=dtype, device=dev)

        params, ok = gauss_newton_batched(yn, ch, p0, model, n_iter=n_iter_gauss_newton)
        ok = ok & ~degenerate

        # --- un-transform back to motor units ---
        params_np = params.detach().cpu().double().numpy()
        ys_np = ys.detach().cpu().double().numpy()
        A_f = params_np[:, 0] * ys_np
        mu_f = xc[None, :] + xs[None, :] * params_np[:, 1 : 1 + D]
        c_f = params_np[:, 1 + D + n_L] * ys_np

        Lf = np.zeros((len(params_np), D, D))
        for t, (r, s) in enumerate(tril):
            Lf[:, r, s] = params_np[:, 1 + D + t]
        precision_f = Lf @ np.swapaxes(Lf, -1, -2)  # L L^T (scaled coords)
        det_f = np.linalg.det(precision_f)
        bad = ~np.isfinite(det_f) | (np.abs(det_f) < 1e-300)
        precision_f[bad] = eye
        cov_f = np.linalg.inv(precision_f)
        cov_f = cov_f * (xs[None, :, None] * xs[None, None, :])  # back to motor units

        # zero the outputs of non-converged / degenerate pixels so no garbage
        # (e.g. a window-mean mu or an inverted dummy covariance) leaks into the
        # result maps or into mosaicity()/orientation(), which do not re-mask
        ok_np = ok.detach().cpu().numpy()
        A_f = np.where(ok_np, A_f, 0.0)
        mu_f = np.where(ok_np[:, None], mu_f, 0.0)
        cov_f = np.where(ok_np[:, None, None], cov_f, 0.0)
        c_f = np.where(ok_np, c_f, 0.0)

        keep = slice(0, len(sel))
        out_A[sel] = A_f[keep]
        out_mu[sel] = mu_f[keep]
        out_cov[sel] = cov_f[keep]
        out_c[sel] = c_f[keep]
        out_s[sel] = ok_np[keep].astype(float)

    return {
        "A": out_A.reshape(ny, nx),
        "mu": out_mu.reshape(ny, nx, D),
        "cov": out_cov.reshape(ny, nx, D, D),
        "c": out_c.reshape(ny, nx),
        "success": out_s.reshape(ny, nx),
    }


def fit_ND_gaussian(data, coordinates, n_iter_gauss_newton=10, mask=None, device=None):
    """Fit an N-D Gaussian + constant background per pixel (batched on GPU).

    A single per-pixel Gaussian fit in an arbitrary number of motor dimensions
    (D = 1, 2, 3, ...). The number of dimensions D is inferred from
    ``coordinates``. A 3-D strain-mosa scan (e.g. chi x mu x 2theta) is fit
    directly. The inverse covariance is Cholesky-parameterised during the fit
    (positive (semi-)definite by construction, no bounds on L); the result
    reports the covariance matrix in motor units.

    Args:
        data (numpy.ndarray): shape (ny, nx, *grid) with ``len(grid) == D``.
        coordinates (numpy.ndarray): shape (D, *grid) motor meshgrids.
        n_iter_gauss_newton (int): Gauss-Newton iterations. Defaults to 10.
        mask (numpy.ndarray): optional (ny, nx) bool; only True pixels are fit.
        device: torch device or name; None auto-detects.

    Returns:
        GaussNDResult: with fields A (ny, nx), mu (ny, nx, D),
        cov (ny, nx, D, D), c (ny, nx), success (ny, nx).
    """
    maps = _fit_ND_engine(data, coordinates, n_iter_gauss_newton, mask, device)
    return GaussNDResult(**maps)


def fit_2D_gaussian(data, coordinates, n_iter_gauss_newton=10, mask=None, device=None):
    """Fit a 2D Gaussian + constant background per pixel over a mosa grid.

    Thin wrapper around :func:`fit_ND_gaussian` (D=2) that returns the legacy
    flat array for back-compat. The inverse covariance is Cholesky-parameterised
    during the fit; the output reports the covariance matrix in motor units.

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
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if data.ndim != 4:
        raise ValueError(f"data must be a 4d numpy array but got {data.ndim} dimensions")
    if coordinates.shape != (2, *data.shape[2:]):
        raise ValueError(
            f"coordinates shape {coordinates.shape} does not match "
            f"(2, {data.shape[2]}, {data.shape[3]})"
        )
    return fit_ND_gaussian(
        data, coordinates, n_iter_gauss_newton=n_iter_gauss_newton,
        mask=mask, device=device,
    ).raw
