"""Edge-constrained refit: recover peak centres for scan-range-truncated pixels.

When a peak is clipped by the scan range, the free N-D Gaussian fit is
degenerate (centre and width trade off along the visible flank) and the
solver diverges or collapses to flat background — the fit_status
EDGE_CLIPPED/FAILED pixels. The flank still pins the centre *given a width*,
so this module refits those pixels with the covariance FIXED at the grain
median of the healthy fits: free parameters are amplitude, centre and
background only, with the centre bounded to the scan window plus a small
margin. The problem is then well-posed and the solver is stable.

Accuracy (cross-validated on MA7031 FN1_BC_001 by artificially truncating
150 healthy single-peak pixels and comparing against their full-data fits;
mu step 25 mdeg, grain sigma_mu 78 mdeg):

    apex 1 sigma inside the edge :  median error  +1 mdeg, p90 32 mdeg
    apex AT the edge             :  median error  -5 mdeg, p90 79 mdeg
    apex 0.5 sigma outside       :  median error -11 mdeg (inward bias)
    apex 1 sigma outside         :  median error -36 mdeg
    apex 1.5 sigma outside       :  median error -74 mdeg

i.e. quantitative while the apex is within ~half a grain-sigma of the edge,
qualitative (systematically biased toward the window) beyond that. Refitted
pixels keep their EDGE_CLIPPED/FAILED status — these are constrained
estimates, not free measurements — and the refit covariance is the fixed
prior, NOT a measured width.
"""

import numpy as np

from ._results import GaussNDResult


def _gated_moment(y, X, base_pct=20.0, nsigma=5.0):
    """Noise-gated weighted first moment per pixel.

    A plain background-subtracted moment is dragged toward the window centre
    by residual baseline noise (thousands of small positive weights across
    the grid can outweigh a sharp 2-3 point peak). Gating the weights at
    ``base + nsigma*sqrt(base)`` (Poisson-scale) keeps only genuine peak
    voxels, which is what makes the moment usable as a centre estimate for
    severely truncated peaks. This is darfix's moment-fallback idea with a
    noise gate added.

    Args:
        y (numpy.ndarray): (P, N) per-pixel curves.
        X (numpy.ndarray): (N, D) motor coordinates.
        base_pct (float): percentile used as the baseline estimate.
        nsigma (float): gate at base + nsigma*sqrt(base + 1).

    Returns:
        tuple: mu (P, D) gated-moment centres (argmax voxel where nothing
        passes the gate), A0 (P,) max-minus-baseline, c0 (P,) baseline.
    """
    base = np.percentile(y, base_pct, axis=1)
    thr = base + nsigma * np.sqrt(np.clip(base, 0.0, None) + 1.0)
    w = np.where(y > thr[:, None], y - base[:, None], 0.0)
    wsum = w.sum(axis=1)
    mu = np.empty((y.shape[0], X.shape[1]))
    ok = wsum > 0
    if ok.any():
        mu[ok] = (w[ok] @ X) / wsum[ok, None]
    if (~ok).any():
        mu[~ok] = X[np.argmax(y[~ok], axis=1)]
    A0 = np.clip(y.max(axis=1) - base, 1e-3, None)
    return mu, A0, base


def median_healthy_cov(result, status, sigma_range=(1e-3, None)):
    """Grain-median covariance from strictly-healthy fitted pixels.

    Args:
        result (GaussNDResult): the free fit.
        status (numpy.ndarray): (ny, nx) fit_status map; OK==1 pixels are used.
        sigma_range (tuple): (lo, hi) sanity band on every per-axis sigma in
            motor units; hi ``None`` uses half the largest motor span estimate
            from the covariance itself (loose upper bound skipped).

    Returns:
        numpy.ndarray: (D, D) median covariance.
    """
    cov = np.asarray(result.cov, dtype=np.float64)
    ok = (np.asarray(status) == 1) & (np.asarray(result.A) > 0)
    diag = np.einsum("...ii->...i", cov)
    sane = ok & (diag > sigma_range[0] ** 2).all(-1)
    if sigma_range[1] is not None:
        sane &= (diag < sigma_range[1] ** 2).all(-1)
    if not sane.any():
        raise ValueError("no healthy pixels to derive the reference covariance")
    return np.median(cov[sane], axis=0)


def fit_ND_fixed_cov(data, coordinates, cov, mask, device=None, n_iter=30,
                     mu_pad_steps=8.0, chunk_px=1024):
    """Batched N-D Gaussian fit with a FIXED covariance (A, mu, c free).

    Args:
        data (numpy.ndarray): (ny, nx, *grid) intensity cube.
        coordinates (numpy.ndarray): (D, *grid) motor meshgrids.
        cov (numpy.ndarray): (D, D) fixed covariance in motor units.
        mask (numpy.ndarray): (ny, nx) bool — pixels to fit.
        device: torch device or name; None auto-detects.
        n_iter (int): maximum Gauss-Newton iterations.
        mu_pad_steps (float): centre bound margin beyond the scan window, in
            motor grid steps per axis.
        chunk_px (int): pixels per device batch.

    Returns:
        dict: A, mu, c (masked pixels filled, others zero), success (ny, nx).
    """
    import torch

    from ..device import get_device
    from ._gaussnewton import gauss_newton_batched

    dev = get_device(device)
    dtype = torch.float32 if dev.type in ("cuda", "mps") else torch.float64

    coordinates = np.asarray(coordinates, dtype=np.float64)
    D = coordinates.shape[0]
    ny, nx = data.shape[:2]
    X = coordinates.reshape(D, -1)  # (D, N)
    N = X.shape[1]

    # whitening: xh = (X - xc) / xs per axis
    xc = X.mean(axis=1)
    xs = np.maximum((X.max(axis=1) - X.min(axis=1)) / 2.0, 1e-12)
    xh = ((X - xc[:, None]) / xs[:, None]).T  # (N, D)

    cov_w = np.diag(1.0 / xs) @ np.asarray(cov, dtype=np.float64) @ np.diag(1.0 / xs)
    Sinv_w = np.linalg.inv(cov_w)

    # per-axis grid steps (whitened) for the centre bounds
    steps = []
    for i in range(D):
        vals = np.unique(coordinates[i].ravel())
        d = np.diff(vals)
        steps.append(float(np.median(d[d > 0])) if (d > 0).any() else 0.0)
    lo_mu = (X.min(axis=1) - mu_pad_steps * np.asarray(steps) - xc) / xs
    hi_mu = (X.max(axis=1) + mu_pad_steps * np.asarray(steps) - xc) / xs

    idx = np.argwhere(np.asarray(mask, dtype=bool))
    P = len(idx)
    A_out = np.zeros((ny, nx)); c_out = np.zeros((ny, nx))
    mu_out = np.zeros((ny, nx, D)); success = np.zeros((ny, nx))
    if P == 0:
        return {"A": A_out, "mu": mu_out, "c": c_out, "success": success}

    Sinv_t = torch.as_tensor(Sinv_w, dtype=dtype, device=dev)
    xh_t = torch.as_tensor(xh, dtype=dtype, device=dev)  # (N, D)

    def model_and_jac(params, x):
        # params (P, 2 + D): [A, mu_0..mu_{D-1}, c]
        A = params[:, 0:1]                       # (P, 1)
        mu = params[:, 1:1 + D]                  # (P, D)
        c = params[:, 1 + D:2 + D]               # (P, 1)
        d = x.unsqueeze(0) - mu.unsqueeze(1)     # (P, N, D)
        Sd = torch.einsum("ij,pnj->pni", Sinv_t, d)
        q = (d * Sd).sum(-1)                     # (P, N)
        e = torch.exp(-0.5 * q.clamp(max=60.0))
        f = A * e + c
        J = torch.empty((*e.shape, 2 + D), dtype=e.dtype, device=e.device)
        J[..., 0] = e
        J[..., 1:1 + D] = (A * e).unsqueeze(-1) * Sd
        J[..., 1 + D] = 1.0
        return f, J

    flat = data.reshape(ny, nx, -1)
    lo = torch.as_tensor(np.concatenate([[0.0], lo_mu, [0.0]]), dtype=dtype, device=dev)
    hi_c = 1.0  # normalized background can't exceed the per-pixel max
    hi = torch.as_tensor(np.concatenate([[50.0], hi_mu, [hi_c]]), dtype=dtype, device=dev)

    for k0 in range(0, P, chunk_px):
        sel = idx[k0:k0 + chunk_px]
        y = flat[sel[:, 0], sel[:, 1]].astype(np.float64)  # (p, N)
        ys = np.maximum(y.max(axis=1), 1.0)
        yn = y / ys[:, None]

        # seeds: per-axis marginal argmax + low-percentile bg. (Gated-moment
        # seeds were tried and moved ~10% of pixels from the validated GN tier
        # into the fallback tier on real data — argmax seeds converge more.)
        seed_mu = np.empty((len(sel), D))
        patch = y.reshape(len(sel), *data.shape[2:])
        for i in range(D):
            other = tuple(1 + j for j in range(D) if j != i)
            prof = patch.sum(axis=other)
            am = prof.argmax(axis=1)
            axis_vals = np.moveaxis(coordinates[i], i, 0).reshape(data.shape[2 + i], -1)[:, 0]
            seed_mu[:, i] = (axis_vals[am] - xc[i]) / xs[i]
        c0 = np.percentile(yn, 20, axis=1)
        A0 = np.clip(yn.max(axis=1) - c0, 1e-3, None)
        p0 = np.concatenate([A0[:, None], seed_mu, c0[:, None]], axis=1)

        y_t = torch.as_tensor(yn, dtype=dtype, device=dev)
        p0_t = torch.as_tensor(p0, dtype=dtype, device=dev)
        p0_t = torch.clamp(p0_t, lo, hi)
        params, ok = gauss_newton_batched(
            y_t, xh_t, p0_t, model_and_jac, n_iter=n_iter, lam=1e-2,
            adaptive=True, bounds=(lo, hi),
        )
        pn = params.detach().cpu().double().numpy()
        A_out[sel[:, 0], sel[:, 1]] = pn[:, 0] * ys
        mu_out[sel[:, 0], sel[:, 1]] = xc[None, :] + xs[None, :] * pn[:, 1:1 + D]
        c_out[sel[:, 0], sel[:, 1]] = pn[:, 1 + D] * ys
        okn = ok.detach().cpu().numpy() & (pn[:, 0] > 0)
        success[sel[:, 0], sel[:, 1]] = okn.astype(np.float64)

    return {"A": A_out, "mu": mu_out, "c": c_out, "success": success}


def refit_edge_pixels(data, coordinates, result, status, cov=None, device=None,
                      n_iter=30, mu_pad_steps=8.0, chunk_px=1024,
                      fallback="moments", return_source=False):
    """Constrained refit of EDGE_CLIPPED/FAILED pixels; returns merged result.

    Pixels with ``status`` 2 (edge-clipped) or 3 (failed) are refit with the
    covariance fixed at the grain median of the healthy fits (or ``cov`` if
    given). The returned result carries the refitted A/mu/c at those pixels
    (their ``cov`` entries become the fixed prior — a width ASSUMPTION, not a
    measurement) and the original values everywhere else. ``status`` is NOT
    modified: refitted pixels remain flagged, because these are constrained
    estimates whose accuracy degrades with truncation depth (see module
    docstring for the measured bias curve).

    Args:
        data, coordinates: the fitted cube and motor grids.
        result (GaussNDResult): the free fit.
        status (numpy.ndarray): (ny, nx) from classify_fit_status.
        cov (numpy.ndarray, optional): (D, D) fixed covariance override.
        device, n_iter, mu_pad_steps, chunk_px: see :func:`fit_ND_fixed_cov`.

    Pixels where even the constrained fit fails get a ``fallback``
    estimate (default "moments"): the noise-gated weighted first moment of
    the raw curve, clamped to the scan window plus margin — darfix writes
    its (ungated) moment seed into the maps silently on fit failure; here
    the same idea is gated against baseline drag and flagged. ``None``
    disables the fallback tier.

    Returns:
        tuple: (merged GaussNDResult, refit_mask (ny, nx) bool — True where a
        replacement value landed). With ``return_source=True`` a third
        (ny, nx) int8 array: 0 untouched, 2 constrained refit, 3 moment
        fallback.
    """
    status = np.asarray(status)
    target = (status == 2) | (status == 3)
    if cov is None:
        cov = median_healthy_cov(result, status)
    cov = np.asarray(cov, dtype=np.float64)

    out = GaussNDResult(
        A=np.array(result.A, copy=True), mu=np.array(result.mu, copy=True),
        cov=np.array(result.cov, copy=True), c=np.array(result.c, copy=True),
        success=np.array(result.success, copy=True),
    )
    source = np.zeros(target.shape, dtype=np.int8)
    if not target.any():
        return (out, target.copy(), source) if return_source else (out, target.copy())

    fit = fit_ND_fixed_cov(
        data, coordinates, cov, target, device=device, n_iter=n_iter,
        mu_pad_steps=mu_pad_steps, chunk_px=chunk_px,
    )
    refit_ok = target & (fit["success"] > 0.5) & (fit["A"] > 0)
    out.A[refit_ok] = fit["A"][refit_ok]
    out.mu[refit_ok] = fit["mu"][refit_ok]
    out.c[refit_ok] = fit["c"][refit_ok]
    out.cov[refit_ok] = cov  # the prior, flagged via refit_mask/source
    source[refit_ok] = 2

    leftover = target & ~refit_ok
    if fallback == "moments" and leftover.any():
        ny, nx = target.shape
        D = np.asarray(coordinates).shape[0]
        X = np.asarray(coordinates, dtype=np.float64).reshape(D, -1).T
        idx = np.argwhere(leftover)
        y = data.reshape(ny, nx, -1)[idx[:, 0], idx[:, 1]].astype(np.float64)
        mu_m, A0, base = _gated_moment(y, X)
        # clamp into the scan window + the same margin the refit allows
        for i in range(D):
            vals = np.unique(np.asarray(coordinates)[i].ravel())
            d = np.diff(vals)
            step = float(np.median(d[d > 0])) if (d > 0).any() else 0.0
            pad = mu_pad_steps * step
            mu_m[:, i] = np.clip(mu_m[:, i], vals[0] - pad, vals[-1] + pad)
        out.A[idx[:, 0], idx[:, 1]] = A0
        out.mu[idx[:, 0], idx[:, 1]] = mu_m
        out.c[idx[:, 0], idx[:, 1]] = base
        out.cov[idx[:, 0], idx[:, 1]] = cov
        source[idx[:, 0], idx[:, 1]] = 3

    refit_mask = source > 0
    return (out, refit_mask, source) if return_source else (out, refit_mask)
