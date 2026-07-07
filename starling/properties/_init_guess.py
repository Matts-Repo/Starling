"""Vectorised initial-guess routines.

Batched linear-trend estimation and Gaussian parameter seed, plus a top-2
local-maxima seed for the two-peak fit.
"""

import torch
import torch.nn.functional as F


def linear_trend(y, x):
    """Least-squares line y = k x + m per curve.

    Args:
        y: (P, N) tensor.
        x: (N,) tensor.

    Returns:
        tuple: k (P,), m (P,).
    """
    n = x.shape[0]
    Sx = x.sum()
    Sxx = (x * x).sum()
    Sy = y.sum(-1)
    Sxy = (y * x[None, :]).sum(-1)
    den = n * Sxx - Sx * Sx
    if den != 0.0:
        k = (n * Sxy - Sx * Sy) / den
        m = (Sy - k * Sx) / n
    else:
        k = torch.zeros_like(Sy)
        m = Sy / n
    return k, m


def gaussian_seed(y, x, k_bg, m_bg):
    """Initial [A, sigma, mu] from weighted moments of the positive residual.

    Degenerate curves (no positive residual mass, non-positive variance, or
    sigma <= 1e-8) are flagged in the returned mask
    with A = sigma = mu = 0.

    Args:
        y: (P, N) tensor.
        x: (N,) tensor.
        k_bg, m_bg: (P,) background estimates.

    Returns:
        tuple: A (P,), sigma (P,), mu (P,), degenerate (P,) bool.
    """
    w = y - (k_bg[:, None] * x[None, :] + m_bg[:, None])
    w = w.clamp_min(0.0)
    sumw = w.sum(-1)
    sumwx = (w * x[None, :]).sum(-1)
    sumwxx = (w * (x * x)[None, :]).sum(-1)
    max_w = w.amax(-1)

    safe = sumw > 0
    mu = torch.where(safe, sumwx / sumw.clamp_min(1e-37), torch.zeros_like(sumw))
    var = torch.where(safe, sumwxx / sumw.clamp_min(1e-37) - mu * mu, torch.zeros_like(sumw))
    sigma = torch.sqrt(var.clamp_min(0.0))

    degenerate = (~safe) | (var <= 0) | (sigma <= 1e-8)
    A = torch.where(max_w > 0, max_w, y.amax(-1))
    A = torch.where(degenerate, torch.zeros_like(A), A)
    sigma = torch.where(degenerate, torch.zeros_like(sigma), sigma)
    mu = torch.where(degenerate, torch.zeros_like(mu), mu)
    return A, sigma, mu, degenerate


def two_peak_seed(y, x, k_bg, m_bg, sigma_smooth=1.5, min_separation_samples=3):
    """Seeds for the two-Gaussian fit from the top-2 smoothed local maxima.

    The background-subtracted curve is smoothed with a small Gaussian kernel
    and its two highest local maxima seed [mu1, mu2]. Curves with only one
    detected maximum get a degenerate second seed next to the first (the
    two-peak model then loses BIC selection naturally).

    Args:
        y: (P, N) tensor (normalised curves).
        x: (N,) tensor (scaled coordinates, assumed near-uniform).
        k_bg, m_bg: (P,) linear background estimates.
        sigma_smooth: smoothing width in samples.

    Returns:
        tuple: A1, sig1, mu1, A2, sig2, mu2 (each (P,)), has2 (P,) bool.
    """
    resid = (y - (k_bg[:, None] * x[None, :] + m_bg[:, None])).clamp_min(0.0)
    half = max(1, int(3 * sigma_smooth))
    t = torch.arange(-half, half + 1, device=y.device, dtype=y.dtype)
    ker = torch.exp(-0.5 * (t / sigma_smooth) ** 2)
    ker = ker / ker.sum()
    s = F.conv1d(resid.unsqueeze(1), ker.view(1, 1, -1), padding=half).squeeze(1)

    ismax = torch.zeros_like(s, dtype=torch.bool)
    ismax[:, 1:-1] = (s[:, 1:-1] > s[:, :-2]) & (s[:, 1:-1] >= s[:, 2:])
    vals = torch.where(ismax, s, torch.full_like(s, -1.0))

    # first peak, then mask an exclusion window around it before picking the
    # second — otherwise Poisson noise puts both maxima on the same peak and
    # the two-Gaussian fit collapses
    v1, i1 = vals.max(dim=-1)
    pos = torch.arange(s.shape[-1], device=s.device)
    excl = (pos[None, :] - i1[:, None]).abs() <= min_separation_samples
    v2, i2 = torch.where(excl, torch.full_like(vals, -1.0), vals).max(dim=-1)

    step = (x[1] - x[0]).abs()
    mu1 = x[i1]
    has2 = v2 > 0
    mu2 = torch.where(has2, x[i2], mu1 + 2 * step)
    A1 = v1.clamp_min(1e-3)
    A2 = torch.where(has2, v2, 0.5 * A1).clamp_min(1e-3)
    sig = ((mu2 - mu1).abs() / 4.0).clamp_min(step)
    return A1, sig, mu1, A2, sig.clone(), mu2, has2


def _smooth3(t, dim):
    """Binomial [1, 2, 1]/4 smoothing along ``dim`` with replicate edges.

    Plain narrow/cat arithmetic (no conv, no pooling) so it behaves identically
    on cpu/cuda/mps.
    """
    n = t.size(dim)
    lo = t.narrow(dim, 0, 1)
    hi = t.narrow(dim, n - 1, 1)
    padded = torch.cat([lo, t, hi], dim=dim)
    return (
        0.25 * padded.narrow(dim, 0, n)
        + 0.5 * padded.narrow(dim, 1, n)
        + 0.25 * padded.narrow(dim, 2, n)
    )


def two_peak_seed_nd(w, grid_shape, min_separation_samples=2):
    """Top-2 separated local maxima of a D-dim curve (batched, deterministic).

    The N-D generalisation of :func:`two_peak_seed`'s peak-finding stage: the
    (already background-subtracted, non-negative) curve is smoothed with a
    small separable binomial kernel per grid axis, interior local maxima are
    detected along every axis, and the second peak is the highest maximum
    outside a Chebyshev (per-axis index) exclusion window around the first —
    otherwise noise puts both maxima on the same peak and the two-Gaussian fit
    collapses.

    Args:
        w: (P, N) tensor, ``N == prod(grid_shape)`` (row-major flattening).
        grid_shape: D-dim motor grid shape of each curve.
        min_separation_samples: half-width (samples, per axis) of the exclusion
            window around peak 1.

    Returns:
        tuple: i1, i2 (P,) long — flat grid indices of the two peaks,
        v1, v2 (P,) — smoothed peak heights, has2 (P,) bool — a valid,
        separated second local maximum exists (when False, i2/v2 are
        placeholders and the caller should fall back to a split seed).
    """
    P, N = w.shape
    grid_shape = tuple(int(g) for g in grid_shape)
    D = len(grid_shape)

    s = w.reshape(P, *grid_shape)
    for ax in range(1, D + 1):
        if s.size(ax) >= 3:
            s = _smooth3(s, ax)

    # interior local maxima along every axis (strict on the low side, >= on
    # the high side, edges excluded — mirrors the 1-D two_peak_seed)
    ismax = torch.ones_like(s, dtype=torch.bool)
    for ax in range(1, D + 1):
        n = s.size(ax)
        if n < 3:
            continue
        core = s.narrow(ax, 1, n - 2)
        prev = s.narrow(ax, 0, n - 2)
        nxt = s.narrow(ax, 2, n - 2)
        m = (core > prev) & (core >= nxt)
        pad_shape = list(m.shape)
        pad_shape[ax] = 1
        edge = torch.zeros(pad_shape, dtype=torch.bool, device=s.device)
        ismax = ismax & torch.cat([edge, m, edge], dim=ax)

    sflat = s.reshape(P, N)
    vals = torch.where(ismax.reshape(P, N), sflat, torch.full_like(sflat, -1.0))
    v1, i1 = vals.max(dim=-1)

    # Chebyshev exclusion window around peak 1: a voxel is excluded only when
    # it is within min_separation_samples along EVERY axis, so peaks separated
    # along any single axis (e.g. only along ccmth in a 3-D scan) survive
    rem = i1
    axis_idx = []
    for size in reversed(grid_shape):
        axis_idx.append(torch.remainder(rem, size))
        rem = torch.div(rem, size, rounding_mode="floor")
    axis_idx.reverse()
    excl = torch.ones(P, *grid_shape, dtype=torch.bool, device=s.device)
    for ax, size in enumerate(grid_shape):
        pos = torch.arange(size, device=s.device)
        view = [1] * (D + 1)
        view[ax + 1] = size
        ctr = [P] + [1] * D
        near = (
            pos.view(view) - axis_idx[ax].view(ctr)
        ).abs() <= min_separation_samples
        excl = excl & near
    v2, i2 = torch.where(
        excl.reshape(P, N), torch.full_like(vals, -1.0), vals
    ).max(dim=-1)

    has2 = (v1 > 0) & (v2 > 0)
    return i1, i2, v1, v2, has2
