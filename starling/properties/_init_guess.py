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
