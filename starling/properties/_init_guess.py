"""Vectorised initial-guess routines.

Batched ports of darling.properties.curvefit.estimate_initial_linear_trend
and _estimate_initial_gaussian_params.
"""

import torch


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
    sigma <= 1e-8 — darling's skip conditions) are flagged in the returned mask
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
