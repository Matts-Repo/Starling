"""Batched torch model + analytic Jacobian evaluators.

Vectorised ports of darling.properties.models. Parameter orders match the
darling *code* (not its docstrings): gauss1d_lin is [A, sigma, mu, k, m].
Built functionally (torch.stack, no slice writes) so torch.compile can fuse
the elementwise chains.
"""

import torch


def gauss1d_lin(params, x):
    """Gaussian + linear background, f(x) = A exp(-(x-mu)^2/(2 sigma^2)) + k x + m.

    Args:
        params: (P, 5) tensor [A, sigma, mu, k, m].
        x: (N,) tensor.

    Returns:
        tuple: f (P, N), J (P, N, 5) ordered [df/dA, df/dsigma, df/dmu, df/dk, df/dm].
    """
    A, sigma, mu, k, m = params.unbind(-1)
    res = mu[:, None] - x[None, :]  # (P, N)
    s2 = (sigma * sigma)[:, None]
    res2 = res * res
    e = torch.exp(-0.5 * res2 / s2)
    f = A[:, None] * e + k[:, None] * x[None, :] + m[:, None]
    Ae = A[:, None] * e
    J = torch.stack(
        [
            e,
            Ae * res2 / (s2 * sigma[:, None]),
            -Ae * res / s2,
            x[None, :].expand(params.shape[0], -1),
            torch.ones_like(e),
        ],
        dim=-1,
    )
    return f, J


def gauss1d_two_lin(params, x):
    """Two Gaussians + linear background.

    Args:
        params: (P, 8) tensor [A1, sigma1, mu1, A2, sigma2, mu2, k, m].
        x: (N,) tensor.

    Returns:
        tuple: f (P, N), J (P, N, 8).
    """
    A1, s1, mu1, A2, s2_, mu2, k, m = params.unbind(-1)
    cols = []
    f = k[:, None] * x[None, :] + m[:, None]
    for A, s, mu in ((A1, s1, mu1), (A2, s2_, mu2)):
        res = mu[:, None] - x[None, :]
        ss = (s * s)[:, None]
        res2 = res * res
        e = torch.exp(-0.5 * res2 / ss)
        Ae = A[:, None] * e
        f = f + Ae
        cols += [e, Ae * res2 / (ss * s[:, None]), -Ae * res / ss]
    cols += [x[None, :].expand(params.shape[0], -1), torch.ones_like(f)]
    return f, torch.stack(cols, dim=-1)


def gauss2d_const(params, x):
    """2D Gaussian with Cholesky-parameterised inverse covariance + constant background.

    f(v) = A exp(-0.5 (v-mu)^T L L^T (v-mu)) + c, with L lower-triangular so
    the precision matrix L L^T is positive (semi-)definite by construction.

    Args:
        params: (P, 7) tensor [A, mu0, mu1, L00, L10, L11, c].
        x: (2, N) tensor of flattened grid coordinates.

    Returns:
        tuple: f (P, N), J (P, N, 7).
    """
    A, mu0, mu1, L00, L10, L11, c = params.unbind(-1)
    d0 = x[0][None, :] - mu0[:, None]  # (P, N)
    d1 = x[1][None, :] - mu1[:, None]
    # u = L^T d  =>  u0 = L00 d0 + L10 d1, u1 = L11 d1
    u0 = L00[:, None] * d0 + L10[:, None] * d1
    u1 = L11[:, None] * d1
    q = u0 * u0 + u1 * u1
    e = torch.exp(-0.5 * q)
    Ae = A[:, None] * e
    f = Ae + c[:, None]
    J = torch.stack(
        [
            e,
            Ae * (L00[:, None] * u0),
            Ae * (L10[:, None] * u0 + L11[:, None] * u1),
            -Ae * u0 * d0,
            -Ae * u0 * d1,
            -Ae * u1 * d1,
            torch.ones_like(e),
        ],
        dim=-1,
    )
    return f, J
