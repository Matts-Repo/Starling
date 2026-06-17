"""Batched torch model + analytic Jacobian evaluators.

Parameter order for gauss1d_lin: [A, sigma, mu, k, m].
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


def pseudovoigt1d_lin(params, x):
    """Pseudo-Voigt (height-normalised) + linear background.

    f(x) = A [ (1 - eta) G(x) + eta L(x) ] + k x + m, with
    G(x) = exp(-(x - mu)^2 / (2 sigma^2)) and L(x) = 1 / (1 + ((x - mu)/gamma)^2),
    both unit-height at x = mu, so A is the peak amplitude and eta in [0, 1] mixes
    Gaussian (0) and Lorentzian (1) lineshapes.

    Args:
        params: (P, 7) tensor [A, sigma, mu, gamma, eta, k, m].
        x: (N,) tensor.

    Returns:
        tuple: f (P, N), J (P, N, 7) ordered
        [df/dA, df/dsigma, df/dmu, df/dgamma, df/deta, df/dk, df/dm].
    """
    A, sigma, mu, gamma, eta, k, m = params.unbind(-1)
    r = x[None, :] - mu[:, None]  # (P, N)
    s2 = (sigma * sigma)[:, None]
    g2 = (gamma * gamma)[:, None]
    G = torch.exp(-0.5 * r * r / s2)
    Lz = 1.0 / (1.0 + r * r / g2)
    Lz2 = Lz * Lz
    eta_ = eta[:, None]
    A_ = A[:, None]
    one_eta = 1.0 - eta_
    S = one_eta * G + eta_ * Lz
    f = A_ * S + k[:, None] * x[None, :] + m[:, None]
    J = torch.stack(
        [
            S,
            A_ * one_eta * G * r * r / (s2 * sigma[:, None]),
            A_ * (one_eta * G * r / s2 + eta_ * Lz2 * 2.0 * r / g2),
            A_ * eta_ * Lz2 * 2.0 * r * r / (g2 * gamma[:, None]),
            A_ * (Lz - G),
            x[None, :].expand(params.shape[0], -1),
            torch.ones_like(G),
        ],
        dim=-1,
    )
    return f, J


def gaussND_const(params, x, D):
    """N-D Gaussian with Cholesky-parameterised inverse covariance + constant bg.

    Generalises ``gauss2d_const`` to an arbitrary number of motor dimensions D.

    f(v) = A exp(-0.5 (v - mu)^T L L^T (v - mu)) + c, with L lower-triangular
    (row-major, diagonal included) so the precision matrix L L^T is positive
    (semi-)definite for *any* real L — no positivity constraints are needed.

    The forward pass and Jacobian are built from elementwise ops + ``torch.stack``
    only (no matmul, no in-place writes): D is a Python constant per call, so the
    unrolled sums specialise cleanly under ``torch.compile`` and fuse on MPS.

    Parameter layout (length ``n_p = 1 + D + D(D+1)/2 + 1``):
        ``[A, mu_0 .. mu_{D-1}, L (lower-tri incl. diagonal, row-major), c]``.

    Args:
        params: (P, n_p) tensor.
        x: (D, N) tensor of flattened grid coordinates.
        D (int): number of motor dimensions (fixed per call).

    Returns:
        tuple: f (P, N), J (P, N, n_p) with columns ordered
        [dA, dmu_0 .. dmu_{D-1}, dL_rs (r>=s, row-major), dc].
    """
    A = params[:, 0]
    mu = [params[:, 1 + i] for i in range(D)]
    n_L = D * (D + 1) // 2
    L_flat = [params[:, 1 + D + t] for t in range(n_L)]
    c = params[:, 1 + D + n_L]

    # row-major lower-triangular index map L[r][s] (r >= s)
    Lmat = [[None] * D for _ in range(D)]
    t = 0
    for r in range(D):
        for s in range(r + 1):
            Lmat[r][s] = L_flat[t]
            t += 1

    d = [x[j][None, :] - mu[j][:, None] for j in range(D)]  # each (P, N)

    # u = L^T d  =>  u_k = sum_{j >= k} L_{jk} d_j
    u = []
    for k in range(D):
        acc = Lmat[k][k][:, None] * d[k]
        for j in range(k + 1, D):
            acc = acc + Lmat[j][k][:, None] * d[j]
        u.append(acc)

    q = u[0] * u[0]
    for k in range(1, D):
        q = q + u[k] * u[k]
    e = torch.exp(-0.5 * q)
    Ae = A[:, None] * e
    f = Ae + c[:, None]

    # Lu_i = sum_{k <= i} L_{ik} u_k  (for the mu derivatives)
    Lu = []
    for i in range(D):
        acc = Lmat[i][0][:, None] * u[0]
        for k in range(1, i + 1):
            acc = acc + Lmat[i][k][:, None] * u[k]
        Lu.append(acc)

    cols = [e]
    for i in range(D):
        cols.append(Ae * Lu[i])  # df/dmu_i = Ae (Lu)_i
    for r in range(D):
        for s in range(r + 1):
            cols.append(-Ae * u[s] * d[r])  # df/dL_rs = -Ae u_s d_r
    cols.append(torch.ones_like(e))  # df/dc
    J = torch.stack(cols, dim=-1)
    return f, J
