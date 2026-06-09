"""Portable batched SPD solve for small (p <= 8) normal-equation systems.

torch.linalg.cholesky_ex is used as a fast path on cuda/cpu; on MPS, where
linalg coverage is patchy across torch versions, an unrolled batched Cholesky
written in plain torch arithmetic gives identical behaviour on every backend
and returns a finite-mask instead of raising.
"""

import torch


def solve_spd(H, g):
    """Solve H x = g for a batch of small SPD systems.

    Args:
        H: (P, p, p) tensor, symmetric positive definite per batch element.
        g: (P, p) tensor.

    Returns:
        tuple: x (P, p) solution (garbage where ok is False), ok (P,) bool mask.
    """
    if H.device.type != "mps":
        L, info = torch.linalg.cholesky_ex(H)
        ok = info == 0
        x = torch.cholesky_solve(g.unsqueeze(-1), L).squeeze(-1)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        ok = ok & torch.isfinite(x).all(-1)
        return x, ok
    return _solve_spd_unrolled(H, g)


def _solve_spd_unrolled(H, g):
    P, p, _ = H.shape
    L = torch.zeros_like(H)
    ok = torch.ones(P, dtype=torch.bool, device=H.device)
    for j in range(p):
        s = H[:, j, j] - (L[:, j, :j] ** 2).sum(-1)
        ok = ok & (s > 0)
        d = torch.sqrt(s.clamp_min(1e-37))
        L[:, j, j] = d
        if j + 1 < p:
            L[:, j + 1 :, j] = (
                H[:, j + 1 :, j] - (L[:, j + 1 :, :j] * L[:, j : j + 1, :j]).sum(-1)
            ) / d.unsqueeze(-1)

    # forward substitution L y = g
    y = torch.zeros_like(g)
    for i in range(p):
        y[:, i] = (g[:, i] - (L[:, i, :i] * y[:, :i]).sum(-1)) / L[:, i, i]
    # back substitution L^T x = y
    x = torch.zeros_like(g)
    for i in range(p - 1, -1, -1):
        x[:, i] = (y[:, i] - (L[:, i + 1 :, i] * x[:, i + 1 :]).sum(-1)) / L[:, i, i]

    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    ok = ok & torch.isfinite(x).all(-1)
    return x, ok
