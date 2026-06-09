"""Generic batched damped Gauss-Newton driver.

Mirrors darling.properties.curvefit.gauss_newton_fit_1D but over a pixel
batch: per iteration H = J^T J, g = J^T r, damped batched solve. Pixels whose
solve fails are frozen with success=0 — the batched equivalent of darling's
per-pixel try/except (darling likewise stops updating a pixel on failure).

The per-iteration step is torch.compile'd: the model/Jacobian evaluation is a
chain of memory-bound elementwise kernels that fusion speeds up ~10x on MPS.
"""

import torch

from ._solve import solve_spd

_STEP_CACHE = {}


def _gn_step(y, x, params, lam, model_and_jac):
    f, J = model_and_jac(params, x)
    r = y - f
    H = J.mT @ J
    g = (J * r.unsqueeze(-1)).sum(1)
    Hd = H + lam * torch.diag_embed(H.diagonal(dim1=-2, dim2=-1))
    return solve_spd(Hd, g)


def _get_step(model_and_jac, device):
    key = (model_and_jac, device.type)
    if key not in _STEP_CACHE:
        fn = lambda y, x, params, lam: _gn_step(y, x, params, lam, model_and_jac)
        try:
            # static shapes compile ~4x faster kernels on MPS than dynamic=True;
            # callers keep chunk shapes uniform so each shape compiles once
            _STEP_CACHE[key] = torch.compile(fn)
        except Exception:
            _STEP_CACHE[key] = fn
    return _STEP_CACHE[key]


def gauss_newton_batched(y, x, params0, model_and_jac, n_iter=7, lam=1e-4):
    """Fit a batch of curves with damped Gauss-Newton.

    Args:
        y: (P, N) data tensor.
        x: (N,) or (d, N) coordinate tensor (passed through to the model).
        params0: (P, p) initial parameters.
        model_and_jac: callable (params, x) -> (f (P, N), J (P, N, p)).
        n_iter: number of iterations.
        lam: Marquardt damping factor on diag(H).

    Returns:
        tuple: params (P, p), success (P,) bool.
    """
    step = _get_step(model_and_jac, y.device)
    params = params0.clone()
    success = torch.ones(params.shape[0], dtype=torch.bool, device=params.device)

    for _ in range(n_iter):
        try:
            delta, ok = step(y, x, params, lam)
        except Exception:
            # compiled path failed at runtime (e.g. unsupported op on this
            # backend) — fall back to eager permanently for this model
            _STEP_CACHE[(model_and_jac, y.device.type)] = lambda y_, x_, p_, l_: _gn_step(
                y_, x_, p_, l_, model_and_jac
            )
            step = _STEP_CACHE[(model_and_jac, y.device.type)]
            delta, ok = step(y, x, params, lam)
        active = success & ok & torch.isfinite(delta).all(-1)
        params = torch.where(active.unsqueeze(-1), params + delta, params)
        success = success & ok

    success = success & torch.isfinite(params).all(-1)
    return params, success
