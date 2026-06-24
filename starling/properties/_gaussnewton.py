"""Generic batched damped Gauss-Newton driver.

Per iteration: H = J^T J, g = J^T r, damped batched solve over a pixel batch.
Pixels whose solve fails are frozen with success=0.

The per-iteration step is torch.compile'd: the model/Jacobian evaluation is a
chain of memory-bound elementwise kernels that fusion speeds up ~10x on MPS.
"""

import torch

from ._solve import solve_spd

_STEP_CACHE = {}
_STEP_CACHE_LAM = {}


def _gn_step(y, x, params, lam, model_and_jac):
    f, J = model_and_jac(params, x)
    r = y - f
    H = J.mT @ J
    g = (J * r.unsqueeze(-1)).sum(1)
    Hd = H + lam * torch.diag_embed(H.diagonal(dim1=-2, dim2=-1))
    return solve_spd(Hd, g)


def _gn_step_lam(y, x, params, lam, model_and_jac):
    """Like ``_gn_step`` but with a per-pixel damping vector ``lam`` (P,)."""
    f, J = model_and_jac(params, x)
    r = y - f
    H = J.mT @ J
    g = (J * r.unsqueeze(-1)).sum(1)
    Hd = H + lam[:, None, None] * torch.diag_embed(H.diagonal(dim1=-2, dim2=-1))
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


def _get_step_lam(model_and_jac, device):
    key = (model_and_jac, device.type)
    if key not in _STEP_CACHE_LAM:
        fn = lambda y, x, params, lam: _gn_step_lam(y, x, params, lam, model_and_jac)
        try:
            _STEP_CACHE_LAM[key] = torch.compile(fn)
        except Exception:
            _STEP_CACHE_LAM[key] = fn
    return _STEP_CACHE_LAM[key]


def gauss_newton_batched(
    y, x, params0, model_and_jac, n_iter=7, lam=1e-4, bounds=None,
    adaptive=False, lam_up=4.0, lam_down=0.5, lam_min=1e-6, lam_max=1e3,
):
    """Fit a batch of curves with damped Gauss-Newton.

    Args:
        y: (P, N) data tensor.
        x: (N,) or (d, N) coordinate tensor (passed through to the model).
        params0: (P, p) initial parameters.
        model_and_jac: callable (params, x) -> (f (P, N), J (P, N, p)).
        n_iter: number of iterations.
        lam: Marquardt damping factor on diag(H). With ``adaptive`` this is the
            initial per-pixel damping.
        bounds: optional (lo, hi) tensors of shape (p,) — parameters are
            projected into the box after every step, which prevents runaway
            divergence in ill-conditioned multi-peak fits.
        adaptive: when True, use per-pixel Levenberg-Marquardt damping that
            grows (``lam_up``) whenever the SPD solve fails — the signature of an
            overshooting step that has driven H non-positive-definite (e.g. a
            sharp peak whose Cholesky-precision parameters blow up) — and shrinks
            (``lam_down``) when the solve succeeds, recovering fast convergence
            near the optimum. Instead of latching ``success`` off on the first
            bad solve (which permanently discards a pixel that merely needed more
            damping), success is judged at the end by whether the residual
            actually decreased relative to the seed.
        lam_up, lam_down: multiplicative damping adjustment per iteration.
        lam_min, lam_max: clamp range for the per-pixel damping.

    Returns:
        tuple: params (P, p), success (P,) bool.
    """
    if not adaptive:
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
            if bounds is not None:
                params = torch.clamp(params, bounds[0], bounds[1])
            success = success & ok

        success = success & torch.isfinite(params).all(-1)
        return params, success

    # --- adaptive per-pixel Levenberg-Marquardt -----------------------------
    step = _get_step_lam(model_and_jac, y.device)
    params = params0.clone()
    lam_vec = torch.full((params.shape[0],), float(lam), dtype=y.dtype, device=y.device)

    def _cost(p):
        f, _ = model_and_jac(p, x)
        r = y - f
        return (r * r).sum(-1)

    cost0 = _cost(params)
    for _ in range(n_iter):
        try:
            delta, ok = step(y, x, params, lam_vec)
        except Exception:
            _STEP_CACHE_LAM[(model_and_jac, y.device.type)] = lambda y_, x_, p_, l_: _gn_step_lam(
                y_, x_, p_, l_, model_and_jac
            )
            step = _STEP_CACHE_LAM[(model_and_jac, y.device.type)]
            delta, ok = step(y, x, params, lam_vec)
        good = ok & torch.isfinite(delta).all(-1)
        params = torch.where(good.unsqueeze(-1), params + delta, params)
        if bounds is not None:
            params = torch.clamp(params, bounds[0], bounds[1])
        lam_vec = torch.where(good, lam_vec * lam_down, lam_vec * lam_up)
        lam_vec = lam_vec.clamp(lam_min, lam_max)

    cost1 = _cost(params)
    success = torch.isfinite(params).all(-1) & (cost1 <= cost0)
    return params, success
