"""Generic batched damped Gauss-Newton driver.

Per iteration: H = J^T J, g = J^T r, damped batched solve over a pixel batch.
Pixels whose solve fails are frozen with success=0.

The per-iteration step is torch.compile'd: the model/Jacobian evaluation is a
chain of memory-bound elementwise kernels that fusion speeds up ~10x on MPS.
"""

import os

import torch

from ._solve import solve_spd

_STEP_CACHE = {}
_STEP_CACHE_LAM = {}


def _should_compile(device):
    """torch.compile the per-iteration step for this backend?

    Measured on real hardware: on MPS the fused kernels are ~10x faster
    than eager, but on CUDA (A40, torch 2.4/inductor) eager BEATS the
    compiled step (89 s vs 95-100 s warm for a 19k-px 3D fit) while
    compilation additionally costs ~40 s on first use and ~5 s per new
    chunk shape. Default: compile everywhere except CUDA; override with
    STARLING_TORCH_COMPILE=1/0.
    """
    env = os.environ.get("STARLING_TORCH_COMPILE")
    if env is not None:
        return env not in ("0", "false", "False")
    return device.type != "cuda"


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
        if _should_compile(device):
            try:
                # static shapes compile ~4x faster kernels on MPS than
                # dynamic=True; callers keep chunk shapes uniform so each
                # shape compiles once
                fn = torch.compile(fn)
            except Exception:
                pass
        _STEP_CACHE[key] = fn
    return _STEP_CACHE[key]


def _get_step_lam(model_and_jac, device):
    key = (model_and_jac, device.type)
    if key not in _STEP_CACHE_LAM:
        fn = lambda y, x, params, lam: _gn_step_lam(y, x, params, lam, model_and_jac)
        if _should_compile(device):
            try:
                fn = torch.compile(fn)
            except Exception:
                pass
        _STEP_CACHE_LAM[key] = fn
    return _STEP_CACHE_LAM[key]


def _rel_step_done(delta, params, xtol):
    """(P,) bool: pixels whose GN step is negligible relative to their params.

    MINPACK-style criterion: ||dx|| <= xtol * (||x|| + xtol). Near a healthy
    optimum GN converges quadratically, so once a step passes this test the
    total remaining parameter motion is a small multiple of that step —
    freezing there changes the result at the xtol level, far below any
    physical precision of the maps.
    """
    dn = torch.linalg.vector_norm(delta, dim=-1)
    pn = torch.linalg.vector_norm(params, dim=-1)
    return dn <= xtol * (pn + xtol)


def gauss_newton_batched(
    y, x, params0, model_and_jac, n_iter=7, lam=1e-4, bounds=None,
    adaptive=False, lam_up=4.0, lam_down=0.5, lam_min=1e-6, lam_max=1e3,
    xtol=1e-5, freeze_fn=None, iter_cb=None,
):
    """Fit a batch of curves with damped Gauss-Newton.

    ``n_iter`` is a **maximum**: a pixel whose relative step falls below
    ``xtol`` is frozen at its converged parameters, and the loop exits as
    soon as every pixel in the batch is frozen. Both decisions are strictly
    per-pixel, so a pixel's result never depends on which other pixels share
    its batch (the mask/batching invariance the test suite asserts
    bit-exactly). Callers chunk pixels, so fully-converged chunks exit early
    even when another chunk holds stubborn (e.g. scan-range-truncated)
    pixels that run to ``n_iter``.

    Args:
        y: (P, N) data tensor.
        x: (N,) or (d, N) coordinate tensor (passed through to the model).
        params0: (P, p) initial parameters.
        model_and_jac: callable (params, x) -> (f (P, N), J (P, N, p)).
        n_iter: maximum number of iterations.
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
        xtol (float, optional): relative step-size convergence tolerance.
            ``None`` disables early termination entirely (legacy fixed-count
            behaviour).
        iter_cb (callable, optional): called once after every completed
            iteration (live progress reporting from callers).
        freeze_fn (callable, optional): ``params (P, p) -> (P,) bool`` marking
            pixels whose parameters have become hopeless (e.g. a fitted centre
            far outside the scan window). Frozen alongside converged pixels so
            a few runaway pixels cannot keep a whole chunk iterating; strictly
            per-pixel, so batch invariance is preserved. Such pixels' final
            values are meaningless either way (they fail the physical-bounds /
            fit_status gates downstream).

    Returns:
        tuple: params (P, p), success (P,) bool.
    """
    if not adaptive:
        step = _get_step(model_and_jac, y.device)
        params = params0.clone()
        success = torch.ones(params.shape[0], dtype=torch.bool, device=params.device)
        done = torch.zeros_like(success)

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
            active = success & ok & torch.isfinite(delta).all(-1) & ~done
            params = torch.where(active.unsqueeze(-1), params + delta, params)
            if bounds is not None:
                params = torch.clamp(params, bounds[0], bounds[1])
            success = success & (ok | done)
            if iter_cb is not None:
                iter_cb()
            if xtol is not None:
                done = done | (active & _rel_step_done(delta, params, xtol)) | ~success
                if freeze_fn is not None:
                    done = done | freeze_fn(params)
                if bool(done.all()):
                    break  # every pixel frozen: further iterations are no-ops

        success = success & torch.isfinite(params).all(-1)
        return params, success

    # --- adaptive per-pixel Levenberg-Marquardt -----------------------------
    step = _get_step_lam(model_and_jac, y.device)
    params = params0.clone()
    lam_vec = torch.full((params.shape[0],), float(lam), dtype=y.dtype, device=y.device)
    done = torch.zeros(params.shape[0], dtype=torch.bool, device=params.device)

    def _cost(p, yy, xx):
        f, _ = model_and_jac(p, xx)
        r = yy - f
        return (r * r).sum(-1)

    cost0 = _cost(params, y, x)
    for it in range(n_iter):
        try:
            delta, ok = step(y, x, params, lam_vec)
        except Exception:
            _STEP_CACHE_LAM[(model_and_jac, y.device.type)] = lambda y_, x_, p_, l_: _gn_step_lam(
                y_, x_, p_, l_, model_and_jac
            )
            step = _STEP_CACHE_LAM[(model_and_jac, y.device.type)]
            delta, ok = step(y, x, params, lam_vec)
        good = ok & torch.isfinite(delta).all(-1) & ~done
        params = torch.where(good.unsqueeze(-1), params + delta, params)
        if bounds is not None:
            params = torch.clamp(params, bounds[0], bounds[1])
        lam_vec = torch.where(good | done, lam_vec * lam_down, lam_vec * lam_up)
        lam_vec = lam_vec.clamp(lam_min, lam_max)
        if iter_cb is not None:
            iter_cb()
        if xtol is not None:
            done = done | (good & _rel_step_done(delta, params, xtol))
            # a pixel whose solve failed AT maximum damping is deterministically
            # stuck: params unchanged -> identical failure next iteration, so
            # freezing it is exactly result-preserving
            done = done | (~ok & (lam_vec >= lam_max))
            if freeze_fn is not None:
                done = done | freeze_fn(params)
            if bool(done.all()):
                break  # every pixel frozen: further iterations are no-ops

    cost1 = _cost(params, y, x)
    success = torch.isfinite(params).all(-1) & (cost1 <= cost0)
    return params, success
