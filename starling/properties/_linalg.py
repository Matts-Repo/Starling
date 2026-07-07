"""Batched host-side inversion that never raises.

``np.linalg.inv`` on a stacked ``(P, D, D)`` array raises ``LinAlgError`` for
the *whole batch* when any single matrix is exactly singular. On real scans a
handful of diverged/degenerate pixels always exist (e.g. a fitted Cholesky
factor with an exactly-zero diagonal after float32 underflow gives a
rank-deficient precision matrix), so the batched un-transform of a full-frame
fit would crash on one bad pixel (observed on MA6278 mosa data).

An absolute det threshold cannot guard this: the computed det of an exactly
rank-deficient matrix is a cancellation residual that scales with the matrix
norm (~eps * ||M||^D, i.e. ~1e23 for precision entries ~1e29), so the
singularity test here is on the det of the max-norm-scaled matrix — a
scale-invariant quantity ~ 1/condition — with a ``pinv`` fallback as a final
safety net. Bad rows are zeroed and reported in the returned mask, matching
starling's "zero degenerate pixels" convention.
"""

import numpy as np

# det(M / ||M||_max) below this is treated as singular. Rounding residuals of
# exactly rank-deficient matrices sit at ~D * eps ~ 1e-15; 1e-13 gives a 100x
# margin while only discarding condition numbers > ~1e13, far beyond anything a
# float32 fit can support.
_DET_TOL = 1e-13


def masked_inv_spd(M):
    """Invert a batch of small symmetric matrices, masking the bad ones.

    Args:
        M: (P, D, D) float64 array (symmetric per batch element).

    Returns:
        tuple: ``Minv`` (P, D, D) with non-invertible rows zeroed, and ``ok``
        (P,) bool — False where the matrix is non-finite or (near-)singular.
    """
    M = np.asarray(M)
    P, D, _ = M.shape
    finite = np.isfinite(M).all(axis=(-1, -2))
    scale = np.abs(np.where(finite[:, None, None], M, 0.0)).max(axis=(-1, -2))
    ok = finite & (scale > 0)
    scale_safe = np.where(ok, scale, 1.0)
    # bad rows are multiplied by 0 (NaN rows stay NaN -> det NaN -> masked);
    # the errstate silences the expected invalid-value warning they produce
    with np.errstate(invalid="ignore", over="ignore"):
        det_n = np.linalg.det(
            M * np.where(ok, 1.0 / scale_safe, 0.0)[:, None, None]
        )
    ok = ok & np.isfinite(det_n) & (np.abs(det_n) > _DET_TOL)

    out = np.zeros_like(M, dtype=np.float64)
    if ok.any():
        good = M[ok]
        try:
            out[ok] = np.linalg.inv(good)
        except np.linalg.LinAlgError:  # pragma: no cover - guarded by det test
            out[ok] = np.linalg.pinv(good)
    return out, ok
