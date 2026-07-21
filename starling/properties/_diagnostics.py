"""Per-pixel fit-status classification: separate "no grain" from "fit failed"
from "peak clipped by the scan range".

Motivation: when a grain rotates partially out of the scanned rocking range
(e.g. charging-induced reorientation), the per-pixel peak sits at/beyond the
edge of the motor grid, the fitted centre lands on the boundary and the
physical-bounds filter rejects it. Rendering those pixels the same way as
"no grain" pixels misleads viewers into thinking the grain was lost there.
This module formalises the bounds filter into a categorical map so displays
and saved files can distinguish the cases.

Status values (int8, persisted in bundles/NeXus — treat as a stable format):

* ``NO_SIGNAL = 0`` — pixel outside the fitted mask (never fitted).
* ``OK = 1`` — converged, centre well inside the scanned range on every axis.
* ``EDGE_CLIPPED = 2`` — converged, but on >= 1 axis the centre is within
  ``edge_tol_steps`` grid steps of a scan-range boundary (or just beyond it):
  the true peak is likely truncated by the scan range, not lost.
* ``FAILED = 3`` — the solver did not converge, or converged to a centre
  beyond the tolerance band (unphysical).
"""

import numpy as np

NO_SIGNAL = 0
OK = 1
EDGE_CLIPPED = 2
FAILED = 3

STATUS_NAMES = {
    NO_SIGNAL: "no signal",
    OK: "ok",
    EDGE_CLIPPED: "edge-clipped",
    FAILED: "failed",
}


def motor_ranges_steps(motors, axes=None):
    """Per-axis (lo, hi) ranges and grid steps from a motor grid.

    Args:
        motors (numpy.ndarray): shape (D, *grid) for an N-motor scan, or a
            1-D array of positions for a single-motor scan.
        axes (sequence, optional): subset of axes to report; default all.

    Returns:
        tuple: ``(ranges, steps)`` — lists of ``(lo, hi)`` and step floats,
        one per (selected) axis. The step is the median positive difference
        of the sorted unique positions (robust to encoder jitter); 0.0 for a
        degenerate single-position axis.
    """
    m = np.asarray(motors, dtype=np.float64)
    if m.ndim == 1:
        m = m[None, ...]
    D = m.shape[0]
    if axes is None:
        axes = range(D)
    ranges, steps = [], []
    for i in axes:
        vals = np.unique(m[i].ravel())
        vals = vals[np.isfinite(vals)]
        lo, hi = (float(vals[0]), float(vals[-1])) if vals.size else (0.0, 0.0)
        d = np.diff(vals)
        d = d[d > 0]
        steps.append(float(np.median(d)) if d.size else 0.0)
        ranges.append((lo, hi))
    return ranges, steps


def edge_peak_mask(data, edge_bins=3, axes=None):
    """(ny, nx) bool: pixels whose rocking profile peaks at a scan-range edge.

    For each motor axis, the per-pixel marginal profile (sum over the other
    motor axes) is computed and its argmax located; the pixel is flagged when
    the argmax falls within ``edge_bins`` grid points of either end on any
    checked axis. This is the *data-driven* truncation signature: when a peak
    is clipped by the scan range the solver typically diverges or fails
    outright, so the fitted centre alone cannot distinguish truncation from a
    genuine failure — the raw profile can.

    Empirical calibration (MA7031 bad-fit scan, 60-point mu axis):
    ``edge_bins=3`` flags ~74% of the unconverged grain pixels and ~1% of the
    well-fitted ones; 5 flags ~96% but ~3% false positives.

    Args:
        data (numpy.ndarray): (ny, nx, *motor_dims) intensity cube.
        edge_bins (int): edge proximity in grid points.
        axes (sequence, optional): motor axes (0-based among motor dims) to
            check; default all.

    Returns:
        numpy.ndarray: (ny, nx) bool.
    """
    n_motor = data.ndim - 2
    if n_motor < 1:
        raise ValueError(f"data must be (ny, nx, *motor_dims), got {data.shape}")
    if axes is None:
        axes = range(n_motor)
    out = np.zeros(data.shape[:2], dtype=bool)
    for ax in axes:
        n = data.shape[2 + ax]
        other = tuple(2 + i for i in range(n_motor) if i != ax)
        prof = data.sum(axis=other, dtype=np.int64) if other else data
        am = np.argmax(prof, axis=-1)
        out |= (am < edge_bins) | (am >= n - edge_bins)
    return out


def classify_fit_status(mu, success, motor_ranges, motor_steps, mask=None,
                        edge_tol_steps=1.0, axes=None, data_edge=None, A=None):
    """Per-pixel fit status: 0 no-signal / 1 ok / 2 edge-clipped / 3 failed.

    Two complementary edge criteria feed EDGE_CLIPPED:

    * fitted-centre proximity: the fit converged with a centre within
      ``edge_tol_steps`` grid steps of a range boundary (or just beyond it);
    * data-driven truncation (``data_edge``, from :func:`edge_peak_mask`):
      the raw profile peaks at the scan edge — applied to pixels whose fit
      failed or diverged, which would otherwise be indistinguishable from
      genuine failures (on truncated peaks the solver usually diverges far
      out of range rather than converging onto the boundary).

    Args:
        mu (numpy.ndarray): fitted centres, (ny, nx, D) or (ny, nx) for D=1.
        success (numpy.ndarray): (ny, nx) fit success flag (> 0.5 = converged).
        motor_ranges (sequence): per-axis (lo, hi) scanned range (motor units).
        motor_steps (sequence): per-axis grid step (motor units).
        mask (numpy.ndarray, optional): (ny, nx) bool — pixels that were
            fitted. ``None`` treats every pixel as fitted (no NO_SIGNAL).
        edge_tol_steps (float): width of the fitted-centre edge band in grid
            steps.
        axes (sequence, optional): subset of mu's axes to check against
            ``motor_ranges``/``motor_steps`` (which must then match in
            length); default all.
        data_edge (numpy.ndarray, optional): (ny, nx) bool from
            :func:`edge_peak_mask`; reclassifies failed/diverged pixels whose
            raw profile peaks at the scan edge as EDGE_CLIPPED.
        A (numpy.ndarray, optional): (ny, nx) fitted amplitudes. A pixel with
            a non-positive or non-finite amplitude is never OK — a degenerate
            "background-only" or negative-dip solution can converge with an
            in-window centre and would otherwise masquerade as a good fit.

    Returns:
        numpy.ndarray: (ny, nx) int8 status map.
    """
    mu = np.asarray(mu, dtype=np.float64)
    if mu.ndim == 2:
        mu = mu[..., None]
    ny, nx = mu.shape[:2]
    if axes is None:
        axes = range(mu.shape[-1])
    axes = list(axes)
    if len(motor_ranges) != len(axes) or len(motor_steps) != len(axes):
        raise ValueError(
            f"need one (lo, hi) range and one step per checked axis: got "
            f"{len(motor_ranges)} ranges / {len(motor_steps)} steps for "
            f"{len(axes)} axes"
        )

    converged = np.asarray(success) > 0.5
    if A is not None:
        Aa = np.asarray(A)
        converged = converged & (Aa > 0) & np.isfinite(Aa)
    near_edge = np.zeros((ny, nx), dtype=bool)
    out_of_bounds = np.zeros((ny, nx), dtype=bool)
    for (lo, hi), step, ax in zip(motor_ranges, motor_steps, axes):
        tol = edge_tol_steps * step
        v = mu[..., ax]
        near_edge |= (v <= lo + tol) | (v >= hi - tol)
        out_of_bounds |= (v < lo - tol) | (v > hi + tol) | ~np.isfinite(v)

    status = np.full((ny, nx), FAILED, dtype=np.int8)
    status[converged & near_edge & ~out_of_bounds] = EDGE_CLIPPED
    status[converged & ~near_edge & ~out_of_bounds] = OK
    if data_edge is not None:
        # rescue failed/diverged pixels whose raw profile peaks at the edge
        status[(status == FAILED) & np.asarray(data_edge, dtype=bool)] = EDGE_CLIPPED
    if mask is not None:
        status[~np.asarray(mask, dtype=bool)] = NO_SIGNAL
    return status


def clamp_edge_estimate(mu, motor_ranges, axes=None):
    """Centres clipped to the scanned range, per axis — for DISPLAY only.

    Lets EDGE_CLIPPED pixels show their best (boundary) estimate instead of a
    hole in the map. Never feed the clamped values into quantitative analysis:
    at these pixels the true centre is outside the scan range and unknown.

    Args:
        mu (numpy.ndarray): (ny, nx, D) or (ny, nx) fitted centres.
        motor_ranges (sequence): per-axis (lo, hi), matching ``axes``.
        axes (sequence, optional): which axes to clamp; default all.

    Returns:
        numpy.ndarray: a clipped copy of ``mu`` (same shape as the input).
    """
    mu = np.asarray(mu, dtype=np.float64)
    squeeze = mu.ndim == 2
    out = (mu[..., None] if squeeze else mu).copy()
    if axes is None:
        axes = range(out.shape[-1])
    for (lo, hi), ax in zip(motor_ranges, axes):
        out[..., ax] = np.clip(out[..., ax], lo, hi)
    return out[..., 0] if squeeze else out
