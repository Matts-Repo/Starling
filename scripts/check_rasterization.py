"""Diagnose how a BLISS fscan2d scan is rasterised and whether starling's
value-sort reconstructs the grid correctly.

starling does not read the beamline's `fast_motor_mode` flag; it re-sorts frames
onto a monotonic grid by their recorded motor values. That is robust to
raster-vs-zigzag *iff* the stored motor arrays are clean (the slow/outer motor is
exactly constant within each fast row). This script checks exactly that on real
data, and compares against the BLISS metadata.

Run when the data drive is mounted:

    python scripts/check_rasterization.py /path/to/master.h5 SCAN_ID
    # e.g. python scripts/check_rasterization.py \
    #   /Volumes/LaCie/ESRF_MA6278/RAW_DATA/<ds>/<ds>.h5 1.1
"""

import sys

import h5py
import numpy as np

from starling.io._metadata import ID03


def main(h5_path, scan_id):
    cfg = ID03(h5_path)
    params, _ = cfg(scan_id)
    cmd = params["scan_command"]
    shape = tuple(int(s) for s in params["scan_shape"])
    motor_names = params["motor_names"]
    print(f"scan        : {scan_id}")
    print(f"command     : {cmd}")
    print(f"scan_shape  : {shape}  (motor1 x motor2)")
    print(f"motor paths : {motor_names}")
    print(f"integrated  : {params['integrated_motors']}  "
          f"(True = fast/continuous fly axis)")

    with h5py.File(h5_path, "r") as f:
        m1 = f[scan_id][motor_names[0]][...].astype(np.float64)
        m2 = f[scan_id][motor_names[1]][...].astype(np.float64)
        # BLISS fscan rasterisation flag that darfix uses (starling ignores it)
        fmm = None
        fp = f[scan_id].get("instrument/fscan_parameters")
        if fp is not None and "fast_motor_mode" in fp:
            v = fp["fast_motor_mode"][()]
            fmm = v.decode() if isinstance(v, bytes) else v

    print(f"\nfast_motor_mode (BLISS) : {fmm!r}"
          + ("   <-- darfix would zigzag-reorder" if fmm == "ZIGZAG" else ""))

    n1, n2 = shape
    if m1.size != n1 * n2:
        print(f"\n[skip] motor array size {m1.size} != {n1}*{n2}; partial/odd scan.")
        return

    # ---- reproduce starling's lexicographic value-sort ----
    s = np.array(list(zip(m1, m2)), dtype=[("m1", "f8"), ("m2", "f8")])
    order = np.argsort(s, order=["m1", "m2"])
    g1 = m1[order].reshape(n1, n2)   # primary key (motor1) on the slow axis
    g2 = m2[order].reshape(n1, n2)

    # ---- is the reconstructed grid separable? (the property the notebook needs) ----
    # motor1 should be constant along each row; motor2 constant down each column.
    row_spread = np.ptp(g1, axis=1).max()      # variation of motor1 within a row
    col_spread = np.ptp(g2, axis=0).max()      # variation of motor2 down a column
    step1 = np.median(np.diff(np.unique(np.round(g1, 9)))) if n1 > 1 else np.nan
    step2 = np.median(np.diff(g2[0])) if n2 > 1 else np.nan

    print(f"\nmotor1 step (median)         : {step1:.3e}")
    print(f"motor2 step (median)         : {step2:.3e}")
    print(f"motor1 spread WITHIN a row    : {row_spread:.3e}   "
          f"({100*row_spread/abs(step1):.1f}% of its step)" if step1 else "")
    print(f"motor2 spread DOWN a column   : {col_spread:.3e}   "
          f"({100*col_spread/abs(step2):.1f}% of its step)" if step2 else "")

    tol1 = 0.25 * abs(step1) if step1 else 0.0
    ok = row_spread < tol1
    print("\nVERDICT")
    if ok:
        print("  ✓ slow motor is ~constant within each fast row "
              "(< 25% of a step): the value-sort reconstructs the grid cleanly; "
              "raster/zigzag handled correctly, no fast-axis scrambling.")
    else:
        print("  ⚠ slow motor VARIES within a row by a large fraction of its step. "
              "The lexicographic value-sort may order the fast axis by slow-motor "
              "JITTER instead of the fast motor, scrambling per-layer slices "
              "(data[:,:,:,k]) and grid-based fits. A tolerance-binned sort "
              "(or darfix's fast_motor_mode-driven index reorder) is needed here.")
    if fmm == "ZIGZAG":
        print("  • BLISS marks this scan ZIGZAG. starling does not read that flag — "
              "it relies on the value-sort above; confirm the verdict is ✓.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
