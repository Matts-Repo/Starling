"""Loader for partial (aborted) fscan/fscan2d scans.

Beamline scans frequently abort mid-acquisition; darling refuses them because
the frame count no longer matches the scan command. The complete prefix is
still good data — e.g. an aborted chi x mu fscan2d whose first chi row is a
full fine mu rocking sweep. This loader truncates to complete fast-motor rows
and returns arrays in darling layout.
"""

import h5py
import numpy as np

_MOTOR_PATHS = (
    "instrument/{m}/value",
    "instrument/{m}/data",
    "instrument/positioners/{m}",
)


def _motor_array(g, name, n_frames):
    for pat in _MOTOR_PATHS:
        path = pat.format(m=name)
        if path in g and g[path].size >= 1:
            arr = np.atleast_1d(np.asarray(g[path][()], dtype=np.float64))
            if arr.size >= n_frames:
                # aborted scans can carry one extra trigger point
                return arr[:n_frames]
            if arr.size == 1:  # motor static during scan
                return np.full(n_frames, arr[0])
    raise ValueError(f"could not find per-frame values for motor '{name}'")


def load_partial_scan(h5file, scan_id, roi=None, detector="pco_ff"):
    """Load a possibly-aborted fscan/fscan2d in darling layout.

    Args:
        h5file (str): BLISS master file path.
        scan_id (str): e.g. "2.1".
        roi (tuple): optional detector (row_min, row_max, col_min, col_max).
        detector (str): detector group name under instrument/.

    Returns:
        tuple: data (a, b, m[, n]) uint16, motors (ndim, m[, n]) float64,
        info dict (scan_command, frames_used, frames_expected, complete).
    """
    with h5py.File(h5file, "r") as f:
        g = f[scan_id]
        title = g["title"][()].decode()
        toks = title.split()
        cmd = toks[0]
        img = g[f"instrument/{detector}/image"]
        n_frames = img.shape[0]

        if cmd == "fscan2d":
            slow_name, n_slow = toks[1], int(toks[4])
            fast_name, n_fast = toks[5], int(toks[8])
            expected = n_slow * n_fast
            rows = n_frames // n_fast
            if rows < 1:
                raise ValueError(
                    f"scan {scan_id} aborted before one complete row "
                    f"({n_frames}/{expected} frames, fast motor {fast_name} x{n_fast})"
                )
            used = rows * n_fast
            slow = _motor_array(g, slow_name, n_frames)[:used].reshape(rows, n_fast)
            fast = _motor_array(g, fast_name, n_frames)[:used].reshape(rows, n_fast)
            sel = (slice(None),) if roi is None else (
                slice(None), slice(roi[0], roi[1]), slice(roi[2], roi[3])
            )
            frames = img[(slice(0, used),) + sel[1:]] if roi else img[:used]
            data = np.ascontiguousarray(
                np.moveaxis(frames.reshape(rows, n_fast, *frames.shape[1:]), (0, 1), (2, 3))
            )
            # fscan2d is a snake scan: sort each row to a monotonic fast-motor
            # grid and rows by slow motor (matches darling's frame reordering)
            for r in range(rows):
                order = np.argsort(fast[r])
                data[:, :, r, :] = data[:, :, r, order]
                fast[r] = fast[r, order]
                slow[r] = slow[r, order]
            row_order = np.argsort(slow[:, 0])
            data = data[:, :, row_order, :]
            slow, fast = slow[row_order], fast[row_order]
            motors = np.stack([slow, fast])
        elif cmd in ("fscan", "ascan", "a2scan"):
            name = toks[1]
            expected = int(toks[4]) + (cmd != "fscan")
            used = n_frames
            mot = _motor_array(g, name, n_frames)
            frames = img[:used] if roi is None else img[:used, roi[0]:roi[1], roi[2]:roi[3]]
            data = np.ascontiguousarray(np.moveaxis(frames, 0, 2))
            order = np.argsort(mot)
            data = data[:, :, order]
            motors = mot[order][None, :]
        else:
            raise ValueError(f"unsupported scan command for partial load: {title}")

    info = {
        "scan_command": title,
        "frames_used": int(used),
        "frames_expected": int(expected),
        "complete": used == expected,
    }
    return data.astype(np.uint16, copy=False), motors, info
