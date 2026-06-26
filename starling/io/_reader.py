"""ID03 scan readers.

Readers return data as (a, b, m[, n]) uint16 with detector dimensions first,
and motors (k, m[, n]). Snake/zigzag scans are re-sorted onto a monotonically
increasing motor grid.

For 2-D ``fscan2d`` scans (the strain-sweep, mosaicity and 3-D strain-mosa
acquisitions) the frames are kept in acquisition order — outer/slow motor on
``axis -2``, inner/fast motor on ``axis -1`` — and each slow row is sorted onto a
monotonic fast-motor grid (which de-zigzags a snake scan), then the rows are
sorted by the slow motor. Because the rows come from acquisition order rather
than from sorting the slow-motor values, this is immune to slow-motor encoder
jitter that a plain lexicographic ``(slow, fast)`` value-sort would otherwise
mistake for the fast-axis order. This mirrors the partial-scan loader exactly.
"""

import warnings

import h5py
import numpy as np

from ._metadata import ID03


def _ascontiguousarrays(data, motors):
    return np.ascontiguousarray(data), np.ascontiguousarray(motors)


def _warn_if_not_separable(motors, scan_id, tol_frac=0.25):
    """Warn if the reconstructed motor grid is not cleanly separable.

    After re-sorting, the slow motor (``motors[0]``) should be ~constant along
    each fast row and the fast motor (``motors[1]``) ~constant down each slow
    column. A large within-row/-column spread means the scan grid is irregular or
    the encoder jitter exceeds the step gap, so per-layer slices and grid-based
    fits cannot be trusted — point the user at the diagnostic script.
    """
    if motors.ndim != 3:
        return
    g_slow, g_fast = motors[0], motors[1]
    m, n = g_slow.shape
    problems = []
    if m > 1:
        step = float(np.median(np.abs(np.diff(g_slow[:, 0]))))
        spread = float(np.ptp(g_slow, axis=1).max())
        if step > 0 and spread > tol_frac * step:
            problems.append(
                f"slow motor varies by {spread:.2e} within a row "
                f"({100 * spread / step:.0f}% of its {step:.2e} step)"
            )
    if n > 1:
        step = float(np.median(np.abs(np.diff(g_fast[0, :]))))
        spread = float(np.ptp(g_fast, axis=0).max())
        if step > 0 and spread > tol_frac * step:
            problems.append(
                f"fast motor varies by {spread:.2e} down a column "
                f"({100 * spread / step:.0f}% of its {step:.2e} step)"
            )
    if problems:
        warnings.warn(
            f"scan {scan_id}: motor grid is not cleanly separable after "
            f"re-sorting ({'; '.join(problems)}). Per-layer slices and grid-based "
            f"fits may be scrambled. Run scripts/check_rasterization.py on this "
            f"scan to investigate (e.g. zigzag/jitter beyond the step gap).",
            stacklevel=3,
        )


class Reader:
    """Base reader. Subclass and implement __call__ for custom acquisition
    schemes; anything returning (data, motors) in the standard layout plugs
    into starling.DataSet."""

    def __init__(self, abs_path_to_h5_file):
        self.abs_path_to_h5_file = abs_path_to_h5_file
        self.config = ID03(abs_path_to_h5_file)
        self.scan_params = None
        self.sensors = None

    def fetch(self, key):
        """Read an arbitrary h5 path (exotic motors, extra metadata)."""
        with h5py.File(self.abs_path_to_h5_file) as h5file:
            return h5file[key][...]

    def _read_stack(self, h5f, scan_id, roi):
        """Read the image stack, reshaped to (*scan_shape, rows, cols) and
        then transposed to detector-first layout."""
        if roi:
            r1, r2, c1, c2 = roi
            data = h5f[scan_id][self.scan_params["data_name"]][:, r1:r2, c1:c2]
        else:
            data = h5f[scan_id][self.scan_params["data_name"]][:, :, :]
        data = data.reshape(
            (*self.scan_params["scan_shape"], data.shape[-2], data.shape[-1])
        )
        data = data.swapaxes(0, -2)
        data = data.swapaxes(1, -1)
        return data

    def __call__(self, scan_id, roi=None):
        raise NotImplementedError


class MosaScan(Reader):
    """2D scan (e.g. fscan2d chi x mu): data (a, b, m, n), motors (2, m, n)."""

    def __call__(self, scan_id, roi=None):
        self.scan_params, self.sensors = self.config(scan_id)

        with h5py.File(self.abs_path_to_h5_file, "r") as h5f:
            motors = [
                h5f[scan_id][mn][...].reshape(*self.scan_params["scan_shape"])
                for mn in self.scan_params["motor_names"]
            ]
            motors = np.array(motors).astype(np.float32)
            data = self._read_stack(h5f, scan_id, roi)

        command = self.scan_params["scan_command"].split()[0]
        if command == "fscan2d":
            # acquisition order: axis -2 = slow (outer) levels, axis -1 = fast
            # (inner) sweep. Jitter-immune (rows come from acquisition order).
            data, motors = self._resort_snake_by_acquisition(data, motors)
        else:
            # arbitrary 2-D acquisition (e.g. amesh): fall back to a lexicographic
            # value sort, which does not assume acquisition order.
            data, motors = self._resort_by_value(data, motors)

        _warn_if_not_separable(motors, scan_id)
        return _ascontiguousarrays(data, motors)

    @staticmethod
    def _resort_snake_by_acquisition(data, motors):
        """Sort each slow row onto a monotonic fast grid, then rows by the slow
        motor — de-zigzagging a snake scan without sorting the slow values."""
        data = np.ascontiguousarray(data)
        a, b, m, n = data.shape
        slow = motors[0].astype(np.float32).copy()
        fast = motors[1].astype(np.float32).copy()
        for r in range(m):
            order = np.argsort(fast[r], kind="stable")
            data[:, :, r, :] = data[:, :, r, order]
            fast[r] = fast[r, order]
            slow[r] = slow[r, order]
        row_order = np.argsort(slow[:, 0], kind="stable")
        data = data[:, :, row_order, :]
        motors = np.stack([slow[row_order], fast[row_order]])
        return data, motors

    @staticmethod
    def _resort_by_value(data, motors):
        """Legacy lexicographic ``(motor1, motor2)`` value sort."""
        s = np.array(
            list(zip(motors[0].flatten(), motors[1].flatten())),
            dtype=[("m1", "f8"), ("m2", "f8")],
        )
        frame_indices = np.argsort(s, order=["m1", "m2"])
        a, b, m, n = data.shape
        data = data.reshape(a, b, m * n)[..., frame_indices].reshape(a, b, m, n)
        motors = motors.copy()
        motors[0, :] = motors[0, :].flatten()[frame_indices].reshape(m, n)
        motors[1, :] = motors[1, :].flatten()[frame_indices].reshape(m, n)
        return data, motors


class RockingScan(Reader):
    """1D scan (e.g. fscan mu): data (a, b, m), motors (1, m)."""

    def __call__(self, scan_id, roi=None):
        self.scan_params, self.sensors = self.config(scan_id)

        with h5py.File(self.abs_path_to_h5_file, "r") as h5f:
            motors = [
                h5f[scan_id][mn][...].reshape(*self.scan_params["scan_shape"])
                for mn in self.scan_params["motor_names"]
            ]
            motors = np.array(motors).astype(np.float32)
            data = self._read_stack(h5f, scan_id, roi)

        frame_indices = np.argsort(motors[0].flatten())
        data = data[..., frame_indices]
        motors[0, :] = motors[0, frame_indices]

        return _ascontiguousarrays(data, motors)


class Darks(Reader):
    """Motorless image series (loopscan darks): data (a, b, m), empty motors."""

    def __call__(self, scan_id, roi=None):
        self.scan_params, self.sensors = self.config(scan_id)

        with h5py.File(self.abs_path_to_h5_file, "r") as h5f:
            motors = np.array([], dtype=np.float32)
            data = self._read_stack(h5f, scan_id, roi)

        return _ascontiguousarrays(data, motors)
