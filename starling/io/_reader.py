"""ID03 scan readers.

Readers return data as (a, b, m[, n]) uint16 with detector dimensions first,
and motors (k, m[, n]). Snake/zigzag scans are re-sorted onto a monotonically
increasing motor grid.
"""

import h5py
import numpy as np

from ._metadata import ID03


def _ascontiguousarrays(data, motors):
    return np.ascontiguousarray(data), np.ascontiguousarray(motors)


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

        # re-sort onto a monotonically increasing grid (snake/zigzag scans)
        s = np.array(
            list(zip(motors[0].flatten(), motors[1].flatten())),
            dtype=[("m1", "f8"), ("m2", "f8")],
        )
        frame_indices = np.argsort(s, order=["m1", "m2"])
        a, b, m, n = data.shape
        data = data.reshape(a, b, m * n)[..., frame_indices].reshape(a, b, m, n)
        motors[0, :] = motors[0, :].flatten()[frame_indices].reshape(m, n)
        motors[1, :] = motors[1, :].flatten()[frame_indices].reshape(m, n)

        return _ascontiguousarrays(data, motors)


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
