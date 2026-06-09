"""starling.DataSet — composes darling.DataSet for ID03 BLISS HDF5 IO.

darling (the local checkout, installed editable — it carries beamline patches
such as amesh scan support) handles all file reading and scan-command parsing.
starling adds GPU-backed analysis and preprocessing convenience methods.
"""

import numpy as np

from .. import preprocess, properties
from ..device import get_device


class DataSet:
    """A DFXM scan with GPU-accelerated analysis.

    Args:
        data_source: h5 file path or a darling reader instance.
        scan_id: scan id ("1.1") or list of ids to stack.
        suffix: suffix pattern matching scan ids (alternative to scan_id).
        scan_motor: h5 path of the motor varying across stacked scan_ids.
        roi: (row_min, row_max, col_min, col_max) detector ROI to load.
        device: torch device or name; None auto-detects (cuda > mps > cpu).
        verbose: print loading progress.
    """

    def __init__(
        self,
        data_source,
        scan_id=None,
        suffix=None,
        scan_motor=None,
        roi=None,
        device=None,
        verbose=True,
    ):
        import darling

        self._dset = darling.DataSet(
            data_source,
            scan_id=scan_id,
            suffix=suffix,
            scan_motor=scan_motor,
            roi=roi,
            verbose=verbose,
        )
        self.device = get_device(device)

    @property
    def data(self):
        return self._dset.data

    @data.setter
    def data(self, value):
        self._dset.data = value

    @property
    def motors(self):
        return self._dset.motors

    @property
    def scan_params(self):
        return self._dset.scan_params

    @property
    def h5file(self):
        return self._dset.h5file

    def info(self):
        self._dset.info()

    # ------------------------------------------------------------------ #
    # preprocessing
    # ------------------------------------------------------------------ #

    def estimate_background(self, n_lowest=5, mode="mean"):
        return preprocess.estimate_background(self.data, n_lowest=n_lowest, mode=mode)

    def subtract(self, background):
        preprocess.subtract(self.data, background)

    def remove_hot_pixels(self, n_sigma=5.0):
        preprocess.remove_hot_pixels(self.data, n_sigma=n_sigma)

    def auto_roi(self, threshold_rel=0.05, pad=20, apply=True):
        """Find (and by default crop to) the grain bounding box."""
        roi = preprocess.auto_roi(self.data, threshold_rel=threshold_rel, pad=pad)
        if apply:
            r1, r2, c1, c2 = roi
            self.data = np.ascontiguousarray(self.data[r1:r2, c1:c2])
        return roi

    # ------------------------------------------------------------------ #
    # analysis (GPU)
    # ------------------------------------------------------------------ #

    def moments(self):
        return properties.moments(self.data, self.motors, device=self.device)

    def fit_1D_gaussian(self, n_iter_gauss_newton=7, mask=None):
        return properties.fit_1D_gaussian(
            self.data,
            self.motors,
            n_iter_gauss_newton=n_iter_gauss_newton,
            mask=mask,
            device=self.device,
        )

    def save_maps(self, path, maps, extra_attrs=None):
        from ._output import save_maps

        save_maps(path, maps, scan_params=self.scan_params, extra_attrs=extra_attrs)
