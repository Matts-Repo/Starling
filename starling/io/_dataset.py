"""starling.DataSet — self-contained ID03 BLISS HDF5 loading + GPU analysis.

The ID03 reading layer (metadata parsing, scan readers) is built into
starling.io._metadata / _reader, so starling installs standalone with no
external DFXM dependencies.
"""

import warnings

import h5py
import numpy as np
from tqdm import tqdm

from .. import preprocess, properties
from ..device import get_device
from ._metadata import ID03
from ._reader import Darks, MosaScan, Reader, RockingScan


class _PartialScan:
    """Minimal reader stand-in for partially acquired scans."""

    def __init__(self, data, motors, h5file, scan_id, info):
        self.data = data
        self.motors = motors
        self.h5file = h5file
        self.scan_params = {"scan_id": scan_id, **info}

    def info(self):
        for k, v in self.scan_params.items():
            print(f"{k:<20} :  {str(v):<30}")


class DataSet:
    """A DFXM scan with GPU-accelerated analysis.

    Args:
        data_source: BLISS master h5 file path, or a Reader instance for
            custom acquisition schemes.
        scan_id: scan id ("1.1") or list of ids to stack along a third motor.
        suffix: suffix pattern matching scan ids (alternative to scan_id),
            e.g. ".1" selects 1.1, 2.1, ...
        scan_motor: h5 path of the motor varying across stacked scan_ids.
        roi: (row_min, row_max, col_min, col_max) detector ROI to load.
        device: torch device or name; None auto-detects (cuda > mps > cpu).
        verbose: print loading progress.
        allow_partial: load aborted scans by keeping complete fast-motor rows.
        n_workers: read-process count for the loader. None (default)
            auto-enables multiprocess reads for large scans; 0/1 forces the
            serial reader; >= 2 forces that many worker processes. Any
            parallel-read failure falls back to the serial path with a
            warning, so results are unaffected.
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
        allow_partial=False,
        n_workers=None,
    ):
        self.device = get_device(device)
        self.partial_info = None
        self.reader = None
        self.data = None
        self.motors = None
        self.roi = None
        self.scan_id = None
        self.scan_motor = None
        self.n_workers = n_workers

        if isinstance(data_source, Reader):
            self.reader = data_source
            self.h5file = data_source.abs_path_to_h5_file
            if n_workers is not None:
                self.reader.n_workers = n_workers
        elif isinstance(data_source, str):
            self.h5file = data_source
        else:
            raise ValueError(
                "data_source must be a starling.io.Reader or a path to a h5 file"
            )

        if suffix is not None and scan_id is not None:
            raise ValueError(
                f"Cannot use both suffix and scan_id, but suffix is {suffix} "
                f"and scan_id is {scan_id}"
            )
        if scan_id is None and suffix is None:
            if roi is not None:
                raise ValueError("Cannot apply a roi without a scan_id")
            return  # defer to load_scan()

        if suffix is not None:
            scan_id = self._get_scan_ids(suffix)

        if allow_partial:
            from ._partial import load_partial_scan

            if not isinstance(scan_id, str):
                raise ValueError("allow_partial supports a single scan_id only")
            data, motors, info = load_partial_scan(self.h5file, scan_id, roi=roi)
            self.reader = _PartialScan(data, motors, self.h5file, scan_id, info)
            self.data, self.motors = data, motors
            self.partial_info = info
            self.roi = roi
            self.scan_id = scan_id
            if verbose and not info["complete"]:
                print(
                    f"partial scan {scan_id}: using {info['frames_used']}/"
                    f"{info['frames_expected']} frames -> shape {data.shape}"
                )
            return

        try:
            self.load_scan(scan_id, scan_motor=scan_motor, roi=roi, verbose=verbose)
        except ValueError as e:
            if "Could not find" in str(e):
                raise ValueError(
                    f"{e}\nThis often means scan {scan_id} was aborted (fewer "
                    f"frames than the scan command declares). Retry with "
                    f"DataSet(..., allow_partial=True) to load the complete prefix."
                ) from e
            raise

    # ------------------------------------------------------------------ #
    # loading
    # ------------------------------------------------------------------ #

    def _get_scan_ids(self, suffix):
        with h5py.File(self.h5file) as h5file:
            scan_ids = [k for k in h5file.keys() if k.endswith(suffix)]
        if len(scan_ids) == 0:
            raise ValueError(
                f"No scan ids found with suffix {suffix} in file {self.h5file}."
            )
        if len(scan_ids) == 1:
            return scan_ids[0]
        try:
            scan_ids.sort(key=float)
        except ValueError:
            scan_ids.sort()
        return scan_ids

    def load_scan(self, scan_id, scan_motor=None, roi=None, verbose=True):
        """Load a scan (or stack of scans) into RAM."""
        if not isinstance(scan_id, (list, str)):
            raise ValueError("scan_id must be a list of strings or a single string")
        if isinstance(scan_id, list) and not isinstance(scan_motor, str):
            raise ValueError("When scan_id is a list, the scan_motor path must be set.")
        if isinstance(scan_id, list) and len(scan_id) == 1:
            raise ValueError("When scan_id is a list, len(scan_id) must be > 1.")

        if self.reader is None:
            config = ID03(self.h5file)
            reference_scan_id = scan_id[0] if isinstance(scan_id, list) else scan_id
            scan_params, _sensors = config(reference_scan_id)
            n_workers = getattr(self, "n_workers", None)
            if scan_params["motor_names"] is None:
                self.reader = Darks(self.h5file, n_workers=n_workers)
            elif len(scan_params["motor_names"]) == 1:
                self.reader = RockingScan(self.h5file, n_workers=n_workers)
            elif len(scan_params["motor_names"]) == 2:
                self.reader = MosaScan(self.h5file, n_workers=n_workers)
            else:
                raise ValueError("Could not find a reader for your h5 file")

        if isinstance(scan_id, str):
            self.data, self.motors = self.reader(scan_id, roi)
        else:
            self._load_stacked_scans(scan_id, scan_motor, roi, verbose)

        self.roi = roi
        self.scan_id = scan_id
        self.scan_motor = scan_motor

    def _load_stacked_scans(self, scan_id, scan_motor, roi, verbose):
        """Stack multiple scans along a third motor dimension.

        For the built-in readers, all per-scan metadata (stack-motor scalars,
        scan commands, motor grids, frame orders, detector keys) is resolved
        in one pass over a single h5py handle, the cube is allocated once and
        every sub-scan is read directly into its slice — via the multiprocess
        read engine (one worker per sub-scan) when enabled, else serially.
        Custom readers keep the legacy one-scan-at-a-time path.
        """
        if type(self.reader) not in (MosaScan, RockingScan):
            return self._load_stacked_scans_generic(scan_id, scan_motor, roi, verbose)

        from ._reader import (
            _dest_chunks,
            _frames_per_chunk,
            _resolve_workers,
            _roi_extent,
            _warn_if_not_separable,
        )

        with h5py.File(self.h5file, "r") as h5f:
            # one metadata pass: stack-motor scalars + per-scan read plans
            scan_motor_values = np.zeros((len(scan_id),))
            for i, sid in enumerate(scan_id):
                scan_motor_values[i] = h5f[sid][scan_motor][()]

            order = np.argsort(scan_motor_values)
            scan_id = [scan_id[idx] for idx in order]
            scan_motor_values = scan_motor_values[order]

            plans = [(sid, *self.reader._prepare(h5f, sid)) for sid in scan_id]
            for sid, _params, _sens, _perm, motors_i, _row_len in plans:
                _warn_if_not_separable(motors_i, sid)

            reference_motors = plans[0][4]
            if reference_motors.ndim == 2:
                motor1 = reference_motors[0, :]
                motors = np.array(
                    np.meshgrid(motor1, scan_motor_values, indexing="ij")
                )
            elif reference_motors.ndim == 3:
                motor1 = reference_motors[0, :, 0]
                motor2 = reference_motors[1, 0, :]
                motors = np.array(
                    np.meshgrid(motor1, motor2, scan_motor_values, indexing="ij")
                )
            else:
                raise ValueError(
                    f"Each scan_id must hold a 1D or 2D scan but "
                    f"{reference_motors.ndim}D was found at scan_id={scan_id[0]}"
                )

            params0 = plans[0][1]
            dset0 = h5f[scan_id[0]][params0["data_name"]]
            rows, cols = _roi_extent(dset0.shape, roi)
            n_frames = int(dset0.shape[0])
            frame_bytes = rows * cols * np.dtype(np.uint16).itemsize
            cube_shape = (
                rows,
                cols,
                *(int(s) for s in params0["scan_shape"]),
                len(scan_id),
            )

            data = None
            n_workers = _resolve_workers(
                self.reader.n_workers,
                int(np.prod(cube_shape)) * np.dtype(np.uint16).itemsize,
                len(scan_id),
            )
            if n_workers >= 2:
                from . import _mpread

                jobs = []
                for k, (sid, params_i, _sens, perm_i, _mot, row_len_i) in enumerate(
                    plans
                ):
                    chunks = _dest_chunks(
                        n_frames, _frames_per_chunk(n_frames, frame_bytes, row_len_i)
                    )
                    jobs.append(
                        (self.h5file, sid, params_i["data_name"], roi, perm_i, k,
                         chunks)
                    )
                try:
                    data = _mpread.read_jobs_shm(
                        cube_shape, np.uint16, jobs, n_workers
                    )
                except Exception as exc:
                    warnings.warn(
                        f"parallel stacked load failed ({exc!r}); falling back "
                        f"to the serial reader.",
                        stacklevel=3,
                    )
            if data is None:
                # serial: each sub-scan is placed straight into its cube slice
                # (no per-scan temporary, no extra strided copy pass)
                data = np.empty(cube_shape, np.uint16)
                for k, (sid, params_i, _sens, perm_i, _mot, row_len_i) in enumerate(
                    tqdm(plans, disable=not verbose)
                ):
                    self.reader._read_placed(
                        h5f, sid, params_i, roi, perm_i, out=data[..., k],
                        row_len=row_len_i,
                    )

        # match the legacy loop, which left the last sub-scan's metadata on
        # the reader before appending the stack motor
        self.reader.scan_params = plans[-1][1]
        self.reader.sensors = plans[-1][2]
        self.reader.scan_params["motor_names"].append(scan_motor)
        self.reader.scan_params["scan_shape"] = np.array(
            [*self.reader.scan_params["scan_shape"], len(scan_id)]
        )
        self.reader.scan_params["integrated_motors"].append(False)
        self.reader.scan_params["scan_id"] = scan_id

        self.motors = motors
        self.data = data

    def _load_stacked_scans_generic(self, scan_id, scan_motor, roi, verbose):
        """Legacy stacked load for custom/duck-typed readers: one reader call
        per sub-scan, copied into the cube."""
        scan_motor_values = np.zeros((len(scan_id),))
        with h5py.File(self.h5file) as h5file:
            for i, sid in enumerate(scan_id):
                scan_motor_values[i] = h5file[sid][scan_motor][()]

        order = np.argsort(scan_motor_values)
        scan_id = [scan_id[idx] for idx in order]
        scan_motor_values = scan_motor_values[order]

        reference_data_block, reference_motors = self.reader(scan_id[0], roi)

        if reference_motors.ndim == 2:
            motor1 = reference_motors[0, :]
            motors = np.array(np.meshgrid(motor1, scan_motor_values, indexing="ij"))
        elif reference_motors.ndim == 3:
            motor1 = reference_motors[0, :, 0]
            motor2 = reference_motors[1, 0, :]
            motors = np.array(
                np.meshgrid(motor1, motor2, scan_motor_values, indexing="ij")
            )
        else:
            raise ValueError(
                f"Each scan_id must hold a 1D or 2D scan but "
                f"{reference_motors.ndim}D was found at scan_id={scan_id[0]}"
            )

        data = np.zeros((*reference_data_block.shape, len(scan_id)), np.uint16)
        data[..., 0] = reference_data_block[...]
        for i, sid in enumerate(tqdm(scan_id[1:], disable=not verbose)):
            data_block, _ = self.reader(sid, roi)
            data[..., i + 1] = data_block[...]

        self.reader.scan_params["motor_names"].append(scan_motor)
        self.reader.scan_params["scan_shape"] = np.array(
            [*self.reader.scan_params["scan_shape"], len(scan_id)]
        )
        self.reader.scan_params["integrated_motors"].append(False)
        self.reader.scan_params["scan_id"] = scan_id

        self.motors = motors
        self.data = data

    # ------------------------------------------------------------------ #
    # metadata
    # ------------------------------------------------------------------ #

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def scan_params(self):
        if self.reader is None:
            raise ValueError("No data has been loaded, use load_scan() to load data.")
        return self.reader.scan_params

    @property
    def sensors(self):
        if self.reader is None:
            raise ValueError("No data has been loaded, use load_scan() to load data.")
        return self.reader.sensors

    def info(self):
        if self.data is not None:
            for k in self.scan_params:
                print(f"{k:<20} :  {str(self.scan_params[k]):<30}")
        else:
            print("No data loaded, use load_scan() to load data.")

    # ------------------------------------------------------------------ #
    # preprocessing
    # ------------------------------------------------------------------ #

    def estimate_background(
        self, n_lowest=5, mode="mean", percentile=10.0, darks=None, chunk_rows=None
    ):
        return preprocess.estimate_background(
            self.data,
            n_lowest=n_lowest,
            mode=mode,
            percentile=percentile,
            darks=darks,
            chunk_rows=chunk_rows,
        )

    def subtract(self, background):
        preprocess.subtract(self.data, background)

    def remove_hot_pixels(
        self,
        n_sigma=5.0,
        one_sided=False,
        protect=None,
        min_sigma=None,
        method="frame",
        backend="torch",
        device=None,
        chunk_frames=None,
    ):
        preprocess.remove_hot_pixels(
            self.data,
            n_sigma=n_sigma,
            one_sided=one_sided,
            protect=protect,
            min_sigma=min_sigma,
            method=method,
            backend=backend,
            device=device,
            chunk_frames=chunk_frames,
        )

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

    @property
    def n_motor_dims(self):
        """Number of motor (scan) dimensions = ``data.ndim - 2``."""
        if self.data is None:
            raise ValueError("No data has been loaded, use load_scan() to load data.")
        return self.data.ndim - 2

    def moments(self, order=2, mask=None):
        return properties.moments(
            self.data, self.motors, order=order, mask=mask, device=self.device
        )

    def fit_1D_gaussian(self, n_iter_gauss_newton=7, mask=None):
        return properties.fit_1D_gaussian(
            self.data,
            self.motors,
            n_iter_gauss_newton=n_iter_gauss_newton,
            mask=mask,
            device=self.device,
        )

    def fit_1D_pseudo_voigt(self, n_iter_gauss_newton=10, mask=None):
        return properties.fit_1D_pseudo_voigt(
            self.data,
            self.motors,
            n_iter_gauss_newton=n_iter_gauss_newton,
            mask=mask,
            device=self.device,
        )

    def fit_two_gaussians_1D(self, n_iter_gauss_newton=12, mask=None, delta_bic=10.0):
        return properties.fit_two_gaussians_1D(
            self.data,
            self.motors,
            n_iter_gauss_newton=n_iter_gauss_newton,
            mask=mask,
            delta_bic=delta_bic,
            device=self.device,
        )

    def fit_2D_gaussian(self, n_iter_gauss_newton=10, mask=None):
        return properties.fit_2D_gaussian(
            self.data,
            self.motors,
            n_iter_gauss_newton=n_iter_gauss_newton,
            mask=mask,
            device=self.device,
        )

    def fit_ND_gaussian(self, n_iter_gauss_newton=10, mask=None, lam=1e-1,
                        adaptive=True, progress=True):
        return properties.fit_ND_gaussian(
            self.data,
            self.motors,
            n_iter_gauss_newton=n_iter_gauss_newton,
            mask=mask,
            device=self.device,
            lam=lam,
            adaptive=adaptive,
            progress=progress,
        )

    def fit_ND_two_gaussians(self, n_iter_gauss_newton=14, mask=None,
                             delta_bic=10.0, single=None, progress=True,
                             **opts):
        """Two-component N-D Gaussian fit with per-pixel BIC model selection.

        Pass ``single=`` a precomputed :class:`~starling.properties.GaussNDResult`
        (from :meth:`fit_ND_gaussian` with the same mask) to skip refitting the
        1-peak model. See :func:`starling.properties.fit_ND_two_gaussians`.
        """
        return properties.fit_ND_two_gaussians(
            self.data,
            self.motors,
            n_iter_gauss_newton=n_iter_gauss_newton,
            mask=mask,
            device=self.device,
            delta_bic=delta_bic,
            single=single,
            progress=progress,
            **opts,
        )

    # ------------------------------------------------------------------ #
    # auto-dispatch + convenience (Section 7)
    # ------------------------------------------------------------------ #

    def _resolve_mask(self, mask):
        """Turn the ``mask`` argument into a bool array or None.

        ``"auto"`` builds a grain mask from the z-sum (speeds up fitting by
        skipping non-grain pixels); ``None`` fits every pixel; an array is used
        verbatim.
        """
        if mask is None:
            return None
        if isinstance(mask, str) and mask == "auto":
            return preprocess.grain_mask(self.data)
        return np.asarray(mask, dtype=bool)

    def analyze(self, method="auto", mask="auto", order=2, two_peak=False, **opts):
        """Run the appropriate analysis for this scan's dimensionality.

        Dispatch is by **motor-dimension count**, so a 1-D, 2-D or 3-D scan all
        "just work" with no ``if SCAN_TYPE`` branching and no dimensionality
        error:

        * ``method="auto"`` and 1 motor dim -> :class:`Gauss1DResult`.
        * ``method="auto"`` and >= 2 motor dims -> :class:`GaussNDResult`
          (the fitted N-D Gaussian; use its ``.mosaicity()`` / ``.orientation()``
          for the spread / mean-orientation maps).

        ``method`` can be forced to ``"moments"`` (-> :class:`MomentResult`),
        ``"gauss1d"``, ``"gauss2p"`` (-> the two-peak dict), ``"gaussND"``,
        ``"gaussND2"`` (-> :class:`GaussNDTwoResult`) or ``"pv"``
        (-> :class:`PseudoVoigtResult`).

        Args:
            method (str): "auto" or a forced method name.
            mask: "auto" (grain mask, default), None (all pixels) or a bool array.
            order (int): moment order when moments are computed (2 or 4).
            two_peak (bool): with ``method="auto"``, run the two-peak fit for
                this dimensionality instead ("gauss2p" for 1 motor dim,
                "gaussND2" otherwise).
            **opts: forwarded to the underlying fit (e.g. ``n_iter_gauss_newton``).

        Returns:
            The result object for the chosen method.
        """
        from ..properties import Gauss1DResult, MomentResult

        m = self._resolve_mask(mask)
        n = self.n_motor_dims
        if method == "auto":
            if two_peak:
                method = "gauss2p" if n == 1 else "gaussND2"
            else:
                method = "gauss1d" if n == 1 else "gaussND"

        if method == "moments":
            res = self.moments(order=order, mask=m)
            if order == 4:
                mean, cov, skew, kurt = res
                return MomentResult(mean, cov, skew, kurt)
            mean, cov = res
            return MomentResult(mean, cov)
        if method == "gauss1d":
            if n != 1:
                raise ValueError(
                    f"gauss1d needs 1 motor dim but this scan has {n}; use "
                    f"method='gaussND' (or 'auto')."
                )
            return Gauss1DResult.from_raw(self.fit_1D_gaussian(mask=m, **opts))
        if method == "gauss2p":
            if n != 1:
                raise ValueError(
                    f"gauss2p needs 1 motor dim but this scan has {n}."
                )
            return self.fit_two_gaussians_1D(mask=m, **opts)
        if method == "pv":
            if n != 1:
                raise ValueError(f"pv needs 1 motor dim but this scan has {n}.")
            return self.fit_1D_pseudo_voigt(mask=m, **opts)
        if method == "gaussND":
            if n < 2:
                raise ValueError(
                    f"gaussND needs >= 2 motor dims but this scan has {n}; use "
                    f"method='gauss1d' (or 'auto')."
                )
            return self.fit_ND_gaussian(mask=m, **opts)
        if method == "gaussND2":
            if n < 2:
                raise ValueError(
                    f"gaussND2 needs >= 2 motor dims but this scan has {n}; use "
                    f"method='gauss2p' (or two_peak=True with 'auto')."
                )
            return self.fit_ND_two_gaussians(mask=m, **opts)
        raise ValueError(
            f"unknown method {method!r}; expected 'auto', 'moments', 'gauss1d', "
            f"'gauss2p', 'gaussND', 'gaussND2' or 'pv'"
        )

    def mosaicity(self, mode="scalar", axes=None, source="fit", mask="auto"):
        """Orientation-spread (mosaicity) map; from the fit (default) or moments.

        ``source="fit"`` uses the fitted covariance (less window/background bias,
        recommended); ``source="moments"`` uses the raw second moment.
        """
        m = self._resolve_mask(mask)
        if source == "fit" and self.n_motor_dims >= 2:
            return self.fit_ND_gaussian(mask=m).mosaicity(mode=mode, axes=axes)
        _, cov = self.moments(mask=m)
        return properties.mosaicity(cov, mode=mode, axes=axes)

    def orientation(self, axes=(0, 1), norm="dynamic", as_rgb=False, mask="auto"):
        """Mean-orientation (centre-of-mass) map for the chosen motor axes."""
        m = self._resolve_mask(mask)
        mean, _ = self.moments(mask=m)
        return properties.orientation_map(mean, axes=axes, norm=norm, as_rgb=as_rgb)

    def strain(self, motor="ccmth", axis=None, reference=None, mask="auto"):
        """Strain map from the orientation/COM along a motor axis.

        ``motor`` selects the formula ("ccmth" or "obpitch"); ``axis`` selects
        which motor dimension to read the COM from (auto-resolved from the scan
        motor names when omitted, defaulting to axis 0 for a 1-motor scan).
        """
        m = self._resolve_mask(mask)
        mean = self.moments(mask=m)[0]
        if mean.ndim == 2:  # single motor
            com = mean
        else:
            if axis is None:
                axis = self._motor_axis(motor)
            com = mean[..., axis]
        if motor == "ccmth":
            return properties.strain_from_ccmth(com, ccmth_0=reference)
        if motor == "obpitch":
            return properties.strain_from_obpitch(com, obpitch_0=reference)
        raise ValueError(f"motor must be 'ccmth' or 'obpitch', got {motor!r}")

    def _motor_axis(self, motor):
        """Best-effort map from a motor name to its axis index (default 0)."""
        try:
            names = self.scan_params.get("motor_names") or []
            for i, nm in enumerate(names):
                if motor in str(nm):
                    return i
        except Exception:
            pass
        return 0

    def kam(self, size=(3, 3), min_neighbors=1, axes=(0, 1), mask="auto"):
        """Kernel average misorientation from the orientation (COM) field."""
        from .. import transforms

        m = self._resolve_mask(mask)
        mean = self.moments(mask=m)[0]
        com = mean if mean.ndim == 2 else mean[..., list(axes)]
        return transforms.kam(com, size=size, min_neighbors=min_neighbors)

    def save_maps(self, path, maps, extra_attrs=None):
        from ._output import save_maps

        save_maps(path, maps, scan_params=self.scan_params, extra_attrs=extra_attrs)

    def save_nexus(self, path, result, *, motors=None, scan_params=None, **kw):
        """Write a result object to a standards-compliant NeXus file.

        Pulls ``self.motors`` / ``self.scan_params`` (the latter is ``None`` when
        no scan has been loaded) unless overridden. See
        :func:`starling.io.save_nexus`.
        """
        from ._nexus import save_nexus

        if motors is None:
            motors = self.motors
        if scan_params is None:
            try:
                scan_params = self.scan_params
            except Exception:
                scan_params = None
        return save_nexus(path, result, motors=motors, scan_params=scan_params, **kw)

    def save_dataset_nexus(self, path, **kw):
        """Persist this dataset's denoised cube + motors + provenance as NeXus.

        See :func:`starling.io.save_dataset_nexus`.
        """
        from ._nexus import save_dataset_nexus

        return save_dataset_nexus(path, self, **kw)
