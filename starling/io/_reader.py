"""ID03 scan readers.

Readers return data as (a, b, m[, n]) uint16 with detector dimensions first,
and motors (k, m[, n]) as float64. Snake/zigzag scans are re-sorted onto a
monotonically increasing motor grid.

For 2-D ``fscan2d`` scans (the strain-sweep, mosaicity and 3-D strain-mosa
acquisitions) the frames are kept in acquisition order — outer/slow motor on
``axis -2``, inner/fast motor on ``axis -1`` — and each slow row is sorted onto a
monotonic fast-motor grid (which de-zigzags a snake scan), then the rows are
sorted by the slow motor. Because the rows come from acquisition order rather
than from sorting the slow-motor values, this is immune to slow-motor encoder
jitter that a plain lexicographic ``(slow, fast)`` value-sort would otherwise
mistake for the fast-axis order. This mirrors the partial-scan loader exactly.

Read path: the frame order is computed from the motor datasets *before* any
frame is read, the final detector-first array is allocated once, and slabs of
frames are placed into it (transposed, re-sort folded into the placement) as
they are decompressed. There is no reshape/swapaxes/contiguous-copy pass and
no separate resort pass, so peak memory is the final array plus one slab.
With ``n_workers`` (or automatically for large scans) the slabs are read by
worker processes into shared memory — libhdf5 holds one process-global lock,
so threads cannot parallelise decompression, but processes can (see
``starling.io._mpread``).
"""

import os
import warnings

import h5py
import numpy as np

from ._metadata import ID03

# target bytes per read slab (serial and per worker); chunks align to whole
# fast-axis rows so the snake re-sort stays local to one slab
_SLAB_BYTES = 192 * 2**20
# auto-enable the multiprocess read engine above this stack size
_MP_MIN_BYTES = int(os.environ.get("STARLING_MP_MIN_BYTES", 2 * 2**30))
_MP_MAX_AUTO_WORKERS = 8


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


# --------------------------------------------------------------------------- #
# acquisition-mode detection (report-only; never changes the sort)
# --------------------------------------------------------------------------- #

# BLISS fast_motor_mode string -> raster/zigzag classification. REWIND rewinds
# to the row start before each fast sweep (all rows same direction = raster);
# ZIGZAG reverses the fast sweep on alternate rows (snake). Anything else is
# reported verbatim but classified "unknown".
_FAST_MOTOR_MODE_MAP = {"REWIND": "raster", "ZIGZAG": "zigzag"}


def _fast_motor_grid(h5f, scan_id, motors=None, row_len=None):
    """Fast-motor values as an (n_slow, n_fast) grid in acquisition order.

    Prefers a caller-supplied acquisition-order ``motors`` array (the reader
    already has it): a (2, m, n) grid is used directly; a flat (2, n_frames)
    array is reshaped with ``row_len``. Otherwise the fast-motor dataset is
    read straight from the file and reshaped from the scan command's grid.
    """
    if motors is not None:
        arr = np.asarray(motors, dtype=np.float64)
        if arr.ndim == 3:
            return arr[1]
        if arr.ndim == 2 and row_len:
            n = int(row_len)
            fast = arr[1].reshape(-1)
            m = fast.size // n
            return fast[: m * n].reshape(m, n)

    cfg = ID03("")
    title = h5f[scan_id]["title"][()]
    title = title.decode("utf-8") if isinstance(title, bytes) else str(title)
    parts = title.split()
    command = parts[0]
    args = parts[1:]
    steps_idx = cfg.scan_arg_pos["motor_steps"][command]
    shape = np.array(args)[steps_idx].astype(int)
    if command in ("a2scan", "ascan", "amesh"):
        shape = shape + 1
    m, n = int(shape[0]), int(shape[1])
    name_idx = cfg.scan_arg_pos["motor_names"][command][-1]  # fast = inner motor
    fast_name = args[name_idx]
    path = cfg.motor_map[fast_name]
    grp = h5f[scan_id]
    if path not in grp:
        path = cfg.fallback_motor_map.get(path)
    return np.asarray(grp[path][...], dtype=np.float64).reshape(m, n)


def _infer_mode_from_grid(grid):
    """raster/zigzag/unknown from the per-row monotonic direction of a grid."""
    m, n = grid.shape
    if m < 2 or n < 2:
        return "unknown"
    dirs = np.sign(grid[:, -1] - grid[:, 0])  # +1 / -1 / 0 per slow row
    if np.any(dirs == 0):
        return "unknown"
    if np.all(dirs == dirs[0]):
        return "raster"
    if np.all(dirs[1:] == -dirs[:-1]):  # each row reverses the previous
        return "zigzag"
    return "unknown"


def detect_acquisition_mode(h5f_or_path, scan_id, motors=None):
    """Report how a 2-motor scan was rastered.

    Returns dict: {"mode": "zigzag"|"raster"|"unknown", "source":
    "metadata"|"inferred"|"none", "fast_motor_mode": <raw str or None>}

    The primary source is the BLISS ``instrument/fscan_parameters/
    fast_motor_mode`` string (REWIND -> raster, ZIGZAG -> zigzag). Without it,
    the mode is inferred from the fast-motor positions in acquisition order.
    This is report-only: it never influences the de-zigzag sort, and never
    raises — odd scans (darks, 1-motor, aborted) yield mode/source unknown/none.
    """
    unknown = {"mode": "unknown", "source": "none", "fast_motor_mode": None}
    try:
        if isinstance(h5f_or_path, (str, os.PathLike)):
            with h5py.File(h5f_or_path, "r") as h5f:
                return detect_acquisition_mode(h5f, scan_id, motors=motors)
        h5f = h5f_or_path

        mode_path = f"{scan_id}/instrument/fscan_parameters/fast_motor_mode"
        if mode_path in h5f:
            raw = h5f[mode_path][()]
            raw = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
            mode = _FAST_MOTOR_MODE_MAP.get(raw.strip().upper(), "unknown")
            return {"mode": mode, "source": "metadata", "fast_motor_mode": raw}

        grid = _fast_motor_grid(h5f, scan_id, motors=motors)
        return {
            "mode": _infer_mode_from_grid(grid),
            "source": "inferred",
            "fast_motor_mode": None,
        }
    except Exception:
        return dict(unknown)


# --------------------------------------------------------------------------- #
# frame-order permutations (computed from motors only, before any frame read)
# --------------------------------------------------------------------------- #


def _snake_perm(motors):
    """Frame permutation + sorted motor grid for an fscan2d acquisition.

    Sorts each slow row onto a monotonic fast grid (de-zigzagging a snake
    scan), then the rows by the slow motor — without sorting the slow values
    inside a row, so slow-motor encoder jitter cannot scramble the grid.

    Args:
        motors: (2, m, n) float [slow, fast] values in acquisition order.

    Returns:
        tuple: perm (m*n,) int64 with ``sorted_flat[k] = acquired_flat[perm[k]]``,
        and motors (2, m, n) float64 on the sorted grid.
    """
    slow = np.asarray(motors[0], dtype=np.float64).copy()
    fast = np.asarray(motors[1], dtype=np.float64).copy()
    m, n = slow.shape
    # sort keys go through float32 so the frame order stays bit-compatible
    # with the legacy loader (which stored motors as float32); only the
    # *stored* motors were upgraded to float64
    fast_key = fast.astype(np.float32)
    src = np.arange(m * n, dtype=np.int64).reshape(m, n)
    for r in range(m):
        order = np.argsort(fast_key[r], kind="stable")
        src[r] = src[r, order]
        fast[r] = fast[r, order]
        slow[r] = slow[r, order]
    row_order = np.argsort(slow[:, 0].astype(np.float32), kind="stable")
    perm = src[row_order].reshape(-1)
    motors_sorted = np.stack([slow[row_order], fast[row_order]])
    return perm, motors_sorted


def _value_perm(motors):
    """Legacy lexicographic ``(motor1, motor2)`` value sort as a permutation.

    Args:
        motors: (2, m, n) float motor values in acquisition order.

    Returns:
        tuple: perm (m*n,) int64, motors (2, m, n) float64 sorted.
    """
    shape = np.asarray(motors[0]).shape
    m1 = np.asarray(motors[0], dtype=np.float64).reshape(-1)
    m2 = np.asarray(motors[1], dtype=np.float64).reshape(-1)
    # float32 sort keys: bit-compatible frame order with the legacy loader
    # (see _snake_perm); stored motors stay float64
    s = np.empty(m1.size, dtype=[("m1", "f8"), ("m2", "f8")])
    s["m1"], s["m2"] = m1.astype(np.float32), m2.astype(np.float32)
    perm = np.argsort(s, order=["m1", "m2"]).astype(np.int64)
    motors_sorted = np.stack([m1[perm].reshape(shape), m2[perm].reshape(shape)])
    return perm, motors_sorted


# --------------------------------------------------------------------------- #
# placement read: frames go straight into the final detector-first layout
# --------------------------------------------------------------------------- #


def _roi_slices(roi):
    if roi:
        r1, r2, c1, c2 = roi
        return slice(r1, r2), slice(c1, c2)
    return slice(None), slice(None)


def _roi_extent(dset_shape, roi):
    rsel, csel = _roi_slices(roi)
    rows = len(range(*rsel.indices(dset_shape[1])))
    cols = len(range(*csel.indices(dset_shape[2])))
    return rows, cols


def _frames_per_chunk(n_frames, frame_bytes, row_len=None):
    """Frames per read slab: ~_SLAB_BYTES, at most ~1/8 of the stack (so peak
    memory stays close to the final array), aligned to whole fast-axis rows."""
    total = n_frames * frame_bytes
    target = min(_SLAB_BYTES, max(frame_bytes, total // 8))
    k = max(1, int(target // frame_bytes))
    if row_len:
        k = max(row_len, (k // row_len) * row_len)
    return min(k, n_frames)


def _dest_chunks(n_frames, frames_per_chunk):
    return [
        (lo, min(lo + frames_per_chunk, n_frames))
        for lo in range(0, n_frames, frames_per_chunk)
    ]


def _place_frames(dset, out_flat, perm, roi, lo, hi):
    """Read the source frames for destination range [lo, hi) and place them
    detector-first: ``out_flat[a, b, k] = frame(perm[k])`` (``frame(k)`` when
    perm is None). The re-sort is folded into the placement — one strided
    write, no global transpose pass, no separate resort pass.

    Reads one contiguous frame span when the permutation is block-local (snake
    rows, jittered sweeps — the common case), else an increasing h5py point
    selection.
    """
    rsel, csel = _roi_slices(roi)
    if perm is None:
        slab = dset[lo:hi, rsel, csel]
    else:
        src = np.asarray(perm[lo:hi])
        smin = int(src.min())
        smax = int(src.max()) + 1
        if smax - smin == hi - lo and np.array_equal(
            src, np.arange(smin, smax, dtype=src.dtype)
        ):
            slab = dset[smin:smax, rsel, csel]
        elif smax - smin <= 4 * (hi - lo):
            slab = dset[smin:smax, rsel, csel]
            slab = slab[src - smin]
        else:
            order = np.argsort(src, kind="stable")
            slab_sorted = dset[src[order], rsel, csel]  # h5py wants increasing
            slab = np.empty_like(slab_sorted)
            slab[order] = slab_sorted
    out_flat[..., lo:hi] = np.moveaxis(slab, 0, -1)


def _resolve_workers(n_workers, stack_bytes, n_frames):
    """Number of read processes; 0 means the serial in-process path.

    ``None`` auto-enables above _MP_MIN_BYTES (override with the
    STARLING_MP_MIN_BYTES env var); an explicit value always wins.
    """
    if n_workers is not None:
        n = int(n_workers)
        return 0 if n < 2 else min(n, n_frames)
    if stack_bytes < _MP_MIN_BYTES:
        return 0
    n = min(_MP_MAX_AUTO_WORKERS, os.cpu_count() or 1, n_frames)
    return 0 if n < 2 else n


class Reader:
    """Base reader. Subclass and implement __call__ for custom acquisition
    schemes; anything returning (data, motors) in the standard layout plugs
    into starling.DataSet.

    Args:
        abs_path_to_h5_file (str): BLISS master file path.
        n_workers: read-process count. None auto-enables multiprocess reads
            for large scans; 0/1 forces the serial path; >= 2 forces that many
            worker processes. Any parallel-read failure falls back to the
            serial path with a warning.
    """

    def __init__(self, abs_path_to_h5_file, n_workers=None):
        self.abs_path_to_h5_file = abs_path_to_h5_file
        self.config = ID03(abs_path_to_h5_file)
        self.scan_params = None
        self.sensors = None
        self.n_workers = n_workers
        # populated by readers that detect rastering (report-only); notebooks
        # can read ``dset.reader.acquisition_mode`` after a load.
        self.acquisition_mode = None

    def fetch(self, key):
        """Read an arbitrary h5 path (exotic motors, extra metadata)."""
        with h5py.File(self.abs_path_to_h5_file) as h5file:
            return h5file[key][...]

    def _read_stack(self, h5f, scan_id, roi):
        """Legacy helper (kept for custom subclasses): read the image stack,
        reshaped to (*scan_shape, rows, cols) and then transposed to
        detector-first layout. The built-in readers use :meth:`_read_placed`,
        which avoids this transpose copy."""
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

    # ---------------- shared metadata/read machinery ---------------- #

    def _read_motors(self, h5f, scan_id, scan_params):
        """Per-frame motor values, (n_motors, *scan_shape) float64."""
        scan_shape = tuple(int(s) for s in scan_params["scan_shape"])
        return np.array(
            [
                np.asarray(h5f[scan_id][mn][...], dtype=np.float64).reshape(scan_shape)
                for mn in scan_params["motor_names"]
            ]
        )

    def _prepare(self, h5f, scan_id):
        """Resolve everything needed to read a scan, from one open handle.

        Returns:
            tuple: (scan_params, sensors, perm, motors, row_len) where perm is
            the frame placement order (None = acquisition order), motors the
            sorted motor grid and row_len the fast-axis length (or None).
        """
        raise NotImplementedError

    def _read_placed(self, h5f, scan_id, scan_params, roi, perm, out=None,
                     row_len=None):
        """Serial placement read into ``out`` (allocated here unless given —
        the stacked loader passes a strided view of the scan cube)."""
        dset = h5f[scan_id][scan_params["data_name"]]
        n_frames = dset.shape[0]
        rows, cols = _roi_extent(dset.shape, roi)
        scan_shape = tuple(int(s) for s in scan_params["scan_shape"])
        if out is None:
            out = np.empty((rows, cols, *scan_shape), dtype=dset.dtype)
        out_flat = out.reshape(rows, cols, n_frames)
        if not np.may_share_memory(out_flat, out):  # must be a view, never a copy
            raise ValueError("output array is not reshapeable to (rows, cols, -1)")
        fpc = _frames_per_chunk(n_frames, rows * cols * dset.dtype.itemsize, row_len)
        for lo, hi in _dest_chunks(n_frames, fpc):
            _place_frames(dset, out_flat, perm, roi, lo, hi)
        return out

    def _dispatch_read(self, h5f, scan_id, scan_params, roi, perm, row_len=None):
        """Read via the multiprocess engine when enabled, else serially.

        Any parallel-read failure (shared-memory setup, worker crash) warns
        and falls back to the serial path.
        """
        dset = h5f[scan_id][scan_params["data_name"]]
        n_frames = dset.shape[0]
        rows, cols = _roi_extent(dset.shape, roi)
        stack_bytes = n_frames * rows * cols * dset.dtype.itemsize
        n_workers = _resolve_workers(self.n_workers, stack_bytes, n_frames)
        if n_workers >= 2:
            from . import _mpread

            scan_shape = tuple(int(s) for s in scan_params["scan_shape"])
            try:
                return _mpread.read_scan_shm(
                    self.abs_path_to_h5_file,
                    scan_id,
                    scan_params["data_name"],
                    (rows, cols, *scan_shape),
                    dset.dtype,
                    roi,
                    perm,
                    n_workers,
                    row_len=row_len,
                )
            except Exception as exc:
                warnings.warn(
                    f"scan {scan_id}: parallel read failed ({exc!r}); falling "
                    f"back to the serial reader.",
                    stacklevel=3,
                )
        return self._read_placed(
            h5f, scan_id, scan_params, roi, perm, row_len=row_len
        )

    def __call__(self, scan_id, roi=None):
        raise NotImplementedError


class MosaScan(Reader):
    """2D scan (e.g. fscan2d chi x mu): data (a, b, m, n), motors (2, m, n)."""

    def _prepare(self, h5f, scan_id):
        scan_params, sensors = self.config(scan_id, h5f=h5f)
        motors = self._read_motors(h5f, scan_id, scan_params)
        # report-only: how the scan was rastered (metadata or inferred from the
        # acquisition-order motors). Never affects the de-zigzag sort below.
        self.acquisition_mode = detect_acquisition_mode(h5f, scan_id, motors=motors)
        command = scan_params["scan_command"].split()[0]
        if command == "fscan2d":
            # acquisition order: axis -2 = slow (outer) levels, axis -1 = fast
            # (inner) sweep. Jitter-immune (rows come from acquisition order).
            perm, motors = _snake_perm(motors)
        else:
            # arbitrary 2-D acquisition (e.g. amesh): fall back to a lexicographic
            # value sort, which does not assume acquisition order.
            perm, motors = _value_perm(motors)
        row_len = int(scan_params["scan_shape"][-1])
        return scan_params, sensors, perm, motors, row_len

    def __call__(self, scan_id, roi=None):
        with h5py.File(self.abs_path_to_h5_file, "r") as h5f:
            scan_params, sensors, perm, motors, row_len = self._prepare(h5f, scan_id)
            self.scan_params, self.sensors = scan_params, sensors
            data = self._dispatch_read(
                h5f, scan_id, scan_params, roi, perm, row_len=row_len
            )
        _warn_if_not_separable(motors, scan_id)
        return _ascontiguousarrays(data, motors)

    @staticmethod
    def _resort_snake_by_acquisition(data, motors):
        """Sort each slow row onto a monotonic fast grid, then rows by the slow
        motor — de-zigzagging a snake scan without sorting the slow values.

        In-memory variant of the permutation the reader applies at placement
        time (kept for diagnostics and tests). Does not mutate its inputs.
        """
        perm, motors_sorted = _snake_perm(motors)
        a, b, m, n = data.shape
        data = data.reshape(a, b, m * n)[..., perm].reshape(a, b, m, n)
        return data, motors_sorted

    @staticmethod
    def _resort_by_value(data, motors):
        """Legacy lexicographic ``(motor1, motor2)`` value sort."""
        perm, motors_sorted = _value_perm(motors)
        a, b, m, n = data.shape
        data = data.reshape(a, b, m * n)[..., perm].reshape(a, b, m, n)
        return data, motors_sorted


class RockingScan(Reader):
    """1D scan (e.g. fscan mu): data (a, b, m), motors (1, m)."""

    def _prepare(self, h5f, scan_id):
        scan_params, sensors = self.config(scan_id, h5f=h5f)
        motors = self._read_motors(h5f, scan_id, scan_params)
        # float32 sort key: bit-compatible frame order with the legacy loader
        perm = np.argsort(motors[0].reshape(-1).astype(np.float32)).astype(np.int64)
        motors = motors[:, perm]
        return scan_params, sensors, perm, motors, None

    def __call__(self, scan_id, roi=None):
        with h5py.File(self.abs_path_to_h5_file, "r") as h5f:
            scan_params, sensors, perm, motors, _ = self._prepare(h5f, scan_id)
            self.scan_params, self.sensors = scan_params, sensors
            data = self._dispatch_read(h5f, scan_id, scan_params, roi, perm)
        return _ascontiguousarrays(data, motors)


class Darks(Reader):
    """Motorless image series (loopscan darks): data (a, b, m), empty motors."""

    def _prepare(self, h5f, scan_id):
        scan_params, sensors = self.config(scan_id, h5f=h5f)
        return scan_params, sensors, None, np.array([], dtype=np.float64), None

    def __call__(self, scan_id, roi=None):
        with h5py.File(self.abs_path_to_h5_file, "r") as h5f:
            scan_params, sensors, perm, motors, _ = self._prepare(h5f, scan_id)
            self.scan_params, self.sensors = scan_params, sensors
            data = self._dispatch_read(h5f, scan_id, scan_params, roi, perm)
        return _ascontiguousarrays(data, motors)
