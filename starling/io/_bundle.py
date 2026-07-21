"""Portable per-scan analysis bundle: ONE self-describing HDF5 per scan.

The use case: analysis runs on the ESRF cluster against the raw BLISS tree,
but further work happens locally without the raw data. :func:`save_bundle`
writes everything a later session needs into a single file — the ROI-cropped,
background-subtracted data cube (Bitshuffle/LZ4), the motor grid, masks, fit
results, a z-sum preview and full provenance (which master file / scans / ROI /
device produced it) — and :func:`load_bundle` reads it back on a machine with
only starling installed.

Layout (format version 1)::

    /                       root attrs: provenance + starling/format versions
    /data/cube              dense uint16 cube (ny, nx, *motor_shape)  [dense mode]
    /data/rocking_curves    (n_px, *motor_shape) uint16               [masked_only]
    /data/pixel_indices     (n_px, 2) int64 (row, col)                [masked_only]
    /data/raw_cube          (ny, nx, *motor_shape) uint16 ROI-cropped RAW data:
                            post-crop, PRE background-subtraction/hot-pixel/
                            threshold (optional; save_raw_crop=True)
    /data/zsum              (ny, nx) float64 z-sum preview (always)
    /motors                 (n_motor, *motor_shape) float64
    /masks/<name>           bool/int arrays, dtypes preserved
    /results/<name>/<field> result dataclass fields, verbatim dtypes;
                            group attr ``result_kind`` drives reconstruction

Results reuse the ``result_kind``/field-dict machinery from
:mod:`starling.io._nexus` (``_RAW_FIELDS`` + each dataclass's
``to_dict``/``from_dict``), so a reloaded result object is field-for-field
identical to the original.
"""

import datetime
import json
import os
from dataclasses import dataclass, field
from typing import Optional

import h5py
import hdf5plugin
import numpy as np

from ._output import _jsonable
from ._nexus import _RAW_FIELDS, _decode, _result_kind, _version
from ..properties import (
    Gauss1DResult,
    GaussNDResult,
    GaussNDTwoResult,
    MomentResult,
    PseudoVoigtResult,
)

__all__ = ["save_bundle", "load_bundle", "Bundle"]

FORMAT_VERSION = 1

_RESULT_CLASSES = {
    "gauss1d": Gauss1DResult,
    "gaussND": GaussNDResult,
    "gaussND_two": GaussNDTwoResult,
    "pseudovoigt": PseudoVoigtResult,
    "moments": MomentResult,
}

# root-attr bookkeeping: which keys were JSON-encoded (so a plain string that
# happens to parse as JSON is never silently retyped on read).
_JSON_KEYS_ATTR = "starling_bundle_json_keys"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _compression_kw(compression):
    """Map the ``compression`` argument to ``create_dataset`` keyword args."""
    if compression is None:
        return {}
    if compression == "bitshuffle":
        return {"compression": hdf5plugin.Bitshuffle()}
    if compression == "gzip":
        return {"compression": "gzip", "shuffle": True}
    raise ValueError(
        f"compression must be 'bitshuffle', 'gzip' or None, got {compression!r}"
    )


def _write_array(group, name, arr, comp_kw):
    """Write an array verbatim (dtype preserved); compress only >= 2-D arrays."""
    arr = np.asarray(arr)
    if arr.ndim >= 2:
        return group.create_dataset(name, data=arr, **comp_kw)
    return group.create_dataset(name, data=arr)


def _dataset_provenance(dset):
    """Everything the DataSet retains about where its cube came from."""
    prov = {}
    prov["data_source"] = getattr(dset, "h5file", None)
    prov["roi"] = getattr(dset, "roi", None)
    device = getattr(dset, "device", None)
    prov["device"] = None if device is None else str(device)
    prov["scan_motor"] = getattr(dset, "scan_motor", None)
    prov["scan_id"] = getattr(dset, "scan_id", None)
    try:
        sp = dset.scan_params
    except Exception:
        sp = None
    if sp:
        if sp.get("scan_id") is not None:
            prov["scan_id"] = sp["scan_id"]
        for key in ("scan_command", "motor_names", "scan_shape",
                    "integrated_motors"):
            if sp.get(key) is not None:
                prov[key] = sp[key]
    return prov


def _write_attrs(f, attrs):
    """Root attrs: scalars verbatim, everything else JSON-encoded + recorded."""
    json_keys = []
    for k, v in attrs.items():
        if v is None:
            continue
        if isinstance(v, (str, bytes, int, float, bool,
                          np.integer, np.floating, np.bool_)):
            f.attrs[k] = v
        else:
            f.attrs[k] = json.dumps(_jsonable(v))
            json_keys.append(k)
    f.attrs[_JSON_KEYS_ATTR] = json.dumps(json_keys)


def _read_attrs(f):
    json_keys = set()
    if _JSON_KEYS_ATTR in f.attrs:
        json_keys = set(json.loads(_decode(f.attrs[_JSON_KEYS_ATTR])))
    out = {}
    for k, v in f.attrs.items():
        if k == _JSON_KEYS_ATTR:
            continue
        v = _decode(v)
        if k in json_keys and isinstance(v, str):
            v = json.loads(v)
        out[k] = v
    return out


def _write_result(parent, name, result, comp_kw):
    """One result -> a group of verbatim field datasets + a ``result_kind`` attr."""
    grp = parent.create_group(name)
    if isinstance(result, np.ndarray):
        grp.attrs["result_kind"] = "array"
        _write_array(grp, "data", result, comp_kw)
        return
    kind = _result_kind(result)  # raises KeyError for unsupported types
    grp.attrs["result_kind"] = kind
    if kind == "gauss1d_sweep":
        if not result:
            raise ValueError(f"result {name!r}: a sweep list must be non-empty")
        if not all(isinstance(r, Gauss1DResult) for r in result):
            raise TypeError(
                f"result {name!r}: a result list must contain only "
                f"Gauss1DResult objects"
            )
        grp.attrs["n_layer"] = len(result)
        for k in _RAW_FIELDS["gauss1d_sweep"]:
            stacked = np.stack([np.asarray(getattr(r, k)) for r in result],
                               axis=-1)
            _write_array(grp, k, stacked, comp_kw)
        return
    if kind == "moments":
        fields = result.to_dict()  # mean, covariance, [skew], [kurtosis]
    else:
        fields = {k: getattr(result, k) for k in _RAW_FIELDS[kind]}
    for k, v in fields.items():
        if v is None:
            continue
        _write_array(grp, k, v, comp_kw)


def _read_result(grp):
    kind = _decode(grp.attrs["result_kind"])
    raw = {k: v[()] for k, v in grp.items()}
    if kind == "array":
        return raw["data"]
    if kind == "gauss1d_sweep":
        n_layer = int(grp.attrs["n_layer"])
        fields = _RAW_FIELDS["gauss1d_sweep"]
        return [
            Gauss1DResult.from_dict({k: raw[k][..., i] for k in fields})
            for i in range(n_layer)
        ]
    cls = _RESULT_CLASSES.get(kind)
    if cls is None:
        return raw  # forward-compat: unknown kind -> plain field dict
    return cls.from_dict(raw)


def _read_raw_crop(dset, expected_shape):
    """Re-read the source h5 at ``dset.final_roi()`` to recover RAW pixel values.

    The preprocessing pipeline mutates ``dset.data`` in place, so the only way
    to get raw (pre-background/hot-pixel/threshold) values at save time is a
    fresh read from the source file. Returns ``(raw_cube, roi)`` where
    ``raw_cube`` matches ``dset.data``'s shape and ``roi`` is the absolute
    detector ROI used (``None`` for the full detector).
    """
    from ._reader import Darks, MosaScan, RockingScan

    reader = dset.reader
    if type(reader) not in (MosaScan, RockingScan, Darks):
        raise ValueError(
            "save_raw_crop requires a full built-in scan load "
            "(MosaScan/RockingScan/Darks); this dataset was loaded with "
            f"{type(reader).__name__}, which cannot be re-read raw."
        )

    roi = dset.final_roi()
    # A fresh reader of the same type: calling the loaded reader again would
    # overwrite the scan_params the DataSet still relies on.
    fresh = type(reader)(dset.h5file, n_workers=getattr(reader, "n_workers", None))

    scan_id = dset.scan_id
    if isinstance(scan_id, str):
        raw, _ = fresh(scan_id, roi)
    elif isinstance(scan_id, list):
        # The loaded cube's last axis is sorted by the stack motor. load_scan
        # leaves dset.scan_id as the ORIGINAL (unsorted) arg; the sorted order
        # lives in scan_params["scan_id"] (set by _load_stacked_scans).
        try:
            ids = dset.scan_params.get("scan_id")
        except Exception:
            ids = None
        if not isinstance(ids, (list, tuple)):
            ids = scan_id
        blocks = [fresh(sid, roi)[0] for sid in ids]
        raw = np.stack(blocks, axis=-1)
    else:
        raise ValueError(
            "save_raw_crop requires a full built-in scan load; dset.scan_id "
            f"is {type(scan_id).__name__}, expected str or list."
        )

    raw = np.asarray(raw)
    if raw.shape != tuple(expected_shape):
        raise ValueError(
            f"raw crop shape {raw.shape} != data shape {tuple(expected_shape)}; "
            "scan ordering or ROI composition drifted."
        )
    return raw, roi


# --------------------------------------------------------------------------- #
# writer
# --------------------------------------------------------------------------- #


def save_bundle(path, dset, results=None, masks=None, provenance=None,
                masked_only=False, mask=None, compression="bitshuffle",
                save_raw_crop=False):
    """Write a self-contained analysis bundle for one scan to ``path``.

    Args:
        path: output ``.h5`` path (truncated/overwritten). Refuses to write
            inside an ESRF ``RAW_DATA`` directory.
        dset: a :class:`~starling.DataSet` with data loaded. Its cube, motors
            and retained provenance (source h5 path, scan id(s), scan motor,
            ROI, device, scan command/shape) are stored.
        results: optional dict of ``name -> result``, where each result is a
            :class:`~starling.properties.Gauss1DResult`,
            :class:`~starling.properties.GaussNDResult`,
            :class:`~starling.properties.GaussNDTwoResult`,
            :class:`~starling.properties.MomentResult`,
            :class:`~starling.properties.PseudoVoigtResult`, a
            ``list[Gauss1DResult]`` (strain sweep, stored stacked on a trailing
            layer axis) or a plain ``numpy.ndarray``.
        masks: optional dict of ``name -> array`` (bool/int; e.g. ``sig_mask``,
            ``ok_mask``, ``grain``, ``fit_mode``). Dtypes are preserved.
        provenance: optional dict of extra root attributes (dataset name,
            pixel size, notes, timestamps, ...). Non-scalars are JSON-encoded.
            starling's version, a bundle format version and a creation
            timestamp are always written.
        masked_only: if True, store only the rocking curves of pixels where
            ``mask`` is True — as ``(n_px, *motor_shape)`` plus the ``(n_px, 2)``
            (row, col) pixel indices — instead of the dense cube. Everything
            else (z-sum, motors, masks, results, attrs) is identical.
        mask: the (ny, nx) bool pixel selector for ``masked_only``; defaults to
            ``masks["sig_mask"]`` when present.
        compression: ``"bitshuffle"`` (default, Bitshuffle/LZ4), ``"gzip"`` or
            ``None``; applied to the cube/curves and every >= 2-D array.
        save_raw_crop: if True, also store an ROI-cropped RAW cube at
            ``/data/raw_cube`` — the source pixel values *before* any
            background subtraction, hot-pixel removal or thresholding, cropped
            to ``dset.final_roi()`` (the composed load-time + auto_roi ROI).
            Because preprocessing mutates ``dset.data`` in place, the raw values
            are recovered by re-reading the source h5, so this costs one extra
            read pass over the source ROI. Requires a full built-in scan load
            (MosaScan/RockingScan/Darks); partial or custom-reader loads raise
            ``ValueError``. The ROI used is stored as ``raw_cube.attrs["roi"]``.

    Returns:
        str: ``path``.
    """
    parts = os.path.normpath(str(path)).split(os.sep)
    if "RAW_DATA" in parts:
        raise PermissionError(f"Write in RAW_DATA dir of ESRF is not allowed: {path}")
    data = np.asarray(dset.data)
    if data.ndim < 3:
        raise ValueError(f"dset.data must be (ny, nx, *motors), got {data.shape}")
    ny, nx = data.shape[:2]
    motor_shape = data.shape[2:]
    comp_kw = _compression_kw(compression)

    if masked_only:
        if mask is None and masks is not None:
            mask = masks.get("sig_mask")
        if mask is None:
            raise ValueError(
                "masked_only=True needs a mask (pass mask=... or include "
                "masks['sig_mask'])"
            )
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (ny, nx):
            raise ValueError(
                f"mask shape {mask.shape} != detector shape {(ny, nx)}"
            )

    raw_cube = None
    raw_roi = None
    if save_raw_crop:
        raw_cube, raw_roi = _read_raw_crop(dset, data.shape)

    attrs = _dataset_provenance(dset)
    attrs.update(provenance or {})
    attrs["detector_shape"] = [int(ny), int(nx)]
    attrs["motor_shape"] = [int(s) for s in motor_shape]
    attrs["masked_only"] = bool(masked_only)
    if save_raw_crop:
        attrs["raw_crop_saved"] = True
        attrs["raw_crop_roi"] = list(raw_roi) if raw_roi is not None else "full"
    attrs["starling_version"] = _version()
    attrs["bundle_format_version"] = FORMAT_VERSION
    attrs.setdefault(
        "created", datetime.datetime.now().replace(microsecond=0).isoformat()
    )

    zsum = data.reshape(ny, nx, -1).sum(axis=-1, dtype=np.float64)

    with h5py.File(path, "w") as f:
        _write_attrs(f, attrs)

        dgrp = f.create_group("data")
        if masked_only:
            idx = np.argwhere(mask)  # (n_px, 2) (row, col)
            curves = data[mask]      # (n_px, *motor_shape)
            n_px = curves.shape[0]
            chunks = None
            if n_px > 0:
                chunks = (min(n_px, 1024), *motor_shape)
            dgrp.create_dataset(
                "rocking_curves", data=curves, chunks=chunks,
                **(comp_kw if n_px > 0 else {}),
            )
            dgrp.create_dataset("pixel_indices", data=idx.astype(np.int64))
        else:
            # chunk = one detector tile per motor point: compresses well
            # (frames are sparse) and slices well spatially.
            chunks = (min(ny, 256), min(nx, 256)) + (1,) * len(motor_shape)
            dgrp.create_dataset("cube", data=data, chunks=chunks, **comp_kw)
        _write_array(dgrp, "zsum", zsum, comp_kw)

        if raw_cube is not None:
            # mirror /data/cube's chunking + compression
            chunks = (min(ny, 256), min(nx, 256)) + (1,) * len(motor_shape)
            rds = dgrp.create_dataset(
                "raw_cube", data=raw_cube, chunks=chunks, **comp_kw
            )
            rds.attrs["roi"] = (
                list(raw_roi) if raw_roi is not None else "full"
            )

        if dset.motors is not None:
            _write_array(f, "motors", np.asarray(dset.motors, np.float64),
                         comp_kw)

        if masks:
            mgrp = f.create_group("masks")
            for name, m in masks.items():
                _write_array(mgrp, name, m, comp_kw)

        if results:
            rgrp = f.create_group("results")
            for name, res in results.items():
                _write_result(rgrp, name, res, comp_kw)
    return path


# --------------------------------------------------------------------------- #
# loader
# --------------------------------------------------------------------------- #


@dataclass
class Bundle:
    """A loaded analysis bundle (see :func:`load_bundle`).

    Exactly one of ``data`` (dense) or ``sparse_data`` + ``pixel_indices``
    (masked_only) is populated; :meth:`dense` always returns a full cube.
    """

    motors: Optional[np.ndarray]
    zsum: np.ndarray
    attrs: dict
    masks: dict = field(default_factory=dict)
    results: dict = field(default_factory=dict)
    data: Optional[np.ndarray] = None
    sparse_data: Optional[np.ndarray] = None
    pixel_indices: Optional[np.ndarray] = None
    raw_data: Optional[np.ndarray] = None

    @property
    def detector_shape(self):
        return tuple(self.attrs["detector_shape"])

    @property
    def motor_shape(self):
        return tuple(self.attrs["motor_shape"])

    def dense(self, fill=0):
        """The full (ny, nx, *motor_shape) cube.

        Dense bundles return the stored cube. masked_only bundles scatter the
        stored rocking curves into a ``fill``-valued cube — ``fill=0`` keeps
        the stored (uint16) dtype; any other fill (e.g. ``np.nan``) returns
        float32.
        """
        if self.data is not None:
            return self.data
        ny, nx = self.detector_shape
        dtype = self.sparse_data.dtype if fill == 0 else np.float32
        out = np.full((ny, nx, *self.motor_shape), fill, dtype=dtype)
        rows, cols = self.pixel_indices[:, 0], self.pixel_indices[:, 1]
        out[rows, cols] = self.sparse_data
        return out


def load_bundle(path):
    """Read a :func:`save_bundle` file back into a :class:`Bundle`.

    Works standalone (no raw data / original filesystem needed): everything is
    materialised into memory. ``bundle.results`` holds reconstructed result
    objects (or a ``list[Gauss1DResult]`` for a sweep, a plain array for kind
    ``"array"``); ``bundle.masks`` the mask arrays with their stored dtypes;
    ``bundle.attrs`` the provenance dict. For a masked_only bundle use
    ``bundle.sparse_data``/``bundle.pixel_indices`` directly or
    ``bundle.dense()`` to rebuild the full cube. ``bundle.raw_data`` holds the
    ROI-cropped RAW cube when the bundle was written with ``save_raw_crop=True``
    (else ``None``).
    """
    with h5py.File(path, "r") as f:
        attrs = _read_attrs(f)
        dgrp = f["data"]
        data = dgrp["cube"][()] if "cube" in dgrp else None
        sparse = dgrp["rocking_curves"][()] if "rocking_curves" in dgrp else None
        idx = dgrp["pixel_indices"][()] if "pixel_indices" in dgrp else None
        raw = dgrp["raw_cube"][()] if "raw_cube" in dgrp else None
        zsum = dgrp["zsum"][()]
        motors = f["motors"][()] if "motors" in f else None
        masks = {k: v[()] for k, v in f["masks"].items()} if "masks" in f else {}
        results = {}
        if "results" in f:
            for name, grp in f["results"].items():
                results[name] = _read_result(grp)
    return Bundle(
        motors=motors, zsum=zsum, attrs=attrs, masks=masks, results=results,
        data=data, sparse_data=sparse, pixel_indices=idx, raw_data=raw,
    )
