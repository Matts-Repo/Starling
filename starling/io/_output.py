"""HDF5 result writer and time-series aggregation."""

import datetime
import json

import h5py
import numpy as np


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    return obj


def save_maps(path, maps, scan_params=None, extra_attrs=None):
    """Write a dict of result maps to an HDF5 file.

    Nested dicts become groups; arrays become compressed datasets. Root attrs
    record provenance (scan params, versions, timestamp).

    Args:
        path (str): output .h5 path (overwritten).
        maps (dict): e.g. {"mean": ..., "gauss1d": {"params": ...}}.
        scan_params (dict): scan metadata stored as a JSON attr.
        extra_attrs (dict): extra root attributes (e.g. recipe_hash, device).
    """
    import torch

    import starling

    with h5py.File(path, "w") as f:
        grp = f.create_group("maps")
        _write_group(grp, maps)
        if scan_params is not None:
            f.attrs["scan_params"] = json.dumps(_jsonable(scan_params))
        f.attrs["starling_version"] = starling.__version__
        f.attrs["torch_version"] = torch.__version__
        f.attrs["timestamp"] = datetime.datetime.now().isoformat()
        for k, v in (extra_attrs or {}).items():
            f.attrs[k] = v


def _write_group(grp, d):
    for k, v in d.items():
        if isinstance(v, dict):
            _write_group(grp.create_group(k), v)
        else:
            arr = np.asarray(v)
            if arr.ndim >= 2:
                grp.create_dataset(k, data=arr, compression="gzip", shuffle=True)
            else:
                grp.create_dataset(k, data=arr)


def load_maps(path):
    """Read a save_maps file back into nested dicts of arrays + attrs."""
    with h5py.File(path, "r") as f:
        maps = _read_group(f["maps"])
        attrs = dict(f.attrs)
    if "scan_params" in attrs:
        attrs["scan_params"] = json.loads(attrs["scan_params"])
    return maps, attrs


def _read_group(grp):
    out = {}
    for k, v in grp.items():
        out[k] = _read_group(v) if isinstance(v, h5py.Group) else v[()]
    return out
