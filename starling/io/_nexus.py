"""Standards-compliant NeXus exporter + loader for starling result objects.

starling's native serialisation (``DataSet.save_maps`` -> a flat ``maps`` group;
``Result.to_h5`` -> root datasets + a ``result_kind`` attr) is compact but opaque
to the wider DFXM tooling. darfix instead writes a ``NXentry/NXprocess/NXdata``
tree with ``@signal``/``@axes``/``@interpretation``/``@default`` attributes so the
file opens directly in *silx view* / *PyMca* and chains in *ewoks*, with
human-readable, darfix-exact map names ("Center of mass", "FWHM", "Mosaicity",
"Color Key", "Kernel Average Misorientation", ...).

This module makes starling output a drop-in for that tooling **and** adds a
round-trippable raw block (an ``NXcollection`` of the verbatim, unmasked,
float64 dataclass fields) so :func:`load_nexus` rebuilds the exact result object
-- something darfix does not do.

Design constraints (kept deliberately): plain ``h5py`` only (no new dependency;
``hdf5plugin`` is already a hard dep), all NeXus attributes set manually, and no
``darling`` import (``tests/test_standalone.py``). silx is used *only in tests*.
"""

import datetime
import json
import warnings

import h5py
import hdf5plugin
import numpy as np

from ._output import _jsonable
from ..properties import (
    Gauss1DResult,
    GaussNDResult,
    GaussNDTwoResult,
    MomentResult,
    PseudoVoigtResult,
)
from ..properties._maps import FWHM_FACTOR
from ..transforms import kam as _kam

__all__ = ["save_nexus", "load_nexus", "save_dataset_nexus"]

# darfix-exact display-map names (so the files are a drop-in for silx/PyMca).
_COM = "Center of mass"
_FWHM = "FWHM"
_SKEW = "Skewness"
_KURT = "Kurtosis"
_MOSAICITY = "Mosaicity"
_COLOR_KEY = "Color Key"
_KAM = "Kernel Average Misorientation"
_MOSAICITY_SPREAD = "Mosaicity spread"  # starling extra: scalar RMS orientation spread
_FIT_STATUS = "Fit status"  # starling extra: per-pixel fit-quality category map
_FIT_STATUS_ENCODING = "0=no_signal 1=ok 2=edge_clipped 3=failed"
# starling extras for the two-peak N-D fit (darfix has no multi-peak maps)
_N_PEAKS = "Number of peaks"
_PEAK_SEP = "Peak separation"
_PEAK_SEP_MAHA = "Peak separation (Mahalanobis)"

# result_kind <-> dataclass field order for the round-trip raw block.
_RAW_FIELDS = {
    "gauss1d": ("A", "sigma", "mu", "k", "m", "success"),
    "gaussND": ("A", "mu", "cov", "c", "success"),
    "gaussND_two": ("A1", "mu1", "cov1", "A2", "mu2", "cov2", "c",
                    "n_peaks", "bic1", "bic2", "success"),
    "pseudovoigt": ("A", "sigma", "mu", "gamma", "eta", "k", "m", "success"),
    "gauss1d_sweep": ("A", "sigma", "mu", "k", "m", "success"),
    # moments fields are dynamic (skew/kurtosis optional) -> handled inline.
}
_PROCESS = "starling_process"


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #


def _now():
    return datetime.datetime.now().replace(microsecond=0).isoformat()


def _version():
    import starling

    return starling.__version__


def _f64(arr):
    """Verbatim float64 view for the round-trip raw block (never masked)."""
    return np.asarray(arr, dtype=np.float64)


def _decode(v):
    """h5py reads string datasets back as bytes -- normalise to ``str``."""
    if isinstance(v, bytes):
        return v.decode("utf-8")
    if isinstance(v, np.ndarray):
        if v.dtype.kind == "S" or v.dtype == object:
            return np.array([_decode(x) for x in v.ravel()]).reshape(v.shape)
        return v
    return v


def _set_str(group, name, value):
    """Write a ``str`` or list-of-``str`` dataset (UTF-8)."""
    if isinstance(value, (list, tuple)):
        group.create_dataset(
            name, data=np.array([str(v) for v in value], dtype=object),
            dtype=h5py.string_dtype("utf-8"),
        )
    else:
        group.create_dataset(name, data=str(value))


def _write_dataset(group, name, arr, *, compress=True):
    arr = np.asarray(arr)
    if compress and arr.ndim >= 2:
        return group.create_dataset(name, data=arr, compression="gzip", shuffle=True)
    return group.create_dataset(name, data=arr)


def _image_axes(ny, nx, pixel_size_mm):
    """Detector image axes (y, x). Real mm if ``pixel_size_mm`` given else index."""
    if pixel_size_mm is None:
        return np.arange(ny, dtype=np.float64), np.arange(nx, dtype=np.float64), "pixel"
    if np.ndim(pixel_size_mm) == 0:  # scalar or 0-d array (np.isscalar misses 0-d)
        py = px = float(pixel_size_mm)
    else:
        flat = np.asarray(pixel_size_mm, dtype=float).ravel()
        if flat.size == 1:
            py = px = float(flat[0])
        elif flat.size >= 2:
            py, px = float(flat[0]), float(flat[1])
        else:
            raise ValueError("pixel_size_mm must be a scalar or (py, px)")
    return np.arange(ny) * py, np.arange(nx) * px, "mm"


def _write_image_nxdata(
    parent, name, signal, y, x, x_units, *,
    interpretation="image", rgba=False, rgba_axes=("y", "x"),
    extra_signal_attrs=None, layer=None,
):
    """Create one ``NXdata`` group holding a single 2-D/3-D image ``signal``.

    Every name listed in ``@axes`` is backed by a real coordinate dataset of the
    matching length (an unbacked name makes silx reject the whole NXdata, which
    would stop ``entry@default`` auto-rendering). Layouts:

    * 2-D map ``(ny, nx)``  -> ``@axes = ["y", "x"]``.
    * rgba image ``(ny, nx, 3)`` -> ``@axes = [*rgba_axes, "."]`` (the channel
      axis is last and unlabelled); silx renders via ``@interpretation =
      "rgba-image"``.
    * strain-sweep stack ``(n_layer, ny, nx)`` -> ``@axes = ["layer", "y", "x"]``
      so silx's frame slider scrubs the layer and each frame is the (ny, nx) map.
    """
    g = parent.create_group(name)
    g.attrs["NX_class"] = "NXdata"
    g.attrs["signal"] = name
    ds = _write_dataset(g, name, signal)
    ds.attrs["interpretation"] = interpretation
    if rgba:
        # (rows, cols, channel): the trailing channel axis carries no coordinate.
        g.attrs["axes"] = [rgba_axes[0], rgba_axes[1], "."]
        ds.attrs["CLASS"] = "IMAGE"
        ad = g.create_dataset(rgba_axes[0], data=_f64(y)); ad.attrs["units"] = x_units
        bd = g.create_dataset(rgba_axes[1], data=_f64(x)); bd.attrs["units"] = x_units
    elif layer is not None:
        g.attrs["axes"] = ["layer", "y", "x"]
        ld = g.create_dataset("layer", data=_f64(layer)); ld.attrs["long_name"] = "strain-sweep layer"
        yd = g.create_dataset("y", data=_f64(y)); yd.attrs["units"] = x_units
        xd = g.create_dataset("x", data=_f64(x)); xd.attrs["units"] = x_units
    else:
        g.attrs["axes"] = ["y", "x"]
        yd = g.create_dataset("y", data=_f64(y)); yd.attrs["units"] = x_units
        xd = g.create_dataset("x", data=_f64(x)); xd.attrs["units"] = x_units
    for k, v in (extra_signal_attrs or {}).items():
        ds.attrs[k] = v
    return g


def _check_separable(motors, tol_frac=0.25):
    """Warn if a >=2-motor grid is not cleanly separable after re-sorting.

    Mirrors ``starling.io._reader._warn_if_not_separable``: for each motor axis
    ``i`` the motor value should vary along ``i`` but stay ~constant across the
    other axes. A large cross-axis spread means per-axis 1-D scan vectors and
    grid-based slices cannot be trusted.
    """
    motors = np.asarray(motors)
    n = motors.shape[0]
    grid = motors.shape[1:]
    if len(grid) != n or n < 2:
        return
    problems = []
    for i in range(n):
        if grid[i] <= 1:
            continue
        idx = tuple(slice(None) if j == i else 0 for j in range(n))
        vec = np.asarray(motors[i][idx], dtype=float)
        step = float(np.median(np.abs(np.diff(vec)))) if vec.size > 1 else 0.0
        # within motors[i] (shape == grid), the motor value should stay constant
        # across every axis other than i -> measure the worst such spread.
        other_axes = tuple(j for j in range(n) if j != i)
        spread = float(np.ptp(motors[i], axis=other_axes).max())
        if step > 0 and spread > tol_frac * step:
            problems.append(
                f"axis {i} varies by {spread:.2e} across the other axes "
                f"({100 * spread / step:.0f}% of its {step:.2e} step)"
            )
    if problems:
        # user -> save_nexus -> _write_scan_group -> _check_separable -> warn
        warnings.warn(
            "motor grid is not cleanly separable after re-sorting "
            f"({'; '.join(problems)}); per-axis scan vectors and per-layer "
            "slices may be scrambled.",
            UserWarning,
            stacklevel=4,
        )


def _leaf_name(path):
    """Short motor name from an h5 leaf path (``.split("/")[-1]``).

    ID03 stores a few motors at ``instrument/<motor>/value`` (chi, phi, ...) and
    the rest at ``instrument/positioners/<motor>``; the trailing ``value``/``data``
    is the BLISS leaf, so fall back to the parent component for a meaningful name.
    """
    parts = str(path).split("/")
    leaf = parts[-1]
    if leaf in ("value", "data") and len(parts) >= 2:
        return parts[-2]
    return leaf


_RESERVED_NAMES = {"scan", "data", _PROCESS}


def _unique_names(names):
    """De-duplicate group names, avoiding collisions with reserved groups.

    Two motor paths can share a leaf (e.g. ``a/chi`` and ``b/chi``) and a leaf
    could collide with ``scan``/``data``/``starling_process``; either would make
    ``h5py.create_group`` raise. Suffix collisions with ``_1``, ``_2``, ...
    """
    seen, out = set(), []
    for nm in names:
        base = f"{nm}_motor" if nm in _RESERVED_NAMES else nm
        cand, i = base, 1
        while cand in seen:
            cand, i = f"{base}_{i}", i + 1
        seen.add(cand)
        out.append(cand)
    return out


def _short_motor_names(scan_params, n_motor):
    names = None
    if scan_params is not None:
        names = scan_params.get("motor_names")
    if names:
        short = [_leaf_name(p) for p in names]
        if len(short) < n_motor:
            short += [f"axis{i}" for i in range(len(short), n_motor)]
        return _unique_names(short[:n_motor])
    return [f"axis{i}" for i in range(n_motor)]


def _write_scan_group(entry, motors, scan_params, motor_units):
    """Write the ``scan/`` NXcollection (per-axis 1-D motor vectors + metadata).

    Returns the list of short motor-axis names (empty when there are no motors).
    """
    motors = None if motors is None else np.asarray(motors)
    has_motors = motors is not None and motors.size > 0 and motors.ndim >= 2
    if not has_motors:
        return []

    n_motor = motors.shape[0]
    _check_separable(motors)
    short = _short_motor_names(scan_params, n_motor)

    scan = entry.create_group("scan")
    scan.attrs["NX_class"] = "NXcollection"
    for i in range(n_motor):
        idx = tuple(slice(None) if j == i else 0 for j in range(n_motor))
        vec = _f64(motors[i][idx])
        d = scan.create_dataset(short[i], data=vec)
        d.attrs["units"] = motor_units
        d.attrs["long_name"] = f"{short[i]} ({motor_units})"

    if scan_params is not None:
        if scan_params.get("scan_command") is not None:
            _set_str(scan, "scan_command", scan_params["scan_command"])
        if scan_params.get("scan_shape") is not None:
            scan.create_dataset(
                "scan_shape", data=np.asarray(scan_params["scan_shape"], dtype=np.int64)
            )
        if scan_params.get("scan_id") is not None:
            _set_str(scan, "scan_id", scan_params["scan_id"])
        _set_str(scan, "motor_names", short)
    return short


def _write_process(entry, result_kind):
    proc = entry.create_group(_PROCESS)
    proc.attrs["NX_class"] = "NXprocess"
    proc.create_dataset("program", data="starling")
    proc.create_dataset("version", data=_version())
    proc.create_dataset("date", data=_now())
    proc.create_dataset("processing_order", data=np.int32(1))
    proc.create_dataset("result_kind", data=result_kind)
    return proc


def _write_raw(proc, fields, *, D, orientation_axes, motor_units, mask, pixel_size_mm,
               fit_status=None):
    """The round-trip NXcollection: verbatim float64 dataclass fields, unmasked.

    ``fit_status`` (if given) is stored verbatim as an int8 sibling dataset so
    the categorical encoding survives round-trip losslessly (the entry-level
    ``Fit status`` NXdata is a float64 display copy). ``from_dict`` selects its
    own keys, so this extra dataset is harmless on result reconstruction.
    """
    raw = proc.create_group("raw")
    raw.attrs["NX_class"] = "NXcollection"
    for k, v in fields.items():
        if v is None:
            continue
        _write_dataset(raw, k, _f64(v))
    raw.attrs["D"] = int(D)
    raw.attrs["orientation_axes"] = list(orientation_axes)
    raw.attrs["motor_units"] = motor_units
    raw.attrs["mask_applied"] = bool(mask is not None)
    if pixel_size_mm is not None:
        raw.attrs["pixel_size_mm"] = np.asarray(pixel_size_mm, dtype=np.float64)
    if mask is not None:
        raw.create_dataset("mask", data=np.asarray(mask, dtype=bool))
    if fit_status is not None:
        ds = _write_dataset(raw, "fit_status", np.asarray(fit_status, dtype=np.int8))
        ds.attrs["encoding"] = _FIT_STATUS_ENCODING
    return raw


def _write_extra_attrs(f, extra_attrs):
    if not extra_attrs:
        return
    keys, json_keys = [], []
    for k, v in extra_attrs.items():
        if isinstance(v, (str, bytes, int, float, bool,
                          np.integer, np.floating, np.bool_)):
            f.attrs[k] = v  # scalar stored verbatim
        else:
            f.attrs[k] = json.dumps(_jsonable(v))
            json_keys.append(k)
        keys.append(k)
    # record which keys were JSON-encoded so the loader only decodes those --
    # otherwise a plain string that happens to be valid JSON ("12", "true")
    # would be silently retyped on read.
    f.attrs["starling_extra_keys"] = json.dumps(keys)
    f.attrs["starling_extra_json_keys"] = json.dumps(json_keys)


def _mask_display(arr2d, keep):
    """Display copy of a 2-D map with off-grain pixels NaN'd (raw stays intact)."""
    if keep is None:
        return np.asarray(arr2d, dtype=np.float64)
    out = np.array(arr2d, dtype=np.float64, copy=True)
    out[~keep] = np.nan
    return out


def _write_fit_status(ent, fit_status, y, x, x_units):
    """Write the categorical ``Fit status`` NXdata at the entry level.

    Never NaN-masked: the whole point of the map is to distinguish the
    ``NO_SIGNAL``/``OK``/``EDGE_CLIPPED``/``FAILED`` categories (see
    :func:`starling.properties.classify_fit_status`). Stored float64 for silx
    friendliness; the verbatim int8 array lives in the raw NXcollection.
    """
    if fit_status is None:
        return
    _write_image_nxdata(
        ent, _FIT_STATUS, _f64(fit_status), y, x, x_units,
        extra_signal_attrs={"quantity": "fit_status",
                            "encoding": _FIT_STATUS_ENCODING},
    )


# --------------------------------------------------------------------------- #
# per-type assembly
# --------------------------------------------------------------------------- #


def _result_kind(result):
    if isinstance(result, (list, tuple)):
        return "gauss1d_sweep"
    return {
        "Gauss1DResult": "gauss1d",
        "GaussNDResult": "gaussND",
        "GaussNDTwoResult": "gaussND_two",
        "PseudoVoigtResult": "pseudovoigt",
        "MomentResult": "moments",
    }[result.__class__.__name__]


def _components(result, kind):
    """Return (D, centre, fwhm, success, extra) for the display maps.

    ``centre``/``fwhm`` are (ny, nx) when D == 1 else (ny, nx, D). ``extra`` holds
    optional per-axis moment maps (skew/kurtosis) keyed by display name.
    """
    extra = {}
    if kind == "gaussND":
        D = result.mu.shape[-1]
        centre, fwhm = np.asarray(result.mu), np.asarray(result.fwhm)
        success = np.asarray(result.success)
    elif kind in ("gauss1d", "pseudovoigt"):
        D = 1
        centre, fwhm = np.asarray(result.mu), np.asarray(result.fwhm)
        success = np.asarray(result.success)
    elif kind == "moments":
        mean = np.asarray(result.mean)
        cov = np.asarray(result.covariance)
        if mean.ndim == 2:  # single-motor scalar moment
            D = 1
            centre = mean
            fwhm = FWHM_FACTOR * np.sqrt(np.clip(cov, 0.0, None))
        else:
            D = mean.shape[-1]
            centre = mean
            diag = np.einsum("...ii->...i", cov)
            fwhm = FWHM_FACTOR * np.sqrt(np.clip(diag, 0.0, None))
        success = None
        if result.skew is not None:
            extra[_SKEW] = np.asarray(result.skew)
        if result.kurtosis is not None:
            extra[_KURT] = np.asarray(result.kurtosis)
    else:
        raise ValueError(f"unsupported result_kind {kind!r}")
    return D, centre, fwhm, success, extra


def _keep_mask(kind, mask, centre, success):
    """``keep`` = on-grain & valid; ``None`` when no mask was supplied."""
    if mask is None:
        return None
    keep = np.asarray(mask, dtype=bool)
    if kind == "moments":
        finite = np.isfinite(centre)
        if finite.ndim == 3:
            finite = finite.all(-1)
        return keep & finite
    return keep & (np.asarray(success) > 0.5)


# --------------------------------------------------------------------------- #
# writer
# --------------------------------------------------------------------------- #


def save_nexus(
    path, result, motors=None, scan_params=None, *, mask=None,
    orientation_axes=(0, 1), kam_size=(3, 3), pixel_size_mm=None,
    motor_units="deg", layer_values=None, entry="entry", extra_attrs=None,
    fit_status=None,
):
    """Write a starling result object to a standards-compliant NeXus file.

    Args:
        path: output ``.nxs``/``.h5`` path (truncated/overwritten).
        result: a :class:`~starling.properties.Gauss1DResult`,
            :class:`~starling.properties.GaussNDResult`,
            :class:`~starling.properties.MomentResult`,
            :class:`~starling.properties.PseudoVoigtResult`, or a
            ``list[Gauss1DResult]`` (a strain sweep -> stacked layer axis).
        motors: the ``DataSet.motors`` array ``(n_motor, *grid)``; used to write
            the ``scan/`` motor axes. ``None``/empty -> index axes, no ``scan/``.
        scan_params: the ``DataSet.scan_params`` dict (scan command, motor names,
            scan shape, scan id); stored under ``scan/``.
        mask: optional grain mask ``(ny, nx)``; off-grain pixels are NaN'd in the
            *display* maps only (the raw block stays unmasked; masked RGB -> white).
        orientation_axes: which motor components drive the orientation/mosaicity
            colour image (D >= 2 only).
        kam_size: KAM kernel window (D >= 2 only).
        pixel_size_mm: scalar or ``(py, px)`` -> real-space mm image axes; else
            the image axes are the pixel index.
        motor_units: angular units of the motor/scan axes.
        layer_values: strain-sweep layer positions (the strain-motor mu per
            layer) used as the ``layer`` axis; defaults to the layer index.
        entry: NXentry group name.
        extra_attrs: extra root attributes (non-scalars JSON-encoded).
        fit_status: optional ``(ny, nx)`` int array from
            :func:`starling.properties.classify_fit_status`
            (``0=no_signal 1=ok 2=edge_clipped 3=failed``). Written as its own
            ``Fit status`` NXdata at the entry level for every result kind, and
            stored verbatim (int8) in the raw block. Never NaN-masked -- the map
            is meant to distinguish the categories, including off-grain pixels.
            (The same categories are also persistable alongside a bundle via
            :func:`starling.io.save_bundle`'s masks.)

    Note:
        For the ``D >= 2`` orientation image this writes darfix's square colour
        stamp (``result.orientation_stamp``, via the ``colorstamps`` package)
        with a matching axed ``Color Key``; if ``colorstamps`` is not installed
        it falls back to the legacy round HSV wheel with a warning.
    """
    kind = _result_kind(result)
    if kind == "gauss1d_sweep":
        return _save_sweep_nexus(
            path, result, motors, scan_params, mask=mask, pixel_size_mm=pixel_size_mm,
            motor_units=motor_units, layer_values=layer_values, entry=entry,
            extra_attrs=extra_attrs, fit_status=fit_status,
        )
    if kind == "gaussND_two":
        return _save_two_peak_nexus(
            path, result, motors, scan_params, mask=mask, pixel_size_mm=pixel_size_mm,
            motor_units=motor_units, orientation_axes=orientation_axes, entry=entry,
            extra_attrs=extra_attrs, fit_status=fit_status,
        )

    D, centre, fwhm, success, extra = _components(result, kind)
    keep = _keep_mask(kind, mask, centre, success)
    is_fit = kind in ("gauss1d", "pseudovoigt", "gaussND")
    com_quantity = "fit_peak_center" if is_fit else "center_of_mass"
    fwhm_quantity = "fit_fwhm" if is_fit else "moment_fwhm"
    source = "fit" if is_fit else "moments"
    ny, nx = centre.shape[:2]
    y, x, x_units = _image_axes(ny, nx, pixel_size_mm)
    extra_attrs_signal = {"peak_model": "pseudovoigt"} if kind == "pseudovoigt" else {}

    with h5py.File(path, "w") as f:
        f.attrs["NX_class"] = "NXroot"
        f.attrs["default"] = entry
        ent = f.create_group(entry)
        ent.attrs["NX_class"] = "NXentry"
        ent.attrs["default"] = _MOSAICITY if D >= 2 else _COM

        proc = _write_process(ent, kind)
        if kind == "moments":
            raw_fields = result.to_dict()  # mean, covariance, [skew], [kurtosis]
        else:
            raw_fields = {k: getattr(result, k) for k in _RAW_FIELDS[kind]}
        _write_raw(
            proc, raw_fields, D=D, orientation_axes=orientation_axes,
            motor_units=motor_units, mask=mask, pixel_size_mm=pixel_size_mm,
            fit_status=fit_status,
        )
        _write_scan_group(ent, motors, scan_params, motor_units)
        _write_fit_status(ent, fit_status, y, x, x_units)

        # ---- centre / FWHM (+ moment skew/kurtosis) -------------------------
        def _write_component_set(parent, i):
            """Write COM/FWHM(/Skewness/Kurtosis) NXdata for axis ``i`` (or 2-D)."""
            comps = [
                (_COM, centre if i is None else centre[..., i],
                 {"quantity": com_quantity, "source": source, **extra_attrs_signal}),
                (_FWHM, fwhm if i is None else fwhm[..., i],
                 {"quantity": fwhm_quantity, "source": source, **extra_attrs_signal}),
            ]
            for ename, evals in extra.items():
                q = "skewness" if ename == _SKEW else "kurtosis"
                comps.append((ename, evals if i is None else evals[..., i],
                              {"quantity": q, "source": source}))
            for cname, cvals, cattrs in comps:
                _write_image_nxdata(
                    parent, cname, _mask_display(cvals, keep), y, x, x_units,
                    extra_signal_attrs=cattrs,
                )

        if D == 1:
            c2 = centre if centre.ndim == 2 else centre[..., 0]
            f2 = fwhm if fwhm.ndim == 2 else fwhm[..., 0]
            ext2 = {k: (v if v.ndim == 2 else v[..., 0]) for k, v in extra.items()}
            for cname, cvals, q in (
                (_COM, c2, com_quantity), (_FWHM, f2, fwhm_quantity),
            ):
                _write_image_nxdata(
                    ent, cname, _mask_display(cvals, keep), y, x, x_units,
                    extra_signal_attrs={"quantity": q, "source": source,
                                        **extra_attrs_signal},
                )
            for ename, evals in ext2.items():
                q = "skewness" if ename == _SKEW else "kurtosis"
                _write_image_nxdata(
                    ent, ename, _mask_display(evals, keep), y, x, x_units,
                    extra_signal_attrs={"quantity": q, "source": source},
                )
        else:
            names = _short_motor_names(scan_params, D)
            for i in range(D):
                coll = ent.create_group(names[i])
                coll.attrs["NX_class"] = "NXcollection"
                _write_component_set(coll, i)

            # ---- orientation RGB / colour key / scalar spread / KAM ---------
            # darfix-parity square colour stamp (colorstamps); the fixed per-axis
            # range spends the full colour square on the grain's own orientation
            # range. Falls back to the legacy round HSV wheel if colorstamps is
            # not installed, so a minimal install never hard-fails on save.
            try:
                rgb, key, vrange = result.orientation_stamp(
                    axes=orientation_axes, mask=keep,
                )
                (klo0, khi0), (klo1, khi1) = vrange
                key_y = np.linspace(klo0, khi0, key.shape[0])
                key_x = np.linspace(klo1, khi1, key.shape[1])
            except ImportError:
                warnings.warn(
                    "colorstamps not installed; the Mosaicity image falls back "
                    "to the legacy round HSV orientation wheel and the Color Key "
                    "uses a fixed -1..1 range.",
                    UserWarning, stacklevel=2,
                )
                _, rgb, key = result.orientation(axes=orientation_axes, as_rgb=True)
                key_y = np.linspace(-1.0, 1.0, key.shape[0])
                key_x = np.linspace(-1.0, 1.0, key.shape[1])
            rgb = np.array(rgb, dtype=np.float64, copy=True)
            if keep is not None:
                # NeXus display override: masked-off pixels are WHITE, not the
                # stamp's black (never NaN; NaN breaks silx rgba rendering).
                rgb[~keep] = 1.0
            _write_image_nxdata(
                ent, _MOSAICITY, rgb, y, x, x_units, interpretation="rgba-image",
                rgba=True, extra_signal_attrs={"source": source},
            )
            _write_image_nxdata(
                ent, _COLOR_KEY, np.asarray(key, dtype=np.float64),
                key_y, key_x, motor_units,
                interpretation="rgba-image", rgba=True, rgba_axes=("ky", "kx"),
            )

            spread = result.mosaicity(mode="scalar", axes=orientation_axes)
            _write_image_nxdata(
                ent, _MOSAICITY_SPREAD, _mask_display(spread, keep), y, x, x_units,
                extra_signal_attrs={"source": source, "units": motor_units,
                                    "long_name": "RMS orientation spread"},
            )

            kam_map = _kam(np.asarray(centre), size=kam_size)
            _write_image_nxdata(
                ent, _KAM, _mask_display(kam_map, keep), y, x, x_units,
                extra_signal_attrs={"source": source, "units": motor_units,
                                    "kernel_size": list(kam_size)},
            )

        _write_extra_attrs(f, extra_attrs)
    return path


def _save_sweep_nexus(
    path, results, motors, scan_params, *, mask, pixel_size_mm, motor_units,
    layer_values, entry, extra_attrs, fit_status=None,
):
    """Strain-sweep branch: a ``list[Gauss1DResult]`` -> stacked layer axis."""
    if not results:
        raise ValueError("a strain-sweep result list must be non-empty")
    if not all(isinstance(r, Gauss1DResult) for r in results):
        raise TypeError("a result list must contain only Gauss1DResult objects")
    n_layer = len(results)
    fields = _RAW_FIELDS["gauss1d_sweep"]
    stacked = {k: np.stack([np.asarray(getattr(r, k)) for r in results], axis=-1)
               for k in fields}
    if layer_values is None:
        layer_mu = np.arange(n_layer, dtype=np.float64)
    else:
        layer_mu = _f64(layer_values)
        if layer_mu.shape != (n_layer,):
            raise ValueError(
                f"layer_values must have length n_layer={n_layer}, got {layer_mu.shape}"
            )

    centre = stacked["mu"]                              # (ny, nx, n_layer)
    fwhm = FWHM_FACTOR * stacked["sigma"]
    ny, nx = centre.shape[:2]
    y, x, x_units = _image_axes(ny, nx, pixel_size_mm)

    keep2d = None if mask is None else np.asarray(mask, dtype=bool)

    def _mask_layers(arr):
        if keep2d is None:
            return np.asarray(arr, dtype=np.float64)
        out = np.array(arr, dtype=np.float64, copy=True)
        out[~keep2d] = np.nan
        return out

    with h5py.File(path, "w") as f:
        f.attrs["NX_class"] = "NXroot"
        f.attrs["default"] = entry
        ent = f.create_group(entry)
        ent.attrs["NX_class"] = "NXentry"
        ent.attrs["default"] = _COM

        proc = _write_process(ent, "gauss1d_sweep")
        raw_fields = dict(stacked)
        raw_fields["layer_mu"] = layer_mu
        _write_raw(
            proc, raw_fields, D=1, orientation_axes=(0, 1), motor_units=motor_units,
            mask=mask, pixel_size_mm=pixel_size_mm, fit_status=fit_status,
        )
        _write_scan_group(ent, motors, scan_params, motor_units)
        _write_fit_status(ent, fit_status, y, x, x_units)

        for cname, cvals, q in (
            (_COM, centre, "fit_peak_center"), (_FWHM, fwhm, "fit_fwhm"),
        ):
            # display stack is layer-FIRST (n_layer, ny, nx) so silx's frame
            # slider scrubs the layer; the raw block keeps the (ny, nx, n_layer)
            # layout the loader splits on the last axis.
            disp = np.moveaxis(_mask_layers(cvals), -1, 0)
            _write_image_nxdata(
                ent, cname, disp, y, x, x_units, layer=layer_mu,
                extra_signal_attrs={"quantity": q, "source": "fit"},
            )

        _write_extra_attrs(f, extra_attrs)
    return path


def _save_two_peak_nexus(
    path, result, motors, scan_params, *, mask, pixel_size_mm, motor_units,
    orientation_axes, entry, extra_attrs, fit_status=None,
):
    """Two-peak N-D fit branch: n_peaks + per-peak/per-axis + separation maps.

    Display policy mirrors the other kinds (raw block always unmasked): with a
    grain ``mask`` the ``Number of peaks`` map is NaN'd off-grain, and the
    per-peak / separation maps are additionally NaN'd where the two-peak model
    was not selected (their raw fields are zero there by construction).
    """
    D = result.D
    ny, nx = np.asarray(result.A1).shape
    y, x, x_units = _image_axes(ny, nx, pixel_size_mm)
    keep = None if mask is None else np.asarray(mask, dtype=bool)
    keep_two = None if keep is None else keep & (np.asarray(result.success) > 0.5)

    with h5py.File(path, "w") as f:
        f.attrs["NX_class"] = "NXroot"
        f.attrs["default"] = entry
        ent = f.create_group(entry)
        ent.attrs["NX_class"] = "NXentry"
        ent.attrs["default"] = _N_PEAKS

        proc = _write_process(ent, "gaussND_two")
        raw_fields = {k: getattr(result, k) for k in _RAW_FIELDS["gaussND_two"]}
        _write_raw(
            proc, raw_fields, D=D, orientation_axes=orientation_axes,
            motor_units=motor_units, mask=mask, pixel_size_mm=pixel_size_mm,
            fit_status=fit_status,
        )
        _write_scan_group(ent, motors, scan_params, motor_units)
        _write_fit_status(ent, fit_status, y, x, x_units)

        _write_image_nxdata(
            ent, _N_PEAKS,
            _mask_display(np.asarray(result.n_peaks, dtype=np.float64), keep),
            y, x, x_units,
            extra_signal_attrs={
                "quantity": "n_peaks", "source": "fit",
                "long_name": "selected number of Gaussian components (0/1/2)",
            },
        )

        dmu, dist = result.separation()
        names = _short_motor_names(scan_params, D)
        for i in range(D):
            coll = ent.create_group(names[i])
            coll.attrs["NX_class"] = "NXcollection"
            for pk, mu_pk in ((1, result.mu1), (2, result.mu2)):
                _write_image_nxdata(
                    coll, f"{_COM} (peak {pk})",
                    _mask_display(np.asarray(mu_pk)[..., i], keep_two),
                    y, x, x_units,
                    extra_signal_attrs={"quantity": "fit_peak_center",
                                        "source": "fit", "peak": pk},
                )
                _write_image_nxdata(
                    coll, f"{_FWHM} (peak {pk})",
                    _mask_display(result.fwhm(peak=pk)[..., i], keep_two),
                    y, x, x_units,
                    extra_signal_attrs={"quantity": "fit_fwhm",
                                        "source": "fit", "peak": pk},
                )
            _write_image_nxdata(
                coll, _PEAK_SEP, _mask_display(dmu[..., i], keep_two),
                y, x, x_units,
                extra_signal_attrs={"quantity": "peak_separation",
                                    "source": "fit", "units": motor_units,
                                    "long_name": "mu1 - mu2 (major - minor)"},
            )

        _write_image_nxdata(
            ent, _PEAK_SEP_MAHA, _mask_display(dist, keep_two), y, x, x_units,
            extra_signal_attrs={
                "quantity": "peak_separation_mahalanobis", "source": "fit",
                "long_name": "pooled-sigma Mahalanobis peak separation",
            },
        )

        _write_extra_attrs(f, extra_attrs)
    return path


# --------------------------------------------------------------------------- #
# loader
# --------------------------------------------------------------------------- #


def _read_group(grp):
    out = {}
    for k, v in grp.items():
        out[k] = _read_group(v) if isinstance(v, h5py.Group) else v[()]
    return out


def load_nexus(path):
    """Read a :func:`save_nexus` file back into ``(result, maps, meta)``.

    ``result`` is the reconstructed dataclass (or ``list[Gauss1DResult]`` for a
    strain sweep), field-for-field equal to the original (the raw block is
    unmasked float64). ``maps`` is the display maps keyed by their darfix names.
    ``meta`` holds provenance (scan params, axis vectors, units, program/version/
    date, D, orientation_axes, pixel_size_mm, extra attrs, the verbatim int8
    ``fit_status`` categories or ``None``, and -- for a sweep -- ``layer_mu``).
    The categorical ``Fit status`` display map also appears in ``maps``.
    """
    with h5py.File(path, "r") as f:
        entry_name = _decode(f.attrs.get("default", "entry"))
        ent = f[entry_name]
        proc = ent[_PROCESS]
        kind = _decode(proc["result_kind"][()])

        raw_grp = proc["raw"]
        raw = {k: v[()] for k, v in raw_grp.items()}
        raw_attrs = dict(raw_grp.attrs)

        # ---- reconstruct the result object (round-trip) --------------------
        if kind == "gauss1d_sweep":
            layer_mu = np.asarray(raw["layer_mu"])
            n_layer = layer_mu.shape[0]
            fields = _RAW_FIELDS["gauss1d_sweep"]
            result = [
                Gauss1DResult.from_dict({k: raw[k][..., i] for k in fields})
                for i in range(n_layer)
            ]
        elif kind == "gauss1d":
            result = Gauss1DResult.from_dict(raw)
        elif kind == "gaussND":
            result = GaussNDResult.from_dict(raw)
        elif kind == "gaussND_two":
            result = GaussNDTwoResult.from_dict(raw)
        elif kind == "pseudovoigt":
            result = PseudoVoigtResult.from_dict(raw)
        elif kind == "moments":
            result = MomentResult.from_dict(raw)
        else:
            raise ValueError(f"unknown result_kind {kind!r} in {path}")

        # ---- display maps ---------------------------------------------------
        maps = {}
        for name, item in ent.items():
            if name in (_PROCESS, "scan", "data") or not isinstance(item, h5py.Group):
                continue
            nxclass = _decode(item.attrs.get("NX_class", ""))
            if nxclass == "NXdata":
                sig = _decode(item.attrs["signal"])
                maps[name] = item[sig][()]
            elif nxclass == "NXcollection":
                sub = {}
                for sname, sitem in item.items():
                    if isinstance(sitem, h5py.Group) and \
                            _decode(sitem.attrs.get("NX_class", "")) == "NXdata":
                        ssig = _decode(sitem.attrs["signal"])
                        sub[sname] = sitem[ssig][()]
                if sub:
                    maps[name] = sub

        # ---- meta -----------------------------------------------------------
        meta = {
            "result_kind": kind,
            "program": _decode(proc["program"][()]),
            "version": _decode(proc["version"][()]),
            "date": _decode(proc["date"][()]),
            "D": int(raw_attrs.get("D", 0)),
            "orientation_axes": tuple(int(a) for a in raw_attrs.get("orientation_axes", ())),
            "motor_units": _decode(raw_attrs.get("motor_units", "")),
            "mask_applied": bool(raw_attrs.get("mask_applied", False)),
            "pixel_size_mm": (np.asarray(raw_attrs["pixel_size_mm"]).tolist()
                              if "pixel_size_mm" in raw_attrs else None),
        }
        if kind == "gauss1d_sweep":
            meta["layer_mu"] = np.asarray(raw["layer_mu"])
        # verbatim int8 fit-status categories (0/1/2/3), if written
        meta["fit_status"] = (np.asarray(raw["fit_status"], dtype=np.int8)
                              if "fit_status" in raw else None)

        if "scan" in ent:
            scan = ent["scan"]
            known = {"scan_command", "scan_shape", "scan_id", "motor_names"}
            scan_axes = {k: v[()] for k, v in scan.items() if k not in known}
            scan_params = {}
            if "scan_command" in scan:
                scan_params["scan_command"] = _decode(scan["scan_command"][()])
            if "motor_names" in scan:
                scan_params["motor_names"] = _decode(scan["motor_names"][()]).tolist()
            if "scan_shape" in scan:
                scan_params["scan_shape"] = scan["scan_shape"][()]
            if "scan_id" in scan:
                scan_params["scan_id"] = _decode(scan["scan_id"][()])
            meta["scan_params"] = scan_params
            meta["scan_axes"] = scan_axes
        else:
            meta["scan_params"] = None
            meta["scan_axes"] = {}

        if _KAM in maps or (_KAM in ent and isinstance(ent[_KAM], h5py.Group)):
            ksize = ent[_KAM][_KAM].attrs.get("kernel_size") if _KAM in ent else None
            if ksize is not None:
                meta["kernel_size"] = tuple(int(k) for k in np.asarray(ksize))

        extra = {}
        if "starling_extra_keys" in f.attrs:
            json_keys = set(json.loads(
                _decode(f.attrs.get("starling_extra_json_keys", "[]"))))
            for k in json.loads(_decode(f.attrs["starling_extra_keys"])):
                v = f.attrs.get(k)
                if k in json_keys and isinstance(v, str):
                    v = json.loads(v)  # only decode keys we actually encoded
                extra[k] = v
        meta["extra_attrs"] = extra

    return result, maps, meta


# --------------------------------------------------------------------------- #
# dataset cube exporter
# --------------------------------------------------------------------------- #


def save_dataset_nexus(
    path, dset, *, motors=None, scan_params=None, entry="entry", extra_attrs=None,
):
    """Persist a denoised detector cube + motors + provenance as NeXus.

    Writes ``dset.data`` to ``entry/data/preprocessed_data`` (Bitshuffle,
    chunked one detector frame per chunk). starling is detector-first
    ``(a, b, *grid)`` -- the opposite of darfix's frame-last layout -- so the
    chunks are singleton on the *motor* axes, NOT on the detector axes.
    """
    import os

    parts = os.path.normpath(str(path)).split(os.sep)
    if "RAW_DATA" in parts:
        raise PermissionError(f"Write in RAW_DATA dir of ESRF is not allowed: {path}")

    data = np.asarray(dset.data)
    if motors is None:
        motors = getattr(dset, "motors", None)
    if scan_params is None:
        try:
            scan_params = dset.scan_params
        except Exception:
            scan_params = None
    motor_units = "deg"
    n_motor_dims = data.ndim - 2
    chunks = data.shape[:2] + (1,) * n_motor_dims

    with h5py.File(path, "w") as f:
        f.attrs["NX_class"] = "NXroot"
        f.attrs["default"] = entry
        ent = f.create_group(entry)
        ent.attrs["NX_class"] = "NXentry"
        ent.attrs["default"] = "data"

        proc = ent.create_group(_PROCESS)
        proc.attrs["NX_class"] = "NXprocess"
        proc.create_dataset("program", data="starling")
        proc.create_dataset("version", data=_version())
        proc.create_dataset("date", data=_now())
        proc.create_dataset("processing_order", data=np.int32(1))
        if getattr(dset, "h5file", None) is not None:
            proc.create_dataset("raw_data_source", data=str(dset.h5file))
        if scan_params is not None:
            if scan_params.get("scan_command") is not None:
                _set_str(proc, "scan_command", scan_params["scan_command"])
            if scan_params.get("motor_names") is not None and scan_params["motor_names"]:
                _set_str(proc, "motor_names",
                         [_leaf_name(p) for p in scan_params["motor_names"]])
            if scan_params.get("scan_shape") is not None:
                proc.create_dataset(
                    "scan_shape", data=np.asarray(scan_params["scan_shape"], np.int64)
                )
            if scan_params.get("scan_id") is not None:
                _set_str(proc, "scan_id", scan_params["scan_id"])

        nxdata = ent.create_group("data")
        nxdata.attrs["NX_class"] = "NXdata"
        nxdata.attrs["signal"] = "preprocessed_data"
        # NB: no @interpretation="image" -- the cube is detector-first
        # (a, b, *motors), so silx's image interpretation (last two dims) would
        # render the motor grid, not the detector frame. Left unhinted: this
        # group is for storage / ewoks chaining, not a zero-click image.
        nxdata.create_dataset(
            "preprocessed_data", data=data, chunks=chunks,
            compression=hdf5plugin.Bitshuffle(),
        )

        _write_scan_group(ent, motors, scan_params, motor_units)
        _write_extra_attrs(f, extra_attrs)
    return path
