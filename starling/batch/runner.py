"""Sequential, resumable batch runner."""

import time
import traceback
from pathlib import Path

import h5py
import numpy as np

from .. import DataSet
from ..io._output import save_maps


def _is_done(out_path, rhash):
    if not Path(out_path).exists():
        return False
    try:
        with h5py.File(out_path, "r") as f:
            return f.attrs.get("recipe_hash") == rhash
    except OSError:
        return False


def run(recipe, force=False, log=print):
    """Process every scan in the recipe; returns the list of failed aliases.

    A scan is skipped when its output file already exists with a matching
    recipe hash (resume after interruption); --force overrides.
    """
    Path(recipe.output_dir).mkdir(parents=True, exist_ok=True)
    rhash = recipe.recipe_hash()
    device = None if recipe.device == "auto" else recipe.device

    failed = []
    for entry in recipe.scans:
        out_path = recipe.output_path(entry.alias)
        if not force and _is_done(out_path, rhash):
            log(f"[skip] {entry.alias} (already processed, hash {rhash})")
            continue
        t0 = time.perf_counter()
        try:
            maps, scan_params, dev_used = process_scan(entry, recipe, device)
            save_maps(
                out_path,
                maps,
                scan_params=scan_params,
                extra_attrs={
                    "recipe_hash": rhash,
                    "alias": entry.alias,
                    "source_file": entry.file,
                    "scan_id": entry.scan_id,
                    "device": dev_used,
                },
            )
            log(f"[done] {entry.alias} in {time.perf_counter() - t0:.1f} s -> {out_path}")
        except Exception:
            failed.append(entry.alias)
            log(f"[FAIL] {entry.alias}:\n{traceback.format_exc()}")

    if recipe.timeseries.get("enabled", True):
        outputs = [
            recipe.output_path(e.alias)
            for e in recipe.scans
            if Path(recipe.output_path(e.alias)).exists()
        ]
        if outputs:
            ts_path = str(Path(recipe.output_dir) / "timeseries.h5")
            aggregate_timeseries(outputs, ts_path)
            log(f"[done] time series -> {ts_path}")

    return failed


def process_scan(entry, recipe, device):
    """Load, preprocess and fit one scan; returns (maps, scan_params, device)."""
    pp = recipe.preprocess
    dset = DataSet(entry.file, scan_id=entry.scan_id, device=device, verbose=False)

    bg_cfg = pp.get("background")
    if bg_cfg:
        bg = dset.estimate_background(
            n_lowest=bg_cfg.get("n", 5), mode=bg_cfg.get("method", "mean")
        )
        dset.subtract(bg)
    if pp.get("hot_pixels", {}).get("enabled", False):
        dset.remove_hot_pixels(n_sigma=pp["hot_pixels"].get("n_sigma", 5.0))
    roi_cfg = pp.get("roi")
    roi = None
    if roi_cfg == "auto":
        roi = dset.auto_roi()
    elif isinstance(roi_cfg, (list, tuple)):
        r1, r2, c1, c2 = roi_cfg
        dset.data = np.ascontiguousarray(dset.data[r1:r2, c1:c2])
        roi = tuple(roi_cfg)

    maps = {}
    opts = recipe.fit_options
    n_motor_dims = dset.data.ndim - 2
    if "moments" in recipe.fits:
        mean, cov = dset.moments()
        maps["mean"] = mean
        maps["covariance"] = cov
    # 1D fits need a single motor dimension, the 2D fit needs two — skip
    # (with a note) rather than failing the whole scan
    if "gauss1d" in recipe.fits:
        if n_motor_dims == 1:
            maps["gauss1d"] = {
                "params": dset.fit_1D_gaussian(**(opts.get("gauss1d") or {}))
            }
        else:
            print(f"  [note] gauss1d skipped: scan has {n_motor_dims} motor dims")
    if "gauss2p" in recipe.fits:
        if n_motor_dims == 1:
            maps["gauss2p"] = dset.fit_two_gaussians_1D(**(opts.get("gauss2p") or {}))
        else:
            print(f"  [note] gauss2p skipped: scan has {n_motor_dims} motor dims")
    if "gauss2d" in recipe.fits:
        if n_motor_dims == 2:
            maps["gauss2d"] = {
                "params": dset.fit_2D_gaussian(**(opts.get("gauss2d") or {}))
            }
        else:
            print(f"  [note] gauss2d skipped: scan has {n_motor_dims} motor dims")
    if roi is not None:
        maps["roi"] = np.asarray(roi)

    return maps, dset.scan_params, str(dset.device)


FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))


def aggregate_timeseries(result_files, out_path):
    """Summarise per-scan maps into per-time-point scalars.

    For each scan output: median peak centre, median FWHM and success
    fraction from the gauss1d fit (where present), the 2-peak pixel fraction
    (gauss2p), and total intensity-weighted mean from moments.
    """
    rows = {}

    def add(key, val):
        rows.setdefault(key, []).append(val)

    aliases = []
    for path in result_files:
        with h5py.File(path, "r") as f:
            aliases.append(f.attrs.get("alias", Path(path).stem))
            maps = f["maps"]
            if "gauss1d" in maps:
                p = maps["gauss1d/params"][()]
                ok = p[..., 5] > 0
                add("gauss1d_mu_median", float(np.median(p[..., 2][ok])) if ok.any() else np.nan)
                add(
                    "gauss1d_fwhm_median",
                    float(np.median(p[..., 1][ok])) * FWHM if ok.any() else np.nan,
                )
                add("gauss1d_success_fraction", float(ok.mean()))
            if "gauss2p" in maps:
                n_peaks = maps["gauss2p/n_peaks"][()]
                add("two_peak_fraction", float((n_peaks == 2).mean()))
            if "gauss2d" in maps:
                p = maps["gauss2d/params"][()]
                ok = p[..., 7] > 0
                for col, name in ((1, "gauss2d_mu0_median"), (2, "gauss2d_mu1_median")):
                    add(name, float(np.median(p[..., col][ok])) if ok.any() else np.nan)
                add("gauss2d_success_fraction", float(ok.mean()))
            if "mean" in maps:
                add("com_mean", float(np.nanmean(maps["mean"][()])))

    with h5py.File(out_path, "w") as f:
        f.create_dataset("alias", data=np.array(aliases, dtype=h5py.string_dtype()))
        for k, v in rows.items():
            f.create_dataset(k, data=np.asarray(v))
