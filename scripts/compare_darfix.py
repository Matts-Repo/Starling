#!/usr/bin/env python
"""Characterise starling vs darfix on the same raw ID03 BLISS scan.

This is a *characterisation* harness, NOT a pass/fail parity test. starling and
darfix use different background estimators, different peak models (starling:
per-axis Gaussian + constant background via Gauss-Newton; darfix: multivariate
Gaussian with a min-background rocking-curve fit), and different hot-pixel
algorithms. Systematic differences are therefore *expected* and the job of this
script is to quantify them, not to assert they are zero.

Both packages read the *same* raw master h5. For every comparable per-pixel map
(centre-of-mass and FWHM per motor axis, plus a scalar orientation-spread /
mosaicity proxy) the script reports, over the intersection of the two valid
masks: median and 95th-percentile absolute difference, Pearson r, and (for COM)
the fraction of pixels agreeing to within one motor grid step.

Outputs (in --output-dir):
  * report.json      -- scan id, n compared pixels, per-map stats, notes
  * maps.h5          -- starling maps, darfix maps and their difference maps
  * fig_<map>.png    -- 3-panel (starling | darfix | difference) figures

Usage:
  python scripts/compare_darfix.py <master.h5> <scan_id> --output-dir OUT \
      [--roi r1 r2 c1 c2] [--bg-mode mean|median|percentile] \
      [--hp-kernel 3] [--device cpu|mps|cuda] [--no-darfix]

darfix is imported lazily; if it is not installed the script prints install
instructions and (unless --no-darfix) exits. On the ESRF cluster darfix is
already installed, so the script runs both sides unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

import h5py
import numpy as np

# --------------------------------------------------------------------------- #
# Scan-command parsing (bridge between starling axis order and darfix dim names)
# --------------------------------------------------------------------------- #

# argument layout inside a BLISS scan-command string, per command, as offsets
# into the token list *after* the command word:
#   fscan   mu  start step n   ...          -> name@0 npoints@3
#   fscan2d chi s1 st1 n1 mu s2 st2 n2 ...  -> names@0,4  npoints@3,7
_CMD_LAYOUT = {
    "fscan": {"names": [0], "npoints": [3], "start": [1], "step": [2]},
    "ascan": {"names": [0], "npoints": [3], "start": [1], "step": [2]},
    "fscan2d": {"names": [0, 4], "npoints": [3, 7], "start": [1, 5], "step": [2, 6]},
    "amesh": {"names": [0, 4], "npoints": [3, 7], "start": [1, 5], "step": [2, 6]},
}


def parse_scan_motors(h5file: str, scan_id: str) -> list[dict]:
    """Return per-motor descriptors in *scan-command order* (== starling axis
    order): ``[{name, npoints, start, step}, ...]``. ``name`` is the short motor
    name (e.g. ``"chi"``) which is also the darfix positioner / dimension key.
    """
    with h5py.File(h5file, "r") as f:
        title = f[scan_id]["title"][()]
    if isinstance(title, bytes):
        title = title.decode("utf-8", "replace")
    toks = title.split()
    cmd, params = toks[0], toks[1:]
    if cmd not in _CMD_LAYOUT:
        raise ValueError(
            f"scan command {cmd!r} is not a rocking/mesh scan this harness "
            f"knows how to map to darfix dimensions (title: {title!r})"
        )
    lay = _CMD_LAYOUT[cmd]
    n_add = 1 if cmd in ("ascan", "amesh") else 0  # interval->point count
    out = []
    for k in range(len(lay["names"])):
        out.append(
            {
                "name": params[lay["names"][k]],
                "npoints": int(params[lay["npoints"][k]]) + n_add,
                "start": float(params[lay["start"][k]]),
                "step": float(params[lay["step"][k]]),
            }
        )
    return out


# --------------------------------------------------------------------------- #
# STARLING side
# --------------------------------------------------------------------------- #

def run_starling(master, scan_id, roi, bg_mode, device, method, verbose):
    """Load + preprocess + fit with starling. Returns a dict of maps and meta."""
    import starling
    from starling import DataSet, properties, preprocess

    dset = DataSet(master, scan_id=scan_id, roi=roi, device=device, verbose=verbose)
    n = dset.n_motor_dims

    # background: estimate + subtract (starling's own estimator)
    bg = dset.estimate_background(mode=bg_mode)
    dset.subtract(bg)
    # hot-pixel removal (grain-safe one-sided, as used in production)
    dset.remove_hot_pixels(one_sided=True, min_sigma=1.0)
    # NOTE: intentionally NO auto_roi -- keep the detector frame aligned with
    # darfix (which loads the full frame / the same load ROI).

    zsum = np.asarray(dset.data).sum(axis=tuple(range(-n, 0))).astype(np.float64)
    grain = preprocess.grain_mask(dset.data)

    # fit (Gaussian; auto-dispatches on motor-dimension count)
    res = dset.analyze(method=method, mask="auto")
    status = dset.fit_status(res, mask="auto")          # 0 nosig,1 ok,2 edge,3 fail
    ok = status == 1

    # moments (for the moment-vs-moment comparison family); moments() does not
    # resolve the "auto" mask keyword, so pass the grain mask explicitly.
    mom = dset.moments(mask=grain)
    mean = np.asarray(mom[0])

    _, steps = properties.motor_ranges_steps(dset.motors)

    # per-axis maps -> keyed by motor short-name
    motors = parse_scan_motors(master, scan_id) if n >= 1 else []
    names = [m["name"] for m in motors]
    if len(names) != n:
        # fall back to axis indices if command parse disagrees with data ndim
        names = [f"axis{i}" for i in range(n)]

    com_fit, com_mom, fwhm_fit = {}, {}, {}
    if n == 1:
        # Gauss1DResult: mu, fwhm are (ny,nx)
        com_fit[names[0]] = np.asarray(res.mu, np.float64)
        fwhm_fit[names[0]] = np.asarray(res.fwhm, np.float64)
        com_mom[names[0]] = mean.astype(np.float64)
    else:
        mu = np.asarray(res.mu, np.float64)             # (ny,nx,D)
        fw = np.asarray(res.fwhm, np.float64)           # (ny,nx,D)
        for i, nm in enumerate(names):
            com_fit[nm] = mu[..., i]
            fwhm_fit[nm] = fw[..., i]
            com_mom[nm] = mean[..., i]

    # scalar spread proxy (see notes): geometric mean of per-axis fit FWHM,
    # plus starling's native covariance-based scalar mosaicity for reference.
    spread = _geo_mean_spread([fwhm_fit[nm] for nm in names])
    if n >= 2:
        native_mos = np.asarray(res.mosaicity(mode="scalar"), np.float64)
    else:
        native_mos = None

    return {
        "n_motor_dims": n,
        "motor_names": names,
        "motor_steps": {nm: float(steps[i]) for i, nm in enumerate(names)},
        "zsum": zsum,
        "grain": grain,
        "ok": ok,
        "fit_status": status,
        "com_fit": com_fit,
        "com_mom": com_mom,
        "fwhm_fit": fwhm_fit,
        "spread": spread,
        "native_mosaicity": native_mos,
        "final_roi": dset.final_roi(),
        "shape": zsum.shape,
    }


def _geo_mean_spread(fwhm_list):
    """Geometric mean of the per-axis FWHM maps -> a scalar spread map.

    A width defined identically for both packages so the mosaicity/spread
    comparison is genuinely apples-to-apples (starling's native scalar
    mosaicity and darfix have no common closed form otherwise)."""
    import warnings
    stk = np.stack([np.abs(np.asarray(f, np.float64)) for f in fwhm_list], axis=0)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN off-grain cols
        return np.exp(np.nanmean(np.log(np.where(stk > 0, stk, np.nan)), axis=0))


# --------------------------------------------------------------------------- #
# DARFIX side (headless ewoks tasks)
# --------------------------------------------------------------------------- #

DARFIX_INSTALL_HINT = (
    "darfix is not importable in this environment.\n"
    "Install it into the starling venv with:\n"
    '    pip install -e "/Users/matt/Lab/projects/DFXM/darfix new" \\\n'
    "        scikit-image opencv-python-headless\n"
    "or, on the ESRF cluster, `module load darfix` (darfix is preinstalled).\n"
    "Re-run with --no-darfix to characterise the starling side only."
)


def _require_darfix():
    try:
        import darfix  # noqa: F401
    except Exception as exc:  # pragma: no cover - env dependent
        raise RuntimeError(DARFIX_INSTALL_HINT) from exc


def run_darfix(master, scan_id, star_motors, star_shape, roi, hp_kernel,
               bg_method, treated_dir, do_rocking=True, verbose=False):
    """Run the darfix ewoks pipeline headless.

    Each stage is wrapped so a runtime API mismatch names the failing stage.
    ``star_motors`` is the parse_scan_motors() list (command order); it is used
    to build explicit darfix ``dims`` (bypassing darfix's fscan auto-dimension
    path, which is broken in darfix 5.1.1 -- it feeds an AcquisitionDims object
    to AcquisitionDims.from_dict). ``roi`` (r1,r2,c1,c2) or None crops the darfix
    output maps to the starling load-ROI so both sit on the same pixel grid.
    """
    _require_darfix()

    detector_path = f"/{scan_id}/measurement/pco_ff/data"
    metadata_path = f"/{scan_id}/instrument/positioners"

    stage = "hdf5_data_selection"
    try:
        from darfix.tasks.hdf5_data_selection import HDF5DataSelection

        # discover the detector data path robustly if the default is absent
        detector_path = _find_detector_path(master, scan_id, default=detector_path)
        t = HDF5DataSelection(inputs={
            "raw_input_file": master,
            "raw_detector_data_path": detector_path,
            "raw_metadata_path": metadata_path,
            "treated_data_dir": treated_dir,
        })
        t.execute()
        ds = t.outputs.dataset
        md = ds.metadata_dict
    except Exception as exc:
        raise RuntimeError(f"[darfix stage '{stage}'] {exc}\n"
                           f"{traceback.format_exc()}") from exc

    stage = "dimension_definition"
    try:
        from darfix.tasks.dimension_definition import DimensionDefinition

        # darfix axis 0 = fastest motor = LAST scan-command motor.
        dims = {}
        for axis, m in enumerate(reversed(star_motors)):
            nm = m["name"]
            vals = np.asarray(md[nm]) if nm in md else None
            if vals is not None and vals.size:
                lo, hi = float(np.min(vals)), float(np.max(vals))
            else:
                lo = m["start"]
                hi = m["start"] + m["step"] * (m["npoints"] - 1)
            dims[axis] = {"name": nm, "size": int(m["npoints"]),
                          "range": [lo, hi, float(m["step"])]}
        t = DimensionDefinition(inputs={"dataset": ds, "dims": dims})
        t.execute()
        ds = t.outputs.dataset
    except Exception as exc:
        raise RuntimeError(f"[darfix stage '{stage}'] {exc}\n"
                           f"{traceback.format_exc()}") from exc

    stage = "noise_removal"
    try:
        from darfix.tasks.noise_removal import NoiseRemoval
        from darfix.core.noise_removal_type import NoiseRemovalType

        ops = [
            {"type": NoiseRemovalType.BS,
             "parameters": {"method": bg_method, "background_type": "Data"}},
            {"type": NoiseRemovalType.HP,
             "parameters": {"kernel_size": int(hp_kernel)}},
        ]
        t = NoiseRemoval(inputs={"dataset": ds, "operations": ops})
        t.execute()
        ds = t.outputs.dataset
    except Exception as exc:
        raise RuntimeError(f"[darfix stage '{stage}'] {exc}\n"
                           f"{traceback.format_exc()}") from exc

    stage = "moments"
    try:
        from darfix.core.moment_types import MomentType

        ds.apply_moments()
        com_mom, fwhm_mom = {}, {}
        for m in star_motors:
            nm = m["name"]
            md_dim = ds.moments_dims.get(nm)
            if md_dim is None:
                continue
            com_mom[nm] = np.asarray(md_dim[MomentType.COM], np.float64)
            fwhm_mom[nm] = np.asarray(md_dim[MomentType.FWHM], np.float64)
    except Exception as exc:
        raise RuntimeError(f"[darfix stage '{stage}'] {exc}\n"
                           f"{traceback.format_exc()}") from exc

    com_fit, fwhm_fit, valid_fit, residuals = {}, {}, None, None
    if do_rocking:
        stage = "rocking_curves"
        try:
            from darfix.tasks.rocking_curves import RockingCurves

            t = RockingCurves(inputs={"dataset": ds, "save_maps": False})
            t.execute()
            maps = t.outputs.maps
            for m in star_motors:
                nm = m["name"]
                ck, fk = f"COM {nm}", f"FWHM {nm}"
                if ck in maps:
                    com_fit[nm] = np.squeeze(np.asarray(maps[ck], np.float64))
                if fk in maps:
                    fwhm_fit[nm] = np.squeeze(np.asarray(maps[fk], np.float64))
            if "Fit successful" in maps:
                valid_fit = np.squeeze(np.asarray(maps["Fit successful"])) > 0.5
            if "Residuals" in maps:
                residuals = np.squeeze(np.asarray(maps["Residuals"], np.float64))
        except Exception as exc:
            raise RuntimeError(f"[darfix stage '{stage}'] {exc}\n"
                               f"{traceback.format_exc()}") from exc

    def crop(a):
        if a is None or roi is None:
            return a
        r1, r2, c1, c2 = roi
        return np.asarray(a)[r1:r2, c1:c2]

    out = {
        "com_fit": {k: crop(v) for k, v in com_fit.items()},
        "fwhm_fit": {k: crop(v) for k, v in fwhm_fit.items()},
        "com_mom": {k: crop(v) for k, v in com_mom.items()},
        "fwhm_mom": {k: crop(v) for k, v in fwhm_mom.items()},
        "valid_fit": crop(valid_fit),
        "residuals": crop(residuals),
    }
    if fwhm_fit:
        out["spread"] = crop(_geo_mean_spread(
            [out["fwhm_fit"][m["name"]] for m in star_motors
             if m["name"] in out["fwhm_fit"]]))
    return out


def _find_detector_path(master, scan_id, default):
    """Return an in-file path to the 3D detector stack for silx/darfix.

    Prefers ``default`` if present; else finds the first 3D dataset under the
    scan group whose first axis is the largest (the frame stack)."""
    with h5py.File(master, "r") as f:
        dp = default.lstrip("/")
        if dp in f and isinstance(f[dp], h5py.Dataset) and f[dp].ndim == 3:
            return default
        best = None
        grp = f[scan_id]

        def visit(name, obj):
            nonlocal best
            if isinstance(obj, h5py.Dataset) and obj.ndim == 3:
                if best is None or obj.shape[0] > best[1]:
                    best = (name, obj.shape[0])

        grp.visititems(visit)
        if best is not None:
            return f"/{scan_id}/{best[0]}"
    return default


# --------------------------------------------------------------------------- #
# COMPARISON
# --------------------------------------------------------------------------- #

def compare_map(a, b, mask, step=None):
    """Difference statistics for two maps over ``mask`` (a - b)."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    m = mask & np.isfinite(a) & np.isfinite(b)
    n = int(m.sum())
    if n == 0:
        return {"n_px": 0, "median_abs_diff": None, "p95_abs_diff": None,
                "pearson_r": None, "frac_within_1_step": None,
                "mean_diff": None, "std_diff": None}
    d = a[m] - b[m]
    ad = np.abs(d)
    out = {
        "n_px": n,
        "median_abs_diff": float(np.median(ad)),
        "p95_abs_diff": float(np.percentile(ad, 95)),
        "mean_diff": float(np.mean(d)),
        "std_diff": float(np.std(d)),
        "frac_within_1_step": None,
    }
    if n >= 2 and np.std(a[m]) > 0 and np.std(b[m]) > 0:
        out["pearson_r"] = float(np.corrcoef(a[m], b[m])[0, 1])
    else:
        out["pearson_r"] = None
    if step is not None and step > 0:
        out["frac_within_1_step"] = float(np.mean(ad <= step))
    return out


def build_report(star, dar, scan_id, args):
    """Assemble the JSON report dict and collect maps for h5/PNG dumping."""
    names = star["motor_names"]
    steps = star["motor_steps"]

    star_ok = star["ok"]
    dar_fit_valid = dar.get("valid_fit")
    if dar_fit_valid is None:
        # moments-only run: validity = finite darfix moment maps
        dar_fit_valid = np.ones_like(star_ok, dtype=bool)

    maps_report = {}
    dump = {}  # name -> (starling_map, darfix_map, diff_map, mask)

    def add(key, a, b, mask, step):
        if a is None or b is None:
            return
        a = np.asarray(a, np.float64)
        b = np.asarray(b, np.float64)
        if a.shape != b.shape:
            maps_report[key] = {"error": f"shape mismatch {a.shape} vs {b.shape}"}
            return
        m = mask & np.isfinite(a) & np.isfinite(b)
        maps_report[key] = compare_map(a, b, m, step=step)
        diff = np.where(m, a - b, np.nan)
        dump[key] = (np.where(m, a, np.nan), np.where(m, b, np.nan), diff, m)

    # --- fit-vs-fit COM / FWHM per axis ---
    fit_mask = star_ok & dar_fit_valid
    for nm in names:
        if nm in star["com_fit"] and nm in dar["com_fit"]:
            add(f"com_fit_{nm}", star["com_fit"][nm], dar["com_fit"][nm],
                fit_mask, steps.get(nm))
        if nm in star["fwhm_fit"] and nm in dar["fwhm_fit"]:
            add(f"fwhm_fit_{nm}", star["fwhm_fit"][nm], dar["fwhm_fit"][nm],
                fit_mask, steps.get(nm))

    # --- moment-vs-moment COM per axis ---
    mom_mask = star["grain"].astype(bool)
    for nm in names:
        if nm in star["com_mom"] and nm in dar.get("com_mom", {}):
            add(f"com_mom_{nm}", star["com_mom"][nm], dar["com_mom"][nm],
                mom_mask, steps.get(nm))

    # --- scalar spread / mosaicity proxy (fit FWHM geometric mean) ---
    if star.get("spread") is not None and dar.get("spread") is not None:
        add("spread_fit", star["spread"], dar["spread"], fit_mask, None)

    # n compared = union of pixels used across fit-COM maps (representative)
    n_compared = 0
    for nm in names:
        k = f"com_fit_{nm}"
        if k in maps_report and isinstance(maps_report[k], dict):
            n_compared = max(n_compared, maps_report[k].get("n_px", 0) or 0)

    report = {
        "scan": scan_id,
        "master": os.path.abspath(args.master),
        "roi": list(args.roi) if args.roi else None,
        "n_motor_dims": star["n_motor_dims"],
        "motor_names": names,
        "motor_steps": steps,
        "n_compared_px": int(n_compared),
        "starling": {
            "bg_mode": args.bg_mode,
            "n_ok_px": int(star_ok.sum()),
            "n_grain_px": int(star["grain"].sum()),
        },
        "darfix": {
            "bg_method": args.bg_method,
            "hp_kernel": args.hp_kernel,
            "n_fit_valid_px": int(np.asarray(dar_fit_valid).sum()),
        },
        "maps": maps_report,
        "notes": [
            "CHARACTERISATION, not parity: nonzero systematic differences are expected by design.",
            "Background estimators differ: starling estimate_background (n-lowest "
            f"'{args.bg_mode}') vs darfix per-frame background subtraction "
            f"(method '{args.bg_method}', background_type 'Data').",
            "Peak model differs: starling per-axis Gaussian + constant background "
            "(Gauss-Newton) vs darfix multivariate Gaussian rocking-curve fit "
            "with min-background.",
            "Hot-pixel algorithms differ: starling one-sided sigma-clip "
            "(min_sigma=1.0) vs darfix median-kernel replacement "
            f"(kernel_size={args.hp_kernel}).",
            "darfix moment maps are median-smoothed (smooth=True); starling "
            "moments are not, so com_mom_* differences include a smoothing term.",
            "spread_fit is a common geometric-mean-of-per-axis-FWHM proxy defined "
            "identically for both packages (neither exposes a shared scalar "
            "mosaicity); starling's native covariance mosaicity is stored "
            "separately in maps.h5 for reference.",
            "COM comparison restricted to starling fit_status==1 AND darfix "
            "'Fit successful'; moment comparison restricted to the starling grain mask.",
        ],
    }
    return report, dump


# --------------------------------------------------------------------------- #
# OUTPUT (h5 + PNG)
# --------------------------------------------------------------------------- #

def write_h5(path, star, dar, dump):
    with h5py.File(path, "w") as f:
        g = f.create_group("starling")
        for fam in ("com_fit", "com_mom", "fwhm_fit"):
            gg = g.create_group(fam)
            for nm, arr in star[fam].items():
                gg.create_dataset(nm, data=np.asarray(arr, np.float64))
        g.create_dataset("zsum", data=star["zsum"])
        g.create_dataset("fit_status", data=star["fit_status"])
        g.create_dataset("grain", data=star["grain"].astype(np.uint8))
        if star.get("native_mosaicity") is not None:
            g.create_dataset("native_mosaicity", data=star["native_mosaicity"])
        if star.get("spread") is not None:
            g.create_dataset("spread_fit", data=star["spread"])

        d = f.create_group("darfix")
        for fam in ("com_fit", "com_mom", "fwhm_fit", "fwhm_mom"):
            if not dar.get(fam):
                continue
            gg = d.create_group(fam)
            for nm, arr in dar[fam].items():
                if arr is not None:
                    gg.create_dataset(nm, data=np.asarray(arr, np.float64))
        if dar.get("valid_fit") is not None:
            d.create_dataset("valid_fit", data=np.asarray(dar["valid_fit"]).astype(np.uint8))
        if dar.get("residuals") is not None:
            d.create_dataset("residuals", data=dar["residuals"])
        if dar.get("spread") is not None:
            d.create_dataset("spread_fit", data=dar["spread"])

        c = f.create_group("difference")
        for key, (a, b, diff, _m) in dump.items():
            c.create_dataset(key, data=diff)


def write_figures(outdir, dump, report):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[warn] matplotlib unavailable, skipping PNGs: {exc}")
        return
    from matplotlib.colors import TwoSlopeNorm

    for key, (a, b, diff, m) in dump.items():
        finite = np.isfinite(a) & np.isfinite(b)
        if not finite.any():
            continue
        vlo = float(np.nanmin([np.nanmin(a), np.nanmin(b)]))
        vhi = float(np.nanmax([np.nanmax(a), np.nanmax(b)]))
        dmax = float(np.nanmax(np.abs(diff))) if np.isfinite(diff).any() else 1.0
        dmax = dmax if dmax > 0 else 1.0

        fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        for ax, arr, title, cmap, norm in (
            (axes[0], a, "starling", "viridis", None),
            (axes[1], b, "darfix", "viridis", None),
            (axes[2], diff, "starling - darfix", "RdBu_r",
             TwoSlopeNorm(vcenter=0.0, vmin=-dmax, vmax=dmax)),
        ):
            cmap_obj = plt.get_cmap(cmap).copy()
            cmap_obj.set_bad("0.6")  # off-mask -> grey
            kw = {"cmap": cmap_obj, "interpolation": "nearest", "origin": "lower"}
            if norm is not None:
                kw["norm"] = norm
            else:
                kw["vmin"], kw["vmax"] = vlo, vhi
            im = ax.imshow(arr, **kw)
            ax.set_title(f"{key}\n{title}")
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, shrink=0.8)
        st = report["maps"].get(key, {})
        cap = (f"n={st.get('n_px')}  med|Δ|={_fmt(st.get('median_abs_diff'))}  "
               f"p95|Δ|={_fmt(st.get('p95_abs_diff'))}  r={_fmt(st.get('pearson_r'))}")
        fig.suptitle(cap, fontsize=9)
        fig.savefig(os.path.join(outdir, f"fig_{key}.png"), dpi=110)
        plt.close(fig)


def _fmt(x):
    return "n/a" if x is None else f"{x:.4g}"


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("master", help="BLISS master h5 file")
    p.add_argument("scan_id", help="scan id, e.g. '1.1'")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--roi", nargs=4, type=int, metavar=("R1", "R2", "C1", "C2"),
                   default=None, help="detector ROI (row/col min/max)")
    p.add_argument("--bg-mode", default="mean",
                   help="starling estimate_background mode (mean|median|percentile)")
    p.add_argument("--bg-method", default="median",
                   help="darfix background-subtraction method (mean|median)")
    p.add_argument("--hp-kernel", type=int, default=3,
                   help="darfix hot-pixel median kernel size")
    p.add_argument("--device", default=None, help="starling torch device")
    p.add_argument("--method", default="auto",
                   help="starling analyze() method (auto|gauss1d|gaussND|...)")
    p.add_argument("--no-rocking", action="store_true",
                   help="skip darfix RockingCurves fit (moments only)")
    p.add_argument("--no-darfix", action="store_true",
                   help="run starling only; still writes its maps")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    roi = tuple(args.roi) if args.roi else None
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[starling] loading {args.master} scan {args.scan_id} ...")
    star = run_starling(args.master, args.scan_id, roi, args.bg_mode,
                        args.device, args.method, args.verbose)
    print(f"[starling] {star['n_motor_dims']}-motor scan, motors={star['motor_names']}, "
          f"ok px={int(star['ok'].sum())}")

    star_motors = parse_scan_motors(args.master, args.scan_id)

    if args.no_darfix:
        dar = {"com_fit": {}, "fwhm_fit": {}, "com_mom": {}, "fwhm_mom": {},
               "valid_fit": None, "residuals": None}
        report = {"scan": args.scan_id, "darfix": "SKIPPED (--no-darfix)",
                  "starling_only": True,
                  "motor_names": star["motor_names"]}
        dump = {}
        _dump_starling_only(args.output_dir, star)
    else:
        treated = os.path.join(args.output_dir, "darfix_treated")
        os.makedirs(treated, exist_ok=True)
        print("[darfix] running headless ewoks pipeline ...")
        dar = run_darfix(args.master, args.scan_id, star_motors, star["shape"],
                         roi, args.hp_kernel, args.bg_method, treated,
                         do_rocking=not args.no_rocking, verbose=args.verbose)
        report, dump = build_report(star, dar, args.scan_id, args)
        write_h5(os.path.join(args.output_dir, "maps.h5"), star, dar, dump)
        write_figures(args.output_dir, dump, report)

    with open(os.path.join(args.output_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[done] wrote {args.output_dir}/report.json "
          f"({report.get('n_compared_px', 0)} compared px)")
    return report


def _dump_starling_only(outdir, star):
    with h5py.File(os.path.join(outdir, "maps.h5"), "w") as f:
        g = f.create_group("starling")
        for fam in ("com_fit", "com_mom", "fwhm_fit"):
            gg = g.create_group(fam)
            for nm, arr in star[fam].items():
                gg.create_dataset(nm, data=np.asarray(arr, np.float64))
        g.create_dataset("zsum", data=star["zsum"])
        g.create_dataset("fit_status", data=star["fit_status"])


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
