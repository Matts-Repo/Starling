#!/usr/bin/env python
"""Back-stamp the sin(2theta) aspect correction onto already-analysed scans.

Analyses written before :func:`starling.io.sample_pixel_size` existed recorded
one isotropic pixel size, so every map they produce is compressed along array
axis 0 by ``1/sin(2theta)`` (2.88x at the ID03-typical 2theta of 20.3 deg) and
carries no way to recover the factor — ``obpitch`` lives in the raw BLISS tree,
which is not on the analysis machine.

This walks a data root and, for every scan directory holding a
``pixel_size.txt``, recovers the geometry from that file and writes the
sample-plane pixel size back into:

* ``pixel_size.txt``      — rewritten with both axes and the aspect factor
* ``bundle*.h5``          — root attrs ``pixel_um_y``/``pixel_um_x``/
                            ``obpitch_deg``/``aspect_factor``/
                            ``topography_axis`` (matching what ``save_bundle``
                            now stamps natively), including ``refined/``
* ``result.nxs``          — the ``y`` axis of every spatial NXdata rescaled,
                            and ``pixel_size_mm`` promoted to ``(py, px)``
* ``review_manifest.json`` — ``pixel_um_y``/``pixel_um_x``/``aspect_factor``

Every edit is additive or in-place on existing arrays: no cube is rewritten and
no result is touched. All of it is idempotent — bundles are re-stamped with the
same values, and the NeXus rescale is guarded by an ``aspect_corrected`` marker
so a second run cannot double-stretch an axis.

Reports by default; pass ``--apply`` to write.

Usage::

    python scripts/apply_aspect_correction.py /Volumes/LaCie/ESRF_MA7031/updated_DFXM
    python scripts/apply_aspect_correction.py <root> --apply
"""

import argparse
import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from starling.io import TOPOGRAPHY_AXIS, aspect_factor, sample_pixel_size  # noqa: E402

_MOTOR_LINE = re.compile(
    r"obx\s*/\s*mainx\s*/\s*obpitch\s*:\s*"
    r"(?P<obx>[-\d.eE+]+)\s*/\s*(?P<mainx>[-\d.eE+]+)\s*/\s*(?P<obpitch>[-\d.eE+]+)"
)
_FFSEL_LINE = re.compile(r"ffsel\s*=\s*(?P<ffsel>[-\d.eE+]+)")
_EFF_LINE = re.compile(r"effective pixel size\s*:\s*[\d.]+\s*nm/px\s*\(\s*(?P<um>[\d.eE+-]+)\s*um")
_OVERRIDE_LINE = re.compile(r"override applied\s*:\s*(?P<flag>\w+)")


class GeometryError(Exception):
    """pixel_size.txt could not be turned into a usable geometry."""


def read_geometry(pixel_size_txt):
    """Recover ``(invariant_motors, recorded_eff_um, override)`` from the record."""
    text = Path(pixel_size_txt).read_text()
    m = _MOTOR_LINE.search(text)
    if not m:
        raise GeometryError("no 'obx / mainx / obpitch' line")
    motors = {k: float(v) for k, v in m.groupdict().items()}
    f = _FFSEL_LINE.search(text)
    if not f:
        raise GeometryError("no ffsel value")
    motors["ffsel"] = float(f.group("ffsel"))

    e = _EFF_LINE.search(text)
    recorded = float(e.group("um")) if e else None
    o = _OVERRIDE_LINE.search(text)
    override = bool(o and o.group("flag").lower() == "true")
    return motors, recorded, override


def plan_for(scan_dir):
    """Compute the correction for one scan directory.

    Returns a dict with the motors, the (dy, dx) pair, the aspect factor and
    the recorded value it supersedes.
    """
    motors, recorded, override = read_geometry(scan_dir / "pixel_size.txt")
    stretch = aspect_factor(motors)
    if not np.isfinite(stretch):
        raise GeometryError(f"unusable obpitch {motors.get('obpitch')!r}")
    dy, dx = sample_pixel_size(motors)
    if override and recorded is not None:
        # a manual override beat the motor geometry at analysis time; keep it
        # as the un-foreshortened axis and stretch that instead of recomputing
        dx = recorded
        dy = recorded * stretch
    if recorded is not None and not override and abs(dx - recorded) > 1e-4:
        raise GeometryError(
            f"recomputed {dx:.5f} um disagrees with recorded {recorded:.5f} um"
        )
    return {
        "motors": motors,
        "dy": float(dy),
        "dx": float(dx),
        "aspect": float(stretch),
        "recorded_um": recorded,
        "override": override,
    }


# --------------------------------------------------------------------------- #
# writers
# --------------------------------------------------------------------------- #


_GENERATED_PREFIXES = (
    "effective pixel size",      # the original isotropic line this supersedes
    "detector-plane pixel",
    "sample-plane pixel size",
    "  axis ",
    "  aspect factor",
    "  2theta (obpitch)",
)


def write_pixel_size_txt(scan_dir, plan, apply):
    """Rewrite pixel_size.txt with both axes, preserving the original fields.

    Idempotent: every line this function generates is stripped before the new
    header is prepended, so re-running does not stack headers.
    """
    path = scan_dir / "pixel_size.txt"
    keep = [ln for ln in path.read_text().splitlines()
            if not ln.startswith(_GENERATED_PREFIXES)]
    while keep and not keep[0].strip():          # blank lines from the old header
        keep.pop(0)

    m, dy, dx, a = plan["motors"], plan["dy"], plan["dx"], plan["aspect"]
    rows = [
        (f"axis {TOPOGRAPHY_AXIS} (rows, detector vertical, along beam)",
         f"{dy*1000:.2f} nm/px  ({dy:.5f} um)"),
        (f"axis {1 - TOPOGRAPHY_AXIS} (cols, detector horizontal)",
         f"{dx*1000:.2f} nm/px  ({dx:.5f} um)"),
        ("aspect factor 1/sin(2theta)", f"{a:.4f}"),
        ("2theta (obpitch)", f"{m['obpitch']:.5f} deg"),
    ]
    w = max(len(label) for label, _ in rows)
    head = [
        f"detector-plane pixel : {dx*1000:.2f} nm/px  ({dx:.5f} um)  [isotropic]",
        "",
        "sample-plane pixel size (sin(2theta) aspect-corrected)",
        *[f"  {label:<{w}} : {value}" for label, value in rows],
        "",
    ]
    if apply:
        path.write_text("\n".join(head + keep) + "\n")
    return "pixel_size.txt"


def stamp_bundle(path, plan, apply):
    """Add the geometry root attrs to one bundle (idempotent)."""
    attrs = {
        "pixel_um_y": plan["dy"],
        "pixel_um_x": plan["dx"],
        "aspect_factor": plan["aspect"],
        "obpitch_deg": plan["motors"]["obpitch"],
        "topography_axis": int(TOPOGRAPHY_AXIS),
    }
    if not apply:
        return path.name
    with h5py.File(path, "r+") as f:
        for k, v in attrs.items():
            f.attrs[k] = v
    return path.name


def patch_nexus(path, plan, apply):
    """Rescale the ``y`` axis of every spatial NXdata; guarded by a marker.

    Returns the action taken: "patched", "already", or "skipped: <why>".
    """
    py_mm, px_mm = plan["dy"] * 1e-3, plan["dx"] * 1e-3
    with h5py.File(path, "r") as f:
        raw = f.get("entry/starling_process/raw")
        if raw is not None and bool(raw.attrs.get("aspect_corrected", False)):
            return "already"
        targets = []
        for grp in _walk_nxdata(f):
            if "y" not in grp or "x" not in grp:
                continue
            y = grp["y"][()]
            if y.ndim != 1 or y.size < 2:
                continue
            step = float(y[1] - y[0])
            if not np.isclose(step, px_mm, rtol=1e-6, atol=0):
                return (f"skipped: {grp.name} y step {step:.6g} mm is not the "
                        f"recorded isotropic {px_mm:.6g} mm")
            targets.append(grp.name)
    if not targets:
        return "skipped: no spatial NXdata"
    if not apply:
        return f"patched ({len(targets)} axes)"
    with h5py.File(path, "r+") as f:
        for name in targets:
            grp = f[name]
            n = grp["y"].shape[0]
            grp["y"][:] = np.arange(n) * py_mm
        raw = f.get("entry/starling_process/raw")
        if raw is not None:
            raw.attrs["pixel_size_mm"] = np.asarray([py_mm, px_mm], dtype=np.float64)
            raw.attrs["aspect_corrected"] = True
            raw.attrs["aspect_factor"] = plan["aspect"]
            raw.attrs["topography_axis"] = int(TOPOGRAPHY_AXIS)
    return f"patched ({len(targets)} axes)"


def _walk_nxdata(f):
    """Yield every group in ``f`` (recursively) — cheap enough at this size."""
    out = []
    f.visititems(lambda _n, o: out.append(o) if isinstance(o, h5py.Group) else None)
    return out


def patch_manifest(path, plan, apply):
    """Add the anisotropic pixel size to review_manifest.json."""
    data = json.loads(path.read_text())
    data["pixel_um_y"] = plan["dy"]
    data["pixel_um_x"] = plan["dx"]
    data["aspect_factor"] = plan["aspect"]
    data["topography_axis"] = int(TOPOGRAPHY_AXIS)
    if apply:
        path.write_text(json.dumps(data, indent=2) + "\n")
    return path.name


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def process(scan_dir, apply, do_nexus=True):
    """Correct one scan directory; returns (plan, list of action strings)."""
    plan = plan_for(scan_dir)
    actions = [write_pixel_size_txt(scan_dir, plan, apply)]

    for bundle in sorted(scan_dir.glob("bundle*.h5")) + sorted(
        scan_dir.glob("refined/bundle*.h5")
    ):
        if bundle.name.startswith("._"):      # macOS resource fork
            continue
        actions.append(stamp_bundle(bundle, plan, apply))

    if do_nexus:
        for nxs in sorted(scan_dir.glob("*.nxs")):
            if nxs.name.startswith("._"):
                continue
            actions.append(f"{nxs.name}: {patch_nexus(nxs, plan, apply)}")

    manifest = scan_dir / "review_manifest.json"
    if manifest.exists():
        actions.append(patch_manifest(manifest, plan, apply))
    return plan, actions


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root", type=Path, help="data root to walk")
    ap.add_argument("--apply", action="store_true",
                    help="write the corrections (default: report only)")
    ap.add_argument("--no-nexus", action="store_true",
                    help="leave result.nxs files alone")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="list every file touched")
    args = ap.parse_args(argv)

    scan_dirs = sorted({p.parent for p in args.root.rglob("pixel_size.txt")
                        if not p.name.startswith("._")
                        and ".ipynb_checkpoints" not in p.parts})
    if not scan_dirs:
        print(f"no scan directories with pixel_size.txt under {args.root}")
        return 1

    mode = "APPLYING" if args.apply else "DRY RUN (pass --apply to write)"
    print(f"{mode} — {len(scan_dirs)} scan directories under {args.root}\n")

    ok, failed, seen = 0, [], {}
    for d in scan_dirs:
        try:
            plan, actions = process(d, args.apply, do_nexus=not args.no_nexus)
        except (GeometryError, OSError) as exc:
            failed.append((d, str(exc)))
            print(f"  !! {d.name}: {exc}")
            continue
        ok += 1
        key = (round(plan["dy"], 6), round(plan["dx"], 6))
        seen.setdefault(key, []).append(d.name)
        skipped = [a for a in actions if "skipped" in a]
        if args.verbose or skipped:
            print(f"  {d.name}")
            for a in actions:
                print(f"      {a}")
        for s in skipped:
            print(f"      !! {s}")

    print(f"\n{ok} scan(s) corrected, {len(failed)} failed\n")
    print(f"{'axis 0 (nm/px)':>16} {'axis 1 (nm/px)':>16} {'aspect':>8}  scans")
    for (dy, dx), names in sorted(seen.items()):
        print(f"{dy*1000:16.2f} {dx*1000:16.2f} {dy/dx:8.4f}  {len(names)}")
    for d, exc in failed:
        print(f"FAILED {d}: {exc}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
