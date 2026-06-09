"""starling CLI: run | validate | aggregate."""

import argparse
import sys
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="starling", description="GPU-accelerated DFXM batch processing"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="process all scans in a recipe")
    p_run.add_argument("recipe", help="path to recipe.yaml")
    p_run.add_argument("--force", action="store_true", help="reprocess existing outputs")

    p_val = sub.add_parser("validate", help="validate a recipe and check scan files exist")
    p_val.add_argument("recipe")

    p_agg = sub.add_parser("aggregate", help="rebuild timeseries.h5 from existing outputs")
    p_agg.add_argument("recipe")

    args = ap.parse_args(argv)

    from .recipe import Recipe

    try:
        recipe = Recipe.load(args.recipe)
    except (ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if args.cmd == "validate":
        missing = [s.file for s in recipe.scans if not Path(s.file).exists()]
        for m in missing:
            print(f"missing scan file: {m}", file=sys.stderr)
        print(
            f"recipe OK: {len(recipe.scans)} scans, fits={recipe.fits}, "
            f"hash={recipe.recipe_hash()}"
            + (f", {len(missing)} missing files" if missing else "")
        )
        return 1 if missing else 0

    from .runner import aggregate_timeseries, run

    if args.cmd == "aggregate":
        outputs = [
            recipe.output_path(e.alias)
            for e in recipe.scans
            if Path(recipe.output_path(e.alias)).exists()
        ]
        if not outputs:
            print("no processed outputs found", file=sys.stderr)
            return 1
        ts = str(Path(recipe.output_dir) / "timeseries.h5")
        aggregate_timeseries(outputs, ts)
        print(f"wrote {ts} ({len(outputs)} scans)")
        return 0

    failed = run(recipe, force=args.force)
    if failed:
        print(f"\n{len(failed)} scans failed: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
