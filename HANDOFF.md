# HANDOFF — starling for the next beamtime

State of the project as of 2026-06-10, and how to pick it up with fresh data.
Audience: future me / a colleague at the beamline.

## What this is

`starling` is a standalone, GPU-accelerated DFXM analysis package for ESRF
ID03 BLISS data. PyTorch backend, auto device selection (CUDA on ESRF GPU
nodes → MPS on Apple Silicon → CPU). The ID03 reading layer
(scan-command parsing, snake-scan readers, amesh support) is built in.

Repo: `/Users/matt/Lab/projects/DFXM/starling` (own git repo).
Local venv: `.venv` (Python 3.12, torch 2.12). Install elsewhere with
`pip install -e .` — nothing else needed.

## Quick start on a fresh scan

```python
import starling

dset = starling.DataSet("/path/to/<dataset>/<dataset>.h5", scan_id="1.1")
dset.info()                                 # scan command, shape, motors
dset.subtract(dset.estimate_background())   # mean of 5 darkest frames
dset.auto_roi()                             # crop to the grain (in place)

mean, cov = dset.moments()                  # per-pixel COM + covariance
params = dset.fit_1D_gaussian()             # (ny,nx,6) [A, sigma, mu, k, m, success]
out2   = dset.fit_two_gaussians_1D()        # 1 vs 2 peaks per px, BIC-selected
p2d    = dset.fit_2D_gaussian()             # (ny,nx,8) full 2D Gaussian on mosa grids
```

Beamline notebook: `notebooks/01_single_scan.ipynb` (cell-by-cell version of
the above). Batch guide: `notebooks/02_batch_recipe_guide.md`.

Scan-type cheat sheet (MA6278 conventions):
- **Strain sweep**: `fscan2d ccmth <start> <step> <n> mu ...` → data
  (a, b, n_ccmth, n_mu). Fit ccmth curves per mu layer: slice
  `data[:, :, :, k]` with `motors[0, :, k]` and call `fit_1D_gaussian`.
- **Mosa**: `fscan2d chi ... mu ...` → (a, b, n_chi, n_mu). Use `moments`
  and/or `fit_2D_gaussian`.
- **3D strain-mosa**: N repeated fscan2d scans stepped in obpitch (or diffry,
  ccmth, …) within one master. Load stacked — **this is the supported 3D route;
  the external VDS/concatenation script is deprecated** (see below):
  ```python
  dset = starling.DataSet(master, scan_id=[f"{i}.1" for i in range(1, 12)],
                          scan_motor="instrument/positioners/obpitch")
  # data (a, b, n_chi, n_mu, n_obpitch)
  res = dset.analyze()        # GaussNDResult (D=3) — fitted 3D maps, not just moments
  res.mosaicity(axes=(0, 1))  # spread in the chi,mu orientation block only
  ```

### Retiring concatenation (the VDS script)

`DataSet(master, scan_id=[...], scan_motor=...)` (`_dataset._load_stacked_scans`)
already assembles the `(ny, nx, n1, n2, n_stack)` cube in memory from the raw
per-scan entries: it sorts sub-scans by the stack motor, builds the motor
meshgrid and reads the detector frames directly — preserving the grid structure
and **avoiding the broken-VDS-reads-zeros problem**. So the external
`3D_strain_mosa_*_concatenationtest.py` (which built a single HDF5 with a VDS
over all frames + concatenated positioners — the darfix route) is **no longer
needed for starling**. Parity is asserted in `tests/test_stacked_load.py`
(stacked load == hand-built concatenation: same frame count, same per-step motor
values, identical detector spectra). The one precondition: each sub-scan entry
must expose its detector dataset and per-frame motor arrays where `io/_reader.py`
expects them (true for the `repaired_cell_charging` datasets). If a future
dataset ships only a pre-concatenated `_concat.h5`, point `DataSet` at it with a
single `scan_id`.
- **Aborted scans** (frame count < scan command): add `allow_partial=True` —
  keeps complete fast-motor rows, fixes snake ordering, sets
  `dset.partial_info`.

## Batch processing (post-experiment, ~30 scans)

```bash
starling validate recipe.yaml   # schema + files exist
starling run recipe.yaml        # sequential, resumable (recipe-hash skip)
starling aggregate recipe.yaml  # rebuild timeseries.h5 only
```

Recipe format in `notebooks/02_batch_recipe_guide.md`. Outputs: one
`<alias>.h5` per scan (`/maps/...` + provenance attrs) plus `timeseries.h5`
(median peak centre / FWHM / 2-peak fraction per time point, in scan order).
Read back with `starling.io.load_maps(path)`.

## Performance expectations (what "working" looks like)

Measured on the M5 Pro 48 GB (MPS), real 2048² PCO data:

| operation | size | time (M5 Pro MPS) |
|---|---|---|
| fit_1D_gaussian | 600²×80 (ROI) | 0.14 s |
| fit_1D_gaussian | 2048²×80 | 3.2 s |
| fit_1D_gaussian | 2048²×25 (strain sweep layer) | 3.0 s |
| fit_2D_gaussian | 2048²×(25×20) | 17 s |
| moments 3D | 2048²×(12×5×11) | 5.8 s |
| batch scan (moments+gauss2d, full frame) | 60 frames | ~12.5 s |

Notes:
- First fit call per (model, chunk shape) pays a ~5–10 s torch.compile
  warm-up; subsequent calls are fast. Don't benchmark the first call.
- `moments` on multi-GB stacks is transfer-bound on MPS; should invert on CUDA.
- **CUDA is untested.** First thing at ESRF: `python benchmarks/bench.py
  --full` on a GPU node. Expect ≥10× over the MPS numbers for the fits. If
  not, suspect the chunk planner (`starling/device.py: plan_chunks`) or
  H2D transfer (data should upload once when it fits in VRAM).
- Sanity thresholds from MA6278: real signal is >50 counts peak (noise σ ≈
  3 counts). Fit success below that threshold is meaningless — always gate
  result maps on amplitude or max-count before interpreting.

## Code map (where to change things)

- `starling/device.py` — device autodetect, memory budget, chunk sizing.
- `starling/properties/_gaussnewton.py` — batched damped GN driver.
  torch.compile'd with **static shapes**: chunks are padded to uniform size
  (`_curvefit.py`), because a ragged last chunk triggers a multi-second
  recompile. Keep it that way.
- `starling/properties/_models.py` — model+Jacobian evaluators (build
  functionally with torch.stack, no slice writes, so compile can fuse).
  Add new lineshapes (pseudo-Voigt etc.) here + a public wrapper in
  `_curvefit.py`.
- `starling/properties/_curvefit.py` — public fit API (`fit_1D_gaussian`,
  `fit_1D_pseudo_voigt`, `fit_two_gaussians_1D`, `fit_2D_gaussian`,
  `fit_ND_gaussian`). Pattern for all fits: centre/scale motor coords to O(1),
  normalise curves by max, fit in float32, un-transform after. Required for MPS
  (no float64) — don't remove. `fit_2D_gaussian` is now a thin wrapper around
  `fit_ND_gaussian(D=2)` (parity asserted in `tests/test_fit_nd.py`); the N-D
  model `gaussND_const` generalises the 2D Cholesky-of-precision parameterisation
  to arbitrary D.
- `starling/properties/_results.py` — named result dataclasses (`Gauss1DResult`,
  `GaussNDResult`, `PseudoVoigtResult`, `MomentResult`); each has `.raw`,
  `.fwhm`, `to_dict`/`to_h5`, and the N-D / moment ones delegate `.mosaicity()`
  and `.orientation()`.
- `starling/properties/_maps.py` — `orientation_map` (first moment, mean) and
  `mosaicity` (second moment, spread). Kept distinct + documented; don't relabel
  a COM map "mosaicity".
- `starling/properties/_strain.py` — `strain_from_ccmth` / `strain_from_obpitch`
  (darfix formulas).
- `starling/transforms/_kam.py` — `kam` (kernel average misorientation, numpy).
- `starling/viz.py` — `denoise_widget(dset)` interactive non-destructive preview
  (ipywidgets, lazily imported).
- `starling/io/_dataset.py` — `DataSet.analyze(method="auto", mask="auto")`
  auto-dispatches by motor-dim count and returns a result object; convenience
  `dset.mosaicity()/.orientation()/.strain()/.kam()`. Grain masking
  (`preprocess.grain_mask` / `polygon_mask`) is threaded through moments + fits.
- `starling/io/_metadata.py` — ID03 scan-command parsing + motor h5-path
  maps. **If a new beamline scan command appears, add it to `scan_arg_pos` /
  `is_integrated` here.** New motors go in `motor_map` (+ fallback).
- `starling/io/_reader.py` — MosaScan/RockingScan/Darks readers; subclass
  `Reader` for custom schemes.
- `starling/io/_partial.py` — aborted-scan loader.
- `starling/batch/` — recipe schema, resumable runner, CLI.

Output param-order (verified, docstrings corrected): `fit_1D_gaussian` returns
`[A, sigma, mu, k, m, success]` — sigma at index 1, mu (peak centre) at
index 2. Prefer the named result objects (`dset.analyze()` →
`Gauss1DResult.mu` / `.fwhm`) over column indices; `.raw` gives the legacy
array for back-compat.

## Tests and how to trust a change

```bash
.venv/bin/pytest tests/ -q
```

- Tests run anywhere: ground-truth recovery, recipe/IO round-trips,
  standalone-ness.
- 1 golden test auto-runs when the LaCie is mounted at
  `/Volumes/LaCie/ESRF_MA6278/RAW_DATA/...` (override with
  `STARLING_GOLDEN_H5` / `STARLING_GOLDEN_SCAN`).
- `tests/test_standalone.py` enforces zero external DFXM package imports.

The LaCie unmounts itself intermittently — check `ls /Volumes/LaCie` before
real-data work.

## Data landscape on the LaCie (audited 2026-06-10)

- Use `/Volumes/LaCie/ESRF_MA6278/RAW_DATA/` — the `MA6278/Raw_Data/DFXM`
  tree is mostly empty skeletons.
- Complete, loadable datasets: everything under
  `DFXM_insitu_repaired_cell/` and `DFXM_insitu_repaired_cell_charging/`
  (strain sweeps, mosa projections, and the 3D strain-mosa layer series
  `..._2x_strain_mosa_layer_charging_{1_0006,2,3}` = chi×mu×obpitch).
- The 13N `3d_mosa_strain` raw data is NOT on the drive (empty dirs, broken
  external links; the `_concat.h5` files in PROCESSED_DATA are VDS shells
  reading zeros). Partial darfix exports with real frames:
  `PROCESSED_DATA/dfxm_insitu_2_hydrogen_charging_13N/data.hdf5`
  (300,1199,867; motors recoverable from the sibling `_concat.h5`) and
  `Matt_Processing/DFXM_insitu_2_hydrogen_charging_13N/data.hdf5`
  (650,1744,1393; no motor metadata). Full-res raw exists only in ESRF's
  archive (`/data/visitor/ma6278/id03`) — retrieve before next beamtime if
  needed.
- mosa_projection scans 1.1/2.1 are aborted; 2.1's first chi row holds the
  Δmu≈0.3° two-spot evidence (starling's 2-peak fit reproduces it: 488
  bimodal px, median sep 259 mdeg).

## Suggested next steps, in order

1. Replug the LaCie, run `pytest tests/test_golden_real_scan.py -v` (closes
   the vendored-loader verification).
2. At ESRF: `python benchmarks/bench.py --full` on a GPU node; record CUDA
   numbers next to the MPS table above.
3. During the experiment: notebook 01 per scan; keep recipes building in
   parallel so the post-experiment batch is one `starling run`.
4. Deferred features when needed: NMF/PCA decomposition, SLURM array
   submission, absolute (crystallographic) strain via unit-cell/hkl, an N-D
   two-peak mixture for overlapping sub-grains in 3D. (Done since the last
   handoff: N-D Gaussian fit, orientation/mosaicity split, strain helpers, KAM,
   skew/kurtosis maps, pseudo-Voigt, auto-dispatch + named results, interactive
   denoise widget, grain masking.)
