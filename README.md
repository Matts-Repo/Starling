# starling

GPU-accelerated DFXM analysis for ESRF ID03 BLISS data. One PyTorch codebase
runs on CUDA (ESRF GPU nodes), Apple Silicon (MPS) and CPU; the device is
detected automatically. Fully standalone — the ID03 BLISS reading layer
(scan-command parsing, snake-scan readers, amesh support) is built in.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Interactive use (beamline)

See `notebooks/01_single_scan.ipynb`. In short:

```python
import starling

dset = starling.DataSet("dataset.h5", scan_id="1.1")
dset.subtract(dset.estimate_background())
dset.auto_roi()

mean, cov = dset.moments()                  # per-pixel COM + covariance
params = dset.fit_1D_gaussian()             # (ny, nx, 6) [A, sigma, mu, k, m, success]
out2 = dset.fit_two_gaussians_1D()          # 1 vs 2 peak per pixel, BIC-selected
p2d = dset.fit_2D_gaussian()                # 2D Gaussian over a mosa grid
```

## Batch use (post-experiment)

```bash
starling validate recipe.yaml
starling run recipe.yaml
```

See `notebooks/02_batch_recipe_guide.md` for the recipe format, resumability
and output layout.

## Notes

- Device override: `device="cpu" | "mps" | "cuda"` kwarg, or `STARLING_DEVICE` env var.
- GPU compute is float32 (MPS has no float64); coordinates are centred/scaled
  and curves normalised internally, so motor values like ccmth ≈ 6.68° with
  1e-3° steps fit stably. CPU uses float64.
- Large scans are processed in pixel chunks sized to the free GPU memory.
- First call per (model, shape) pays a torch.compile warm-up of a few seconds;
  subsequent calls are fast. Benchmarks: `python benchmarks/bench.py [--full]`.
- Golden test against real data: `STARLING_GOLDEN_H5=/path/to/master.h5 pytest tests/test_golden_real_scan.py`.

## Aborted scans

Beamline scans that abort mid-acquisition (frame count < scan command) load
with `DataSet(..., allow_partial=True)`: the complete fast-motor rows are
kept, snake-ordering is fixed, and `dset.partial_info` reports
frames_used/frames_expected.
