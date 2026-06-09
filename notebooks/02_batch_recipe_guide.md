# Batch processing with `starling run`

Post-experiment workflow: one YAML recipe describes the preprocessing and
fits once; the CLI applies it to every scan and aggregates a time series.

## 1. Write a recipe

```yaml
# recipe.yaml
output_dir: /path/to/results
device: auto              # auto | cuda | mps | cpu

preprocess:
  background: {method: mean, n: 5}     # mean of the 5 darkest frames
  hot_pixels: {enabled: true, n_sigma: 5.0}
  roi: auto               # auto | [r1, r2, c1, c2] | omit for full frame

fits: [moments, gauss1d, gauss2p]      # any of: moments gauss1d gauss2p gauss2d
fit_options:
  gauss2p: {delta_bic: 10.0}

scans:
  - {file: /data/MA6278/RAW_DATA/.../dataset.h5, scan_id: "1.1", alias: t00}
  - {file: /data/MA6278/RAW_DATA/.../dataset.h5, scan_id: "2.1", alias: t01}
  # ... ~30 entries; alias order defines the time axis

timeseries:
  enabled: true
```

## 2. Validate, run, aggregate

```bash
starling validate recipe.yaml   # checks schema + that scan files exist
starling run recipe.yaml        # processes every scan sequentially
starling run recipe.yaml --force          # reprocess everything
starling aggregate recipe.yaml  # rebuild timeseries.h5 only
```

`run` is resumable: each output records a hash of the processing definition,
and scans whose output already matches are skipped. Edit the preprocessing or
fits and re-run — only then is everything recomputed. Failures are logged and
the run continues; the exit code reports them at the end.

## 3. Outputs

One `<alias>.h5` per scan:

```
/maps/mean, /maps/covariance          # moments
/maps/gauss1d/params                  # (ny, nx, 6) [A, sigma, mu, k, m, success]
/maps/gauss2p/{params1,params2,n_peaks,bic1,bic2}
/maps/gauss2d/params                  # (ny, nx, 8) [A, mu0, mu1, cov00, cov01, cov11, c, success]
attrs: scan_params, recipe_hash, device, versions, timestamp
```

plus `timeseries.h5` with per-scan scalars (`gauss1d_mu_median`,
`gauss1d_fwhm_median`, `two_peak_fraction`, ...) in scan-list order.

## 4. Read results back

```python
from starling.io import load_maps
maps, attrs = load_maps("results/t00.h5")

import h5py
with h5py.File("results/timeseries.h5") as f:
    fwhm = f["gauss1d_fwhm_median"][()]
    aliases = [a.decode() for a in f["alias"][()]]
```
