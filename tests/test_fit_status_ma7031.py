"""Regression: fit_status on the MA7031 "bad fit" bundle.

The mosa_strain_2x_layer_scan_FN1_BC_002 scan has its grain rotated partially
out of the scanned mu range on the left side of the field of view: raw
profiles there peak at the mu boundary and the solver diverges/fails.
classify_fit_status + edge_peak_mask must label that region EDGE_CLIPPED
(not just FAILED), concentrated in the left half.

Skipped when the local example bundle is not present. Reads the full 10 GB
cube in row chunks to build the data-driven edge mask (~minutes) — a local
validation gate, not a CI test.
"""

import os

import numpy as np
import pytest

BUNDLE = (
    "/Users/matt/Lab/projects/DFXM/MA7031 example data/26-07-20 bad fit/"
    "mosa_strain_2x_layer_scan_FN1_BC_002/bundle.h5"
)

pytestmark = [
    pytest.mark.skipif(
        not os.path.exists(BUNDLE), reason="MA7031 example bundle not available"
    ),
    pytest.mark.slow,
]


def _chunked_edge_mask(cube, edge_bins, step=32):
    """edge_peak_mask equivalent, streamed over detector-row chunks."""
    ny, nx = cube.shape[:2]
    dims = cube.shape[2:]
    out = np.zeros((ny, nx), dtype=bool)
    for r0 in range(0, ny, step):
        blk = cube[r0:r0 + step][()]
        for ax, n in enumerate(dims):
            other = tuple(2 + i for i in range(len(dims)) if i != ax)
            prof = blk.sum(axis=other, dtype=np.int64)
            am = np.argmax(prof, axis=-1)
            out[r0:r0 + step] |= (am < edge_bins) | (am >= n - edge_bins)
    return out


def test_edge_clipped_dominates_left_half():
    import h5py

    try:
        import hdf5plugin  # noqa: F401  (registers bitshuffle for the cube)
    except ImportError:
        pass

    from starling.properties import (
        EDGE_CLIPPED,
        FAILED,
        classify_fit_status,
        motor_ranges_steps,
    )

    with h5py.File(BUNDLE, "r") as f:
        mu = f["results/fit_3d/mu"][()]
        success = f["results/fit_3d/success"][()]
        sig_mask = f["masks/sig_mask"][()].astype(bool)
        motors = f["motors"][()]
        data_edge = _chunked_edge_mask(f["data/cube"], edge_bins=3)

    ranges, steps = motor_ranges_steps(motors)
    status = classify_fit_status(
        mu, success, ranges, steps, mask=sig_mask, data_edge=data_edge
    )

    grain = sig_mask
    n_edge = int((status == EDGE_CLIPPED)[grain].sum())
    n_failed = int((status == FAILED)[grain].sum())
    n_grain = int(grain.sum())

    # ~40% of this grain failed; the bulk must now be labelled edge-clipped
    assert n_edge > 0.15 * n_grain
    assert n_edge > n_failed

    # spatial pattern: edge-clipped pixels concentrate in the left half
    cols = np.nonzero(status == EDGE_CLIPPED)[1]
    mid = status.shape[1] // 2
    assert (cols < mid).mean() > 0.6
