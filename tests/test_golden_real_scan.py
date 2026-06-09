"""Golden tests on real MA6278 scans — run only when the LaCie is mounted.

Defaults to the verified RAW_DATA paths; override with STARLING_GOLDEN_H5 /
STARLING_GOLDEN_SCAN to point at another BLISS master.
"""

import os

import numpy as np
import pytest

_RAW = "/Volumes/LaCie/ESRF_MA6278/RAW_DATA/DFXM_insitu_repaired_cell"
GOLDEN = os.environ.get(
    "STARLING_GOLDEN_H5",
    f"{_RAW}/DFXM_insitu_repaired_cell_g1_strain_sweep_test_before_electrochem_0004/"
    "DFXM_insitu_repaired_cell_g1_strain_sweep_test_before_electrochem_0004.h5",
)
SCAN = os.environ.get("STARLING_GOLDEN_SCAN", "1.1")
MOSA = f"{_RAW}/DFXM_insitu_repaired_cell_mosa_projection/DFXM_insitu_repaired_cell_mosa_projection.h5"

pytestmark = pytest.mark.skipif(
    not os.path.exists(GOLDEN), reason="real MA6278 data not reachable"
)


def test_golden_vendored_loader_matches_darling():
    """The vendored BLISS reader loads byte-identically to darling.DataSet."""
    darling = pytest.importorskip("darling")
    import starling

    roi = (900, 1100, 900, 1100)
    s = starling.DataSet(GOLDEN, scan_id=SCAN, roi=roi, verbose=False)
    d = darling.DataSet(GOLDEN, scan_id=SCAN, roi=roi, verbose=False)
    np.testing.assert_array_equal(s.data, d.data)
    np.testing.assert_allclose(s.motors, d.motors)
    assert list(s.scan_params["motor_names"]) == list(d.scan_params["motor_names"])
    assert list(s.scan_params["scan_shape"]) == list(d.scan_params["scan_shape"])
    assert s.scan_params["data_name"] == d.scan_params["data_name"]


def test_golden_strain_sweep_parity():
    """Strain sweep (ccmth x mu): moments + per-layer 1D fit vs darling."""
    darling = pytest.importorskip("darling")
    import starling

    sset = starling.DataSet(GOLDEN, scan_id=SCAN, verbose=False)
    sset.subtract(sset.estimate_background())

    mu_s, cov_s = sset.moments()
    mu_d, cov_d = darling.properties.moments(sset.data, sset.motors)
    np.testing.assert_allclose(mu_s, mu_d, rtol=1e-3, atol=1e-4)

    # fit ccmth rocking curves on the first mu layer
    layer = np.ascontiguousarray(sset.data[:, :, :, 0])
    ccmth = np.ascontiguousarray(sset.motors[0, :, 0])
    out_s = starling.properties.fit_1D_gaussian(layer, (ccmth,))
    out_d = darling.properties.curvefit.fit_1D_gaussian(layer, (ccmth,))

    # compare only pixels with real signal (the user's 50-count threshold)
    strong = layer.max(-1) > 50
    both = strong & (out_s[..., 5] > 0) & (out_d[..., 5] > 0)
    sane = both
    for o in (out_s, out_d):
        sane = sane & (o[..., 2] > ccmth[0]) & (o[..., 2] < ccmth[-1])
    assert sane.sum() > 500
    step = ccmth[1] - ccmth[0]
    dmu = np.abs(out_s[..., 2] - out_d[..., 2])[sane] / step
    assert np.percentile(dmu, 99) < 0.05  # measured: exact to float32 precision


def test_golden_partial_scan_loader():
    """The aborted mosa scan 2.1 loads as one complete chi row of 80 mu pts."""
    if not os.path.exists(MOSA):
        pytest.skip("mosa master not reachable")
    import starling

    dset = starling.DataSet(
        MOSA, scan_id="2.1", roi=(900, 1100, 900, 1100), allow_partial=True, verbose=False
    )
    assert dset.partial_info["complete"] is False
    assert dset.partial_info["frames_used"] == 80
    assert dset.data.shape == (200, 200, 1, 80)
    mu = dset.motors[1, 0]
    assert np.all(np.diff(mu) > 0)  # monotonic after snake-sort


def test_golden_partial_matches_darling_on_complete_scan():
    """On a complete scan the partial loader is byte-identical to darling."""
    if not os.path.exists(MOSA):
        pytest.skip("mosa master not reachable")
    darling = pytest.importorskip("darling")
    from starling.io import load_partial_scan

    roi = (900, 1100, 900, 1100)
    d = darling.DataSet(MOSA, scan_id="4.1", roi=roi, verbose=False)
    data, motors, info = load_partial_scan(MOSA, "4.1", roi=roi)
    assert info["complete"]
    np.testing.assert_array_equal(d.data, data)
    np.testing.assert_allclose(d.motors, motors)
