"""2024-era (diffry) backward-compatibility validation on a real scan.

The MA6043 316H in-situ dataset predates the current beamline layout: its
fscan2d sweeps use ``diffry`` as the fast motor and the raster mode is recorded
as REWIND. This test loads a real scan through the full starling stack and
checks the grid reconstruction, acquisition-mode report and imaging geometry.

Slow + drive-gated: skipped unless the LaCie volume is mounted.
"""

import os

import numpy as np
import pytest

H5 = (
    "/Volumes/LaCie/ESRF_MA6043/316H_insitu_DCT_47N/"
    "316H_insitu_DCT_47N_maybe_g2_47N_3Dstrain_midlayer/"
    "316H_insitu_DCT_47N_maybe_g2_47N_3Dstrain_midlayer.h5"
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not os.path.exists(H5), reason="MA6043 2024 data not reachable"),
]

ROI = (900, 1100, 900, 1100)
N_CHI = 15
N_DIFFRY = 20
DIFFRY_START = -4.25
DIFFRY_STEP = 0.15


def test_2024_diffry_scan_loads_and_reconstructs():
    import starling

    dset = starling.DataSet(H5, scan_id="1.1", roi=ROI, verbose=False)

    # detector ROI (200x200) x scan grid (15 chi x 20 diffry)
    assert dset.data.shape == (200, 200, N_CHI, N_DIFFRY)
    assert dset.data.dtype == np.uint16
    assert dset.motors.shape == (2, N_CHI, N_DIFFRY)

    # fast motor (diffry) spans its commanded range and is monotonic per row
    # after the de-zigzag sort. Recorded encoder values, so allow one step of
    # slack at each end.
    fast = dset.motors[1]
    expected_max = DIFFRY_START + (N_DIFFRY - 1) * DIFFRY_STEP  # -1.4
    assert np.all(np.diff(fast, axis=1) > 0)  # every row monotonic increasing
    assert fast.min() == pytest.approx(DIFFRY_START, abs=DIFFRY_STEP)
    assert fast.max() == pytest.approx(expected_max, abs=DIFFRY_STEP)
    span = fast[0, -1] - fast[0, 0]
    assert span == pytest.approx((N_DIFFRY - 1) * DIFFRY_STEP, rel=0.02)

    # acquisition mode is read straight from BLISS metadata: REWIND -> raster
    mode = dset.reader.acquisition_mode
    assert mode["mode"] == "raster"
    assert mode["source"] == "metadata"
    assert mode["fast_motor_mode"] == "REWIND"

    # imaging geometry resolves from the invariant motors -> magnified pixel
    invariant = dset.scan_params["invariant_motors"]
    for name in ("obx", "mainx", "obpitch"):
        assert name in invariant
    px = dset.pixel_size_um
    assert np.isfinite(px)
    # ffsel=0 -> 10x objective in: beamline-quoted ~35 nm for this setup
    assert px * 1000 == pytest.approx(35.5, abs=1.0)
