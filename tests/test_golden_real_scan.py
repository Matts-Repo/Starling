"""Golden tests on real MA6278 scans — run only when the drive is mounted.

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
