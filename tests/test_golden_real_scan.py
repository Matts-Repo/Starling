"""Golden test on a real MA6278 scan — runs only when the data is reachable.

Point STARLING_GOLDEN_H5 at a BLISS master .h5 file and STARLING_GOLDEN_SCAN
at a scan id (default "1.1"), e.g.:

    export STARLING_GOLDEN_H5=/Volumes/LaCie/ESRF_MA6278/RAW_DATA/.../dataset.h5
    pytest tests/test_golden_real_scan.py -v
"""

import os

import numpy as np
import pytest

GOLDEN = os.environ.get("STARLING_GOLDEN_H5")
SCAN = os.environ.get("STARLING_GOLDEN_SCAN", "1.1")

pytestmark = pytest.mark.skipif(
    not (GOLDEN and os.path.exists(GOLDEN)),
    reason="STARLING_GOLDEN_H5 not set or file not reachable",
)


def test_golden_scan_parity():
    import darling
    import starling

    sset = starling.DataSet(GOLDEN, scan_id=SCAN, verbose=False)
    bg = sset.estimate_background()
    sset.subtract(bg)
    sset.auto_roi()

    mu_s, cov_s = sset.moments()
    mu_d, cov_d = darling.properties.moments(sset.data, sset.motors)
    np.testing.assert_allclose(mu_s, mu_d, rtol=1e-3, atol=1e-5)

    if sset.data.ndim == 3:  # rocking scan: compare the 1D fit too
        out_s = sset.fit_1D_gaussian()
        out_d = darling.properties.curvefit.fit_1D_gaussian(sset.data, sset.motors)
        x = np.asarray(sset.motors[0])
        both = (out_s[..., 5] > 0) & (out_d[..., 5] > 0)
        strong = both & (out_d[..., 0] > 10 * float(np.median(out_d[..., 0][both])))
        if strong.sum() > 100:
            step = float(np.abs(np.diff(x)).mean())
            dmu = np.abs(out_s[..., 2] - out_d[..., 2])[strong] / step
            assert np.median(dmu) < 0.05
