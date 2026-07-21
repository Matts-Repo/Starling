"""Effective pixel size from the ID03 thin-lens geometry + ffsel objective."""

import numpy as np
import pytest

from starling.io import NOMINAL_PIXEL_UM, effective_pixel_size, magnification
from starling.io._geometry import objective_magnification


def _motors(obx, mainx, obpitch, ffsel=-90.0):
    return {"obx": obx, "mainx": mainx, "obpitch": obpitch, "ffsel": ffsel}


def test_magnification_hand_computed():
    # d1 = 250/cos(20 deg), d2 = 5000/cos(20 deg) - d1 -> M = (5000-250)/250 = 19
    m = magnification(_motors(250.0, -5000.0, 20.0))
    assert m == pytest.approx((5000.0 - 250.0) / 250.0)


def test_magnification_pitch_invariant_ratio():
    # the cos(obpitch) factor cancels in d2/d1 when both distances share it
    m0 = magnification(_motors(300.0, -6000.0, 0.0))
    m1 = magnification(_motors(300.0, -6000.0, 25.0))
    assert m0 == pytest.approx(m1)
    assert m0 == pytest.approx((6000.0 - 300.0) / 300.0)


def test_objective_magnification_ffsel():
    assert objective_magnification({"ffsel": 0.0}) == 10.0
    assert objective_magnification({"ffsel": 0.3}) == 10.0   # within tolerance
    assert objective_magnification({"ffsel": -90.0}) == 2.0
    assert objective_magnification({"ffsel": 45.0}) == 2.0
    assert np.isnan(objective_magnification({}))


def test_effective_pixel_size_2x():
    eff = effective_pixel_size(_motors(250.0, -5000.0, 20.0, ffsel=-90.0))
    assert eff == pytest.approx(6.5 / (19.0 * 2.0))


def test_effective_pixel_size_10x():
    eff = effective_pixel_size(_motors(250.0, -5000.0, 20.0, ffsel=0.0))
    assert eff == pytest.approx(6.5 / (19.0 * 10.0))


def test_effective_pixel_size_real_2024_scan_values():
    # MA6043 316H scan 1.1: obx=258.9024, mainx=-5000, obpitch=20.1945,
    # ffsel=0 (10x in) -> beamline-quoted ~35 nm
    eff = effective_pixel_size(_motors(258.9024, -5000.0, 20.1945, ffsel=0.0))
    assert eff * 1000 == pytest.approx(35.5, abs=0.5)


def test_missing_ffsel_warns_assumes_2x():
    m = {"obx": 250.0, "mainx": -5000.0, "obpitch": 20.0}
    with pytest.warns(UserWarning, match="ffsel"):
        eff = effective_pixel_size(m)
    assert eff == pytest.approx(6.5 / (19.0 * 2.0))


def test_missing_motors_falls_back_to_nominal():
    with pytest.warns(UserWarning, match="falling back"):
        eff = effective_pixel_size({"obx": 250.0})
    assert eff == NOMINAL_PIXEL_UM


def test_degenerate_geometry_falls_back():
    # mainx too close: d2 <= 0
    with pytest.warns(UserWarning):
        eff = effective_pixel_size(_motors(300.0, -100.0, 0.0))
    assert eff == NOMINAL_PIXEL_UM
    assert not np.isfinite(magnification(_motors(300.0, -100.0, 0.0)))


def test_array_valued_motors_accepted():
    # invariant motors can arrive as 0-d numpy arrays from h5py
    m = {"obx": np.array(250.0), "mainx": np.array(-5000.0),
         "obpitch": np.array(20.0), "ffsel": np.array(0.0)}
    assert magnification(m) == pytest.approx(19.0)
    assert effective_pixel_size(m) == pytest.approx(6.5 / 190.0)
