"""Unit tests for scripts/compare_darfix.py -- no darfix dependency.

These exercise the pure helpers (scan-command parsing, the difference
statistics, the scalar-spread proxy, report assembly) so they run in CI where
darfix is not installed. The darfix orchestration itself (run_darfix) is
integration-tested manually against a synthetic BLISS scan; see the module
docstring in scripts/compare_darfix.py.
"""
import importlib.util
import os
import tempfile

import h5py
import numpy as np
import pytest

# load the script as a module (it lives in scripts/, not an importable package)
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPEC = importlib.util.spec_from_file_location(
    "compare_darfix", os.path.join(_HERE, "scripts", "compare_darfix.py")
)
cd = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cd)


def _write_title(path, scan_id, title):
    with h5py.File(path, "w") as f:
        f[f"{scan_id}/title"] = title


def test_parse_scan_motors_fscan2d():
    tmp = tempfile.mkdtemp()
    p = os.path.join(tmp, "m.h5")
    _write_title(p, "1.1", "fscan2d chi -0.1 0.05 5 mu 0.4 0.01 7 0.1")
    motors = cd.parse_scan_motors(p, "1.1")
    assert [m["name"] for m in motors] == ["chi", "mu"]
    assert [m["npoints"] for m in motors] == [5, 7]
    assert motors[0]["start"] == pytest.approx(-0.1)
    assert motors[1]["step"] == pytest.approx(0.01)


def test_parse_scan_motors_fscan1d():
    tmp = tempfile.mkdtemp()
    p = os.path.join(tmp, "m.h5")
    _write_title(p, "2.1", "fscan mu 0.4 0.004 15 0.1")
    motors = cd.parse_scan_motors(p, "2.1")
    assert len(motors) == 1
    assert motors[0]["name"] == "mu"
    assert motors[0]["npoints"] == 15


def test_parse_scan_motors_ascan_interval_to_points():
    tmp = tempfile.mkdtemp()
    p = os.path.join(tmp, "m.h5")
    _write_title(p, "1.1", "ascan mu 0.0 1.0 10 0.1")  # 10 intervals -> 11 pts
    motors = cd.parse_scan_motors(p, "1.1")
    assert motors[0]["npoints"] == 11


def test_parse_scan_motors_rejects_unknown_command():
    tmp = tempfile.mkdtemp()
    p = os.path.join(tmp, "m.h5")
    _write_title(p, "1.1", "loopscan 100 0.1")
    with pytest.raises(ValueError):
        cd.parse_scan_motors(p, "1.1")


def test_compare_map_identical():
    a = np.linspace(0, 1, 100).reshape(10, 10)
    mask = np.ones_like(a, bool)
    st = cd.compare_map(a, a.copy(), mask, step=0.01)
    assert st["n_px"] == 100
    assert st["median_abs_diff"] == 0.0
    assert st["p95_abs_diff"] == 0.0
    assert st["pearson_r"] == pytest.approx(1.0)
    assert st["frac_within_1_step"] == 1.0


def test_compare_map_constant_offset():
    a = np.random.default_rng(0).normal(size=(8, 8))
    b = a + 0.5
    mask = np.ones_like(a, bool)
    st = cd.compare_map(a, b, mask, step=None)
    assert st["median_abs_diff"] == pytest.approx(0.5, abs=1e-9)
    assert st["mean_diff"] == pytest.approx(-0.5, abs=1e-9)
    assert st["pearson_r"] == pytest.approx(1.0)  # perfectly correlated
    assert st["frac_within_1_step"] is None


def test_compare_map_within_step_fraction():
    a = np.zeros((10, 10))
    b = np.zeros((10, 10))
    b.flat[:50] = 2.0  # half the pixels differ by 2, half by 0
    mask = np.ones_like(a, bool)
    st = cd.compare_map(a, b, mask, step=1.0)  # step 1 -> only the 0-diff half pass
    assert st["frac_within_1_step"] == pytest.approx(0.5)


def test_compare_map_respects_mask_and_nan():
    a = np.ones((4, 4))
    b = np.ones((4, 4))
    b[0, 0] = np.nan  # excluded by finiteness
    mask = np.ones_like(a, bool)
    mask[1, 1] = False  # excluded by mask
    st = cd.compare_map(a, b, mask)
    assert st["n_px"] == 16 - 2


def test_compare_map_empty_mask():
    a = np.ones((4, 4))
    st = cd.compare_map(a, a, np.zeros_like(a, bool))
    assert st["n_px"] == 0
    assert st["median_abs_diff"] is None
    assert st["pearson_r"] is None


def test_geo_mean_spread():
    f1 = np.full((3, 3), 4.0)
    f2 = np.full((3, 3), 9.0)
    sp = cd._geo_mean_spread([f1, f2])
    assert np.allclose(sp, 6.0)  # sqrt(4*9)


def test_geo_mean_spread_ignores_nonpositive():
    f1 = np.array([[4.0, 0.0]])
    f2 = np.array([[9.0, 9.0]])
    sp = cd._geo_mean_spread([f1, f2])
    assert sp[0, 0] == pytest.approx(6.0)
    # column 1: f1 is 0 (dropped) -> geo mean of the single positive value
    assert sp[0, 1] == pytest.approx(9.0)


def test_build_report_shapes_and_masks():
    ny, nx = 6, 5
    names = ["chi", "mu"]
    rng = np.random.default_rng(1)

    def m():
        return rng.normal(size=(ny, nx))

    ok = np.zeros((ny, nx), bool)
    ok[1:4, 1:4] = True
    grain = np.zeros((ny, nx), bool)
    grain[1:5, 1:4] = True

    star = {
        "n_motor_dims": 2,
        "motor_names": names,
        "motor_steps": {"chi": 0.05, "mu": 0.01},
        "zsum": m(),
        "grain": grain,
        "ok": ok,
        "fit_status": ok.astype(np.int8),
        "com_fit": {"chi": m(), "mu": m()},
        "com_mom": {"chi": m(), "mu": m()},
        "fwhm_fit": {"chi": np.abs(m()) + 1, "mu": np.abs(m()) + 1},
        "spread": np.abs(m()) + 1,
        "native_mosaicity": m(),
        "shape": (ny, nx),
    }
    valid = np.zeros((ny, nx), bool)
    valid[1:4, 1:4] = True
    dar = {
        "com_fit": {"chi": m(), "mu": m()},
        "fwhm_fit": {"chi": np.abs(m()) + 1, "mu": np.abs(m()) + 1},
        "com_mom": {"chi": m(), "mu": m()},
        "fwhm_mom": {"chi": np.abs(m()) + 1, "mu": np.abs(m()) + 1},
        "valid_fit": valid,
        "residuals": np.abs(m()),
        "spread": np.abs(m()) + 1,
    }

    class Args:
        master = "x.h5"
        roi = None
        bg_mode = "mean"
        bg_method = "median"
        hp_kernel = 3

    report, dump = cd.build_report(star, dar, "1.1", Args())
    assert report["scan"] == "1.1"
    assert set(report["motor_names"]) == set(names)
    # fit COM masks = starling ok AND darfix valid (both the 3x3 centre block)
    assert report["maps"]["com_fit_chi"]["n_px"] == int((ok & valid).sum())
    # moment COM mask = grain
    assert report["maps"]["com_mom_chi"]["n_px"] == int(grain.sum())
    assert "spread_fit" in report["maps"]
    assert any(n.startswith("CHARACTERISATION") for n in report["notes"])
    # every reported map yields a diff-map triple for dumping
    for key in report["maps"]:
        assert key in dump
        a, b, diff, msk = dump[key]
        assert a.shape == (ny, nx)


def test_build_report_shape_mismatch_is_flagged():
    ny, nx = 5, 5
    ones = np.ones((ny, nx))
    star = {
        "n_motor_dims": 1, "motor_names": ["mu"],
        "motor_steps": {"mu": 0.01}, "zsum": ones, "grain": ones.astype(bool),
        "ok": ones.astype(bool), "fit_status": ones.astype(np.int8),
        "com_fit": {"mu": ones}, "com_mom": {"mu": ones},
        "fwhm_fit": {"mu": ones}, "spread": ones,
        "native_mosaicity": None, "shape": (ny, nx),
    }
    dar = {
        "com_fit": {"mu": np.ones((4, 4))},  # wrong shape on purpose
        "fwhm_fit": {}, "com_mom": {}, "fwhm_mom": {},
        "valid_fit": np.ones((ny, nx), bool), "residuals": None,
    }

    class Args:
        master = "x.h5"; roi = None; bg_mode = "mean"
        bg_method = "median"; hp_kernel = 3

    report, _ = cd.build_report(star, dar, "1.1", Args())
    assert "error" in report["maps"]["com_fit_mu"]
