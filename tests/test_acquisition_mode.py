"""Acquisition-mode detection (report-only) in the 2-D reader.

``detect_acquisition_mode`` reports how an fscan2d was rastered: from the BLISS
``instrument/fscan_parameters/fast_motor_mode`` string when present (REWIND ->
raster, ZIGZAG -> zigzag), else inferred from the fast-motor positions in
acquisition order (all rows same direction -> raster; adjacent rows alternate
-> zigzag). It never changes the de-zigzag sort and never raises on odd scans.
"""

import h5py
import numpy as np
import pytest

from starling.io._reader import (
    _infer_mode_from_grid,
    detect_acquisition_mode,
)


def _write_fscan_params_scan(path, scan_id, fast_motor_mode):
    """Minimal h5 with only the fscan_parameters/fast_motor_mode string."""
    with h5py.File(path, "w") as f:
        grp = f.create_group(scan_id)
        fp = grp.create_group("instrument/fscan_parameters")
        fp.create_dataset("fast_motor_mode", data=np.bytes_(fast_motor_mode))


def _fast_grid(m, n, zigzag):
    """(m, n) fast-motor grid in acquisition order (diffry-like ramp)."""
    ramp = -4.25 + 0.15 * np.arange(n)
    grid = np.empty((m, n), dtype=np.float64)
    for i in range(m):
        grid[i] = ramp[::-1] if (zigzag and i % 2 == 1) else ramp
    return grid


def _write_inference_scan(path, scan_id, m, n, zigzag):
    """h5 with a title + fast-motor dataset but NO fscan_parameters."""
    grid = _fast_grid(m, n, zigzag)
    slow = np.repeat(0.1 + 0.2 * np.arange(m), n)  # chi (slow), acquisition order
    with h5py.File(path, "w") as f:
        grp = f.create_group(scan_id)
        grp.create_dataset(
            "title",
            data=np.bytes_(f"fscan2d chi 0.1 0.2 {m} diffry -4.25 0.15 {n} 1 1.0"),
        )
        grp.create_dataset("instrument/diffry/data", data=grid.reshape(-1))
        grp.create_dataset("instrument/chi/value", data=slow)
    return grid


# ------------------------------------------------------------------ #
# metadata-sourced
# ------------------------------------------------------------------ #


@pytest.mark.parametrize(
    "raw,expected",
    [("ZIGZAG", "zigzag"), ("REWIND", "raster")],
)
def test_metadata_mode(tmp_path, raw, expected):
    path = str(tmp_path / "meta.h5")
    _write_fscan_params_scan(path, "1.1", raw)
    res = detect_acquisition_mode(path, "1.1")
    assert res == {"mode": expected, "source": "metadata", "fast_motor_mode": raw}


def test_metadata_unknown_string_kept_raw(tmp_path):
    path = str(tmp_path / "meta.h5")
    _write_fscan_params_scan(path, "1.1", "SPIRAL")
    res = detect_acquisition_mode(path, "1.1")
    assert res["mode"] == "unknown"
    assert res["source"] == "metadata"
    assert res["fast_motor_mode"] == "SPIRAL"  # raw preserved


def test_metadata_accepts_open_handle(tmp_path):
    path = str(tmp_path / "meta.h5")
    _write_fscan_params_scan(path, "3.1", "ZIGZAG")
    with h5py.File(path, "r") as f:
        res = detect_acquisition_mode(f, "3.1")
    assert res["mode"] == "zigzag"
    assert res["source"] == "metadata"


# ------------------------------------------------------------------ #
# inferred (no fscan_parameters)
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("zigzag,expected", [(False, "raster"), (True, "zigzag")])
def test_inference_from_file(tmp_path, zigzag, expected):
    path = str(tmp_path / "infer.h5")
    _write_inference_scan(path, "1.1", m=15, n=20, zigzag=zigzag)
    res = detect_acquisition_mode(path, "1.1")
    assert res == {"mode": expected, "source": "inferred", "fast_motor_mode": None}


@pytest.mark.parametrize("zigzag,expected", [(False, "raster"), (True, "zigzag")])
def test_inference_from_supplied_motor_grid(tmp_path, zigzag, expected):
    # no fscan_parameters; pass acquisition-order motors (2, m, n) directly,
    # exactly as the reader hands them in from _read_motors.
    path = str(tmp_path / "infer2.h5")
    grid = _write_inference_scan(path, "1.1", m=6, n=8, zigzag=zigzag)
    motors = np.stack([np.zeros_like(grid), grid])  # (2, m, n): slow, fast
    res = detect_acquisition_mode(path, "1.1", motors=motors)
    assert res["mode"] == expected
    assert res["source"] == "inferred"


def test_infer_helper_directly():
    assert _infer_mode_from_grid(_fast_grid(4, 5, zigzag=False)) == "raster"
    assert _infer_mode_from_grid(_fast_grid(4, 5, zigzag=True)) == "zigzag"
    # single row / single column -> unknown (nothing to compare)
    assert _infer_mode_from_grid(_fast_grid(1, 5, zigzag=False)) == "unknown"
    assert _infer_mode_from_grid(_fast_grid(4, 1, zigzag=False)) == "unknown"


# ------------------------------------------------------------------ #
# robustness: never raises
# ------------------------------------------------------------------ #


def test_missing_scan_returns_none(tmp_path):
    path = str(tmp_path / "empty.h5")
    with h5py.File(path, "w") as f:
        f.create_group("1.1")  # no title, no motors, no fscan_parameters
    res = detect_acquisition_mode(path, "1.1")
    assert res == {"mode": "unknown", "source": "none", "fast_motor_mode": None}


def test_bad_path_returns_none():
    res = detect_acquisition_mode("/no/such/file.h5", "1.1")
    assert res == {"mode": "unknown", "source": "none", "fast_motor_mode": None}
