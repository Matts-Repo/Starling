"""Loader rework tests: multiprocess==serial parity, ROI parity, shared-memory
cleanup, and fallback-to-serial on any parallel-read failure.

All scans are synthetic BLISS-like masters (title + per-frame motors + a 3-D
detector stack), so these run without the data drive.
"""

import gc
import warnings

import h5py
import numpy as np
import pytest

from starling import DataSet
from starling.io import _mpread
from starling.io import _reader

DET = (16, 20)


def _write_snake(path, m=6, n=8, seed=0):
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as f:
        for k, obp in enumerate([0.30, 0.10, 0.20]):
            g = f.create_group(f"{k + 1}.1")
            g["title"] = f"fscan2d chi -1.0 0.05 {m} mu 2.0 0.01 {n} 0.1"
            chi = np.zeros((m, n))
            mu = np.zeros((m, n))
            for i in range(m):
                sweep = 2.0 + 0.01 * np.arange(n)
                if i % 2 == 1:
                    sweep = sweep[::-1]  # snake
                chi[i] = -1.0 + 0.05 * i + rng.normal(0, 1e-4, n)
                mu[i] = sweep + rng.normal(0, 1e-5, n)
            g["instrument/chi/value"] = chi.ravel()
            g["instrument/mu/data"] = mu.ravel()
            g["instrument/positioners/mu"] = 2.05  # scalar -> fallback used
            g["instrument/positioners/obpitch"] = obp
            data = rng.integers(0, 60000, (m * n, *DET), dtype=np.uint16)
            g.create_dataset("instrument/pco_ff/image", data=data, chunks=(1, *DET))
    return path


@pytest.fixture(scope="module")
def snake_master(tmp_path_factory):
    return str(_write_snake(tmp_path_factory.mktemp("mp") / "snake.h5"))


def _load(master, n_workers, **kw):
    return DataSet(master, scan_id="1.1", device="cpu", verbose=False,
                   n_workers=n_workers, **kw)


# --------------------------- mp == serial parity --------------------------- #


def test_multiprocess_matches_serial(snake_master):
    ser = _load(snake_master, n_workers=0)
    par = _load(snake_master, n_workers=2)
    assert np.array_equal(ser.data, par.data)
    assert np.array_equal(ser.motors, par.motors)
    assert ser.data.dtype == par.data.dtype == np.uint16
    assert par.motors.dtype == np.float64
    # parallel-loaded data must be writable in place (preprocessing mutates it)
    par.data[0, 0, 0, 0] += 1


def test_multiprocess_stacked_matches_serial(snake_master):
    ids = ["1.1", "2.1", "3.1"]
    kw = dict(scan_motor="instrument/positioners/obpitch", device="cpu",
              verbose=False)
    ser = DataSet(snake_master, scan_id=list(ids), n_workers=0, **kw)
    par = DataSet(snake_master, scan_id=list(ids), n_workers=2, **kw)
    assert np.array_equal(ser.data, par.data)
    assert np.array_equal(ser.motors, par.motors)
    # stack axis sorted by the stack motor
    assert np.allclose(ser.motors[2, 0, 0, :], [0.10, 0.20, 0.30])
    assert repr(ser.scan_params) == repr(par.scan_params)


def test_auto_enable_threshold(snake_master, monkeypatch):
    # auto (n_workers=None) goes parallel above the size threshold ...
    monkeypatch.setattr(_reader, "_MP_MIN_BYTES", 1)
    auto = _load(snake_master, n_workers=None)
    ser = _load(snake_master, n_workers=0)
    assert np.array_equal(auto.data, ser.data)
    # ... and stays serial below it (any _mpread use would raise -> warn)
    monkeypatch.setattr(_reader, "_MP_MIN_BYTES", 2**60)
    monkeypatch.setattr(_mpread, "read_scan_shm", _boom)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        below = _load(snake_master, n_workers=None)
    assert np.array_equal(below.data, ser.data)


def _boom(*a, **k):
    raise RuntimeError("must not be called")


# ------------------------------- ROI parity -------------------------------- #


@pytest.mark.parametrize("n_workers", [0, 2])
def test_roi_parity(snake_master, n_workers):
    roi = (3, 11, 5, 17)
    full = _load(snake_master, n_workers=0)
    part = _load(snake_master, n_workers=n_workers, roi=roi)
    r1, r2, c1, c2 = roi
    assert np.array_equal(part.data, full.data[r1:r2, c1:c2])
    assert np.array_equal(part.motors, full.motors)


# --------------------------- shared-memory cleanup ------------------------- #


def test_shared_memory_cleanup(snake_master, monkeypatch):
    from multiprocessing import shared_memory

    created, closed = [], []
    orig_shm = shared_memory.SharedMemory
    orig_close = _mpread._close_shm

    def spy_shm(*a, **kw):
        shm = orig_shm(*a, **kw)
        if kw.get("create") or (len(a) > 1 and a[1]):
            created.append(shm.name)
        return shm

    monkeypatch.setattr(_mpread.shared_memory, "SharedMemory", spy_shm)
    monkeypatch.setattr(_mpread, "_close_shm",
                        lambda shm: (closed.append(shm.name), orig_close(shm)))

    ds = _load(snake_master, n_workers=2)
    ref = ds.data.copy()
    assert len(created) == 1
    # the segment name is unlinked as soon as the load returns ...
    with pytest.raises(FileNotFoundError):
        orig_shm(name=created[0])
    # ... but the mapping stays valid while the array is referenced
    gc.collect()
    assert np.array_equal(ds.data, ref)
    assert not closed
    # dropping the last reference closes the mapping deterministically
    del ds
    gc.collect()
    assert closed == created


# --------------------------- fallback on failure --------------------------- #


def test_fallback_when_shared_memory_setup_fails(snake_master, monkeypatch):
    def broken(*a, **kw):
        raise OSError("no shared memory for you")

    monkeypatch.setattr(_mpread.shared_memory, "SharedMemory", broken)
    ser = _load(snake_master, n_workers=0)
    with pytest.warns(UserWarning, match="parallel read failed"):
        par = _load(snake_master, n_workers=2)
    assert np.array_equal(ser.data, par.data)
    assert np.array_equal(ser.motors, par.motors)


def test_fallback_when_worker_crashes(snake_master, monkeypatch):
    # empty destination chunks crash the workers themselves (the jobs are
    # built parent-side); the loader must warn and re-read serially
    monkeypatch.setattr(_mpread, "_split_contiguous",
                        lambda chunks, n: [[(10 ** 9, 10 ** 9 + 5)]])
    ser = _load(snake_master, n_workers=0)
    with pytest.warns(UserWarning, match="parallel read failed"):
        par = _load(snake_master, n_workers=2)
    assert np.array_equal(ser.data, par.data)


def test_fallback_when_stacked_pool_fails(snake_master, monkeypatch):
    monkeypatch.setattr(_mpread, "read_jobs_shm", _boom)
    ids = ["1.1", "2.1", "3.1"]
    kw = dict(scan_motor="instrument/positioners/obpitch", device="cpu",
              verbose=False)
    ser = DataSet(snake_master, scan_id=list(ids), n_workers=0, **kw)
    with pytest.warns(UserWarning, match="parallel stacked load failed"):
        par = DataSet(snake_master, scan_id=list(ids), n_workers=2, **kw)
    assert np.array_equal(ser.data, par.data)
    assert np.array_equal(ser.motors, par.motors)


# ------------------------------- motor dtype ------------------------------- #


def test_motor_dtype_unified_float64(snake_master):
    ds = _load(snake_master, n_workers=0)
    assert ds.motors.dtype == np.float64
    ids = ["1.1", "2.1", "3.1"]
    stacked = DataSet(snake_master, scan_id=ids,
                      scan_motor="instrument/positioners/obpitch",
                      device="cpu", verbose=False, n_workers=0)
    assert stacked.motors.dtype == np.float64
