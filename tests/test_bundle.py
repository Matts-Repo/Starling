"""Portable bundle round-trip + support_count (synthetic, no real data).

DataSets are fabricated bare (``DataSet.__new__`` style of
tests/test_analyze_dispatch.py) or with a stub reader carrying scan_params,
so no h5 master files are needed.
"""

import h5py
import numpy as np
import pytest

import starling
from starling import DataSet, load_bundle, save_bundle
from starling.preprocess import support_count
from starling.properties import (
    Gauss1DResult,
    GaussNDResult,
    GaussNDTwoResult,
    MomentResult,
)

NY, NX = 7, 5
MOTOR_SHAPE = (4, 3)  # 2 motor dims, non-square on purpose


class _StubReader:
    def __init__(self, scan_params):
        self.scan_params = scan_params
        self.sensors = None


def _dset(seed=0, with_reader=True):
    rng = np.random.default_rng(seed)
    ds = DataSet.__new__(DataSet)
    ds.data = rng.integers(0, 900, (NY, NX, *MOTOR_SHAPE), dtype=np.uint16)
    m1 = np.linspace(-1.0, 1.0, MOTOR_SHAPE[0])
    m2 = np.linspace(4.7, 5.3, MOTOR_SHAPE[1])
    ds.motors = np.array(np.meshgrid(m1, m2, indexing="ij"))
    ds.device = starling.get_device("cpu")
    ds.roi = (900, 1100, 800, 1000)
    ds.scan_id = None
    ds.scan_motor = "instrument/positioners/obpitch"
    ds.h5file = "/data/visitor/ma6278/id03/master.h5"
    ds.reader = None
    if with_reader:
        ds.reader = _StubReader({
            "scan_command": "fscan2d chi -2.7 0.2 4 mu 4.7 0.3 3 1",
            "scan_shape": np.array(MOTOR_SHAPE),
            "motor_names": ["chi", "mu"],
            "integrated_motors": [False, False],
            "scan_id": ["2.1", "3.1", "4.1"],
        })
    return ds


def _spd_cov(ny, nx, D, rng, scale=0.05, floor=0.02):
    L = rng.normal(0.0, scale, (ny, nx, D, D))
    return np.einsum("...ij,...kj->...ik", L, L) + floor * np.eye(D)


def _gauss1d(seed=1):
    rng = np.random.default_rng(seed)
    return Gauss1DResult(
        A=rng.uniform(1.0, 2.0, (NY, NX)),
        sigma=rng.uniform(0.05, 0.2, (NY, NX)),
        mu=rng.normal(0.0, 0.1, (NY, NX)),
        k=rng.normal(0.0, 0.01, (NY, NX)),
        m=rng.uniform(0.0, 0.1, (NY, NX)),
        success=np.ones((NY, NX)),
    )


def _gaussnd(D=2, seed=2):
    rng = np.random.default_rng(seed)
    return GaussNDResult(
        A=rng.uniform(1.0, 2.0, (NY, NX)),
        mu=rng.normal(0.0, 0.1, (NY, NX, D)),
        cov=_spd_cov(NY, NX, D, rng),
        c=rng.uniform(0.0, 0.1, (NY, NX)),
        success=np.ones((NY, NX)),
    )


def _gaussnd_two(D=2, seed=5):
    rng = np.random.default_rng(seed)
    n_peaks = rng.integers(0, 3, (NY, NX)).astype(np.uint8)
    two = (n_peaks == 2).astype(np.float64)
    return GaussNDTwoResult(
        A1=rng.uniform(1.5, 2.0, (NY, NX)) * two,
        mu1=rng.normal(0.0, 0.1, (NY, NX, D)) * two[..., None],
        cov1=_spd_cov(NY, NX, D, rng) * two[..., None, None],
        A2=rng.uniform(0.5, 1.4, (NY, NX)) * two,
        mu2=rng.normal(0.3, 0.1, (NY, NX, D)) * two[..., None],
        cov2=_spd_cov(NY, NX, D, rng) * two[..., None, None],
        c=rng.uniform(0.0, 0.1, (NY, NX)) * two,
        n_peaks=n_peaks,
        bic1=rng.normal(-100.0, 10.0, (NY, NX)),
        bic2=rng.normal(-120.0, 10.0, (NY, NX)),
        success=two,
    )


def _moments(D=2, seed=3):
    rng = np.random.default_rng(seed)
    return MomentResult(
        mean=rng.normal(0.0, 0.1, (NY, NX, D)),
        covariance=_spd_cov(NY, NX, D, rng),
        skew=rng.normal(0.0, 0.2, (NY, NX, D)),
        kurtosis=rng.uniform(2.0, 4.0, (NY, NX, D)),
    )


def _results():
    sweep = [_gauss1d(seed=10 + i) for i in range(3)]
    rng = np.random.default_rng(7)
    return {
        "gauss1d": _gauss1d(),
        "gaussND": _gaussnd(),
        "gaussND_two": _gaussnd_two(),
        "moments": _moments(),
        "sweep": sweep,
        "strain_map": rng.normal(0.0, 1e-4, (NY, NX)).astype(np.float32),
    }


def _masks(seed=4):
    rng = np.random.default_rng(seed)
    return {
        "sig_mask": rng.random((NY, NX)) > 0.4,
        "ok_mask": rng.random((NY, NX)) > 0.2,
        "fit_mode": rng.integers(0, 3, (NY, NX)).astype(np.int8),
    }


def _assert_result_equal(loaded, orig):
    if isinstance(orig, list):
        assert isinstance(loaded, list) and len(loaded) == len(orig)
        for lo, o in zip(loaded, orig):
            _assert_result_equal(lo, o)
        return
    if isinstance(orig, np.ndarray):
        assert isinstance(loaded, np.ndarray)
        assert loaded.dtype == orig.dtype
        assert np.array_equal(loaded, orig)
        return
    assert type(loaded) is type(orig)
    for k, v in orig.to_dict().items():
        if v is None:
            continue
        lv = getattr(loaded, k)
        assert np.array_equal(np.asarray(lv), np.asarray(v)), k


PROV = {
    "dataset_name": "g1_strain_mosa_layer",
    "scan_type": "strain-mosa",
    "pixel_size_mm": [0.00065, 0.00065],
    "notes": "synthetic round-trip",
}


# ------------------------------- dense ------------------------------------- #


def test_dense_roundtrip(tmp_path):
    ds = _dset()
    results, masks = _results(), _masks()
    p = str(tmp_path / "bundle.h5")
    save_bundle(p, ds, results=results, masks=masks, provenance=PROV)
    b = load_bundle(p)

    # data cube byte-identical, dtype preserved
    assert b.data.dtype == np.uint16
    assert np.array_equal(b.data, ds.data)
    assert np.array_equal(b.dense(), ds.data)
    # motors
    assert np.array_equal(b.motors, np.asarray(ds.motors, np.float64))
    # z-sum preview always present
    zsum = ds.data.reshape(NY, NX, -1).sum(axis=-1, dtype=np.float64)
    assert np.array_equal(b.zsum, zsum)
    # masks: values and dtypes
    assert set(b.masks) == set(masks)
    for k in masks:
        assert b.masks[k].dtype == masks[k].dtype
        assert np.array_equal(b.masks[k], masks[k])
    # results: every kind reconstructed field-for-field
    assert set(b.results) == set(results)
    for k in results:
        _assert_result_equal(b.results[k], results[k])
    # provenance attrs
    for k, v in PROV.items():
        assert b.attrs[k] == v
    assert b.attrs["data_source"] == ds.h5file
    assert b.attrs["roi"] == [900, 1100, 800, 1000]
    assert b.attrs["scan_id"] == ["2.1", "3.1", "4.1"]
    assert b.attrs["scan_motor"] == ds.scan_motor
    assert b.attrs["scan_command"].startswith("fscan2d")
    assert b.attrs["scan_shape"] == list(MOTOR_SHAPE)
    assert b.attrs["device"] == "cpu"
    assert b.attrs["starling_version"] == starling.__version__
    assert b.attrs["bundle_format_version"] == 1
    assert b.attrs["masked_only"] is False or b.attrs["masked_only"] == 0
    assert b.detector_shape == (NY, NX)
    assert b.motor_shape == MOTOR_SHAPE
    # a bundle written without save_raw_crop has no raw cube
    assert b.raw_data is None


def test_dense_roundtrip_no_reader_no_compression(tmp_path):
    """A bare DataSet (no scan_params) and compression=None still round-trip."""
    ds = _dset(with_reader=False)
    p = str(tmp_path / "bundle.h5")
    save_bundle(p, ds, compression=None)
    b = load_bundle(p)
    assert np.array_equal(b.data, ds.data)
    assert b.attrs["data_source"] == ds.h5file
    assert "scan_command" not in b.attrs
    assert b.results == {} and b.masks == {}


def test_gauss1d_sweep_kind_and_types(tmp_path):
    ds = _dset()
    sweep = [_gauss1d(seed=20 + i) for i in range(4)]
    p = str(tmp_path / "bundle.h5")
    save_bundle(p, ds, results={"sweep": sweep})
    b = load_bundle(p)
    assert isinstance(b.results["sweep"], list)
    assert all(isinstance(r, Gauss1DResult) for r in b.results["sweep"])
    assert len(b.results["sweep"]) == 4


def test_refuses_raw_data_path(tmp_path):
    ds = _dset()
    bad = str(tmp_path / "RAW_DATA" / "b.h5")
    with pytest.raises(PermissionError):
        save_bundle(bad, ds)


# ----------------------------- masked_only --------------------------------- #


def test_masked_only_roundtrip(tmp_path):
    ds = _dset()
    masks = _masks()
    results = {"gaussND": _gaussnd()}
    p = str(tmp_path / "sparse.h5")
    # default mask = masks["sig_mask"]
    save_bundle(p, ds, results=results, masks=masks, provenance=PROV,
                masked_only=True)
    b = load_bundle(p)

    sig = masks["sig_mask"]
    assert b.data is None
    assert b.sparse_data.shape == (int(sig.sum()), *MOTOR_SHAPE)
    assert b.sparse_data.dtype == np.uint16
    assert np.array_equal(b.sparse_data, ds.data[sig])
    assert np.array_equal(b.pixel_indices, np.argwhere(sig))

    # dense reconstruction: stored curves in place, zero elsewhere
    dense = b.dense()
    assert dense.dtype == np.uint16
    assert np.array_equal(dense[sig], ds.data[sig])
    assert np.all(dense[~sig] == 0)
    # NaN fill
    dnan = b.dense(fill=np.nan)
    assert dnan.dtype == np.float32
    assert np.array_equal(dnan[sig], ds.data[sig].astype(np.float32))
    assert np.all(np.isnan(dnan[~sig]))

    # everything else identical to the dense form
    zsum = ds.data.reshape(NY, NX, -1).sum(axis=-1, dtype=np.float64)
    assert np.array_equal(b.zsum, zsum)  # z-sum from the FULL cube
    assert np.array_equal(b.motors, np.asarray(ds.motors, np.float64))
    for k in masks:
        assert np.array_equal(b.masks[k], masks[k])
    _assert_result_equal(b.results["gaussND"], results["gaussND"])
    assert b.attrs["masked_only"]
    for k, v in PROV.items():
        assert b.attrs[k] == v


def test_masked_only_explicit_mask_overrides(tmp_path):
    ds = _dset()
    masks = _masks()
    explicit = np.zeros((NY, NX), bool)
    explicit[2, 3] = explicit[5, 1] = True
    p = str(tmp_path / "sparse.h5")
    save_bundle(p, ds, masks=masks, masked_only=True, mask=explicit)
    b = load_bundle(p)
    assert b.sparse_data.shape == (2, *MOTOR_SHAPE)
    assert np.array_equal(b.pixel_indices, np.argwhere(explicit))
    assert np.array_equal(b.sparse_data, ds.data[explicit])


def test_masked_only_requires_a_mask(tmp_path):
    ds = _dset()
    with pytest.raises(ValueError, match="mask"):
        save_bundle(str(tmp_path / "b.h5"), ds, masked_only=True)
    with pytest.raises(ValueError, match="mask"):
        save_bundle(str(tmp_path / "b.h5"), ds, masked_only=True,
                    masks={"grain": np.ones((NY, NX), bool)})  # no sig_mask


# ------------------------------ save_raw_crop ------------------------------ #


def _write_mosa_master(path, scan_specs, det_shape=(44, 52),
                       blob=(24, 32, 6000.0, 2.0), m=4, n=5):
    """A synthetic BLISS-like fscan2d master readable by MosaScan.

    Each scan's detector stack is a CONSTANT image across all frames (a low
    ``base`` pedestal + an off-centre bright Gaussian ``blob``), so the frame
    permutation is irrelevant to pixel values and the raw cube can be verified
    by cropping the base image at a known absolute ROI. ``scan_specs`` maps
    ``scan_id -> (obpitch_value, base_level)``.
    """
    ny, nx = det_shape
    br, bc, amp, sigma = blob
    yy, xx = np.mgrid[0:ny, 0:nx]
    g2d = amp * np.exp(-((yy - br) ** 2 + (xx - bc) ** 2) / (2 * sigma ** 2))
    with h5py.File(path, "w") as f:
        for sid, (obp, base) in scan_specs.items():
            g = f.create_group(sid)
            g["title"] = f"fscan2d chi -1.0 0.05 {m} mu 2.0 0.01 {n} 0.1"
            chi = np.zeros((m, n))
            mu = np.zeros((m, n))
            for i in range(m):
                sweep = 2.0 + 0.01 * np.arange(n)
                if i % 2 == 1:
                    sweep = sweep[::-1]  # snake
                chi[i] = -1.0 + 0.05 * i
                mu[i] = sweep
            g["instrument/chi/value"] = chi.ravel()
            g["instrument/mu/data"] = mu.ravel()
            g["instrument/positioners/mu"] = 2.05
            g["instrument/positioners/obpitch"] = obp
            image = np.clip(base + g2d, 0, 65535).astype(np.uint16)
            data = np.broadcast_to(image, (m * n, ny, nx)).astype(np.uint16)
            g.create_dataset("instrument/pco_ff/image", data=data,
                             chunks=(1, ny, nx))
    return path


def _base_image(path, sid):
    """The (constant) raw detector image of one scan, read straight from h5."""
    with h5py.File(path, "r") as f:
        return f[f"{sid}/instrument/pco_ff/image"][0].astype(np.uint16)


def test_raw_crop_roundtrip_composes_two_crops(tmp_path):
    """save_raw_crop stores the RAW (pre-subtract) values at the composed
    LOAD_ROI + auto_roi ROI — verified against an independently-computed ROI."""
    h5 = str(tmp_path / "master.h5")
    _write_mosa_master(h5, {"1.1": (0.2, 40)})
    LOAD_ROI = (8, 36, 12, 46)

    ds = DataSet(h5, scan_id="1.1", roi=LOAD_ROI, device="cpu", verbose=False)
    # crop to the off-centre blob (a genuine second crop on top of LOAD_ROI)
    local = ds.auto_roi(threshold_rel=0.05, pad=4)
    lr1, lr2, lc1, lc2 = local
    # the TRUE absolute ROI, composed here without trusting final_roi()
    true_abs = (LOAD_ROI[0] + lr1, LOAD_ROI[0] + lr2,
                LOAD_ROI[2] + lc1, LOAD_ROI[2] + lc2)
    assert ds.final_roi() == true_abs          # final_roi composes correctly
    assert true_abs != LOAD_ROI                # the second crop actually bit

    ds.subtract(30)                            # processed data now != raw
    p = str(tmp_path / "b.h5")
    save_bundle(p, ds, save_raw_crop=True)
    b = load_bundle(p)

    assert b.raw_data is not None
    assert b.raw_data.dtype == np.uint16
    assert b.raw_data.shape == b.data.shape
    # independently: crop the base image at the true composed ROI
    r1, r2, c1, c2 = true_abs
    expected2d = _base_image(h5, "1.1")[r1:r2, c1:c2]
    assert np.array_equal(b.raw_data, np.broadcast_to(
        expected2d[:, :, None, None], b.raw_data.shape))
    # raw really is pre-subtraction (differs from the stored, processed cube)
    assert np.any(b.raw_data != b.data)
    # ROI recorded on the dataset attr + in provenance
    assert list(b.attrs["raw_crop_roi"]) == list(true_abs)


def test_raw_crop_stacked_preserves_stack_order(tmp_path):
    """Stacked raw cube is assembled in the loaded cube's (obp-sorted) order,
    catching the scan_id-ordering bug."""
    h5 = str(tmp_path / "master.h5")
    # obp deliberately unsorted; distinct base levels identify each sub-scan
    specs = {"1.1": (0.30, 100), "2.1": (0.10, 200), "3.1": (0.20, 300)}
    _write_mosa_master(h5, specs)
    LOAD_ROI = (8, 36, 12, 46)

    ds = DataSet(h5, scan_id=["1.1", "2.1", "3.1"],
                 scan_motor="instrument/positioners/obpitch",
                 roi=LOAD_ROI, device="cpu", verbose=False)
    local = ds.auto_roi(threshold_rel=0.05, pad=4)
    lr1, lr2, lc1, lc2 = local
    r1, r2 = LOAD_ROI[0] + lr1, LOAD_ROI[0] + lr2
    c1, c2 = LOAD_ROI[2] + lc1, LOAD_ROI[2] + lc2

    ds.subtract(50)
    p = str(tmp_path / "b.h5")
    save_bundle(p, ds, save_raw_crop=True)
    b = load_bundle(p)

    assert b.raw_data is not None
    assert b.raw_data.shape == b.data.shape
    # loaded stack axis is sorted by obp: 2.1 (0.10), 3.1 (0.20), 1.1 (0.30)
    order = ["2.1", "3.1", "1.1"]
    for k, sid in enumerate(order):
        expected2d = _base_image(h5, sid)[r1:r2, c1:c2]
        assert np.array_equal(
            b.raw_data[..., k],
            np.broadcast_to(expected2d[:, :, None, None], b.raw_data[..., k].shape),
        ), f"stack layer {k} != scan {sid}"
    assert np.any(b.raw_data != b.data)


def test_raw_crop_requires_builtin_reader(tmp_path):
    ds = _dset()  # _StubReader, not a built-in scan reader
    with pytest.raises(ValueError, match="save_raw_crop requires"):
        save_bundle(str(tmp_path / "b.h5"), ds, save_raw_crop=True)


def test_raw_crop_rejects_partial_scan(tmp_path):
    from starling.io._dataset import _PartialScan

    ds = _dset()
    ds.reader = _PartialScan(ds.data, ds.motors, ds.h5file, "1.1",
                             {"complete": False})
    ds.scan_id = "1.1"
    with pytest.raises(ValueError, match="save_raw_crop requires"):
        save_bundle(str(tmp_path / "b.h5"), ds, save_raw_crop=True)


# ------------------------------ support_count ------------------------------ #


def test_support_count_known_pattern():
    # (ny, nx, n_ccmth=4, n_chi=2, n_mu=3)
    data = np.zeros((2, 2, 4, 2, 3), np.uint16)
    # pixel (0,0): lit somewhere in ccmth planes 0 and 2
    data[0, 0, 0, 1, 2] = 10
    data[0, 0, 2, 0, 0] = 10
    # pixel (0,1): lit in every ccmth plane
    data[0, 1, :, 0, 1] = 5
    # pixel (1,0): lit in one plane only
    data[1, 0, 3, 1, 1] = 7
    # pixel (1,1): all dark
    out = support_count(data, motor_axis=0, threshold=0)
    assert out.dtype == np.int16
    assert out.shape == (2, 2)
    assert out[0, 0] == 2
    assert out[0, 1] == 4
    assert out[1, 0] == 1
    assert out[1, 1] == 0  # all-dark pixel -> 0


def test_support_count_other_axes():
    data = np.zeros((1, 1, 3, 4), np.uint16)
    data[0, 0, 1, :] = 9   # motor-0 plane 1 fully lit
    data[0, 0, :, 2] = 9   # motor-1 plane 2 lit at every motor-0 step
    assert support_count(data, motor_axis=0, threshold=0)[0, 0] == 3
    assert support_count(data, motor_axis=1, threshold=0)[0, 0] == 4


def test_support_count_threshold_exact_value_not_counted():
    data = np.full((1, 1, 3), 5, np.uint16)
    assert support_count(data, motor_axis=0, threshold=5)[0, 0] == 0  # strict >
    assert support_count(data, motor_axis=0, threshold=4)[0, 0] == 3


def test_support_count_single_motor_dim():
    data = np.zeros((2, 1, 5), np.uint16)
    data[0, 0, [0, 4]] = 3
    out = support_count(data, motor_axis=0, threshold=1)
    assert out[0, 0] == 2 and out[1, 0] == 0


def test_support_count_bad_axis():
    data = np.zeros((2, 2, 3, 4), np.uint16)
    with pytest.raises(ValueError, match="motor_axis"):
        support_count(data, motor_axis=2, threshold=0)
    with pytest.raises(ValueError, match="motor_axis"):
        support_count(data, motor_axis=-1, threshold=0)
