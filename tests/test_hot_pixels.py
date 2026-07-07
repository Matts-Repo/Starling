"""Hot-pixel removal: torch-vs-scipy parity, static/hybrid semantics, devices.

The torch backend must be pixel-identical (not just close) to the scipy
reference for integer inputs, across every option combination, on every
available device. CUDA tests are collected but skipped where unavailable —
they are the ESRF validation gate.
"""

import numpy as np
import pytest
import torch

from starling import preprocess


def _devices():
    devs = ["cpu"]
    if torch.backends.mps.is_available():
        devs.append("mps")
    devs.append(
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        )
    )
    return devs


def _stack(dtype, ny=44, nx=52, nf=14, seed=3):
    """Rocking grain + pedestal with planted single-frame zingers and a void.

    Values are integer-valued even for float dtypes, so threshold decisions
    are never borderline and scipy/torch parity is exact by construction.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:ny, 0:nx]
    foot = (yy - ny // 2) ** 2 / 40.0 + (xx - nx // 2) ** 2 / 55.0 < 1
    k = np.arange(nf)
    prof = np.exp(-0.5 * ((k - nf // 2) / 2.5) ** 2)
    clean = np.full((ny, nx, nf), 20.0)
    clean[foot] += 900.0 * prof[None, :]
    data = rng.poisson(clean).astype(np.float64)
    # single-frame zingers, off- and on-grain
    data[3, 5, 1] += 30000.0
    data[30, 40, 7] += 30000.0
    data[ny // 2, nx // 2, nf // 2] += 30000.0
    # a persistent dark void inside the grain (dead-ish pixel)
    data[ny // 2 + 2, nx // 2 + 1, :] = 1.0
    return data.astype(dtype), foot


# ------------------------- torch vs scipy parity ---------------------------- #


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("dtype", [np.uint16, np.float32])
@pytest.mark.parametrize("one_sided", [False, True])
@pytest.mark.parametrize("min_sigma", [None, 1.0])
@pytest.mark.parametrize("use_protect", [False, True])
def test_torch_scipy_parity(device, dtype, one_sided, min_sigma, use_protect):
    data, foot = _stack(dtype)
    protect = foot if use_protect else None
    ref = data.copy()
    preprocess.remove_hot_pixels(
        ref, n_sigma=5.0, one_sided=one_sided, min_sigma=min_sigma,
        protect=protect, backend="scipy",
    )
    out = data.copy()
    preprocess.remove_hot_pixels(
        out, n_sigma=5.0, one_sided=one_sided, min_sigma=min_sigma,
        protect=protect, backend="torch", device=device,
    )
    assert (ref != data).any()  # the filter actually removed something
    assert np.array_equal(out, ref)  # pixel-identical to the scipy reference


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("shape", [(1, 30, 30), (9, 9, 5), (2, 3, 4), (16, 18, 3, 4)])
def test_torch_scipy_parity_odd_shapes(device, shape):
    """Degenerate/odd shapes: 1-row detector, odd pixel count (odd-n MAD
    branch), tiny frames, and a 4-D (two-motor) stack."""
    rng = np.random.default_rng(1)
    data = rng.integers(0, 2000, shape).astype(np.uint16)
    data.reshape(shape[0], shape[1], -1)[
        shape[0] // 2, shape[1] // 2, 0
    ] = 60000  # one spike
    ref = data.copy()
    preprocess.remove_hot_pixels(ref, n_sigma=5.0, backend="scipy")
    out = data.copy()
    preprocess.remove_hot_pixels(out, n_sigma=5.0, backend="torch", device=device)
    assert np.array_equal(out, ref)


@pytest.mark.parametrize("device", _devices())
def test_torch_chunking_invariance(device):
    """Chunk boundaries must not change the result (per-frame statistics)."""
    data, _ = _stack(np.uint16)
    a = data.copy()
    preprocess.remove_hot_pixels(a, backend="torch", device=device)
    b = data.copy()
    preprocess.remove_hot_pixels(b, backend="torch", device=device, chunk_frames=3)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("device", _devices())
def test_median9_network_matches_numpy(device):
    """The min/max sorting network is an exact median-of-9 on every device."""
    rng = np.random.default_rng(0)
    x = rng.integers(0, 65535, (9, 63, 77)).astype(np.float32)
    planes = [torch.from_numpy(p.copy()).to(device) for p in x]
    got = preprocess._median9(planes).cpu().numpy()
    assert np.array_equal(got, np.median(x, axis=0).astype(np.float32))


def test_invalid_backend_and_method():
    data, _ = _stack(np.uint16)
    with pytest.raises(ValueError, match="backend"):
        preprocess.remove_hot_pixels(data.copy(), backend="numpy")
    with pytest.raises(ValueError, match="method"):
        preprocess.remove_hot_pixels(data.copy(), method="temporal")


# ------------------------- static / hybrid semantics ------------------------ #


def _static_stack(ny=40, nx=40, nf=16, seed=5):
    rng = np.random.default_rng(seed)
    data = rng.poisson(50.0, (ny, nx, nf)).astype(np.uint16)
    data[5, 7, :] = 60000  # persistent hot detector defect
    data[9, 9, :] = 0  # persistent dead pixel (dark defect)
    data[30, 33, 4] += 30000  # single-frame zinger
    return data


def test_static_fixes_defects_but_misses_zingers():
    data = _static_stack()
    preprocess.remove_hot_pixels(
        data, n_sigma=5.0, one_sided=True, min_sigma=1.0, method="static"
    )
    assert (data[5, 7, :] < 200).all()  # static defect filled in every frame
    assert data[9, 9, 0] == 0  # one-sided: dark defect untouched
    assert data[30, 33, 4] > 10000  # documented: static misses zingers


def test_static_two_sided_fills_dead_pixel():
    data = _static_stack()
    preprocess.remove_hot_pixels(
        data, n_sigma=5.0, one_sided=False, min_sigma=1.0, method="static"
    )
    assert (data[9, 9, :] > 10).all()  # two-sided fills the dead pixel


def test_static_respects_protect():
    data = _static_stack()
    protect = np.zeros(data.shape[:2], bool)
    protect[5, 7] = True
    preprocess.remove_hot_pixels(
        data, n_sigma=5.0, one_sided=True, min_sigma=1.0, method="static",
        protect=protect,
    )
    assert (data[5, 7, :] == 60000).all()  # protected defect untouched


@pytest.mark.parametrize("device", _devices())
def test_hybrid_fixes_defects_and_zingers(device):
    data = _static_stack()
    preprocess.remove_hot_pixels(
        data, n_sigma=5.0, one_sided=True, min_sigma=1.0, method="hybrid",
        device=device,
    )
    assert (data[5, 7, :] < 200).all()  # static defect gone
    assert data[30, 33, 4] < 1000  # zinger gone too


def test_frame_mode_kills_zinger_reference():
    """Sanity: the default per-frame mode does catch the single-frame zinger."""
    data = _static_stack()
    preprocess.remove_hot_pixels(
        data, n_sigma=5.0, one_sided=True, min_sigma=1.0, method="frame",
        device="cpu",
    )
    assert data[30, 33, 4] < 1000


def test_hot_pixels_removed_passthrough_nondestructive():
    data = _static_stack()
    before = data.copy()
    out = preprocess.hot_pixels_removed(
        data, n_sigma=5.0, one_sided=True, min_sigma=1.0, method="hybrid"
    )
    assert np.array_equal(data, before)
    assert out is not data
    assert (out[5, 7, :] < 200).all()


# ------------------- batch runner defaults reconciliation ------------------- #


class _FakeDataSet:
    """Minimal stand-in for DataSet so process_scan's preprocess block runs."""

    def __init__(self, file, scan_id=None, device=None, verbose=False):
        frame = np.full((30, 30), 500, np.uint16)
        data = np.repeat(frame[:, :, None], 6, axis=2)
        data[15, 15, :] = 50  # genuine interior dark void (every frame)
        data[3, 4, 2] = 60000  # single-frame zinger
        self.data = data
        self.device = "cpu"
        self.scan_params = {}


def _run_process_scan(monkeypatch, hot_cfg):
    from starling.batch import runner
    from starling.batch.recipe import Recipe

    monkeypatch.setattr(runner, "DataSet", _FakeDataSet)
    recipe = Recipe.from_dict({
        "output_dir": "/tmp/unused",
        "device": "cpu",
        "preprocess": {"hot_pixels": hot_cfg},
        "fits": [],
        "scans": [{"file": "/nonexistent.h5", "scan_id": "1.1"}],
    })
    dset_holder = {}
    orig = preprocess.remove_hot_pixels

    def spy(data, **kw):
        dset_holder["data"] = data
        return orig(data, **kw)

    monkeypatch.setattr(runner.preprocess, "remove_hot_pixels", spy)
    runner.process_scan(recipe.scans[0], recipe, "cpu")
    return dset_holder["data"]


def test_runner_uses_grain_safe_defaults(monkeypatch):
    """runner.process_scan must match the notebook/widget grain-safe settings:
    one_sided (dark void preserved) with the zinger still removed."""
    data = _run_process_scan(monkeypatch, {"enabled": True, "n_sigma": 5.0})
    assert data[15, 15, 0] == 50  # dark void NOT filled (one_sided=True)
    assert data[3, 4, 2] < 5000  # zinger removed


def test_runner_legacy_settings_reachable(monkeypatch):
    """Explicit recipe keys restore the legacy two-sided behaviour."""
    data = _run_process_scan(
        monkeypatch,
        {"enabled": True, "n_sigma": 5.0, "one_sided": False, "min_sigma": None},
    )
    assert data[15, 15, 0] > 100  # two-sided legacy fills the dark void
