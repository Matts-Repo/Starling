"""Auto-dispatch + named result objects (Section 7)."""

import numpy as np
import pytest

import starling
from starling import DataSet
from starling.properties import (
    Gauss1DResult,
    GaussNDResult,
    MomentResult,
    PseudoVoigtResult,
)


def _bare(data, motors, device="cpu"):
    ds = DataSet.__new__(DataSet)
    ds.data = data
    ds.motors = motors
    ds.device = starling.get_device(device)
    ds.reader = None
    ds.roi = None
    return ds


def _rock_stack(ny=10, nx=10, N=61):
    rng = np.random.default_rng(0)
    x = np.linspace(6.6, 7.4, N)
    prof = 3000 * np.exp(-0.5 * ((x - 7.0) / 0.05) ** 2) + 20
    data = rng.poisson(np.clip(np.tile(prof, (ny, nx, 1)), 0, None)).astype(np.uint16)
    return data, np.array([x])


def test_dispatch_1d_returns_gauss1d():
    data, motors = _rock_stack()
    res = _bare(data, motors).analyze(mask=None)
    assert isinstance(res, Gauss1DResult)
    assert res.mu.shape == (10, 10)
    assert res.fwhm.shape == (10, 10)


def test_dispatch_2d_returns_gaussND():
    from test_fit_synthetic import mosa_stack

    data, coords, _ = mosa_stack()
    res = _bare(data, coords).analyze(mask=None)
    assert isinstance(res, GaussNDResult)
    assert res.D == 2


def test_dispatch_3d_returns_gaussND_no_error():
    from test_fit_nd import gaussian_nd_stack

    data, coords, _ = gaussian_nd_stack()
    res = _bare(data, coords).analyze(mask=None)  # must NOT raise
    assert isinstance(res, GaussNDResult)
    assert res.D == 3
    assert (res.success > 0).mean() > 0.9


def test_forced_methods():
    data, motors = _rock_stack()
    ds = _bare(data, motors)
    assert isinstance(ds.analyze(method="moments", mask=None), MomentResult)
    assert isinstance(ds.analyze(method="gauss1d", mask=None), Gauss1DResult)
    assert isinstance(ds.analyze(method="pv", mask=None), PseudoVoigtResult)
    g2 = ds.analyze(method="gauss2p", mask=None)
    assert "n_peaks" in g2


def test_raw_matches_legacy_array():
    data, motors = _rock_stack()
    ds = _bare(data, motors)
    res = ds.analyze(method="gauss1d", mask=None)
    legacy = ds.fit_1D_gaussian()
    assert np.array_equal(res.raw, legacy)
    # named field access mirrors the column layout
    assert np.allclose(res.A, legacy[..., 0])
    assert np.allclose(res.sigma, legacy[..., 1])
    assert np.allclose(res.mu, legacy[..., 2])


def test_moment_result_order4():
    data, motors = _rock_stack()
    res = _bare(data, motors).analyze(method="moments", order=4, mask=None)
    assert res.skew is not None and res.kurtosis is not None


def test_wrong_dimensionality_errors_point_to_right_fn():
    from test_fit_synthetic import mosa_stack

    data, coords, _ = mosa_stack()
    ds = _bare(data, coords)
    with pytest.raises(ValueError, match="gaussND"):
        ds.analyze(method="gauss1d", mask=None)
    data1, motors1 = _rock_stack()
    with pytest.raises(ValueError, match="gauss1d"):
        _bare(data1, motors1).analyze(method="gaussND", mask=None)
