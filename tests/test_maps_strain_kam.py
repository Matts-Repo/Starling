"""orientation vs mosaicity (S2), strain helpers (S3), KAM (S4)."""

import numpy as np
import pytest

import starling
from starling.properties import (
    mosaicity,
    orientation_map,
    strain_from_ccmth,
    strain_from_obpitch,
)
from starling.transforms import kam


# --------------------------- Section 2: maps ------------------------------- #


def test_orientation_recovers_mean():
    rng = np.random.default_rng(0)
    mean = np.stack([rng.normal(0.0, 0.1, (16, 16)), rng.normal(7.5, 0.1, (16, 16))], -1)
    om = orientation_map(mean, axes=(0, 1))
    assert om.shape == (16, 16, 2)
    assert np.allclose(om, mean)


def test_orientation_rgb_and_key():
    mean = np.zeros((8, 8, 2))
    sel, rgb, key = orientation_map(mean, as_rgb=True)
    assert rgb.shape == (8, 8, 3)
    assert key.ndim == 3 and key.shape[-1] == 3
    assert (rgb >= 0).all() and (rgb <= 1).all()


def test_mosaicity_scalar_recovers_total_spread():
    # diagonal covariance with known sigma_x, sigma_y
    sx, sy = 0.2, 0.3
    cov = np.zeros((10, 10, 2, 2))
    cov[..., 0, 0] = sx**2
    cov[..., 1, 1] = sy**2
    m = mosaicity(cov, mode="scalar")
    assert np.allclose(m, np.sqrt(sx**2 + sy**2))


def test_mosaicity_scalar_rotation_invariant():
    # adding chi-mu correlation must not change sqrt(trace)
    base = np.array([[0.04, 0.0], [0.0, 0.09]])
    corr = np.array([[0.04, 0.02], [0.02, 0.09]])
    a = mosaicity(np.broadcast_to(base, (4, 4, 2, 2)).copy(), mode="scalar")
    b = mosaicity(np.broadcast_to(corr, (4, 4, 2, 2)).copy(), mode="scalar")
    assert np.allclose(a, b)


def test_mosaicity_distinct_from_orientation():
    # same mean, different spread -> orientation identical, mosaicity differs
    mean = np.full((5, 5, 2), 1.0)
    cov_small = np.broadcast_to(0.01 * np.eye(2), (5, 5, 2, 2)).copy()
    cov_big = np.broadcast_to(0.25 * np.eye(2), (5, 5, 2, 2)).copy()
    assert np.allclose(orientation_map(mean), mean)
    assert mosaicity(cov_big, mode="scalar").mean() > mosaicity(cov_small, mode="scalar").mean()


def test_mosaicity_ellipse():
    cov = np.zeros((3, 3, 2, 2))
    cov[..., 0, 0] = 0.09  # larger -> major axis along x
    cov[..., 1, 1] = 0.04
    maj, mn, ang = mosaicity(cov, mode="ellipse")
    fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0))
    assert np.allclose(maj, fwhm * 0.3)
    assert np.allclose(mn, fwhm * 0.2)
    assert np.allclose(np.abs(ang) % 180, 0.0, atol=1e-6)  # major axis along x


def test_mosaicity_axes_subblock():
    # 3D cov mixing orientation (0,1) and strain (2): use only the (0,1) block
    cov = np.zeros((2, 2, 3, 3))
    cov[..., 0, 0] = 0.04
    cov[..., 1, 1] = 0.09
    cov[..., 2, 2] = 100.0  # strain axis, must be excluded
    m = mosaicity(cov, mode="scalar", axes=(0, 1))
    assert np.allclose(m, np.sqrt(0.13))


def test_mosaicity_1d_reduces_to_sqrt_var():
    var = np.full((4, 4), 0.0025)  # sigma = 0.05
    m = mosaicity(var, mode="scalar")
    assert np.allclose(m, 0.05)


# --------------------------- Section 3: strain ----------------------------- #


def test_strain_from_ccmth_formula():
    rng = np.random.default_rng(1)
    ccmth = 6.7227 + rng.normal(0, 1e-3, (8, 8))
    ref = 6.7227
    eps = strain_from_ccmth(ccmth, ccmth_0=ref)
    expect = (np.deg2rad(ccmth) - np.deg2rad(ref)) / np.tan(np.deg2rad(ref))
    assert np.allclose(eps, expect)


def test_strain_from_ccmth_default_median():
    ccmth = np.array([[6.70, 6.72], [6.74, 6.76]])
    eps = strain_from_ccmth(ccmth)
    ref = np.nanmedian(ccmth)
    assert np.allclose(eps, (np.deg2rad(ccmth) - np.deg2rad(ref)) / np.tan(np.deg2rad(ref)))


def test_strain_from_obpitch_formula():
    ob = np.array([[4.70, 4.72], [4.74, 4.76]])
    ref = 4.73
    eps = strain_from_obpitch(ob, obpitch_0=ref)
    expect = -(np.deg2rad(ob) - np.deg2rad(ref)) / (2.0 * np.tan(np.deg2rad(ref) / 2.0))
    assert np.allclose(eps, expect)


def test_strain_nan_propagates():
    ccmth = np.array([[6.70, np.nan], [6.74, 6.76]])
    eps = strain_from_ccmth(ccmth, ccmth_0=6.72)
    assert np.isnan(eps[0, 1])
    assert np.isfinite(eps[0, 0])


# --------------------------- Section 4: KAM -------------------------------- #


def test_kam_hand_computed():
    # 3x3 scalar orientation field, centre pixel
    field = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=float)
    out = kam(field, size=(3, 3))
    # centre pixel: 8 neighbours all 0, |1-0|=1 averaged over 8 -> 1.0
    assert np.isclose(out[1, 1], 1.0)


def test_kam_vector_field():
    field = np.zeros((5, 5, 2))
    field[2, 2] = [3.0, 4.0]  # L2 distance 5 to all-zero neighbours
    out = kam(field, size=(3, 3))
    assert np.isclose(out[2, 2], 5.0)


def test_kam_nan_skipped():
    field = np.zeros((4, 4))
    field[0, 0] = np.nan
    out = kam(field, size=(3, 3), min_neighbors=1)
    assert np.isnan(out[0, 0])  # NaN centre -> NaN
    assert np.isfinite(out[2, 2])


def test_kam_min_neighbors():
    field = np.zeros((3, 3))
    out = kam(field, size=(3, 3), min_neighbors=100)  # impossible -> all NaN
    assert np.isnan(out).all()
