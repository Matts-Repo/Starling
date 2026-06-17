"""Grain masking (S9) and non-destructive preview (S8)."""

import numpy as np
import pytest

import starling
from starling import preprocess


def _grain_stack(ny=40, nx=40, N=20):
    rng = np.random.default_rng(0)
    data = rng.poisson(3.0, (ny, nx, N)).astype(np.uint16)  # dim background
    # bright disk grain in the centre
    yy, xx = np.mgrid[0:ny, 0:nx]
    disk = (yy - 20) ** 2 + (xx - 18) ** 2 < 70
    data[disk] += rng.poisson(400, (disk.sum(), N)).astype(np.uint16)
    return data, disk


# ------------------------------ Section 9 ---------------------------------- #


def test_grain_mask_captures_grain():
    data, disk = _grain_stack()
    m = preprocess.grain_mask(data, threshold_rel=0.1)
    assert m.dtype == bool
    # the grain disk should be (almost) fully inside the mask
    assert m[disk].mean() > 0.95
    # and the mask should not flood the whole frame
    assert m.mean() < 0.4


def test_grain_mask_otsu():
    data, disk = _grain_stack()
    m = preprocess.grain_mask(data, method="otsu")
    assert m[disk].mean() > 0.9


def test_polygon_mask_square():
    # a square polygon from (5,5) to (15,15) in (x, y)
    verts = [(5, 5), (15, 5), (15, 15), (5, 15)]
    m = preprocess.polygon_mask((20, 20), verts)
    assert m.dtype == bool
    assert m[10, 10]  # inside
    assert not m[0, 0]  # outside
    # area roughly the square (10x10), allow boundary tolerance
    assert 80 < m.sum() < 130


@pytest.mark.parametrize("device", ["cpu", None])
def test_moments_mask_identical_on_grain(device):
    """moments(mask=) matches the unmasked result on grain pixels, zeros elsewhere."""
    rng = np.random.default_rng(2)
    chi = np.linspace(-0.5, 0.5, 9)
    mu = np.linspace(7.0, 9.0, 11)
    coords = np.array(np.meshgrid(chi, mu, indexing="ij"))
    data = rng.integers(0, 500, (10, 10, 9, 11)).astype(np.uint16)
    mask = np.zeros((10, 10), bool)
    mask[3:7, 3:7] = True

    mu_f, cov_f = starling.properties.moments(data, coords, device=device)
    mu_m, cov_m = starling.properties.moments(data, coords, mask=mask, device=device)
    assert np.allclose(mu_m[mask], mu_f[mask])
    assert np.allclose(cov_m[mask], cov_f[mask])
    assert np.allclose(mu_m[~mask], 0.0)
    assert np.allclose(cov_m[~mask], 0.0)

    # order=4: skew/kurtosis identical on grain too
    _, _, sk_f, ku_f = starling.properties.moments(data, coords, order=4, device=device)
    _, _, sk_m, ku_m = starling.properties.moments(
        data, coords, order=4, mask=mask, device=device
    )
    assert np.allclose(sk_m[mask], sk_f[mask])
    assert np.allclose(ku_m[mask], ku_f[mask])


@pytest.mark.parametrize("device", ["cpu", None])
def test_fit_mask_identical_on_grain(device):
    """Masked fit must match the unmasked fit on grain pixels, zeros elsewhere."""
    from test_fit_synthetic import mosa_stack

    data, coords, _ = mosa_stack()
    mask = np.zeros(data.shape[:2], bool)
    mask[8:16, 8:16] = True
    full = starling.properties.fit_2D_gaussian(data, coords, device=device)
    masked = starling.properties.fit_2D_gaussian(data, coords, mask=mask, device=device)
    assert np.allclose(masked[mask], full[mask])
    assert np.allclose(masked[~mask], 0.0)


# ------------------------------ Section 8 ---------------------------------- #


def test_preview_non_destructive():
    data, _ = _grain_stack()
    before = data.copy()
    pv = preprocess.preview(data, bg_n=5, hot_sigma=5.0, roi_threshold=0.05)
    assert np.array_equal(data, before)  # untouched
    for key in ("raw_zsum", "proc_zsum", "raw_frame", "proc_frame", "background", "roi", "hist"):
        assert key in pv
    assert pv["raw_zsum"].shape == data.shape[:2]
    # background subtraction lowers the z-sum
    assert pv["proc_zsum"].sum() < pv["raw_zsum"].sum()


def test_subtracted_hot_pixels_removed_non_destructive():
    data, _ = _grain_stack()
    before = data.copy()
    bg = preprocess.estimate_background(data, n_lowest=5)
    out = preprocess.subtracted(data, bg)
    assert np.array_equal(data, before)
    assert out is not data
    out2 = preprocess.hot_pixels_removed(data, n_sigma=5.0)
    assert np.array_equal(data, before)
    assert out2 is not data
