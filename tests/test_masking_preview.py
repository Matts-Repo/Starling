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


# ------------- grain-safe background / hot-pixel / diagnostics -------------- #


def _rocking_grain(ny=40, nx=44, nframes=20, amp=600, ped=20, seed=1):
    """Grain whose pixels peak in a MINORITY of frames, on a flat pedestal.

    Returns (uint16 data, footprint bool, true background-free integral / pixel).
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:ny, 0:nx]
    foot = (yy - ny // 2) ** 2 / 50.0 + (xx - nx // 2) ** 2 / 60.0 < 1
    k = np.arange(nframes)
    prof = np.exp(-0.5 * ((k - nframes // 2) / 2.0) ** 2)  # lit in ~7/20 frames
    clean = np.zeros((ny, nx, nframes))
    clean[foot] = amp * prof[None, :]
    data = rng.poisson(ped + clean).astype(np.uint16)
    return data, foot, clean.sum(-1)


def test_estimate_background_modes_shape_dtype():
    data, _, _ = _rocking_grain()
    for mode in ("mean", "median", "lowest", "pmedian", "percentile"):
        bg = preprocess.estimate_background(data, n_lowest=5, mode=mode, percentile=10)
        assert bg.shape == data.shape[:2]
        assert bg.dtype == data.dtype


def test_estimate_background_rounds_not_floors():
    # mean of (10, 11, 11) = 10.67 must ROUND to 11, not floor to 10 (NUM-01)
    d = np.empty((1, 1, 3), np.uint16)
    d[..., 0], d[..., 1], d[..., 2] = 10, 11, 11
    assert preprocess.estimate_background(d, n_lowest=3, mode="mean")[0, 0] == 11


def test_estimate_background_darks_overrides_mode():
    data, _, _ = _rocking_grain()
    darks = np.full((data.shape[0], data.shape[1], 4), 7, np.uint16)
    assert np.all(preprocess.estimate_background(data, darks=darks) == 7)
    img = np.full(data.shape[:2], 9, np.uint16)
    assert np.all(preprocess.estimate_background(data, darks=img) == 9)


def test_estimate_background_percentile_chunking_invariant():
    data, _, _ = _rocking_grain()
    a = preprocess.estimate_background(data, mode="percentile", percentile=10, chunk_rows=3)
    b = preprocess.estimate_background(data, mode="percentile", percentile=10, chunk_rows=10_000)
    assert np.array_equal(a, b)


def test_grain_signal_retained_flags_inflated_background():
    data, foot, _ = _rocking_grain()
    before = data.copy()
    bg_ok = preprocess.estimate_background(data, n_lowest=5, mode="lowest")
    d_ok = preprocess.grain_signal_retained(data, bg_ok, mask=foot)
    assert d_ok["retained"] > 0.9          # darkest-frames bg ~ as good as the floor
    bg_hi = np.full(data.shape[:2], 200, np.uint16)
    d_hi = preprocess.grain_signal_retained(data, bg_hi, mask=foot)
    assert d_hi["retained"] < 0.9          # an inflated bg eats grain vs the floor
    assert np.array_equal(data, before)    # non-destructive


def test_signed_zsum_clean_baseline_and_nondestructive():
    data, foot, true_int = _rocking_grain()
    before = data.copy()
    bg = preprocess.estimate_background(data, n_lowest=5, mode="lowest")
    sz = preprocess.signed_zsum(data, bg)
    assert np.array_equal(data, before)
    off = ~preprocess.grain_mask(data)
    assert abs(np.median(sz[off])) < 5            # noise cancels -> ~0 off-grain
    assert sz[foot].sum() > 0.8 * true_int[foot].sum()  # recovers most grain signal


def test_remove_hot_pixels_one_sided_preserves_interior_dark_feature():
    frame = np.full((1, 30, 30), 500, np.uint16)
    frame[0, 15, 15] = 50                          # genuine sharp dark void
    two = frame.copy(); preprocess.remove_hot_pixels(two, n_sigma=5)
    one = frame.copy(); preprocess.remove_hot_pixels(one, n_sigma=5, one_sided=True)
    assert two[0, 15, 15] > 100                    # two-sided FILLS the void (HP-02)
    assert one[0, 15, 15] == 50                    # one-sided leaves it intact


def test_remove_hot_pixels_min_sigma_prevents_floored_frame_flagging():
    # mostly-zero (floored) frame; one interior pixel 3 cts above its neighbours.
    frame = np.zeros((1, 30, 30), np.uint16)
    frame[0, 12:18, 12:18] = 100
    frame[0, 15, 15] = 103
    legacy = frame.copy(); preprocess.remove_hot_pixels(legacy, n_sigma=5)
    safe = frame.copy(); preprocess.remove_hot_pixels(safe, n_sigma=5, min_sigma=1.0)
    assert legacy[0, 15, 15] == 100   # collapsed MAD flags a 3-ct grain-texture pixel
    assert safe[0, 15, 15] == 103     # min_sigma floor leaves it (HP-01 fix)


def test_remove_hot_pixels_still_kills_real_zinger_on_grain():
    data, foot, _ = _rocking_grain()
    gi = np.argwhere(foot)[0]
    data[gi[0], gi[1], data.shape[2] // 2] = 60000   # zinger on a grain pixel
    preprocess.remove_hot_pixels(data, n_sigma=5, one_sided=True, min_sigma=1.0)
    assert data[gi[0], gi[1], data.shape[2] // 2] < 5000  # removed despite being on-grain
