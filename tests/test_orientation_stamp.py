"""orientation_stamp (darfix-parity 2-D colour) + _style helpers."""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
pytest.importorskip("colorstamps")

from starling.properties import (  # noqa: E402
    EDGE_CLIPPED,
    FAILED,
    NO_SIGNAL,
    OK,
    imshow_map,
    masked_for_display,
    orientation_map,
    orientation_stamp,
    robust_limits,
    status_cmap,
)


def _mean_field(ny=20, nx=30):
    """Smooth synthetic COM field over 2 axes with a known gradient."""
    yy, xx = np.mgrid[0:ny, 0:nx]
    mean = np.zeros((ny, nx, 2))
    mean[..., 0] = 0.01 * xx  # chi COM ramps left-right
    mean[..., 1] = 0.02 * yy  # mu COM ramps top-bottom
    return mean


def test_stamp_shapes_and_range():
    mean = _mean_field()
    rgb, key, vrange = orientation_stamp(mean, key_size=64)
    assert rgb.shape == (20, 30, 3)
    assert key.shape == (64, 64, 3)
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0
    (lo0, hi0), (lo1, hi1) = vrange
    assert lo0 == pytest.approx(0.0) and hi0 == pytest.approx(0.01 * 29)
    assert lo1 == pytest.approx(0.0) and hi1 == pytest.approx(0.02 * 19)


def test_stamp_no_collapse_with_outlier():
    """Regression for the sat-scale collapse: one huge outlier must not wash
    the rest of the map into a single colour block (as the dynamic-percentile
    round wheel does when unmasked)."""
    mean = _mean_field()
    mask = np.ones(mean.shape[:2], dtype=bool)
    mean[0, 0] = (1e3, 1e3)  # absurd outlier
    mask[0, 0] = False       # ... excluded by the mask

    rgb, _, _ = orientation_stamp(mean, mask=mask)
    grain = rgb[mask]
    # colour variation across the grain survives: many distinct colours
    spread = grain.std(axis=0)
    assert (spread > 0.05).any()
    # the masked outlier is black
    assert (rgb[0, 0] == 0.0).all()


def test_stamp_nan_pixels_black():
    mean = _mean_field()
    mean[3, 4] = np.nan
    rgb, _, _ = orientation_stamp(mean)
    assert (rgb[3, 4] == 0.0).all()


def test_stamp_out_of_range_grey():
    mean = _mean_field()
    rgb, _, _ = orientation_stamp(mean, vrange=((0.0, 0.1), (0.0, 0.2)))
    hot = _mean_field()
    hot[5, 5] = (0.5, 0.5)  # outside the fixed range
    rgb2, _, _ = orientation_stamp(hot, vrange=((0.0, 0.1), (0.0, 0.2)))
    assert np.allclose(rgb2[5, 5], 0.2)


def test_stamp_rejects_single_motor():
    with pytest.raises(ValueError):
        orientation_stamp(np.zeros((4, 5)))


def test_orientation_map_mask_restores_contrast():
    """Regression for the "single block of colour" symptom: with starling's
    zeroed off-grain convention, the map-wide median sits at 0, every grain
    pixel deviates in the same direction, and the whole grain saturates into
    one hue. Passing the grain mask must restore in-grain colour variation."""
    ny, nx = 30, 40
    mean = np.zeros((ny, nx, 2))
    mask = np.zeros((ny, nx), dtype=bool)
    mask[10:20, 10:30] = True
    yy, xx = np.mgrid[0:10, 0:20]
    mean[10:20, 10:30, 0] = 0.5 + 0.001 * xx  # offset >> in-grain gradient
    mean[10:20, 10:30, 1] = 0.8 + 0.001 * yy

    _, rgb_masked, _ = orientation_map(mean, as_rgb=True, mask=mask)
    _, rgb_unmasked, _ = orientation_map(mean, as_rgb=True)

    var_masked = rgb_masked[mask].std(axis=0).max()
    var_unmasked = rgb_unmasked[mask].std(axis=0).max()
    assert var_masked > 3 * var_unmasked
    assert var_masked > 0.05


def test_status_cmap_bad_color():
    cm = status_cmap("viridis", bad_color="0.85")
    bad = cm(np.ma.masked_invalid([np.nan]))[0]
    assert bad[:3] == pytest.approx((0.85, 0.85, 0.85), abs=1e-6)


def test_masked_for_display():
    arr = np.arange(6.0).reshape(2, 3)
    ok = arr > 2
    out = masked_for_display(arr, ok)
    assert np.isnan(out[0]).all() and not np.isnan(out[1]).any()
    assert arr[0, 0] == 0.0  # input untouched


def test_robust_limits_symmetric():
    arr = np.random.default_rng(0).normal(0.0, 1.0, (50, 50))
    lo, hi = robust_limits(arr, symmetric=True)
    assert lo == -hi and hi > 0


def test_imshow_map_smoke():
    import matplotlib.pyplot as plt

    arr = np.random.default_rng(1).normal(size=(10, 12))
    st = np.full((10, 12), OK, dtype=np.int8)
    st[:, 0] = NO_SIGNAL
    st[:, 1] = FAILED
    st[:, 2] = EDGE_CLIPPED
    clamped = np.zeros((10, 12))

    fig, ax = plt.subplots()
    im = imshow_map(ax, arr, fit_status=st, cmap="strain",
                    show_clamped=clamped, cbar_label="test")
    assert im is not None
    disp = np.ma.filled(im.get_array().astype(float), np.nan)
    assert np.isnan(disp[:, 0]).all()   # no-signal hidden
    assert np.isnan(disp[:, 1]).all()   # failed hidden (grey overlay on top)
    assert (disp[:, 2] == 0.0).all()    # edge-clipped shows clamped value
    plt.close(fig)
