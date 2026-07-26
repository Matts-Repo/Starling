"""Physical map axes: extent, aspect ratio and scale bars.

The failure this guards against is silent: a map drawn on square pixels looks
perfectly plausible while being compressed ~2.9x along the beam direction.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from starling.properties import add_scalebar, imshow_map, map_extent, pixel_pair_um


PX_YX = (0.52255, 0.18117)   # MA7031 fn1 2x, aspect-corrected (um)


@pytest.fixture
def ax():
    fig, a = plt.subplots()
    yield a
    plt.close(fig)


# --------------------------------------------------------------------------- #
# pixel_pair_um
# --------------------------------------------------------------------------- #


def test_pixel_pair_scalar_is_isotropic():
    assert pixel_pair_um(0.2) == (0.2, 0.2)


def test_pixel_pair_passes_through_a_pair():
    assert pixel_pair_um(PX_YX) == pytest.approx(PX_YX)


def test_pixel_pair_accepts_arrays():
    assert pixel_pair_um(np.array(PX_YX)) == pytest.approx(PX_YX)


@pytest.mark.parametrize("bad", [(1.0, 2.0, 3.0), (0.0, 0.2), (0.2, -1.0),
                                 (np.nan, 0.2), np.inf])
def test_pixel_pair_rejects_unusable(bad):
    with pytest.raises(ValueError):
        pixel_pair_um(bad)


# --------------------------------------------------------------------------- #
# map_extent
# --------------------------------------------------------------------------- #


def test_map_extent_corner_origin_row0_at_top():
    left, right, bottom, top = map_extent((300, 750), PX_YX)
    assert (left, top) == (0.0, 0.0)
    assert right == pytest.approx(750 * PX_YX[1])
    assert bottom == pytest.approx(300 * PX_YX[0])
    assert bottom > top          # origin='upper' convention


def test_map_extent_encodes_the_aspect_correction():
    # 300 x 750 px of MA7031 fn1: looks 2.5:1 wide on square pixels, is
    # actually slightly TALLER than wide in the sample.
    left, right, bottom, top = map_extent((300, 750), PX_YX)
    assert (bottom - top) / (right - left) == pytest.approx(1.153, abs=1e-3)


def test_map_extent_isotropic_scalar_matches_raw_pixel_ratio():
    left, right, bottom, top = map_extent((300, 750), 0.18117)
    assert (right - left) / (bottom - top) == pytest.approx(2.5)


def test_map_extent_units():
    um = map_extent((10, 20), (1.0, 2.0), unit="um")
    mm = map_extent((10, 20), (1.0, 2.0), unit="mm")
    nm = map_extent((10, 20), (1.0, 2.0), unit="nm")
    assert mm == pytest.approx(tuple(v * 1e-3 for v in um))
    assert nm == pytest.approx(tuple(v * 1e3 for v in um))


def test_map_extent_centre_origin():
    left, right, bottom, top = map_extent((10, 20), (1.0, 2.0), centre=True)
    assert left == pytest.approx(-right)
    assert top == pytest.approx(-bottom)
    assert right - left == pytest.approx(40.0)


def test_map_extent_ignores_trailing_dims():
    assert map_extent((10, 20, 4), (1.0, 2.0)) == map_extent((10, 20), (1.0, 2.0))


def test_map_extent_bad_unit():
    with pytest.raises(ValueError, match="unit"):
        map_extent((10, 20), 1.0, unit="inch")


# --------------------------------------------------------------------------- #
# imshow_map integration
# --------------------------------------------------------------------------- #


def test_imshow_map_pixel_size_sets_physical_extent(ax):
    arr = np.zeros((300, 750))
    im = imshow_map(ax, arr, pixel_size=PX_YX, colorbar=False)
    assert im.get_extent() == pytest.approx(map_extent(arr.shape, PX_YX))


def test_imshow_map_renders_equal_aspect(ax):
    # aspect='equal' + a physical extent is what actually un-squashes the grain
    im = imshow_map(ax, np.zeros((300, 750)), pixel_size=PX_YX, colorbar=False)
    assert im.axes.get_aspect() in ("equal", 1.0)


def test_imshow_map_explicit_extent_wins(ax):
    im = imshow_map(ax, np.zeros((4, 4)), pixel_size=PX_YX,
                    extent=(0, 1, 1, 0), colorbar=False)
    assert im.get_extent() == pytest.approx((0, 1, 1, 0))


def test_imshow_map_without_pixel_size_stays_in_pixels(ax):
    im = imshow_map(ax, np.zeros((300, 750)), colorbar=False)
    left, right, bottom, top = im.get_extent()
    assert (right - left) == pytest.approx(750, abs=1.0)


def test_imshow_map_contour_overlay_shares_the_extent(ax):
    from starling.properties import EDGE_CLIPPED, OK

    status = np.full((20, 40), OK, dtype=np.int8)
    status[5:10, 5:10] = EDGE_CLIPPED
    imshow_map(ax, np.zeros((20, 40)), fit_status=status, pixel_size=PX_YX,
               colorbar=False)
    # the orange edge outline must land inside the physical axes, not at px 5
    right = 40 * PX_YX[1]
    assert ax.get_xlim()[1] == pytest.approx(right, rel=0.05)


# --------------------------------------------------------------------------- #
# add_scalebar
# --------------------------------------------------------------------------- #


def test_scalebar_on_physical_axes(ax):
    imshow_map(ax, np.zeros((300, 750)), pixel_size=PX_YX, colorbar=False)
    bar = add_scalebar(ax, length=20, unit="um")
    assert bar is not None
    span_um = 750 * PX_YX[1]
    assert bar.get_width() == pytest.approx(20 / span_um)


def test_scalebar_on_pixel_axes_needs_pixel_size(ax):
    imshow_map(ax, np.zeros((300, 750)), colorbar=False)
    bar = add_scalebar(ax, length=20, unit="um", pixel_size=PX_YX)
    span_um = 750 * PX_YX[1]
    assert bar.get_width() == pytest.approx(20 / span_um)


def test_scalebar_only_uses_the_horizontal_pixel_size(ax):
    # the bar is horizontal, so the sin(2theta)-stretched axis 0 is irrelevant
    imshow_map(ax, np.zeros((300, 750)), colorbar=False)
    b1 = add_scalebar(ax, length=20, pixel_size=PX_YX)
    b2 = add_scalebar(ax, length=20, pixel_size=(99.0, PX_YX[1]))
    assert b1.get_width() == pytest.approx(b2.get_width())


def test_scalebar_auto_length_is_a_round_number(ax):
    imshow_map(ax, np.zeros((300, 750)), pixel_size=PX_YX, colorbar=False)
    add_scalebar(ax)
    label = ax.texts[-1].get_text()
    value = float(label.split()[0])
    assert value in (1, 2, 5, 10, 20, 50, 100)
    assert value <= 750 * PX_YX[1] / 4


def test_scalebar_declines_when_it_would_not_fit(ax):
    imshow_map(ax, np.zeros((300, 750)), pixel_size=PX_YX, colorbar=False)
    assert add_scalebar(ax, length=1e6) is None


def test_scalebar_bad_unit(ax):
    imshow_map(ax, np.zeros((10, 10)), colorbar=False)
    with pytest.raises(ValueError, match="unit"):
        add_scalebar(ax, length=1, unit="inch", pixel_size=PX_YX)
