"""fit_status classification: synthetic 4-category cases + range helpers."""

import numpy as np
import pytest

from starling.properties import (
    EDGE_CLIPPED,
    FAILED,
    NO_SIGNAL,
    OK,
    clamp_edge_estimate,
    classify_fit_status,
    edge_peak_mask,
    motor_ranges_steps,
)


def test_motor_ranges_steps_from_grid():
    chi = np.linspace(0.0, 1.2, 26)
    mu = np.linspace(-4.1, -2.6, 60)
    motors = np.array(np.meshgrid(chi, mu, indexing="ij"))
    ranges, steps = motor_ranges_steps(motors)
    assert ranges[0] == pytest.approx((0.0, 1.2))
    assert ranges[1] == pytest.approx((-4.1, -2.6))
    assert steps[0] == pytest.approx(1.2 / 25)
    assert steps[1] == pytest.approx(1.5 / 59)


def test_motor_ranges_steps_1d():
    ranges, steps = motor_ranges_steps(np.linspace(5.0, 6.0, 11))
    assert ranges == [pytest.approx((5.0, 6.0))]
    assert steps[0] == pytest.approx(0.1)


def test_four_categories_1d():
    # scan range [0, 10], step 1, tol 1 step
    ranges, steps = [(0.0, 10.0)], [1.0]
    mu = np.array([[5.0, 0.5, 10.5, 20.0, 5.0, 5.0]])
    success = np.array([[1.0, 1.0, 1.0, 1.0, 0.0, 1.0]])
    mask = np.array([[True, True, True, True, True, False]])
    st = classify_fit_status(mu, success, ranges, steps, mask=mask)
    assert st[0, 0] == OK            # centre mid-range
    assert st[0, 1] == EDGE_CLIPPED  # within 1 step of lo
    assert st[0, 2] == EDGE_CLIPPED  # just beyond hi, inside tol band
    assert st[0, 3] == FAILED        # far outside
    assert st[0, 4] == FAILED        # solver did not converge
    assert st[0, 5] == NO_SIGNAL     # outside mask


def test_nan_centre_is_failed():
    st = classify_fit_status(
        np.array([[np.nan]]), np.array([[1.0]]), [(0.0, 10.0)], [1.0]
    )
    assert st[0, 0] == FAILED


def test_nd_any_axis_flags_edge():
    # axis 0 fine, axis 1 at the boundary
    mu = np.zeros((1, 1, 2))
    mu[0, 0] = (5.0, 10.0)
    st = classify_fit_status(
        mu, np.ones((1, 1)), [(0.0, 10.0), (0.0, 10.0)], [1.0, 1.0]
    )
    assert st[0, 0] == EDGE_CLIPPED


def test_axes_subset():
    # only check axis 0; axis 1 out of range must be ignored
    mu = np.zeros((1, 1, 2))
    mu[0, 0] = (5.0, 99.0)
    st = classify_fit_status(
        mu, np.ones((1, 1)), [(0.0, 10.0)], [1.0], axes=[0]
    )
    assert st[0, 0] == OK


def test_clamp_edge_estimate():
    mu = np.array([[[-1.0, 12.0], [5.0, 5.0]]])
    out = clamp_edge_estimate(mu, [(0.0, 10.0), (0.0, 10.0)])
    assert out[0, 0].tolist() == [0.0, 10.0]
    assert out[0, 1].tolist() == [5.0, 5.0]
    assert mu[0, 0, 0] == -1.0  # input untouched


def test_edge_peak_mask_2d_motor_grid():
    # (1, 3, 10, 8) cube: pixel 0 peaks mid-grid, pixel 1 at the mu end,
    # pixel 2 at the chi start
    data = np.zeros((1, 3, 10, 8), dtype=np.uint16)
    data[0, 0, 5, 4] = 100
    data[0, 1, 5, 7] = 100
    data[0, 2, 0, 4] = 100
    em = edge_peak_mask(data, edge_bins=2)
    assert em.tolist() == [[False, True, True]]
    # restricting to axis 1 (the 8-long axis) drops the chi-edge pixel
    em1 = edge_peak_mask(data, edge_bins=2, axes=[1])
    assert em1.tolist() == [[False, True, False]]


def test_data_edge_rescues_failed_pixels():
    # two failed pixels: one with a truncated (edge-peaking) profile, one not
    mu = np.full((1, 2, 1), 50.0)  # diverged far outside [0, 10]
    success = np.zeros((1, 2))
    data_edge = np.array([[True, False]])
    st = classify_fit_status(
        mu, success, [(0.0, 10.0)], [1.0], data_edge=data_edge
    )
    assert st[0, 0] == EDGE_CLIPPED
    assert st[0, 1] == FAILED


def test_clamp_1d_shape_preserved():
    mu = np.array([[11.0, 5.0]])
    out = clamp_edge_estimate(mu, [(0.0, 10.0)])
    assert out.shape == mu.shape
    assert out[0, 0] == 10.0
