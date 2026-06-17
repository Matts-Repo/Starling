"""Skewness & kurtosis (Section 5)."""

import numpy as np
import pytest

import starling


def _stack(profile, ny=8, nx=8):
    return np.tile(profile.astype(np.uint16), (ny, nx, 1))


@pytest.mark.parametrize("device", ["cpu", None])
def test_gaussian_skew_kurt_near_zero(device):
    x = np.linspace(-4, 4, 121)
    prof = (5000 * np.exp(-0.5 * (x / 0.8) ** 2)).round()
    data = _stack(prof)
    coords = np.array([x])
    mean, cov, skew, kurt = starling.properties.moments(data, coords, order=4, device=device)
    assert np.allclose(skew, 0.0, atol=1e-3)
    assert np.allclose(kurt, 0.0, atol=0.05)  # finite window -> small bias


def test_order4_does_not_change_mean_cov():
    x = np.linspace(-3, 3, 81)
    prof = (4000 * np.exp(-0.5 * (x / 0.7) ** 2)).round()
    data = _stack(prof)
    coords = np.array([x])
    mean2, cov2 = starling.properties.moments(data, coords, order=2, device="cpu")
    mean4, cov4, _, _ = starling.properties.moments(data, coords, order=4, device="cpu")
    assert np.array_equal(mean2, mean4)
    assert np.array_equal(cov2, cov4)


def test_skewed_distribution_positive_skew():
    x = np.linspace(-3, 4, 121)
    # right-skewed: sharp core + long right tail
    prof = (np.exp(-0.5 * (x / 0.5) ** 2) * (x > -0.6)
            + 0.4 * np.exp(-0.5 * ((x - 1.8) / 0.7) ** 2))
    data = _stack((5000 * prof).round())
    coords = np.array([x])
    _, _, skew, _ = starling.properties.moments(data, coords, order=4, device="cpu")
    assert np.median(skew) > 0.2


def test_order_validation():
    x = np.linspace(0, 1, 10)
    data = np.zeros((4, 4, 10), dtype=np.uint16)
    data[0, 0] = 100
    with pytest.raises(ValueError):
        starling.properties.moments(data, np.array([x]), order=3, device="cpu")
