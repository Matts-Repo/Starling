import numpy as np
import pytest

import darling
import starling


@pytest.mark.parametrize("device", ["cpu", None])
def test_moments_1d_parity(device):
    rng = np.random.default_rng(7)
    data = (64000 * rng.random((64, 64, 40))).astype(np.uint16)
    x = np.linspace(4.7, 8.6, 40)
    coords = np.array([x])

    mu_d, cov_d = darling.properties.moments(data, coords)
    mu_s, cov_s = starling.properties.moments(data, coords, device=device)

    assert mu_s.shape == mu_d.shape
    assert cov_s.shape == cov_d.shape
    np.testing.assert_allclose(mu_s, mu_d, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(cov_s, cov_d, rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize("device", ["cpu", None])
def test_moments_2d_parity(device):
    rng = np.random.default_rng(11)
    data = (64000 * rng.random((32, 32, 13, 11))).astype(np.uint16)
    phi = np.linspace(-0.5, 0.5, 13)
    chi = np.linspace(-1.0, 1.0, 11)
    coords = np.array(np.meshgrid(phi, chi, indexing="ij"))

    mu_d, cov_d = darling.properties.moments(data, coords)
    mu_s, cov_s = starling.properties.moments(data, coords, device=device)

    assert mu_s.shape == mu_d.shape == (32, 32, 2)
    assert cov_s.shape == cov_d.shape == (32, 32, 2, 2)
    np.testing.assert_allclose(mu_s, mu_d, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(cov_s, cov_d, rtol=1e-3, atol=1e-6)


def test_moments_zero_pixels():
    data = np.zeros((4, 4, 10), dtype=np.uint16)
    data[0, 0] = 100
    x = np.linspace(0, 1, 10)
    coords = np.array([x])
    mu, cov = starling.properties.moments(data, coords, device="cpu")
    assert mu[1, 1] == 0.0
    assert cov[1, 1] == 0.0
