import numpy as np

import starling


def test_moments_zero_pixels():
    data = np.zeros((4, 4, 10), dtype=np.uint16)
    data[0, 0] = 100
    x = np.linspace(0, 1, 10)
    coords = np.array([x])
    mu, cov = starling.properties.moments(data, coords, device="cpu")
    assert mu[1, 1] == 0.0
    assert cov[1, 1] == 0.0
