"""N-D Gaussian fit: 3-D recovery and fit_ND(D=2) == fit_2D parity (Section 1)."""

import numpy as np
import pytest

import starling


def gaussian_nd_stack(seed=0, ny=10, nx=10, steps=(7, 8, 6)):
    """Synthetic 3-D strain-mosa cube with known per-pixel mu / covariance."""
    rng = np.random.default_rng(seed)
    a = np.linspace(-0.4, 0.4, steps[0])
    b = np.linspace(7.0, 8.2, steps[1])
    c = np.linspace(-0.3, 0.3, steps[2])
    coords = np.array(np.meshgrid(a, b, c, indexing="ij"))  # (3, *steps)

    c_a = rng.uniform(-0.15, 0.15, (ny, nx))
    c_b = rng.uniform(7.4, 7.8, (ny, nx))
    c_c = rng.uniform(-0.1, 0.1, (ny, nx))
    s_a = rng.uniform(0.08, 0.12, (ny, nx))
    s_b = rng.uniform(0.15, 0.25, (ny, nx))
    s_c = rng.uniform(0.06, 0.10, (ny, nx))
    A = rng.uniform(2000, 5000, (ny, nx))

    da = (coords[0] - c_a[..., None, None, None]) / s_a[..., None, None, None]
    db = (coords[1] - c_b[..., None, None, None]) / s_b[..., None, None, None]
    dc = (coords[2] - c_c[..., None, None, None]) / s_c[..., None, None, None]
    q = da**2 + db**2 + dc**2
    f = A[..., None, None, None] * np.exp(-0.5 * q) + 25
    data = rng.poisson(np.clip(f, 0, None)).astype(np.uint16)
    truth = dict(mu=np.stack([c_a, c_b, c_c], -1),
                 var=np.stack([s_a**2, s_b**2, s_c**2], -1))
    return data, coords, truth


@pytest.mark.parametrize("device", ["cpu", None])
def test_3d_gaussian_recovery(device):
    data, coords, truth = gaussian_nd_stack()
    res = starling.properties.fit_ND_gaussian(data, coords, device=device)
    assert res.D == 3
    assert res.mu.shape == (10, 10, 3)
    assert res.cov.shape == (10, 10, 3, 3)
    ok = res.success > 0
    assert ok.mean() > 0.9

    for k in range(3):
        cerr = np.abs(res.mu[..., k] - truth["mu"][..., k])[ok]
        assert np.median(cerr) < 0.01, f"centre axis {k} median err {np.median(cerr):.4f}"
        verr = np.abs(res.cov[..., k, k] - truth["var"][..., k])[ok]
        scale = np.median(truth["var"][..., k])
        assert np.median(verr) < 0.15 * scale, f"var axis {k} median err {np.median(verr):.2e}"


@pytest.mark.parametrize("device", ["cpu", None])
def test_fit_nd_degenerate_pixels_zeroed(device):
    """All-zero / single-bright voxels are flagged success=0 with zeroed outputs."""
    a = np.linspace(-0.4, 0.4, 7)
    b = np.linspace(7.0, 8.2, 8)
    c = np.linspace(-0.3, 0.3, 6)
    coords = np.array(np.meshgrid(a, b, c, indexing="ij"))
    data = np.zeros((6, 6, 7, 8, 6), dtype=np.uint16)
    data[1, 1, 3, 4, 2] = 1000  # single bright voxel -> zero-variance degenerate
    res = starling.properties.fit_ND_gaussian(data, coords, device=device)
    assert np.isfinite(res.raw).all()  # no NaN/inf leak
    assert (res.success == 0).all()
    assert np.allclose(res.A, 0.0)
    assert np.allclose(res.mu, 0.0)
    assert np.allclose(res.cov, 0.0)
    # zeroed covariance => mosaicity reports no spurious spread
    assert np.allclose(res.mosaicity(mode="scalar"), 0.0)


def test_fit_2d_degenerate_zeroed():
    """Legacy fit_2D_gaussian zeros degenerate-pixel params (cov cols 3,4,5)."""
    chi = np.linspace(-0.5, 0.5, 9)
    mu = np.linspace(7.0, 9.0, 11)
    coords = np.array(np.meshgrid(chi, mu, indexing="ij"))
    out = starling.properties.fit_2D_gaussian(
        np.zeros((6, 6, 9, 11), dtype=np.uint16), coords, device="cpu"
    )
    assert np.isfinite(out).all()
    assert (out[..., 7] == 0).all()
    assert np.allclose(out[..., 3:6], 0.0)  # covariance entries zeroed


@pytest.mark.parametrize("device", ["cpu", None])
def test_fit_nd_d2_matches_fit_2d(device):
    """fit_ND_gaussian(D=2).raw must reproduce fit_2D_gaussian exactly."""
    from test_fit_synthetic import mosa_stack

    data, coords, _ = mosa_stack()
    res = starling.properties.fit_ND_gaussian(data, coords, device=device)
    arr = starling.properties.fit_2D_gaussian(data, coords, device=device)
    assert np.array_equal(res.raw, arr)
    # field access mirrors the legacy column layout
    assert np.allclose(res.A, arr[..., 0])
    assert np.allclose(res.mu[..., 0], arr[..., 1])
    assert np.allclose(res.cov[..., 0, 1], arr[..., 4])
    assert np.allclose(res.success, arr[..., 7])
