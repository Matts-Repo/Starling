import numpy as np
import pytest

import starling

darling = pytest.importorskip("darling", reason="parity reference (optional dev dep)")


def synthetic_stack(seed=3, ny=48, nx=48, N=40):
    """Gaussians + linear background + noise, in realistic motor coordinates."""
    rng = np.random.default_rng(seed)
    x = np.linspace(4.7385, 8.6385, N)  # mu motor range from MA6278
    A = rng.uniform(200, 5000, (ny, nx))
    mu = rng.uniform(x[5], x[-5], (ny, nx))
    sigma = rng.uniform(0.1, 0.5, (ny, nx))
    k = rng.uniform(-5, 5, (ny, nx))
    m = rng.uniform(10, 60, (ny, nx))
    f = A[..., None] * np.exp(
        -0.5 * (x - mu[..., None]) ** 2 / sigma[..., None] ** 2
    ) + k[..., None] * x + m[..., None]
    noisy = rng.poisson(np.clip(f, 0, None)).astype(np.uint16)
    return noisy, x, dict(A=A, mu=mu, sigma=sigma)


@pytest.mark.parametrize("device", ["cpu", None])
def test_gauss1d_parity_vs_darling(device):
    data, x, _truth = synthetic_stack()
    coords = (x,)

    out_d = darling.properties.curvefit.fit_1D_gaussian(data, coords)
    out_s = starling.properties.fit_1D_gaussian(data, coords, device=device)

    assert out_s.shape == out_d.shape == (*data.shape[:2], 6)

    both = (out_d[..., 5] > 0) & (out_s[..., 5] > 0)
    assert both.mean() > 0.85, f"joint success rate too low: {both.mean():.3f}"

    # darling's success flag only means "no singular solve" — diverged fits
    # (params at 1e12) are still flagged 1 by both codes. Compare on pixels
    # where both fits are physically sane, with robust quantiles: the rare
    # remaining outliers are pixels where the two implementations walked to
    # different (equally bad) local solutions on near-degenerate curves.
    span = x[-1] - x[0]

    def is_sane(out):
        return (
            (out[..., 2] > x[0])
            & (out[..., 2] < x[-1])
            & (np.abs(out[..., 1]) < span)
            & (np.abs(out[..., 1]) > 1e-4)
            & (out[..., 0] > 0)
            & (out[..., 0] < 10 * float(data.max()))
        )

    sane = both & is_sane(out_d) & is_sane(out_s)
    assert sane.mean() > 0.55, f"sane-fit fraction too low: {sane.mean():.3f}"

    step = x[1] - x[0]
    dmu = np.abs(out_s[..., 2] - out_d[..., 2])[sane] / step
    dsig = np.abs(out_s[..., 1] - np.abs(out_d[..., 1]))[sane] / np.abs(out_d[..., 1])[sane]
    dA = np.abs(out_s[..., 0] - out_d[..., 0])[sane] / out_d[..., 0][sane]
    assert np.percentile(dmu, 99) < 0.05, f"p99 mu error {np.percentile(dmu, 99):.2e} steps"
    assert np.percentile(dsig, 99) < 0.01
    assert np.percentile(dA, 99) < 0.01


@pytest.mark.parametrize("device", ["cpu", None])
def test_gauss1d_recovers_ground_truth(device):
    data, x, truth = synthetic_stack(seed=12)
    out = starling.properties.fit_1D_gaussian(data, (x,), device=device)
    ok = out[..., 5] > 0
    assert ok.mean() > 0.9
    # strong peaks should localise to well under a motor step
    strong = ok & (truth["A"] > 1000)
    err = np.abs(out[..., 2] - truth["mu"])[strong]
    assert np.median(err) < 0.1 * (x[1] - x[0])


def test_gauss1d_mask():
    data, x, _ = synthetic_stack(seed=5, ny=16, nx=16)
    mask = np.zeros(data.shape[:2], dtype=bool)
    mask[:8] = True
    out = starling.properties.fit_1D_gaussian(data, (x,), mask=mask, device="cpu")
    assert np.all(out[8:] == 0.0)
    assert (out[:8, :, 5] > 0).mean() > 0.9


def test_gauss1d_degenerate_pixels():
    data = np.zeros((4, 4, 20), dtype=np.uint16)
    x = np.linspace(0, 1, 20)
    out = starling.properties.fit_1D_gaussian(data, (x,), device="cpu")
    assert np.all(out[..., 5] == 0)
    assert np.all(out[..., 0] == 0)
