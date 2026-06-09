"""Ground-truth recovery tests for the 2-peak and 2D Gaussian fits."""

import numpy as np
import pytest

import starling


def two_peak_stack(seed=4, ny=32, nx=32, N=80, sep_steps=8.0, sigma_steps=(1.5, 2.5)):
    """Half the pixels bimodal with known separation, half single-peak."""
    rng = np.random.default_rng(seed)
    x = np.linspace(4.7385, 8.6385, N)
    step = x[1] - x[0]
    two = np.zeros((ny, nx), dtype=bool)
    two[:, nx // 2 :] = True

    A1 = rng.uniform(1000, 4000, (ny, nx))
    mu1 = rng.uniform(x[15], x[N // 2 - 5], (ny, nx))
    sigma = rng.uniform(sigma_steps[0] * step, sigma_steps[1] * step, (ny, nx))
    mu2 = mu1 + sep_steps * step
    A2 = A1 * rng.uniform(0.5, 1.0, (ny, nx))

    f = A1[..., None] * np.exp(-0.5 * (x - mu1[..., None]) ** 2 / sigma[..., None] ** 2) + 30
    f = f + np.where(
        two[..., None],
        A2[..., None] * np.exp(-0.5 * (x - mu2[..., None]) ** 2 / sigma[..., None] ** 2),
        0.0,
    )
    data = rng.poisson(np.clip(f, 0, None)).astype(np.uint16)
    return data, x, dict(two=two, mu1=mu1, mu2=mu2, sep=sep_steps * step)


@pytest.mark.parametrize("device", ["cpu", None])
def test_two_peak_classification(device):
    data, x, truth = two_peak_stack()
    out = starling.properties.fit_two_gaussians_1D(data, (x,), device=device)
    n_peaks = out["n_peaks"]

    two_detected = n_peaks == 2
    # resolved bimodal pixels (sep ~ 4 sigma) overwhelmingly classified
    # 2-peak, unimodal ones 1-peak
    assert two_detected[truth["two"]].mean() > 0.9
    assert two_detected[~truth["two"]].mean() < 0.05


@pytest.mark.parametrize("device", ["cpu"])
def test_two_peak_merged_limit(device):
    # peaks separated by only ~2 sigma blend into a single hump — detection
    # is fundamentally ambiguous there; require better-than-chance, not
    # perfection, and zero runaway false positives on unimodal pixels
    data, x, truth = two_peak_stack(sigma_steps=(3.5, 4.0))
    out = starling.properties.fit_two_gaussians_1D(data, (x,), device=device)
    two_detected = out["n_peaks"] == 2
    assert two_detected[truth["two"]].mean() > 0.5
    assert two_detected[~truth["two"]].mean() < 0.05


@pytest.mark.parametrize("device", ["cpu", None])
def test_two_peak_separation_recovery(device):
    data, x, truth = two_peak_stack(seed=9)
    out = starling.properties.fit_two_gaussians_1D(data, (x,), device=device)
    sel = (out["n_peaks"] == 2) & truth["two"]
    p2 = out["params2"]
    sep = p2[..., 5] - p2[..., 2]  # mu_hi - mu_lo (peaks are mu-sorted)
    err = np.abs(sep[sel] - truth["sep"])
    step = x[1] - x[0]
    assert np.median(err) < 0.2 * step
    # peak positions themselves
    err1 = np.abs(p2[..., 2] - truth["mu1"])[sel]
    assert np.median(err1) < 0.2 * step


def mosa_stack(seed=6, ny=24, nx=24, m=26, n=37):
    rng = np.random.default_rng(seed)
    chi = np.linspace(-0.5, 0.5, m)
    mu = np.linspace(7.0, 9.16, n)
    coords = np.array(np.meshgrid(chi, mu, indexing="ij"))
    c_chi = rng.uniform(-0.2, 0.2, (ny, nx))
    c_mu = rng.uniform(7.5, 8.6, (ny, nx))
    s_chi = rng.uniform(0.05, 0.12, (ny, nx))
    s_mu = rng.uniform(0.1, 0.25, (ny, nx))
    rho = rng.uniform(-0.5, 0.5, (ny, nx))
    A = rng.uniform(1000, 5000, (ny, nx))

    dchi = (coords[0] - c_chi[..., None, None]) / s_chi[..., None, None]
    dmu = (coords[1] - c_mu[..., None, None]) / s_mu[..., None, None]
    r = rho[..., None, None]
    q = (dchi**2 - 2 * r * dchi * dmu + dmu**2) / (1 - r**2)
    f = A[..., None, None] * np.exp(-0.5 * q) + 20
    data = rng.poisson(np.clip(f, 0, None)).astype(np.uint16)
    truth = dict(
        c_chi=c_chi,
        c_mu=c_mu,
        cov00=s_chi**2,
        cov11=s_mu**2,
        cov01=rho * s_chi * s_mu,
    )
    return data, coords, truth


@pytest.mark.parametrize("device", ["cpu", None])
def test_2d_gaussian_recovery(device):
    data, coords, truth = mosa_stack()
    out = starling.properties.fit_2D_gaussian(data, coords, device=device)
    ok = out[..., 7] > 0
    assert ok.mean() > 0.9

    for idx, key in ((1, "c_chi"), (2, "c_mu")):
        err = np.abs(out[..., idx] - truth[key])[ok]
        assert np.median(err) < 0.01, f"centre {key} median err {np.median(err):.4f}"

    for idx, key in ((3, "cov00"), (4, "cov01"), (5, "cov11")):
        err = np.abs(out[..., idx] - truth[key])[ok]
        scale = np.median(np.abs(truth[key]))
        assert np.median(err) < 0.1 * max(scale, 1e-3), f"{key} median err {np.median(err):.2e}"
