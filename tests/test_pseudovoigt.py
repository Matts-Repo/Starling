"""Pseudo-Voigt fit (Section 6)."""

import numpy as np
import pytest

import starling


def pv_stack(seed=5, ny=12, nx=12, N=121, eta_range=(0.3, 0.6)):
    rng = np.random.default_rng(seed)
    x = np.linspace(6.5, 7.5, N)
    A = rng.uniform(3000, 6000, (ny, nx))
    mu = rng.uniform(6.95, 7.05, (ny, nx))
    sig = rng.uniform(0.04, 0.06, (ny, nx))
    gam = rng.uniform(0.04, 0.06, (ny, nx))
    eta = rng.uniform(*eta_range, (ny, nx))
    r = x - mu[..., None]
    G = np.exp(-0.5 * r**2 / sig[..., None] ** 2)
    L = 1.0 / (1.0 + (r / gam[..., None]) ** 2)
    f = A[..., None] * ((1 - eta[..., None]) * G + eta[..., None] * L) + 20
    data = rng.poisson(np.clip(f, 0, None)).astype(np.uint16)
    return data, x, dict(A=A, mu=mu, sigma=sig, gamma=gam, eta=eta)


@pytest.mark.parametrize("device", ["cpu", None])
def test_pseudovoigt_recovery(device):
    data, x, truth = pv_stack()
    res = starling.properties.fit_1D_pseudo_voigt(data, (x,), device=device)
    ok = res.success > 0
    assert ok.mean() > 0.6
    assert np.median(np.abs(res.mu - truth["mu"])[ok]) < 0.01
    assert np.median(np.abs(res.eta - truth["eta"])[ok]) < 0.15
    assert res.fwhm.shape == (12, 12)


def test_pseudovoigt_reduces_to_gaussian():
    # pure Gaussian data: eta should collapse near 0 and sigma match the
    # Gaussian fit
    rng = np.random.default_rng(7)
    x = np.linspace(6.5, 7.5, 121)
    prof = 4000 * np.exp(-0.5 * ((x - 7.0) / 0.05) ** 2) + 15
    data = rng.poisson(np.clip(np.tile(prof, (10, 10, 1)), 0, None)).astype(np.uint16)
    pv = starling.properties.fit_1D_pseudo_voigt(data, (x,), device="cpu")
    g = starling.properties.fit_1D_gaussian(data, (x,), device="cpu")
    okp = pv.success > 0
    okg = g[..., 5] > 0
    assert np.median(pv.eta[okp]) < 0.2
    assert abs(np.median(pv.sigma[okp]) - np.median(g[..., 1][okg])) < 0.01


def test_pseudovoigt_eta_bounded():
    data, x, _ = pv_stack()
    res = starling.properties.fit_1D_pseudo_voigt(data, (x,), device="cpu")
    assert (res.eta >= 0).all() and (res.eta <= 1).all()
