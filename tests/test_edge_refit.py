"""Edge-constrained refit: truncated peaks recover centres; classifier A-gate."""

import numpy as np
import pytest

from starling.properties import (
    EDGE_CLIPPED,
    FAILED,
    OK,
    GaussNDResult,
    classify_fit_status,
    fit_ND_fixed_cov,
    median_healthy_cov,
    refit_edge_pixels,
)


def _truncated_mosa(ny=6, nx=8, nchi=24, nmu=40, cut=6, seed=0):
    """Synthetic chi x mu scan whose mu peaks sit near/beyond the top edge.

    The full grid would span mu [0, 1] but only the first (nmu - cut) columns
    are 'acquired', so the true centres (drawn in [0.80, 0.95]) fall at or
    just beyond the acquired range's top edge.
    """
    rng = np.random.default_rng(seed)
    chi = np.linspace(0.0, 1.0, nchi)
    mu_full = np.linspace(0.0, 1.0, nmu)
    mu = mu_full[: nmu - cut]
    CHI, MU = np.meshgrid(chi, mu, indexing="ij")
    s_chi, s_mu = 0.10, 0.06
    c0 = rng.uniform(0.35, 0.65, (ny, nx))
    m0 = rng.uniform(0.80, 0.95, (ny, nx))
    data = np.empty((ny, nx, nchi, len(mu)))
    for j in range(ny):
        for i in range(nx):
            g = 4000 * np.exp(
                -0.5 * (((CHI - c0[j, i]) / s_chi) ** 2 + ((MU - m0[j, i]) / s_mu) ** 2)
            )
            data[j, i] = rng.poisson(g + 15)
    coords = np.array(np.meshgrid(chi, mu, indexing="ij"))
    cov = np.diag([s_chi ** 2, s_mu ** 2])
    return data.astype(np.uint16), coords, cov, c0, m0


def test_fixed_cov_recovers_truncated_centres():
    data, coords, cov, c0, m0 = _truncated_mosa()
    mask = np.ones(data.shape[:2], bool)
    fit = fit_ND_fixed_cov(data, coords, cov, mask, device="cpu")
    ok = fit["success"] > 0.5
    assert ok.mean() > 0.9
    mu_step = coords[1, 0, 1] - coords[1, 0, 0]
    err_chi = np.abs(fit["mu"][..., 0] - c0)[ok]
    err_mu = np.abs(fit["mu"][..., 1] - m0)[ok]
    # chi is fully contained: sub-step accuracy. mu is truncated: within ~1.5
    # steps for apexes up to ~1 sigma beyond the edge (matches the measured
    # real-data bias curve).
    assert np.median(err_chi) < 0.3 * mu_step
    assert np.median(err_mu) < 1.0 * mu_step
    assert np.percentile(err_mu, 90) < 2.5 * mu_step
    assert (fit["A"][ok] > 0).all()


def test_refit_edge_pixels_merges_and_flags():
    data, coords, cov, c0, m0 = _truncated_mosa()
    ny, nx = data.shape[:2]
    # a fake free-fit result: half the pixels "healthy", half diverged garbage
    A = np.full((ny, nx), 3000.0)
    mu = np.stack([c0, m0], axis=-1)
    covs = np.broadcast_to(cov, (ny, nx, 2, 2)).copy()
    c = np.full((ny, nx), 15.0)
    success = np.ones((ny, nx))
    bad = np.zeros((ny, nx), bool)
    bad[:, nx // 2:] = True
    mu = mu.copy()
    mu[bad] = (55.0, -40.0)  # runaway centres
    A[bad] = -5.0
    res = GaussNDResult(A=A, mu=mu, cov=covs, c=c, success=success)

    status = np.full((ny, nx), OK, dtype=np.int8)
    status[bad] = FAILED

    merged, refit_mask = refit_edge_pixels(data, coords, res, status, device="cpu")
    assert refit_mask.sum() > 0
    assert (refit_mask <= bad).all()          # only flagged pixels touched
    # healthy pixels untouched
    assert np.array_equal(merged.mu[~bad], res.mu[~bad])
    # refitted centres are physical now
    mu_step = coords[1, 0, 1] - coords[1, 0, 0]
    err = np.abs(merged.mu[refit_mask][:, 1] - m0[refit_mask])
    assert np.median(err) < 1.5 * mu_step
    assert (merged.A[refit_mask] > 0).all()


def test_median_healthy_cov_requires_ok_pixels():
    res = GaussNDResult(
        A=np.zeros((2, 2)), mu=np.zeros((2, 2, 2)),
        cov=np.zeros((2, 2, 2, 2)), c=np.zeros((2, 2)),
        success=np.zeros((2, 2)),
    )
    with pytest.raises(ValueError):
        median_healthy_cov(res, np.zeros((2, 2), np.int8))


def test_classifier_negative_amplitude_never_ok():
    # converged, centre mid-window, but amplitude negative -> not OK
    mu = np.array([[[0.5]]])
    success = np.array([[1.0]])
    st = classify_fit_status(mu[..., None][..., 0], success, [(0.0, 1.0)], [0.05],
                             A=np.array([[-8.2]]))
    assert st[0, 0] == FAILED
    # with a data-edge signature it degrades to EDGE_CLIPPED, not OK
    st2 = classify_fit_status(mu[..., None][..., 0], success, [(0.0, 1.0)], [0.05],
                              A=np.array([[-8.2]]), data_edge=np.array([[True]]))
    assert st2[0, 0] == EDGE_CLIPPED
    # positive amplitude stays OK
    st3 = classify_fit_status(mu[..., None][..., 0], success, [(0.0, 1.0)], [0.05],
                              A=np.array([[100.0]]))
    assert st3[0, 0] == OK


def test_gated_moment_resists_baseline_drag():
    from starling.properties._refit import _gated_moment

    rng = np.random.default_rng(3)
    n = 20000  # real cubes have ~20k voxels: aggregate baseline outweighs a spike
    x = np.linspace(0.0, 1.0, n)[:, None]
    # flat Poisson baseline + a sharp 2-point spike at the top edge
    y = rng.poisson(50, size=(1, n)).astype(float)
    y[0, -2:] += (20000.0, 30000.0)
    mu, A0, base = _gated_moment(y, x)
    # plain moment is dragged toward mid-window by baseline residuals; the
    # gated one must sit at the spike
    plain = ((y[0] - np.percentile(y[0], 20)).clip(0) @ x[:, 0]) / \
            (y[0] - np.percentile(y[0], 20)).clip(0).sum()
    assert abs(plain - 0.5) < 0.2           # demonstrates the drag
    assert mu[0, 0] > 0.95                  # gated moment sits at the spike
    assert A0[0] > 1000


def test_fallback_tier_covers_all_targets():
    data, coords, cov, c0, m0 = _truncated_mosa()
    ny, nx = data.shape[:2]
    # everything flagged FAILED with garbage free-fit values
    res = GaussNDResult(
        A=np.full((ny, nx), -1.0), mu=np.full((ny, nx, 2), 99.0),
        cov=np.broadcast_to(np.eye(2) * 1e-8, (ny, nx, 2, 2)).copy(),
        c=np.zeros((ny, nx)), success=np.zeros((ny, nx)),
    )
    status = np.full((ny, nx), FAILED, dtype=np.int8)
    merged, rmask, source = refit_edge_pixels(
        data, coords, res, status, cov=cov, device="cpu", return_source=True
    )
    # with the fallback tier every target carries a value
    assert rmask.all()
    assert set(np.unique(source[rmask])) <= {2, 3}
    # all centres inside window + margin
    for i in range(2):
        g = np.unique(coords[i])
        step = np.median(np.diff(g))
        assert (merged.mu[..., i] >= g.min() - 8 * step - 1e-9).all()
        assert (merged.mu[..., i] <= g.max() + 8 * step + 1e-9).all()
    # disabling the fallback reverts to GN-only coverage
    _, rmask2 = refit_edge_pixels(data, coords, res, status, cov=cov,
                                  device="cpu", fallback=None)
    assert rmask2.sum() <= rmask.sum()


def test_width_refinement_recovers_contained_axis_widths():
    """Stage 2 must recover per-pixel chi widths (contained axis) even though
    the prior covariance carries a single grain-median width."""
    rng = np.random.default_rng(7)
    ny, nx, nchi, nmu = 5, 8, 30, 34
    chi = np.linspace(0.0, 1.0, nchi)
    mu = np.linspace(0.0, 1.0, nmu)[:28]     # mu truncated at the top
    CHI, MU = np.meshgrid(chi, mu, indexing="ij")
    s_mu = 0.06
    s_chi_true = rng.uniform(0.05, 0.20, (ny, nx))   # per-pixel chi widths
    c0 = rng.uniform(0.35, 0.65, (ny, nx))
    m0 = rng.uniform(0.85, 0.95, (ny, nx))           # at/beyond the mu edge
    data = np.empty((ny, nx, nchi, len(mu)))
    for j in range(ny):
        for i in range(nx):
            g = 5000 * np.exp(-0.5 * (((CHI - c0[j, i]) / s_chi_true[j, i]) ** 2
                                      + ((MU - m0[j, i]) / s_mu) ** 2))
            data[j, i] = rng.poisson(g + 12)
    coords = np.array(np.meshgrid(chi, mu, indexing="ij"))
    prior = np.diag([0.10 ** 2, s_mu ** 2])   # grain-median chi width: 0.10

    res = GaussNDResult(
        A=np.full((ny, nx), -1.0), mu=np.full((ny, nx, 2), 99.0),
        cov=np.broadcast_to(prior, (ny, nx, 2, 2)).copy(),
        c=np.zeros((ny, nx)), success=np.zeros((ny, nx)),
    )
    status = np.full((ny, nx), FAILED, dtype=np.int8)

    merged, rmask, source = refit_edge_pixels(
        data.astype(np.uint16), coords, res, status, cov=prior, device="cpu",
        return_source=True,
    )
    ok = source == 2
    assert ok.mean() > 0.8
    s_chi_fit = np.sqrt(merged.cov[..., 0, 0])
    rel = np.abs(s_chi_fit - s_chi_true)[ok] / s_chi_true[ok]
    # widths measurably per-pixel, not stuck at the 0.10 prior
    assert np.median(rel) < 0.25
    spread = np.std(s_chi_fit[ok])
    assert spread > 0.02  # varies across pixels (prior would give spread 0)
    # truncated mu axis keeps the prior width exactly
    assert np.allclose(np.sqrt(merged.cov[ok][:, 1, 1]), s_mu, rtol=1e-6)
