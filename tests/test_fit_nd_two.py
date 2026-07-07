"""Two-component N-D Gaussian fit: ground-truth recovery, BIC/gate selection,
Jacobian correctness, and the singular-precision crash regression.

CUDA-first, device-parametrised: "cuda" runs explicitly on the GPU nodes and
auto-skips elsewhere; ``None`` auto-detects (mps locally, cuda in production).
"""

import numpy as np
import pytest
import torch

import starling
from starling.properties import GaussNDTwoResult
from starling.properties._init_guess import two_peak_seed_nd
from starling.properties._linalg import masked_inv_spd
from starling.properties._models import gaussND_two_const

DEVICES = [
    "cpu",
    None,  # auto-detect: cuda > mps > cpu
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA not available"
        ),
    ),
]


# ------------------------------- builders ---------------------------------- #


def two_peak_mosa_stack(seed=3, ny=16, nx=16, m=21, n=25, sep=(0.25, 0.45),
                        ratio=(0.6, 0.9), A_lo=2000, A_hi=4000):
    """2-D mosa grid; right half bimodal with known separation, left single."""
    rng = np.random.default_rng(seed)
    chi = np.linspace(-0.5, 0.5, m)
    mu = np.linspace(7.0, 8.2, n)
    coords = np.array(np.meshgrid(chi, mu, indexing="ij"))
    two = np.zeros((ny, nx), bool)
    two[:, nx // 2:] = True

    c1 = np.stack([rng.uniform(-0.25, -0.05, (ny, nx)),
                   rng.uniform(7.2, 7.5, (ny, nx))], -1)
    c2 = c1 + np.asarray(sep)
    s = np.stack([rng.uniform(0.04, 0.07, (ny, nx)),
                  rng.uniform(0.06, 0.10, (ny, nx))], -1)
    A1 = rng.uniform(A_lo, A_hi, (ny, nx))
    A2 = A1 * rng.uniform(*ratio, (ny, nx))

    def bump(c, A):
        d0 = (coords[0] - c[..., 0, None, None]) / s[..., 0, None, None]
        d1 = (coords[1] - c[..., 1, None, None]) / s[..., 1, None, None]
        return A[..., None, None] * np.exp(-0.5 * (d0 ** 2 + d1 ** 2))

    f = bump(c1, A1) + np.where(two[..., None, None], bump(c2, A2), 0.0) + 30
    data = rng.poisson(np.clip(f, 0, None)).astype(np.uint16)
    return data, coords, dict(two=two, c1=c1, c2=c2, sep=np.asarray(sep),
                              A1=A1, A2=A2, sig=s)


def two_peak_3d_stack(seed=0, ny=8, nx=8, steps=(9, 8, 11), sep_ax=2,
                      sep_frac=0.45, ratio=(0.6, 0.9)):
    """3-D strain-mosa cube; right half bimodal, separated ONLY along sep_ax.

    sep_ax=2 is the ccmth-split case: two lattice parameters at the same
    (chi, mu) orientation — the configuration that matters scientifically.
    """
    rng = np.random.default_rng(seed)
    axes = [np.linspace(-0.4, 0.4, steps[0]), np.linspace(7.0, 8.2, steps[1]),
            np.linspace(-0.3, 0.3, steps[2])]
    coords = np.array(np.meshgrid(*axes, indexing="ij"))
    two = np.zeros((ny, nx), bool)
    two[:, nx // 2:] = True
    spans = np.array([a.max() - a.min() for a in axes])
    c1 = np.stack([rng.uniform(a.min() + 0.3 * sp, a.min() + 0.45 * sp, (ny, nx))
                   for a, sp in zip(axes, spans)], -1)
    dvec = np.zeros(3)
    dvec[sep_ax] = sep_frac * spans[sep_ax]
    c2 = c1 + dvec
    s = np.stack([np.full((ny, nx), 0.07 * sp) for sp in spans], -1)
    A1 = rng.uniform(2000, 4000, (ny, nx))
    A2 = A1 * rng.uniform(*ratio, (ny, nx))

    def bump(c, A):
        q = sum(((coords[k] - c[..., k, None, None, None])
                 / s[..., k, None, None, None]) ** 2 for k in range(3))
        return A[..., None, None, None] * np.exp(-0.5 * q)

    f = bump(c1, A1) + np.where(two[..., None, None, None], bump(c2, A2), 0.0) + 25
    data = rng.poisson(np.clip(f, 0, None)).astype(np.uint16)
    return data, coords, dict(two=two, c1=c1, c2=c2, dvec=dvec, A1=A1, A2=A2)


# --------------------------- classification -------------------------------- #


@pytest.mark.parametrize("device", DEVICES)
def test_2d_two_peak_classification(device):
    data, coords, t = two_peak_mosa_stack()
    res = starling.properties.fit_ND_two_gaussians(
        data, coords, device=device, progress=False
    )
    assert isinstance(res, GaussNDTwoResult)
    det = res.n_peaks == 2
    assert det[t["two"]].mean() > 0.9
    assert det[~t["two"]].mean() < 0.05
    # unimodal pixels resolve to the single-Gaussian model, not to nothing
    assert (res.n_peaks[~t["two"]] == 1).mean() > 0.9
    # success is exactly the two-peak selection; non-selected fields are zeroed
    assert np.array_equal(res.success > 0.5, det)
    assert np.allclose(res.A1[~det], 0.0) and np.allclose(res.A2[~det], 0.0)
    assert np.allclose(res.mu1[~det], 0.0) and np.allclose(res.cov2[~det], 0.0)
    assert np.isfinite(res.mu1).all() and np.isfinite(res.cov1).all()


@pytest.mark.parametrize("device", DEVICES)
def test_2d_two_peak_recovery(device):
    data, coords, t = two_peak_mosa_stack(seed=7)
    res = starling.properties.fit_ND_two_gaussians(
        data, coords, device=device, progress=False
    )
    sel = (res.n_peaks == 2) & t["two"]
    assert sel.mean() > 0.4  # half the frame is bimodal and detected
    # peaks are amplitude-sorted (descending) => peak 1 is the A1 component
    assert (res.A1[sel] >= res.A2[sel]).all()
    for ax in range(2):
        err1 = np.abs(res.mu1[..., ax] - t["c1"][..., ax])[sel]
        err2 = np.abs(res.mu2[..., ax] - t["c2"][..., ax])[sel]
        assert np.median(err1) < 0.01
        assert np.median(err2) < 0.01
        # separation vector recovered (mu1 - mu2 = c1 - c2 = -sep)
        dmu, _ = res.separation()
        assert abs(np.median(dmu[sel][:, ax]) + t["sep"][ax]) < 0.01
    # amplitudes and widths in physical units
    assert np.median(np.abs(res.A1[sel] - t["A1"][sel]) / t["A1"][sel]) < 0.1
    var_truth = t["sig"][..., 0] ** 2
    assert np.median(
        np.abs(res.cov1[..., 0, 0][sel] - var_truth[sel]) / var_truth[sel]
    ) < 0.2
    # Mahalanobis separation is strongly super-critical for a 4-sigma split
    _, dist = res.separation()
    assert np.median(dist[sel]) > 3.0
    assert np.allclose(dist[~sel & ~t["two"]], 0.0)


@pytest.mark.parametrize("device", DEVICES)
def test_3d_ccmth_split(device):
    """Peaks differing only along the 3rd (strain) axis are found and the
    separation vector points along that axis only."""
    data, coords, t = two_peak_3d_stack()
    res = starling.properties.fit_ND_two_gaussians(
        data, coords, device=device, progress=False
    )
    assert res.D == 3
    det = res.n_peaks == 2
    assert det[t["two"]].mean() > 0.8
    assert det[~t["two"]].mean() < 0.05
    sel = det & t["two"]
    dmu, dist = res.separation()
    # separation lives on axis 2; axes 0/1 are consistent with zero
    assert abs(np.median(np.abs(dmu[sel][:, 2])) - t["dvec"][2]) < 0.02
    assert np.median(np.abs(dmu[sel][:, 0])) < 0.02
    assert np.median(np.abs(dmu[sel][:, 1])) < 0.05
    assert np.median(dist[sel]) > 2.0


@pytest.mark.parametrize("device", DEVICES)
def test_extreme_amplitude_ratio(device):
    """A 10:1 major:minor pair, well separated, is still classified 2-peak."""
    data, coords, t = two_peak_mosa_stack(
        seed=11, ratio=(0.1, 0.1), A_lo=3500, A_hi=4500
    )
    res = starling.properties.fit_ND_two_gaussians(
        data, coords, device=device, progress=False
    )
    det = res.n_peaks == 2
    assert det[t["two"]].mean() > 0.8
    assert det[~t["two"]].mean() < 0.05
    sel = det & t["two"]
    # amplitude sort puts the strong component first
    assert np.median(np.abs(res.A1[sel] - t["A1"][sel]) / t["A1"][sel]) < 0.15
    ratio = res.A2[sel] / res.A1[sel]
    assert 0.05 < np.median(ratio) < 0.2


def test_merged_single_limit():
    """Peaks 1.5 sigma apart blend into one hump: the conservative N-D BIC
    (10 extra parameters in 2-D) must select the single-Gaussian model, and a
    clean single peak must never be split."""
    rng = np.random.default_rng(5)
    ny, nx, m, n = 10, 10, 21, 25
    chi = np.linspace(-0.5, 0.5, m)
    mus = np.linspace(7.0, 8.2, n)
    coords = np.array(np.meshgrid(chi, mus, indexing="ij"))
    c1 = np.stack([np.full((ny, nx), -0.1), np.full((ny, nx), 7.5)], -1)
    sig = 0.09
    c2 = c1 + np.array([0.0, 1.5 * sig])
    s = np.stack([np.full((ny, nx), 0.06), np.full((ny, nx), sig)], -1)
    A = np.full((ny, nx), 3000.0)

    def bump(c, A_):
        d0 = (coords[0] - c[..., 0, None, None]) / s[..., 0, None, None]
        d1 = (coords[1] - c[..., 1, None, None]) / s[..., 1, None, None]
        return A_[..., None, None] * np.exp(-0.5 * (d0 ** 2 + d1 ** 2))

    merged = rng.poisson(bump(c1, A) + bump(c2, 0.9 * A) + 30).astype(np.uint16)
    res = starling.properties.fit_ND_two_gaussians(
        merged, coords, device="cpu", progress=False
    )
    assert (res.n_peaks == 2).mean() < 0.05
    assert (res.n_peaks == 1).mean() > 0.9

    single = rng.poisson(bump(c1, A) + 30).astype(np.uint16)
    res1 = starling.properties.fit_ND_two_gaussians(
        single, coords, device="cpu", progress=False
    )
    assert (res1.n_peaks == 2).mean() < 0.05


# --------------------------------- gates ----------------------------------- #


def test_separation_gate_blocks_subresolution():
    """True peaks 1 motor step apart: whatever BIC says, the > 2-motor-step
    separation gate must keep n_peaks at 1."""
    rng = np.random.default_rng(5)
    ny, nx, m, n = 10, 10, 21, 25
    chi = np.linspace(-0.5, 0.5, m)
    mus = np.linspace(7.0, 8.2, n)
    coords = np.array(np.meshgrid(chi, mus, indexing="ij"))
    step_mu = mus[1] - mus[0]
    c1 = np.stack([np.full((ny, nx), -0.1), np.full((ny, nx), 7.5)], -1)
    c2 = c1 + np.array([0.0, step_mu])
    s = np.stack([np.full((ny, nx), 0.06), np.full((ny, nx), 0.09)], -1)
    A = np.full((ny, nx), 3000.0)
    d0 = (coords[0] - c1[..., 0, None, None]) / s[..., 0, None, None]
    d1 = (coords[1] - c1[..., 1, None, None]) / s[..., 1, None, None]
    d0b = (coords[0] - c2[..., 0, None, None]) / s[..., 0, None, None]
    d1b = (coords[1] - c2[..., 1, None, None]) / s[..., 1, None, None]
    f = (A[..., None, None] * np.exp(-0.5 * (d0 ** 2 + d1 ** 2))
         + 0.8 * A[..., None, None] * np.exp(-0.5 * (d0b ** 2 + d1b ** 2)) + 30)
    data = rng.poisson(f).astype(np.uint16)
    res = starling.properties.fit_ND_two_gaussians(
        data, coords, device="cpu", progress=False
    )
    assert (res.n_peaks == 2).mean() == 0.0


def test_delta_bic_margin_honoured():
    """An absurd delta_bic margin turns every bimodal pixel into n_peaks=1."""
    data, coords, t = two_peak_mosa_stack(ny=8, nx=8)
    res = starling.properties.fit_ND_two_gaussians(
        data, coords, device="cpu", progress=False, delta_bic=1e12
    )
    assert (res.n_peaks == 2).sum() == 0
    assert (res.n_peaks == 1).mean() > 0.9
    # and the BIC maps still show the two-peak model winning on bimodal pixels
    assert np.median((res.bic1 - res.bic2)[t["two"]]) > 10.0


def test_out_of_span_component_rejected():
    """A bright tail from a component centred beyond the scanned window must
    not be reported as a resolved second peak (in-span/width gates)."""
    rng = np.random.default_rng(5)
    ny, nx, m, n = 10, 10, 21, 25
    chi = np.linspace(-0.5, 0.5, m)
    mus = np.linspace(7.0, 8.2, n)
    coords = np.array(np.meshgrid(chi, mus, indexing="ij"))
    c_in = np.stack([np.full((ny, nx), -0.1), np.full((ny, nx), 7.5)], -1)
    c_out = np.stack([np.full((ny, nx), -0.1), np.full((ny, nx), mus[-1] + 0.15)], -1)
    s = np.stack([np.full((ny, nx), 0.06), np.full((ny, nx), 0.09)], -1)
    A = np.full((ny, nx), 3000.0)

    def bump(c, A_):
        d0 = (coords[0] - c[..., 0, None, None]) / s[..., 0, None, None]
        d1 = (coords[1] - c[..., 1, None, None]) / s[..., 1, None, None]
        return A_[..., None, None] * np.exp(-0.5 * (d0 ** 2 + d1 ** 2))

    data = rng.poisson(bump(c_in, A) + bump(c_out, 2.0 * A) + 30).astype(np.uint16)
    res = starling.properties.fit_ND_two_gaussians(
        data, coords, device="cpu", progress=False
    )
    det = res.n_peaks == 2
    if det.any():  # anything that slips through must at least be in-span
        for ax, lo_v, hi_v in ((0, chi[0], chi[-1]), (1, mus[0], mus[-1])):
            span = hi_v - lo_v
            assert (res.mu2[det][:, ax] > lo_v - 0.55 * span).all()
            assert (res.mu2[det][:, ax] < hi_v + 0.55 * span).all()
    assert det.mean() < 0.2


# ------------------------------ BIC sanity ---------------------------------- #


def test_bic_sanity():
    data, coords, t = two_peak_mosa_stack(ny=10, nx=10)
    res = starling.properties.fit_ND_two_gaussians(
        data, coords, device="cpu", progress=False
    )
    # resolved bimodal pixels: the 2-peak model wins by far more than the margin
    assert np.median((res.bic1 - res.bic2)[t["two"]]) > 100.0
    # unimodal pixels: the 2-peak model must not beat the margin
    assert (((res.bic2 + 10.0) < res.bic1)[~t["two"]]).mean() < 0.05
    # failed/unfit pixels carry the sentinel, fitted ones are finite
    assert np.isfinite(res.bic1[t["two"]]).all()


# ------------------------- single= precomputed path ------------------------- #


def test_precomputed_single_result_matches_internal():
    data, coords, t = two_peak_mosa_stack(ny=10, nx=10)
    single = starling.properties.fit_ND_gaussian(
        data, coords, device="cpu", progress=False
    )
    a = starling.properties.fit_ND_two_gaussians(
        data, coords, device="cpu", progress=False
    )
    b = starling.properties.fit_ND_two_gaussians(
        data, coords, device="cpu", progress=False, single=single
    )
    assert np.array_equal(a.n_peaks, b.n_peaks)
    for f in ("A1", "mu1", "cov1", "A2", "mu2", "cov2", "c", "success"):
        assert np.array_equal(getattr(a, f), getattr(b, f)), f

    with pytest.raises(ValueError, match="single"):
        starling.properties.fit_ND_two_gaussians(
            data[:4, :4], coords, device="cpu", progress=False, single=single
        )


# ------------------------------- seeding ------------------------------------ #


def test_two_peak_seed_nd_finds_separated_blobs():
    m, n = 15, 17
    grid = np.zeros((m, n))
    yy, xx = np.mgrid[0:m, 0:n]
    b1 = np.exp(-0.5 * (((yy - 4) / 1.5) ** 2 + ((xx - 5) / 1.5) ** 2))
    b2 = 0.7 * np.exp(-0.5 * (((yy - 10) / 1.5) ** 2 + ((xx - 12) / 1.5) ** 2))
    w = torch.as_tensor((b1 + b2).reshape(1, -1), dtype=torch.float64)
    i1, i2, v1, v2, has2 = two_peak_seed_nd(w, (m, n))
    assert bool(has2[0])
    r1, c1 = divmod(int(i1[0]), n)
    r2, c2 = divmod(int(i2[0]), n)
    assert abs(r1 - 4) <= 1 and abs(c1 - 5) <= 1
    assert abs(r2 - 10) <= 1 and abs(c2 - 12) <= 1
    assert float(v1[0]) > float(v2[0]) > 0

    # single blob -> no second maximum -> fall back path is signalled
    w1 = torch.as_tensor(b1.reshape(1, -1), dtype=torch.float64)
    _, _, _, _, has2b = two_peak_seed_nd(w1, (m, n))
    assert not bool(has2b[0])

    # peaks separated only along ONE axis of a 3-D grid still resolve
    g3 = np.zeros((7, 6, 11))
    zz = np.arange(11)
    g3[3, 3] = np.exp(-0.5 * ((zz - 2) / 1.0) ** 2) + \
        0.8 * np.exp(-0.5 * ((zz - 8) / 1.0) ** 2)
    w3 = torch.as_tensor(g3.reshape(1, -1), dtype=torch.float64)
    i1, i2, _, _, has2c = two_peak_seed_nd(w3, (7, 6, 11))
    assert bool(has2c[0])
    z1 = int(i1[0]) % 11
    z2 = int(i2[0]) % 11
    assert {min(z1, z2), max(z1, z2)} <= {1, 2, 3, 7, 8, 9}
    assert abs(z1 - z2) >= 4


# ----------------------- Jacobian vs autograd ------------------------------- #


@pytest.mark.parametrize("D", [1, 2, 3])
def test_gaussnd_two_jacobian_matches_autograd(D):
    torch.manual_seed(0)
    n_L = D * (D + 1) // 2
    n_p = 2 * (1 + D + n_L) + 1
    P, N = 5, 40
    params = torch.randn(P, n_p, dtype=torch.float64)
    x = torch.randn(D, N, dtype=torch.float64)
    f, J = gaussND_two_const(params, x, D)
    assert f.shape == (P, N) and J.shape == (P, N, n_p)

    def fwd(p):
        return gaussND_two_const(p, x, D)[0]

    Jauto = torch.autograd.functional.jacobian(fwd, params)
    idx = torch.arange(P)
    Jauto = Jauto[idx, :, idx, :]
    assert (J - Jauto).abs().max().item() < 1e-10


# ------------------- singular-precision crash regression -------------------- #


def test_masked_inv_spd_never_raises_and_classifies():
    """Regression for the fit_2D_gaussian LinAlgError crash on real MA6278
    data: an adversarial batch of L L^T precisions (float32-quantised entries,
    exact zeros, huge norms) must invert without raising, with rank-deficient
    rows masked out and well-conditioned rows exactly matching np.linalg.inv.

    The old absolute |det| < 1e-300 guard is provably insufficient here: the
    det residual of an exactly rank-deficient matrix scales with the matrix
    norm, so batches like this one raised LinAlgError from the batched inv.
    """
    rng = np.random.default_rng(0)
    P = 50000
    L = np.zeros((P, 2, 2))
    raw = rng.normal(0, 1, (P, 3)).astype(np.float32) * \
        10.0 ** rng.integers(-25, 25, (P, 3))
    z = rng.random((P, 3)) < 0.3
    raw[z] = 0.0
    L[:, 0, 0] = raw[:, 0]
    L[:, 1, 0] = raw[:, 1]
    L[:, 1, 1] = raw[:, 2]
    M = L @ np.swapaxes(L, -1, -2)
    M[0] = np.nan  # non-finite row
    M[1] = 0.0     # zero row

    Minv, ok = masked_inv_spd(M)  # must not raise
    assert not ok[0] and not ok[1]
    assert np.isfinite(Minv).all()
    # exactly rank-deficient rows (a zero Cholesky diagonal) are masked out
    rank_def = (L[:, 0, 0] == 0.0) | (L[:, 1, 1] == 0.0)
    assert not ok[rank_def].any()
    # zeroed, not garbage
    assert np.allclose(Minv[~ok], 0.0)
    # well-conditioned rows agree exactly with a plain per-row inverse
    good = np.flatnonzero(ok)[:200]
    for i in good:
        assert np.allclose(Minv[i], np.linalg.inv(M[i]), rtol=1e-9, atol=0.0)
        assert np.allclose(M[i] @ Minv[i], np.eye(2), atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_fit_nd_gaussian_singular_pixels_no_crash(device):
    """fit_ND_gaussian on a batch holding flat/zero/spike/ridge pixels must
    not raise, must zero + success=0 the degenerate pixels, and must leave
    every good pixel's result exactly as if the bad pixels were absent."""
    from test_fit_synthetic import mosa_stack

    data, coords, _ = mosa_stack(ny=12, nx=12)
    data = data.copy()
    data[0, 0] = 0                              # all-zero curve
    data[0, 1] = 500                            # perfectly flat curve
    data[0, 2] = 0
    data[0, 2, 3, 5] = 4000                     # single-voxel spike
    prof = 3000 * np.exp(-0.5 * ((np.linspace(-1, 1, data.shape[2]) / 0.2) ** 2))
    data[0, 3] = (prof[:, None] * np.ones((1, data.shape[3]))).astype(np.uint16)

    good = np.ones((12, 12), bool)
    good[0, :4] = False

    full = starling.properties.fit_ND_gaussian(
        data, coords, device=device, progress=False
    )  # must NOT raise
    assert np.isfinite(full.raw).all()
    # the three definitely-degenerate pixels: zeroed with success=0
    assert (full.success[0, :3] == 0).all()
    assert np.allclose(full.raw[0, :3], 0.0)

    # good pixels unchanged vs a run without the bad pixels in the batch
    sub = starling.properties.fit_ND_gaussian(
        data, coords, device=device, mask=good, progress=False
    )
    assert np.array_equal(full.raw[good], sub.raw[good])


@pytest.mark.parametrize("device", DEVICES)
def test_fit_nd_two_gaussians_singular_pixels_no_crash(device):
    from test_fit_synthetic import mosa_stack

    data, coords, _ = mosa_stack(ny=10, nx=10)
    data = data.copy()
    data[0, 0] = 0
    data[0, 1] = 700
    data[0, 2] = 0
    data[0, 2, 4, 6] = 3000
    res = starling.properties.fit_ND_two_gaussians(
        data, coords, device=device, progress=False
    )  # must NOT raise
    assert (res.n_peaks[0, :3] == 0).all()
    for f in ("A1", "mu1", "cov1", "A2", "mu2", "cov2", "c", "bic1", "bic2"):
        assert np.isfinite(getattr(res, f)).all(), f


@pytest.mark.parametrize("device", DEVICES)
def test_single_peak_results_healthy_batch_all_converge(device):
    """Guard against an over-aggressive singularity tolerance: on a healthy
    synthetic every pixel must still converge with success=1 (the parity of
    the actual values with the pre-fix code was verified bitwise)."""
    from test_fit_synthetic import mosa_stack

    data, coords, _ = mosa_stack(ny=10, nx=10)
    res = starling.properties.fit_ND_gaussian(
        data, coords, device=device, progress=False
    )
    assert (res.success > 0).all()


# --------------------------- result object / mask --------------------------- #


@pytest.mark.parametrize("device", ["cpu"])
def test_mask_only_computes_true_pixels(device):
    data, coords, t = two_peak_mosa_stack(ny=10, nx=10)
    mask = np.zeros((10, 10), bool)
    mask[2:8, 2:8] = True
    res = starling.properties.fit_ND_two_gaussians(
        data, coords, device=device, mask=mask, progress=False
    )
    assert (res.n_peaks[~mask] == 0).all()
    assert np.allclose(res.A1[~mask], 0.0)
    full = starling.properties.fit_ND_two_gaussians(
        data, coords, device=device, progress=False
    )
    assert np.array_equal(res.n_peaks[mask], full.n_peaks[mask])
    assert np.array_equal(res.mu1[mask], full.mu1[mask])


def test_result_object_helpers():
    data, coords, t = two_peak_mosa_stack(ny=10, nx=10)
    res = starling.properties.fit_ND_two_gaussians(
        data, coords, device="cpu", progress=False
    )
    sel = res.n_peaks == 2
    assert res.D == 2
    fw1 = res.fwhm(peak=1)
    assert fw1.shape == (10, 10, 2)
    assert (fw1[sel] > 0).all()
    mos = res.mosaicity(peak=1, mode="scalar")
    assert mos.shape == (10, 10)
    ori = res.orientation(peak=2, axes=(0, 1))
    assert ori.shape == (10, 10, 2)
    with pytest.raises(ValueError, match="peak"):
        res.fwhm(peak=3)
    d = res.to_dict()
    rebuilt = GaussNDTwoResult.from_dict(d)
    assert rebuilt.n_peaks.dtype == np.uint8
    assert np.array_equal(rebuilt.mu1, res.mu1)


def test_analyze_dispatch_two_peak():
    from test_analyze_dispatch import _bare

    data, coords, t = two_peak_mosa_stack(ny=8, nx=8)
    ds = _bare(data, coords)
    res = ds.analyze(two_peak=True, mask=None, progress=False)
    assert isinstance(res, GaussNDTwoResult)
    res2 = ds.analyze(method="gaussND2", mask=None, progress=False)
    assert isinstance(res2, GaussNDTwoResult)
    assert np.array_equal(res.n_peaks, res2.n_peaks)
    with pytest.raises(ValueError, match="gaussND2"):
        rock = np.zeros((4, 4, 9), np.uint16)
        _bare(rock, np.array([np.linspace(0, 1, 9)])).analyze(
            method="gaussND2", mask=None
        )
