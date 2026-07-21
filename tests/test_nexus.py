"""NeXus exporter + loader round-trip and standards-compliance tests.

Synthetic, inline result objects (style of ``tests/test_maps_strain_kam.py``);
no real scan files. silx is optional (``pytest.importorskip``) and only used to
confirm the emitted tree is a valid NXdata that silx can parse.
"""

import ast

import h5py
import numpy as np
import pytest

import starling
from starling import load_nexus, save_dataset_nexus, save_nexus
from starling.properties import (
    Gauss1DResult,
    GaussNDResult,
    GaussNDTwoResult,
    MomentResult,
    PseudoVoigtResult,
)

NY, NX = 7, 5  # non-square on purpose: catches y/x axis transposition bugs


# ------------------------------- builders ---------------------------------- #


def _spd_cov(ny, nx, D, rng, scale=0.05, floor=0.02):
    L = rng.normal(0.0, scale, (ny, nx, D, D))
    return np.einsum("...ij,...kj->...ik", L, L) + floor * np.eye(D)


def _gaussnd(ny=NY, nx=NX, D=2, seed=0):
    rng = np.random.default_rng(seed)
    return GaussNDResult(
        A=rng.uniform(1.0, 2.0, (ny, nx)),
        mu=rng.normal(0.0, 0.1, (ny, nx, D)),
        cov=_spd_cov(ny, nx, D, rng),
        c=rng.uniform(0.0, 0.1, (ny, nx)),
        success=np.ones((ny, nx)),
    )


def _gauss1d(ny=NY, nx=NX, seed=1):
    rng = np.random.default_rng(seed)
    return Gauss1DResult(
        A=rng.uniform(1.0, 2.0, (ny, nx)),
        sigma=rng.uniform(0.05, 0.2, (ny, nx)),
        mu=rng.normal(0.0, 0.1, (ny, nx)),
        k=rng.normal(0.0, 0.01, (ny, nx)),
        m=rng.uniform(0.0, 0.1, (ny, nx)),
        success=np.ones((ny, nx)),
    )


def _pseudovoigt(ny=NY, nx=NX, seed=2):
    rng = np.random.default_rng(seed)
    return PseudoVoigtResult(
        A=rng.uniform(1.0, 2.0, (ny, nx)),
        sigma=rng.uniform(0.05, 0.2, (ny, nx)),
        mu=rng.normal(0.0, 0.1, (ny, nx)),
        gamma=rng.uniform(0.05, 0.2, (ny, nx)),
        eta=rng.uniform(0.0, 1.0, (ny, nx)),
        k=rng.normal(0.0, 0.01, (ny, nx)),
        m=rng.uniform(0.0, 0.1, (ny, nx)),
        success=np.ones((ny, nx)),
    )


def _moments(ny=NY, nx=NX, D=2, order=2, seed=3):
    rng = np.random.default_rng(seed)
    mean = rng.normal(0.0, 0.1, (ny, nx, D))
    cov = _spd_cov(ny, nx, D, rng)
    if order == 4:
        return MomentResult(mean, cov, rng.normal(0, 0.2, (ny, nx, D)),
                            rng.uniform(2.0, 4.0, (ny, nx, D)))
    return MomentResult(mean, cov)


def _moments_1d(ny=NY, nx=NX, seed=4):
    rng = np.random.default_rng(seed)
    return MomentResult(rng.normal(0.0, 0.1, (ny, nx)),
                        rng.uniform(0.01, 0.1, (ny, nx)))


def _gaussnd_two(ny=NY, nx=NX, D=2, seed=5):
    """Mirror the fit output invariants: n_peaks in {0,1,2}, success == 2-peak
    selection, all per-peak fields zeroed where the model was not selected."""
    rng = np.random.default_rng(seed)
    n_peaks = rng.integers(0, 3, (ny, nx)).astype(np.uint8)
    two = (n_peaks == 2).astype(np.float64)
    sel2 = two[..., None]
    sel3 = two[..., None, None]
    return GaussNDTwoResult(
        A1=rng.uniform(1.5, 2.0, (ny, nx)) * two,
        mu1=rng.normal(0.0, 0.1, (ny, nx, D)) * sel2,
        cov1=_spd_cov(ny, nx, D, rng) * sel3,
        A2=rng.uniform(0.5, 1.4, (ny, nx)) * two,
        mu2=rng.normal(0.3, 0.1, (ny, nx, D)) * sel2,
        cov2=_spd_cov(ny, nx, D, rng) * sel3,
        c=rng.uniform(0.0, 0.1, (ny, nx)) * two,
        n_peaks=n_peaks,
        bic1=rng.normal(-100.0, 10.0, (ny, nx)),
        bic2=rng.normal(-120.0, 10.0, (ny, nx)),
        success=two,
    )


def _save(tmp_path, result, **kw):
    p = tmp_path / "out.nxs"
    p.parent.mkdir(parents=True, exist_ok=True)
    save_nexus(str(p), result, **kw)
    return str(p)


# ------------------------------ round-trip --------------------------------- #


def test_roundtrip_gaussnd(tmp_path):
    g = _gaussnd()
    res, _, _ = load_nexus(_save(tmp_path, g))
    assert isinstance(res, GaussNDResult)
    for f in ("A", "mu", "cov", "c", "success"):
        assert np.allclose(getattr(res, f), getattr(g, f)), f


def test_roundtrip_gauss1d(tmp_path):
    g = _gauss1d()
    res, _, _ = load_nexus(_save(tmp_path, g))
    assert isinstance(res, Gauss1DResult)
    for f in ("A", "sigma", "mu", "k", "m", "success"):
        assert np.allclose(getattr(res, f), getattr(g, f)), f


def test_roundtrip_pseudovoigt(tmp_path):
    g = _pseudovoigt()
    res, _, _ = load_nexus(_save(tmp_path, g))
    assert isinstance(res, PseudoVoigtResult)
    for f in ("A", "sigma", "mu", "gamma", "eta", "k", "m", "success"):
        assert np.allclose(getattr(res, f), getattr(g, f)), f


def test_roundtrip_moments_order2(tmp_path):
    m = _moments(order=2)
    res, _, _ = load_nexus(_save(tmp_path, m))
    assert isinstance(res, MomentResult)
    assert np.allclose(res.mean, m.mean)
    assert np.allclose(res.covariance, m.covariance)
    assert res.skew is None and res.kurtosis is None


def test_roundtrip_moments_order4(tmp_path):
    m = _moments(order=4)
    res, _, _ = load_nexus(_save(tmp_path, m))
    assert np.allclose(res.mean, m.mean)
    assert np.allclose(res.covariance, m.covariance)
    assert np.allclose(res.skew, m.skew)
    assert np.allclose(res.kurtosis, m.kurtosis)


def test_roundtrip_moments_1d(tmp_path):
    m = _moments_1d()
    res, maps, meta = load_nexus(_save(tmp_path, m))
    assert np.allclose(res.mean, m.mean)
    assert np.allclose(res.covariance, m.covariance)
    assert meta["D"] == 1
    assert "Center of mass" in maps and "FWHM" in maps


def test_roundtrip_gaussnd_two(tmp_path):
    g = _gaussnd_two()
    res, maps, meta = load_nexus(_save(tmp_path, g))
    assert isinstance(res, GaussNDTwoResult)
    for f in ("A1", "mu1", "cov1", "A2", "mu2", "cov2", "c",
              "bic1", "bic2", "success"):
        assert np.allclose(getattr(res, f), getattr(g, f)), f
    assert np.array_equal(res.n_peaks, g.n_peaks)
    assert res.n_peaks.dtype == np.uint8
    assert meta["result_kind"] == "gaussND_two"
    assert meta["D"] == 2


def test_roundtrip_gaussnd_two_d3(tmp_path):
    g = _gaussnd_two(ny=5, nx=4, D=3)
    res, _, meta = load_nexus(_save(tmp_path, g))
    assert meta["D"] == 3
    assert np.allclose(res.mu2, g.mu2)
    assert np.allclose(res.cov1, g.cov1)


def test_gaussnd_two_display_maps(tmp_path):
    g = _gaussnd_two()
    p = _save(tmp_path, g)
    with h5py.File(p) as f:
        ent = f["entry"]
        assert ent.attrs["default"] == "Number of peaks"
        npk = ent["Number of peaks/Number of peaks"]
        assert npk.shape == (NY, NX)
        assert np.array_equal(npk[()], g.n_peaks.astype(np.float64))
        assert npk.attrs["quantity"] == "n_peaks"
        assert "Peak separation (Mahalanobis)" in ent
        for motor in ("axis0", "axis1"):
            coll = ent[motor]
            for name in ("Center of mass (peak 1)", "Center of mass (peak 2)",
                         "FWHM (peak 1)", "FWHM (peak 2)", "Peak separation"):
                assert name in coll, (motor, name)
            com1 = coll["Center of mass (peak 1)/Center of mass (peak 1)"]
            i = 0 if motor == "axis0" else 1
            assert np.allclose(com1[()], g.mu1[..., i])
            sep = coll["Peak separation/Peak separation"][()]
            assert np.allclose(sep, (g.mu1 - g.mu2)[..., i])


def test_gaussnd_two_mask_display_only(tmp_path):
    g = _gaussnd_two()
    # ensure at least one on-grain 2-peak and one on-grain non-2-peak pixel
    g.n_peaks[1, 1] = 2
    g.success[1, 1] = 1.0
    g.mu1[1, 1] = 0.05
    g.n_peaks[2, 2] = 1
    g.success[2, 2] = 0.0
    mask = np.ones((NY, NX), dtype=bool)
    mask[0, 0] = False
    p = _save(tmp_path, g, mask=mask)
    with h5py.File(p) as f:
        ent = f["entry"]
        npk = ent["Number of peaks/Number of peaks"][()]
        assert np.isnan(npk[0, 0])           # off-grain -> NaN
        assert np.isfinite(npk[2, 2])        # on-grain single-peak visible
        com1 = ent["axis0/Center of mass (peak 1)/Center of mass (peak 1)"][()]
        assert np.isnan(com1[0, 0])          # off-grain
        assert np.isnan(com1[2, 2])          # on-grain but not 2-peak
        assert np.isfinite(com1[1, 1])       # bimodal pixel shown
        # raw block unmasked and exact
        assert np.allclose(f["entry/starling_process/raw/mu1"][()], g.mu1)
    res, _, _ = load_nexus(p)
    assert np.allclose(res.mu1, g.mu1)


def test_gaussnd_two_silx_can_parse(tmp_path):
    silx = pytest.importorskip("silx")  # noqa: F841
    from silx.io.nxdata import get_default

    with h5py.File(_save(tmp_path, _gaussnd_two())) as f:  # silx wants a group
        default = get_default(f)
        assert default is not None
        assert default.signal is not None


# ------------------------------- NeXus attrs ------------------------------- #


def test_nexus_structural_attrs(tmp_path):
    with h5py.File(_save(tmp_path, _gaussnd())) as f:
        assert f.attrs["NX_class"] == "NXroot"
        assert f.attrs["default"] == "entry"
        ent = f["entry"]
        assert ent.attrs["NX_class"] == "NXentry"
        assert ent.attrs["default"] == "Mosaicity"
        proc = ent["starling_process"]
        assert proc.attrs["NX_class"] == "NXprocess"
        assert proc["program"][()].decode() == "starling"
        assert proc["version"][()].decode() == starling.__version__
        assert proc["processing_order"][()] == 1
        assert proc["processing_order"].dtype == np.int32  # spec pins int32
        assert proc["raw"].attrs["NX_class"] == "NXcollection"
        com = ent["axis0"]["Center of mass"]
        assert com.attrs["NX_class"] == "NXdata"
        assert com.attrs["signal"] == "Center of mass"
        assert list(com.attrs["axes"]) == ["y", "x"]
        assert com["Center of mass"].attrs["interpretation"] == "image"


def test_darfix_names_present(tmp_path):
    with h5py.File(_save(tmp_path, _gaussnd())) as f:
        ent = f["entry"]
        for name in ("Mosaicity", "Color Key", "Kernel Average Misorientation"):
            assert name in ent, name
        for motor in ("axis0", "axis1"):
            assert "Center of mass" in ent[motor]
            assert "FWHM" in ent[motor]


def test_quantity_fit_vs_moment(tmp_path):
    with h5py.File(_save(tmp_path, _gaussnd())) as f:
        com = f["entry/axis0/Center of mass/Center of mass"]
        fwhm = f["entry/axis0/FWHM/FWHM"]
        assert com.attrs["quantity"] == "fit_peak_center"
        assert fwhm.attrs["quantity"] == "fit_fwhm"
        assert com.attrs["source"] == "fit"
    with h5py.File(_save(tmp_path / "m", _moments(order=2))) as f:
        com = f["entry/axis0/Center of mass/Center of mass"]
        fwhm = f["entry/axis0/FWHM/FWHM"]
        assert com.attrs["quantity"] == "center_of_mass"
        assert fwhm.attrs["quantity"] == "moment_fwhm"
        assert com.attrs["source"] == "moments"


def test_pseudovoigt_peak_model(tmp_path):
    with h5py.File(_save(tmp_path, _pseudovoigt())) as f:
        com = f["entry/Center of mass/Center of mass"]
        assert com.attrs["peak_model"] == "pseudovoigt"
        assert com.attrs["quantity"] == "fit_peak_center"


def test_rgb_shapes_and_attrs(tmp_path):
    with h5py.File(_save(tmp_path, _gaussnd())) as f:
        mos = f["entry/Mosaicity"]
        sig = mos["Mosaicity"]
        assert sig.shape == (NY, NX, 3)
        assert (sig[()] >= 0).all() and (sig[()] <= 1).all()
        assert sig.attrs["interpretation"] == "rgba-image"
        # channel axis last + unlabelled; named axes are backed by real coords
        # (an unbacked @axes name makes silx reject the NXdata -> no auto-render)
        assert list(mos.attrs["axes"]) == ["y", "x", "."]
        assert mos["y"].shape == (NY,) and mos["x"].shape == (NX,)
        # darfix-parity square colour stamp -> (S, S, 3) key at key_size=256
        key = f["entry/Color Key/Color Key"]
        assert key.shape == (256, 256, 3)
        assert key.attrs["interpretation"] == "rgba-image"
        ck = f["entry/Color Key"]
        assert list(ck.attrs["axes"]) == ["ky", "kx", "."]
        assert ck["ky"].shape == (256,) and ck["kx"].shape == (256,)


def test_kam_present_d2_absent_d1(tmp_path):
    with h5py.File(_save(tmp_path / "nd", _gaussnd())) as f:
        kam = f["entry/Kernel Average Misorientation/Kernel Average Misorientation"]
        assert kam.shape == (NY, NX)
        assert kam.attrs["units"] == "deg"
        assert list(kam.attrs["kernel_size"]) == [3, 3]
    with h5py.File(_save(tmp_path / "1d", _gauss1d())) as f:
        assert "Kernel Average Misorientation" not in f["entry"]
        assert "Mosaicity" not in f["entry"]


def test_default_chain(tmp_path):
    with h5py.File(_save(tmp_path / "nd", _gaussnd())) as f:
        assert f.attrs["default"] == "entry"
        assert f["entry"].attrs["default"] == "Mosaicity"
        assert "Mosaicity" in f["entry"]
    with h5py.File(_save(tmp_path / "1d", _gauss1d())) as f:
        assert f["entry"].attrs["default"] == "Center of mass"
        assert "Center of mass" in f["entry"]


# ------------------------------ strain sweep ------------------------------- #


def test_strain_sweep_roundtrip(tmp_path):
    layers = [_gauss1d(seed=s) for s in range(3)]
    layer_values = np.array([0.10, 0.20, 0.30])
    p = str(tmp_path / "sweep.nxs")
    save_nexus(p, layers, layer_values=layer_values)

    with h5py.File(p) as f:
        raw = f["entry/starling_process/raw"]
        assert f["entry/starling_process/result_kind"][()].decode() == "gauss1d_sweep"
        assert raw["mu"].shape == (NY, NX, 3)  # raw keeps (ny, nx, n_layer)
        assert np.allclose(raw["layer_mu"][()], layer_values)
        com = f["entry/Center of mass"]
        # display stack is layer-FIRST so silx scrubs layers (not spatial Y)
        assert com["Center of mass"].shape == (3, NY, NX)
        assert list(com.attrs["axes"]) == ["layer", "y", "x"]
        assert np.allclose(com["layer"][()], layer_values)
        # layer 1's displayed COM equals that layer's peak centre
        assert np.allclose(com["Center of mass"][1], layers[1].mu)

    res, _, meta = load_nexus(p)
    assert isinstance(res, list) and len(res) == 3
    assert np.allclose(meta["layer_mu"], layer_values)
    for orig, got in zip(layers, res):
        for fld in ("A", "sigma", "mu", "k", "m", "success"):
            assert np.allclose(getattr(got, fld), getattr(orig, fld)), fld


# --------------------------------- masking --------------------------------- #


def test_mask_display_only_raw_intact(tmp_path):
    g = _gaussnd()
    mask = np.ones((NY, NX), dtype=bool)
    mask[0, 0] = False  # off-grain pixel
    p = _save(tmp_path, g, mask=mask)

    with h5py.File(p) as f:
        # display map: off-grain -> NaN, on-grain finite
        disp = f["entry/axis0/Center of mass/Center of mass"][()]
        assert np.isnan(disp[0, 0])
        assert np.isfinite(disp[1, 1])
        # raw block: unmasked, exact
        assert np.allclose(f["entry/starling_process/raw/mu"][()], g.mu)
        assert not np.isnan(f["entry/starling_process/raw/mu"][()]).any()
        assert f["entry/starling_process/raw"].attrs["mask_applied"]
        assert "mask" in f["entry/starling_process/raw"]
        # masked RGB -> white (1.0), never NaN
        rgb = f["entry/Mosaicity/Mosaicity"][()]
        assert np.allclose(rgb[0, 0], 1.0)
        assert not np.isnan(rgb).any()

    # round-trip still exact (raw is unmasked)
    res, _, _ = load_nexus(p)
    assert np.allclose(res.mu, g.mu)


# ------------------------------ pixel size --------------------------------- #


def test_pixel_size_mm_axes(tmp_path):
    g = _gauss1d()
    with h5py.File(_save(tmp_path / "mm", g, pixel_size_mm=0.65)) as f:
        x = f["entry/Center of mass/x"]
        y = f["entry/Center of mass/y"]
        assert x.attrs["units"] == "mm"
        assert np.allclose(x[()], np.arange(NX) * 0.65)
        assert np.allclose(y[()], np.arange(NY) * 0.65)
    with h5py.File(_save(tmp_path / "px", g)) as f:
        x = f["entry/Center of mass/x"]
        assert x.attrs["units"] == "pixel"
        assert np.allclose(x[()], np.arange(NX))


def test_pixel_size_anisotropic(tmp_path):
    g = _gauss1d()
    with h5py.File(_save(tmp_path, g, pixel_size_mm=(0.4, 0.8))) as f:
        assert np.allclose(f["entry/Center of mass/y"][()], np.arange(NY) * 0.4)
        assert np.allclose(f["entry/Center of mass/x"][()], np.arange(NX) * 0.8)


# ------------------------------ scan axes ---------------------------------- #


def _separable_motors(m=4, n=3):
    slow = np.arange(m, dtype=np.float32)[:, None] * np.ones((1, n), np.float32)
    fast = np.ones((m, 1), np.float32) * np.arange(n, dtype=np.float32)[None, :]
    return np.stack([slow, fast])  # (2, m, n)


def test_scan_group_motor_axes(tmp_path):
    g = _gaussnd(ny=4, nx=3)  # ny,nx irrelevant; motors drive scan/
    motors = _separable_motors(4, 3)
    scan_params = {
        "scan_command": "fscan2d chi 0 3 4 mu 0 2 3",
        "motor_names": ["instrument/chi/value", "instrument/positioners/mu"],
        "scan_shape": np.array([4, 3]),
        "scan_id": "1.1",
    }
    p = str(tmp_path / "scan.nxs")
    save_nexus(p, g, motors=motors, scan_params=scan_params)
    with h5py.File(p) as f:
        scan = f["entry/scan"]
        assert scan.attrs["NX_class"] == "NXcollection"
        assert np.allclose(scan["chi"][()], np.arange(4))
        assert np.allclose(scan["mu"][()], np.arange(3))
        assert scan["chi"].attrs["units"] == "deg"
        assert scan["scan_command"][()].decode().startswith("fscan2d")
    # the per-motor display collections use the short motor names
    with h5py.File(p) as f:
        assert "chi" in f["entry"] and "mu" in f["entry"]


def test_non_separable_grid_warns(tmp_path):
    g = _gaussnd(ny=4, nx=5)
    m, n = 4, 5
    # slow motor drifts strongly across the fast axis -> not separable
    slow = (np.arange(m, dtype=np.float32)[:, None]
            + 0.7 * np.arange(n, dtype=np.float32)[None, :])
    fast = np.ones((m, 1), np.float32) * np.arange(n, dtype=np.float32)[None, :]
    motors = np.stack([slow, fast])
    with pytest.warns(UserWarning, match="separable"):
        save_nexus(str(tmp_path / "ns.nxs"), g, motors=motors)


# --------------------------- dataset cube save ----------------------------- #


class _FakeDset:
    def __init__(self, data, motors, scan_params, h5file):
        self.data = data
        self.motors = motors
        self._sp = scan_params
        self.h5file = h5file

    @property
    def scan_params(self):
        return self._sp


def test_save_dataset_cube(tmp_path):
    a, b, m, n = 4, 3, 5, 6  # detector-first (a, b, *grid)
    data = (1000 * np.random.default_rng(0).random((a, b, m, n))).astype(np.uint16)
    motors = _separable_motors(m, n)
    scan_params = {
        "scan_command": "fscan2d chi 0 4 5 mu 0 5 6",
        "motor_names": ["instrument/chi/value", "instrument/positioners/mu"],
        "scan_shape": np.array([m, n]),
        "scan_id": "1.1",
    }
    dset = _FakeDset(data, motors, scan_params, "/data/raw/sample.h5")
    p = str(tmp_path / "cube.nxs")
    save_dataset_nexus(p, dset)

    with h5py.File(p) as f:
        assert f["entry"].attrs["default"] == "data"
        d = f["entry/data/preprocessed_data"]
        assert d.shape == (a, b, m, n)
        assert d.chunks == (a, b, 1, 1)  # singleton on the MOTOR axes
        plist = d.id.get_create_plist()
        filters = [plist.get_filter(i)[0] for i in range(plist.get_nfilters())]
        assert 32008 in filters  # Bitshuffle
        assert np.array_equal(d[()], data)
        proc = f["entry/starling_process"]
        assert proc["raw_data_source"][()].decode() == "/data/raw/sample.h5"
        assert "scan" in f["entry"]


def test_dataset_save_refuses_raw_data_dir(tmp_path):
    data = np.zeros((2, 2, 3), np.uint16)
    dset = _FakeDset(data, None, None, None)
    with pytest.raises(PermissionError, match="RAW_DATA"):
        save_dataset_nexus(str(tmp_path / "RAW_DATA" / "x.nxs"), dset)


# ------------------------------ DataSet wrapper ---------------------------- #


def test_dataset_wrapper_no_scan(tmp_path):
    """DataSet.save_nexus works before any scan is loaded (scan_params -> None)."""
    dset = starling.DataSet.__new__(starling.DataSet)
    dset.reader = None
    dset.motors = None
    p = str(tmp_path / "w.nxs")
    dset.save_nexus(p, _gauss1d())
    res, _, _ = load_nexus(p)
    assert isinstance(res, Gauss1DResult)


# --------------------------- display-map values ---------------------------- #


def test_display_map_values_match_source(tmp_path):
    """Display NXdata signals must equal the source field component (no swap)."""
    g = _gaussnd()  # NY != NX, so a y/x transpose would change the shape/values
    with h5py.File(_save(tmp_path, g)) as f:
        for i, motor in enumerate(("axis0", "axis1")):
            com = f[f"entry/{motor}/Center of mass/Center of mass"][()]
            fwhm = f[f"entry/{motor}/FWHM/FWHM"][()]
            assert com.shape == (NY, NX)
            assert np.allclose(com, g.mu[..., i])
            assert np.allclose(fwhm, g.fwhm[..., i])
    g1 = _gauss1d()
    with h5py.File(_save(tmp_path / "d1", g1)) as f:
        assert np.allclose(f["entry/Center of mass/Center of mass"][()], g1.mu)
        assert np.allclose(f["entry/FWHM/FWHM"][()], g1.fwhm)


def test_moments_order4_d2_skew_kurtosis_groups(tmp_path):
    m = _moments(order=4)
    with h5py.File(_save(tmp_path, m)) as f:
        for motor in ("axis0", "axis1"):
            assert "Skewness" in f[f"entry/{motor}"]
            assert "Kurtosis" in f[f"entry/{motor}"]
        sk = f["entry/axis0/Skewness/Skewness"]
        assert sk.attrs["quantity"] == "skewness"
        assert sk.attrs["source"] == "moments"
        assert np.allclose(sk[()], m.skew[..., 0])


def test_gaussnd_d3(tmp_path):
    g = _gaussnd(ny=5, nx=4, D=3)
    res, _, meta = load_nexus(_save(tmp_path, g, orientation_axes=(0, 1)))
    assert np.allclose(res.mu, g.mu) and np.allclose(res.cov, g.cov)
    assert meta["D"] == 3
    with h5py.File(_save(tmp_path / "d3", g, orientation_axes=(0, 1))) as f:
        for ax in ("axis0", "axis1", "axis2"):
            assert ax in f["entry"]
        assert "Mosaicity" in f["entry"]
        assert "Kernel Average Misorientation" in f["entry"]


# --------------------------- masking (moments) ----------------------------- #


def test_mask_on_moments(tmp_path):
    m = _moments(order=2)
    mask = np.ones((NY, NX), dtype=bool)
    mask[2, 3] = False
    with h5py.File(_save(tmp_path, m, mask=mask)) as f:
        disp = f["entry/axis0/Center of mass/Center of mass"][()]
        assert np.isnan(disp[2, 3]) and np.isfinite(disp[0, 0])
        raw = f["entry/starling_process/raw"]
        assert raw.attrs["mask_applied"]
        assert np.allclose(raw["mean"][()], m.mean)  # raw unmasked
    res, _, _ = load_nexus(_save(tmp_path / "rt", m, mask=mask))
    assert np.allclose(res.mean, m.mean)


# ------------------------- Darks / empty motors ---------------------------- #


def test_no_motors_skips_scan_group(tmp_path):
    g = _gauss1d()
    with h5py.File(_save(tmp_path, g)) as f:  # motors defaults to None
        assert "scan" not in f["entry"]
        assert f["entry/Center of mass/x"].attrs["units"] == "pixel"
    p = str(tmp_path / "darks.nxs")
    save_nexus(p, g, motors=np.array([], dtype=np.float32))  # empty (Darks)
    with h5py.File(p) as f:
        assert "scan" not in f["entry"]


# --------------------------- extra_attrs fidelity -------------------------- #


def test_extra_attrs_roundtrip(tmp_path):
    extra = {
        "recipe_hash": "abc123",
        "grain_id": "12",          # numeric-looking string must stay a string
        "device": "mps",
        "params": {"a": 1, "b": [2, 3]},  # non-scalar -> JSON-encoded
    }
    _, _, meta = load_nexus(_save(tmp_path, _gauss1d(), extra_attrs=extra))
    ea = meta["extra_attrs"]
    assert ea["recipe_hash"] == "abc123"
    assert ea["grain_id"] == "12" and isinstance(ea["grain_id"], str)
    assert ea["device"] == "mps"
    assert ea["params"] == {"a": 1, "b": [2, 3]}


# ------------------------------ fit status --------------------------------- #


def _fit_status(ny=NY, nx=NX, seed=7):
    """A (ny, nx) int8 map spanning all four categories (0/1/2/3)."""
    rng = np.random.default_rng(seed)
    st = rng.integers(0, 4, (ny, nx)).astype(np.int8)
    st[0, 0], st[0, 1], st[1, 0], st[1, 1] = 0, 1, 2, 3  # guarantee all present
    return st


def test_fit_status_nxdata_roundtrip(tmp_path):
    """fit_status -> its own 'Fit status' NXdata + verbatim int8 raw round-trip."""
    g = _gaussnd()
    st = _fit_status()
    p = _save(tmp_path, g, fit_status=st)
    with h5py.File(p) as f:
        ent = f["entry"]
        assert "Fit status" in ent
        sig = ent["Fit status/Fit status"]
        assert sig.shape == (NY, NX)
        assert sig.dtype == np.float64             # float cast for silx
        assert sig.attrs["quantity"] == "fit_status"
        assert sig.attrs["encoding"] == "0=no_signal 1=ok 2=edge_clipped 3=failed"
        assert not np.isnan(sig[()]).any()          # never NaN-masked
        assert np.array_equal(sig[()].astype(np.int8), st)  # int equality
        # verbatim int8 in the raw block
        raw_fs = f["entry/starling_process/raw/fit_status"]
        assert raw_fs.dtype == np.int8
        assert np.array_equal(raw_fs[()], st)
    # round-trips through load_nexus: display map + meta int8, both exact
    _, maps, meta = load_nexus(p)
    assert "Fit status" in maps
    assert np.array_equal(maps["Fit status"].astype(np.int8), st)
    assert meta["fit_status"].dtype == np.int8
    assert np.array_equal(meta["fit_status"], st)


def test_fit_status_not_masked(tmp_path):
    """Off-grain pixels stay categorical (0), not NaN, even with a grain mask."""
    g = _gaussnd()
    st = _fit_status()
    st[0, 0] = 0  # off-grain / no-signal
    mask = np.ones((NY, NX), dtype=bool)
    mask[0, 0] = False
    with h5py.File(_save(tmp_path, g, fit_status=st, mask=mask)) as f:
        sig = f["entry/Fit status/Fit status"][()]
        assert sig[0, 0] == 0.0 and not np.isnan(sig[0, 0])
        assert np.array_equal(sig.astype(np.int8), st)


def test_fit_status_all_kinds(tmp_path):
    """fit_status threads through moments, 1-D, sweep and two-peak branches."""
    st = _fit_status()
    cases = {
        "moments": _moments(order=2),
        "gauss1d": _gauss1d(),
        "pvoigt": _pseudovoigt(),
        "two": _gaussnd_two(),
    }
    for name, res in cases.items():
        with h5py.File(_save(tmp_path / name, res, fit_status=st)) as f:
            assert np.array_equal(
                f["entry/Fit status/Fit status"][()].astype(np.int8), st)
            assert np.array_equal(
                f["entry/starling_process/raw/fit_status"][()], st)
    # strain sweep (list of Gauss1DResult)
    layers = [_gauss1d(seed=s) for s in range(3)]
    p = str(tmp_path / "sweep.nxs")
    save_nexus(p, layers, fit_status=st)
    with h5py.File(p) as f:
        assert np.array_equal(
            f["entry/Fit status/Fit status"][()].astype(np.int8), st)


def test_no_fit_status_absent(tmp_path):
    """Without the kwarg no 'Fit status' group is written and meta is None."""
    with h5py.File(_save(tmp_path, _gaussnd())) as f:
        assert "Fit status" not in f["entry"]
        assert "fit_status" not in f["entry/starling_process/raw"]
    _, maps, meta = load_nexus(_save(tmp_path / "b", _gaussnd()))
    assert "Fit status" not in maps
    assert meta["fit_status"] is None


# --------------------------- square colour stamp --------------------------- #


def test_color_key_stamp_axes_match_vrange(tmp_path):
    """Color Key axes are the stamp's motor-unit vrange (not the old -1..1)."""
    g = _gaussnd()
    # replicate the exact stamp the writer builds (no mask -> keep is None)
    _, key, vrange = g.orientation_stamp(axes=(0, 1), mask=None)
    (lo0, hi0), (lo1, hi1) = vrange
    S = key.shape[0]
    with h5py.File(_save(tmp_path, g, motor_units="deg")) as f:
        ck = f["entry/Color Key"]
        sig = ck["Color Key"]
        assert sig.shape == (S, S, 3)               # (S, S, 3) rgba convention
        ky, kx = ck["ky"], ck["kx"]
        assert ky.shape == (S,) and kx.shape == (S,)
        assert ky.attrs["units"] == "deg" and kx.attrs["units"] == "deg"
        assert np.allclose(ky[()], np.linspace(lo0, hi0, S))
        assert np.allclose(kx[()], np.linspace(lo1, hi1, S))
        # axes must NOT be the legacy fixed -1..1 range
        assert not np.allclose(ky[()], np.linspace(-1.0, 1.0, S))


def test_color_key_stamp_axes_moments(tmp_path):
    """Moments D>=2 branch also routes through the square stamp Color Key."""
    m = _moments(order=2)
    _, key, vrange = m.orientation_stamp(axes=(0, 1), mask=None)
    (lo0, hi0), (lo1, hi1) = vrange
    S = key.shape[0]
    with h5py.File(_save(tmp_path, m)) as f:
        ck = f["entry/Color Key"]
        assert ck["Color Key"].shape == (S, S, 3)
        assert np.allclose(ck["ky"][()], np.linspace(lo0, hi0, S))


def test_masked_stamp_white_offgrain(tmp_path):
    """A masked save yields WHITE (1.0) rgb at off-mask pixels in the stamp."""
    g = _gaussnd()
    mask = np.ones((NY, NX), dtype=bool)
    mask[0, 0] = False
    mask[3, 2] = False
    with h5py.File(_save(tmp_path, g, mask=mask)) as f:
        rgb = f["entry/Mosaicity/Mosaicity"][()]
        assert rgb.shape == (NY, NX, 3)
        assert np.allclose(rgb[0, 0], 1.0)          # off-grain -> white
        assert np.allclose(rgb[3, 2], 1.0)
        assert not np.isnan(rgb).any()               # never NaN (breaks silx)
        assert np.isfinite(rgb[1, 1]).all()          # on-grain keeps stamp colour


# ------------------------------ standalone --------------------------------- #


def test_nexus_imports_are_standalone():
    """_nexus.py must import only stdlib + h5py/hdf5plugin/numpy + starling.*."""
    import starling.io._nexus as nx

    src = open(nx.__file__).read()
    allowed_third_party = {"h5py", "hdf5plugin", "numpy"}
    allowed_stdlib = {"datetime", "json", "warnings", "os", "ast"}
    forbidden = {"darling", "darfix", "torch", "silx", "scipy", "pandas"}
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots = {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:  # relative -> starling-internal
                continue
            roots = {(node.module or "").split(".")[0]}
        else:
            continue
        assert not (roots & forbidden), f"forbidden import: {roots & forbidden}"
        unknown = roots - allowed_third_party - allowed_stdlib - {"starling", ""}
        assert not unknown, f"unexpected import root(s): {unknown}"


# ----------------------------- silx (optional) ----------------------------- #


def test_silx_can_parse(tmp_path):
    silx = pytest.importorskip("silx")  # noqa: F841 (skip if not installed)
    from silx.io.nxdata import get_default

    p = _save(tmp_path, _gaussnd())
    with h5py.File(p) as f:  # silx get_default wants a group, not a path
        default = get_default(f)
        assert default is not None  # silx resolved entry@default -> a valid NXdata
        assert default.signal is not None
