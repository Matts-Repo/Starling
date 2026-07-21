"""Smoke test for the pixel-click rocking-curve inspector (viz.pixel_inspector).

Headless: forces the Agg matplotlib backend and skips cleanly when ipywidgets
is not installed. A click is synthesised by calling the handler that
pixel_inspector registers with fig.canvas.mpl_connect (exposed on the figure
as ``fig._starling_on_click``) with a mock event, then the right-hand
rocking-curve axes are checked to have gained lines.
"""

import types

import matplotlib
matplotlib.use("Agg")  # headless, before pyplot is imported anywhere

import numpy as np
import pytest


def _synthetic_dset(ny=8, nx=9, na=5, nb=6):
    """A tiny dset-like object: data (ny, nx, na, nb), motors (2, na, nb)."""
    a = np.linspace(-0.4, 0.4, na)
    b = np.linspace(7.0, 8.0, nb)
    motors = np.array(np.meshgrid(a, b, indexing="ij"))  # (2, na, nb)

    rng = np.random.default_rng(0)
    ca, cb = 0.05, 7.5
    sa, sb = 0.15, 0.3
    da = (motors[0] - ca) / sa
    db = (motors[1] - cb) / sb
    peak = 3000.0 * np.exp(-0.5 * (da ** 2 + db ** 2)) + 20.0  # (na, nb)
    data = rng.poisson(np.broadcast_to(peak, (ny, nx, na, nb))).astype(np.uint16)
    return types.SimpleNamespace(data=data, motors=motors), (ca, cb, sa, sb)


def _gaussnd_result(dset, centre):
    """Build a GaussNDResult (D=2) with plausible per-pixel fields."""
    from starling.properties import GaussNDResult

    ny, nx = dset.data.shape[:2]
    ca, cb, sa, sb = centre
    A = np.full((ny, nx), 3000.0)
    mu = np.zeros((ny, nx, 2))
    mu[..., 0] = ca
    mu[..., 1] = cb
    cov = np.zeros((ny, nx, 2, 2))
    cov[..., 0, 0] = sa ** 2
    cov[..., 1, 1] = sb ** 2
    c = np.full((ny, nx), 20.0)
    success = np.ones((ny, nx))
    return GaussNDResult(A=A, mu=mu, cov=cov, c=c, success=success)


def test_pixel_inspector_click_draws_curves():
    pytest.importorskip("ipywidgets")
    pytest.importorskip("IPython")
    import matplotlib.pyplot as plt

    from starling import viz

    dset, centre = _synthetic_dset()
    result = _gaussnd_result(dset, centre)
    fit_status = np.full(dset.data.shape[:2], 1, dtype=int)
    fit_status[0, 0] = 2  # edge-clipped
    fit_status[1, 1] = 3  # failed

    container = viz.pixel_inspector(dset, result=result, fit_status=fit_status)
    assert container is not None

    fig = plt.gcf()
    # axes[0] is the map, the rest are the per-motor-axis rocking curves
    ax_map = fig.axes[0]
    curve_axes = fig.axes[1:]
    assert len(curve_axes) == 2  # D == 2 motor axes
    assert all(len(ax.lines) == 0 for ax in curve_axes)  # nothing drawn yet

    # synthesise a click on the map: xdata->col, ydata->row (rounded to int)
    event = types.SimpleNamespace(inaxes=ax_map, xdata=4.4, ydata=3.6)
    fig._starling_on_click(event)

    # each rocking-curve axis now has the raw marginal + the fit overlay
    for ax in curve_axes:
        assert len(ax.lines) >= 2

    plt.close(fig)


def test_pixel_inspector_out_of_bounds_click_is_ignored():
    pytest.importorskip("ipywidgets")
    pytest.importorskip("IPython")
    import matplotlib.pyplot as plt

    from starling import viz

    dset, _ = _synthetic_dset()
    container = viz.pixel_inspector(dset)  # no result / no status
    assert container is not None

    fig = plt.gcf()
    ax_map = fig.axes[0]
    curve_axes = fig.axes[1:]

    # click outside the data bounds -> guarded, no curves drawn
    event = types.SimpleNamespace(inaxes=ax_map, xdata=999.0, ydata=999.0)
    fig._starling_on_click(event)
    assert all(len(ax.lines) == 0 for ax in curve_axes)

    # click off the map axes entirely -> ignored
    event2 = types.SimpleNamespace(inaxes=None, xdata=1.0, ydata=1.0)
    fig._starling_on_click(event2)
    assert all(len(ax.lines) == 0 for ax in curve_axes)

    # a valid click draws the raw marginal (no fit, since result is None)
    event3 = types.SimpleNamespace(inaxes=ax_map, xdata=2.0, ydata=2.0)
    fig._starling_on_click(event3)
    for ax in curve_axes:
        assert len(ax.lines) >= 1

    plt.close(fig)


def test_pixel_inspector_single_motor_gauss1d():
    pytest.importorskip("ipywidgets")
    pytest.importorskip("IPython")
    import matplotlib.pyplot as plt

    from starling import viz
    from starling.properties import Gauss1DResult

    ny, nx, n = 6, 7, 9
    x = np.linspace(-1.0, 1.0, n)
    motors = x.copy()  # 1-D motors for a single-motor scan
    rng = np.random.default_rng(1)
    peak = 2000.0 * np.exp(-0.5 * (x / 0.3) ** 2) + 10.0
    data = rng.poisson(np.broadcast_to(peak, (ny, nx, n))).astype(np.uint16)
    dset = types.SimpleNamespace(data=data, motors=motors)

    result = Gauss1DResult(
        A=np.full((ny, nx), 2000.0),
        sigma=np.full((ny, nx), 0.3),
        mu=np.zeros((ny, nx)),
        k=np.zeros((ny, nx)),
        m=np.full((ny, nx), 10.0),
        success=np.ones((ny, nx)),
    )

    container = viz.pixel_inspector(dset, result=result)
    assert container is not None

    fig = plt.gcf()
    ax_map = fig.axes[0]
    curve_axes = fig.axes[1:]
    assert len(curve_axes) == 1  # single motor axis

    event = types.SimpleNamespace(inaxes=ax_map, xdata=3.0, ydata=2.0)
    fig._starling_on_click(event)
    assert len(curve_axes[0].lines) >= 2  # raw + fit

    plt.close(fig)
