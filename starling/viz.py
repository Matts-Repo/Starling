"""Interactive notebook widgets.

``roi_picker``    — stream a z-sum preview per scan without loading the full
                    dataset; drag or type to pick an ROI before loading.
``denoise_widget`` — non-destructive noise-reduction preview; only commits to
                    the dataset when Apply is pressed.
``pixel_inspector`` — click a map pixel to inspect its per-motor-axis rocking
                    curve(s) and the fitted-model overlay.

ipywidgets and matplotlib are imported lazily so importing ``starling`` never
requires them.
"""

import numpy as np

from . import preprocess

# Detector dataset paths tried in order when auto-detecting
_DETECTOR_CANDIDATES = [
    "instrument/pco_ff/image",
    "measurement/pco_ff",
    "instrument/eiger_4m/data",
    "measurement/eiger_4m",
]


def roi_picker(h5_path, scan_ids, n_preview_frames=20, pixel_size_mm=6.5e-3):
    """Lazy ROI picker — stream z-sum previews without loading the full dataset.

    Opens the BLISS HDF5 master file directly and reads a strided subset of
    frames per scan to build a quick z-sum.  Navigate scans with the slider,
    drag a box on the detector image or type pixel coordinates, then press
    **Confirm ROI** to print the ``LOAD_ROI`` tuple ready to paste into the
    config cell.

    Requires ``%matplotlib widget`` to be active in the calling cell.

    Args:
        h5_path: path to the BLISS master HDF5 file.
        scan_ids: single scan id string or list (e.g. ``['1.1', '2.1', ...]``).
        n_preview_frames: frames read per scan (evenly strided); default 20.
        pixel_size_mm: detector pixel size in mm for the info panel display.

    Returns:
        The ipywidgets container (also displayed in the notebook).
    """
    import h5py
    import hdf5plugin  # noqa: F401 — registers decompressors
    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.widgets import RectangleSelector
    from IPython.display import display

    scan_ids = [scan_ids] if isinstance(scan_ids, str) else list(scan_ids)

    # ── lazy z-sum cache ──────────────────────────────────────────────────────
    _cache = {}
    _data_name = [None]

    def _detect_data_name(h5f, sid):
        for c in _DETECTOR_CANDIDATES:
            if f"{sid}/{c}" in h5f:
                return c
        result = [None]
        def _visit(name, obj):
            if result[0] is None and isinstance(obj, h5py.Dataset) \
                    and obj.ndim == 3 and obj.shape[1] > 100:
                result[0] = name[len(sid) + 1:]
                raise StopIteration
        try:
            h5f[sid].visititems(_visit)
        except StopIteration:
            pass
        return result[0]

    def _get_zsum(sid):
        if sid not in _cache:
            with h5py.File(h5_path) as f:
                if _data_name[0] is None:
                    _data_name[0] = _detect_data_name(f, sid)
                    if _data_name[0] is None:
                        raise RuntimeError(f"No detector dataset found in scan {sid}")
                ds = f[f"{sid}/{_data_name[0]}"]
                n = ds.shape[0]
                step = max(1, n // n_preview_frames)
                frames = ds[::step].astype(np.float32)
            _cache[sid] = frames.sum(0)
        return _cache[sid]

    # ── figure ────────────────────────────────────────────────────────────────
    fig, (ax, ax_info) = plt.subplots(1, 2, figsize=(13, 6),
                                      gridspec_kw={"width_ratios": [3, 1]})
    ax_info.axis("off")

    state = {"r1": 0, "r2": 2048, "c1": 0, "c2": 2048, "_patch": None}
    _busy = [False]  # re-entrancy guard for coord ↔ text-box sync

    def _draw_rect():
        if state["_patch"] is not None:
            state["_patch"].remove()
        p = Rectangle(
            (state["c1"], state["r1"]),
            state["c2"] - state["c1"],
            state["r2"] - state["r1"],
            edgecolor="cyan", facecolor="none", lw=2,
        )
        ax.add_patch(p)
        state["_patch"] = p
        _update_info()
        fig.canvas.draw_idle()

    def _update_info():
        ax_info.clear(); ax_info.axis("off")
        r1, r2, c1, c2 = state["r1"], state["r2"], state["c1"], state["c2"]
        h, w = r2 - r1, c2 - c1
        txt = (
            f"ROI\n"
            f"  rows : {r1} – {r2}\n"
            f"           {h} px  ·  {h * pixel_size_mm:.2f} mm\n"
            f"  cols : {c1} – {c2}\n"
            f"           {w} px  ·  {w * pixel_size_mm:.2f} mm\n\n"
            f"LOAD_ROI =\n  ({r1}, {r2},\n   {c1}, {c2})"
        )
        ax_info.text(
            0.05, 0.5, txt, transform=ax_info.transAxes,
            fontsize=9, family="monospace", va="center",
            bbox=dict(fc="#f8f8f8", ec="gray", lw=0.8, pad=8),
        )

    # ── widgets ───────────────────────────────────────────────────────────────
    lbl = widgets.Label(value="")
    scan_w = widgets.SelectionSlider(
        options=scan_ids, description="scan:",
        continuous_update=False,
        style={"description_width": "initial"},
        layout=widgets.Layout(width="420px"),
    )
    _iw = dict(layout=widgets.Layout(width="155px"))
    r1_w = widgets.BoundedIntText(value=0, min=0, max=4096, description="row min:", **_iw)
    r2_w = widgets.BoundedIntText(value=2048, min=1, max=4096, description="row max:", **_iw)
    c1_w = widgets.BoundedIntText(value=0, min=0, max=4096, description="col min:", **_iw)
    c2_w = widgets.BoundedIntText(value=2048, min=1, max=4096, description="col max:", **_iw)
    confirm_btn = widgets.Button(description="Confirm ROI", button_style="success", icon="check")
    out = widgets.Output()

    def _sync_widgets_to_state():
        _busy[0] = True
        r1_w.value = state["r1"]; r2_w.value = state["r2"]
        c1_w.value = state["c1"]; c2_w.value = state["c2"]
        _busy[0] = False

    def _show_scan(sid):
        ax.clear(); state["_patch"] = None
        lbl.value = f"Loading {sid} …"
        zsum = _get_zsum(sid)
        h, w = zsum.shape
        p1, p99 = np.percentile(zsum, [1, 99])
        ax.imshow(zsum, cmap="hot", interpolation="nearest",
                  vmin=max(p1, 0), vmax=p99)
        ax.set_title(
            f"{sid}  |  z-sum from {max(1, zsum.shape[0]//max(1,zsum.shape[0]//n_preview_frames))} frames"
            f"  |  {w} × {h} px"
        )
        ax.set_xlabel("detector column"); ax.set_ylabel("detector row")
        r2_w.max = h; c2_w.max = w
        state["r2"] = min(state["r2"], h); state["c2"] = min(state["c2"], w)
        _sync_widgets_to_state()
        _draw_rect()
        lbl.value = f"{sid}  ✓"

    # rectangle selector — left-click drag on the detector image
    def _on_rect_select(eclick, erelease):
        if _busy[0]:
            return
        state["c1"] = int(round(min(eclick.xdata, erelease.xdata)))
        state["c2"] = int(round(max(eclick.xdata, erelease.xdata)))
        state["r1"] = int(round(min(eclick.ydata, erelease.ydata)))
        state["r2"] = int(round(max(eclick.ydata, erelease.ydata)))
        _sync_widgets_to_state()
        _draw_rect()

    _rs = RectangleSelector(
        ax, _on_rect_select, useblit=True, button=[1],
        interactive=True, spancoords="pixels",
    )

    def _on_coord_change(_):
        if _busy[0]:
            return
        state["r1"] = r1_w.value; state["r2"] = r2_w.value
        state["c1"] = c1_w.value; state["c2"] = c2_w.value
        _draw_rect()

    for w in (r1_w, r2_w, c1_w, c2_w):
        w.observe(_on_coord_change, names="value")

    def _on_scan_change(change):
        _show_scan(change["new"])

    scan_w.observe(_on_scan_change, names="value")

    def _on_confirm(_):
        roi = (state["r1"], state["r2"], state["c1"], state["c2"])
        with out:
            out.clear_output()
            print(f"LOAD_ROI = {roi}")
            print(f"Paste this into the CONFIG cell, then re-run from cell 3.")

    confirm_btn.on_click(_on_confirm)

    _show_scan(scan_ids[0])
    plt.tight_layout()
    plt.show()

    panel = widgets.VBox([
        scan_w,
        lbl,
        widgets.HBox([r1_w, r2_w, c1_w, c2_w, confirm_btn]),
        out,
    ])
    display(panel)
    return panel


def denoise_widget(dset, bg_n=5, bg_mode="mean", bg_percentile=10.0, hot_sigma=5.0,
                   roi_threshold=0.05, hot_one_sided=True, hot_min_sigma=1.0):
    """Interactive denoise preview for a DataSet (JupyterLab, %matplotlib widget).

    Sliders/inputs control the background estimator, the hot-pixel sigma and the
    ROI threshold. Every change recomputes a cheap preview and redraws the
    before/after z-sum, the **counts removed inside the grain** (so you can see
    directly whether the background is eating grain signal), a representative
    frame and a signal histogram, with the proposed ROI box overlaid and the
    in-grain *signal-retained %* in the figure title. The dataset is **not**
    modified until *Apply* is pressed; re-opening the widget always previews from
    the original data. *Apply* uses the grain-safe hot-pixel filter (one-sided +
    robust-sigma floor) so genuine grain pixels are not flagged and interior dark
    features are never filled, while real zingers (even on the grain) are removed.

    Args:
        dset: a ``starling.DataSet`` (uses ``dset.data``).
        bg_n, bg_mode, bg_percentile, hot_sigma, roi_threshold: initial values.
            ``bg_mode`` is one of "mean"/"median"/"lowest"/"pmedian"/"percentile".
        hot_one_sided, hot_min_sigma: grain-safe hot-pixel settings applied on
            *Apply* (one-sided so dark features are never filled; ``min_sigma``
            floors the robust sigma so mostly-zero frames cannot flag grain pixels).

    Returns:
        The ipywidgets container (also displayed in a notebook).
    """
    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    from IPython.display import display
    from matplotlib.patches import Rectangle

    # keep an immutable reference to the original so previews are always fresh
    if not hasattr(dset, "_raw_data"):
        dset._raw_data = dset.data.copy()
    raw = dset._raw_data

    bg_n_w = widgets.IntSlider(value=bg_n, min=1, max=30, description="bg n")
    bg_mode_w = widgets.Dropdown(
        options=["mean", "median", "lowest", "pmedian", "percentile"],
        value=bg_mode, description="bg mode",
    )
    pct_w = widgets.FloatSlider(value=bg_percentile, min=1.0, max=50.0, step=1.0,
                                description="pctile")
    hot_w = widgets.FloatSlider(value=hot_sigma, min=2.0, max=15.0, step=0.5, description="hot σ")
    roi_w = widgets.FloatSlider(value=roi_threshold, min=0.0, max=0.5, step=0.01, description="roi thr")
    apply_btn = widgets.Button(description="Apply", button_style="success")
    status = widgets.Output()

    fig, axes = plt.subplots(1, 5, figsize=(18, 3.2))

    def render(*_):
        pv = preprocess.preview(
            raw, bg_n=bg_n_w.value, bg_mode=bg_mode_w.value,
            bg_percentile=pct_w.value, hot_sigma=hot_w.value,
            roi_threshold=roi_w.value,
        )
        for ax in axes:
            ax.clear()
        axes[0].imshow(np.log1p(pv["raw_zsum"]), cmap="hot", origin="lower")
        axes[0].set_title("raw z-sum")
        axes[1].imshow(np.log1p(pv["proc_zsum"]), cmap="hot", origin="lower")
        axes[1].set_title("denoised z-sum")
        r1, r2, c1, c2 = pv["roi"]
        axes[1].add_patch(Rectangle((c1, r1), c2 - c1, r2 - r1,
                                    ec="cyan", fc="none", lw=1.5))
        # counts removed inside the grain — should be ~0 within the grain core
        removed_in = np.where(pv["grain"], pv["removed_zsum"], np.nan)
        im = axes[2].imshow(removed_in, cmap="magma", origin="lower")
        axes[2].set_title("removed inside grain")
        fig.colorbar(im, ax=axes[2], fraction=0.046)
        axes[3].imshow(np.log1p(pv["proc_frame"]), cmap="hot", origin="lower")
        axes[3].set_title("brightest frame (denoised)")
        counts, edges = pv["hist"]
        axes[4].plot(0.5 * (edges[:-1] + edges[1:]), counts, lw=1.2)
        axes[4].set_yscale("log")
        axes[4].set_title("signal histogram")
        ret = pv["retained"]
        warn = "  ⚠ background may be eating grain" if ret == ret and ret < 0.90 else ""
        fig.suptitle(f"in-grain signal retained: {100 * ret:.1f}%{warn}", fontsize=11)
        fig.canvas.draw_idle()

    def on_apply(_):
        with status:
            status.clear_output()
            bg = preprocess.estimate_background(
                raw, n_lowest=bg_n_w.value, mode=bg_mode_w.value,
                percentile=pct_w.value,
            )
            diag = preprocess.grain_signal_retained(raw, bg)
            data = preprocess.subtracted(raw, bg)
            # grain-safe hot pixels: one-sided (never fills dark features) +
            # robust-sigma floor (mostly-zero frames can't flag grain pixels)
            preprocess.remove_hot_pixels(
                data, n_sigma=hot_w.value, one_sided=hot_one_sided,
                min_sigma=hot_min_sigma,
            )
            roi = preprocess.auto_roi(data, threshold_rel=roi_w.value)
            r1, r2, c1, c2 = roi
            dset.data = np.ascontiguousarray(data[r1:r2, c1:c2])
            dset.roi = roi
            print(
                f"Applied: bg(n={bg_n_w.value}, {bg_mode_w.value}"
                f"{f', pct={pct_w.value:.0f}' if bg_mode_w.value == 'percentile' else ''}), "
                f"hot σ={hot_w.value} (one-sided, σ-floored), roi={roi} "
                f"-> data {dset.data.shape}"
            )
            print(
                f"In-grain signal retained: {100 * diag['retained']:.1f}%  "
                f"({diag['grain_px']:,} grain px, {diag['floored_px']:,} floored to 0)"
            )

    for w in (bg_n_w, bg_mode_w, pct_w, hot_w, roi_w):
        w.observe(render, names="value")
    apply_btn.on_click(on_apply)
    render()

    controls = widgets.HBox([bg_n_w, bg_mode_w, pct_w, hot_w, roi_w, apply_btn])
    box = widgets.VBox([controls, status])
    display(box)
    return box


# ── pixel rocking-curve inspector ───────────────────────────────────────────


def _axis_grid_values(motors, i, D):
    """Motor grid positions along axis ``i`` (others held at index 0).

    ``motors`` is ``(D, *grid)`` for a multi-motor scan or a 1-D array of
    positions for a single motor. For axis ``i`` the returned 1-D values are
    ``motors[i]`` sliced with axis ``i`` free and every other grid axis at 0.
    """
    m = np.asarray(motors, dtype=np.float64)
    if m.ndim == 1:  # single-motor scan
        return m
    sel = [0] * D
    sel[i] = slice(None)
    return np.asarray(m[i][tuple(sel)], dtype=np.float64)


def pixel_inspector(dset, result=None, fit_status=None, cmap="hot", map_data=None):
    """Click a pixel on the map to inspect its rocking curve(s) and fit.

    Left: z-sum (or ``map_data``) image. Click any pixel: right panel(s) show,
    for each motor axis, the 1-D marginal profile of the raw data at that
    pixel (sum over the other motor axes) with, when ``result`` is given, the
    fitted model's matching 1-D slice through the fitted centre overlaid,
    plus a text readout of fitted parameters and fit_status.
    Requires %matplotlib widget (ipympl).

    The raw curve for axis ``i`` is a **marginal** (the pixel's data summed
    over every other motor axis), while the overlaid fit line is a **centre
    slice** — the fitted model evaluated along axis ``i`` with the other
    coordinates pinned to the fitted centre. These are different reductions,
    so their amplitudes are not expected to match; the fit slice is therefore
    rescaled by ``raw.max() / fit_slice.max()`` (shown as "scaled" in the
    legend) so the *shapes* can be compared. ``dset`` is never mutated.

    Args:
        dset: a ``starling.DataSet`` (or any object exposing ``.data`` of shape
            ``(ny, nx, *motor_dims)`` and ``.motors`` of shape ``(D, *grid)`` /
            1-D for a single motor).
        result: optional ``Gauss1DResult`` or ``GaussNDResult`` to overlay.
        fit_status: optional ``(ny, nx)`` integer status map (see
            ``starling.properties.STATUS_NAMES``); enables the status readout
            and the overlay checkbox.
        cmap: initial colormap for the map (also the Dropdown default).
        map_data: optional ``(ny, nx)`` array to show on the left instead of
            the z-sum (e.g. a mosaicity or amplitude map).

    Returns:
        The ipywidgets container (also displayed in the notebook).
    """
    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    from IPython.display import display

    from .properties import STATUS_NAMES

    data = np.asarray(dset.data)
    ny, nx = data.shape[:2]
    D = data.ndim - 2  # number of motor axes
    if D < 1:
        raise ValueError("dset.data must have at least one motor axis")

    # per-axis motor grid values (x-axes for the rocking curves)
    x_vals = [_axis_grid_values(dset.motors, i, D) for i in range(D)]

    # left-panel background map (never mutates dset)
    if map_data is not None:
        base_map = np.asarray(map_data, dtype=np.float64)
    else:
        base_map = data.sum(axis=tuple(2 + i for i in range(D))).astype(np.float64)

    fstat = None if fit_status is None else np.asarray(fit_status)

    # classify the result object (only Gauss1D / GaussND are overlaid)
    is_nd = result is not None and hasattr(result, "cov") and hasattr(result, "mu")
    is_1d = (
        result is not None
        and not is_nd
        and all(hasattr(result, a) for a in ("A", "sigma", "mu", "k", "m"))
    )

    cmap_options = ["hot", "viridis", "magma", "Greys_r"]
    if cmap not in cmap_options:
        cmap_options = [cmap] + cmap_options

    # ── figure: map on the left, D stacked rocking-curve axes on the right ──
    fig = plt.figure(figsize=(11, max(3.0, 2.4 * D)))
    gs = fig.add_gridspec(D, 2, width_ratios=[1.25, 1.0])
    ax_map = fig.add_subplot(gs[:, 0])
    curve_axes = [fig.add_subplot(gs[i, 1]) for i in range(D)]

    state = {"row": None, "col": None, "marker": None, "overlay": []}

    def _draw_map():
        ax_map.clear()
        ax_map.imshow(base_map, cmap=cmap_w.value, origin="lower",
                      interpolation="nearest")
        ax_map.set_title("z-sum" if map_data is None else "map")
        ax_map.set_xlabel("column (nx)")
        ax_map.set_ylabel("row (ny)")
        state["marker"] = None
        state["overlay"] = []
        if fstat is not None and status_w is not None and status_w.value:
            _add_status_overlay()
        if state["row"] is not None:
            _draw_marker()
        fig.canvas.draw_idle()

    def _add_status_overlay():
        # orange contour of edge-clipped (==2), grey wash of failed (==3)
        edge = (fstat == 2).astype(float)
        if edge.any():
            cs = ax_map.contour(edge, levels=[0.5], colors="orange", linewidths=1.5)
            state["overlay"].append(cs)
        failed = np.where(fstat == 3, 1.0, np.nan)
        if np.isfinite(failed).any():
            im = ax_map.imshow(failed, cmap="Greys", origin="lower",
                               vmin=0.0, vmax=1.0, alpha=0.5,
                               interpolation="nearest")
            state["overlay"].append(im)

    def _clear_status_overlay():
        for art in state["overlay"]:
            try:
                art.remove()
            except (ValueError, AttributeError):
                # older mpl QuadContourSet: remove member collections
                for coll in getattr(art, "collections", []):
                    coll.remove()
        state["overlay"] = []
        fig.canvas.draw_idle()

    def _draw_marker():
        if state["marker"] is not None:
            try:
                state["marker"].remove()
            except ValueError:
                pass
        (state["marker"],) = ax_map.plot(
            state["col"], state["row"], "+", color="cyan", ms=12, mew=2
        )

    def _eval_fit_slice(row, col, i, xf):
        """Fitted-model centre slice along axis ``i`` at pixel (row, col)."""
        if is_nd:
            var_i = float(result.cov[row, col, i, i])
            if not np.isfinite(var_i) or var_i <= 0:
                return None
            A = float(result.A[row, col])
            mu_i = float(result.mu[row, col, i])
            c = float(result.c[row, col])
            return A * np.exp(-0.5 * (xf - mu_i) ** 2 / var_i) + c
        if is_1d:  # gauss1d_lin, single motor axis
            sigma = float(result.sigma[row, col])
            if not np.isfinite(sigma) or sigma <= 0:
                return None
            A = float(result.A[row, col])
            mu = float(result.mu[row, col])
            k = float(result.k[row, col])
            m = float(result.m[row, col])
            return A * np.exp(-0.5 * (xf - mu) ** 2 / sigma ** 2) + k * xf + m
        return None

    def _param_text(row, col):
        lines = [f"pixel (row={row}, col={col})"]
        if fstat is not None:
            s = int(fstat[row, col])
            name = STATUS_NAMES.get(s, str(s))
            edge = "  (peak at scan-range edge)" if s == 2 else ""
            lines.append(f"status: {s} — {name}{edge}")
        if is_nd:
            mu = np.asarray(result.mu[row, col])
            lines.append(f"A = {float(result.A[row, col]):.4g}")
            lines.append("mu = [" + ", ".join(f"{v:.4g}" for v in mu) + "]")
            diag = np.diagonal(np.asarray(result.cov[row, col]))
            lines.append("var = [" + ", ".join(f"{v:.3g}" for v in diag) + "]")
            lines.append(f"c = {float(result.c[row, col]):.4g}")
            lines.append(f"success = {float(result.success[row, col]):.3g}")
        elif is_1d:
            lines.append(f"A = {float(result.A[row, col]):.4g}")
            lines.append(f"sigma = {float(result.sigma[row, col]):.4g}")
            lines.append(f"mu = {float(result.mu[row, col]):.4g}")
            lines.append(f"k = {float(result.k[row, col]):.4g}")
            lines.append(f"m = {float(result.m[row, col]):.4g}")
            lines.append(f"success = {float(result.success[row, col]):.3g}")
        return "\n".join(lines)

    def _draw_curves(row, col):
        cube = data[row, col].astype(np.float64)  # shape motor_dims
        title = f"pixel ({row}, {col})"
        if fstat is not None:
            s = int(fstat[row, col])
            title += f"  |  {STATUS_NAMES.get(s, s)}"
            if s == 2:
                title += " (peak at scan-range edge)"
        for i, ax in enumerate(curve_axes):
            ax.clear()
            other = tuple(j for j in range(D) if j != i)
            raw = cube.sum(axis=other) if other else cube
            x = x_vals[i]
            ax.plot(x, raw, "o-", ms=3, lw=1.2, color="C0", label="raw (marginal)")
            if result is not None:
                xf = np.linspace(float(np.min(x)), float(np.max(x)), 200)
                fit = _eval_fit_slice(row, col, i, xf)
                if fit is not None:
                    fmax = float(np.max(fit))
                    rmax = float(np.max(raw))
                    if is_nd and np.isfinite(fmax) and fmax != 0:
                        fit = fit * (rmax / fmax)
                        lbl = "fit (centre slice, scaled)"
                    else:
                        lbl = "fit"
                    ax.plot(xf, fit, "-", lw=1.5, color="C3", label=lbl)
            ax.set_xlabel(f"motor {i}")
            ax.set_ylabel("counts")
            ax.legend(fontsize=7, loc="best")
        curve_axes[0].set_title(title, fontsize=9)
        fig.canvas.draw_idle()

    # ── widgets ────────────────────────────────────────────────────────────
    cmap_w = widgets.Dropdown(options=cmap_options, value=cmap, description="cmap:")
    status_w = None
    if fstat is not None:
        status_w = widgets.Checkbox(value=False, description="overlay status")
    readout = widgets.Output()

    def _on_cmap(_):
        _draw_map()

    cmap_w.observe(_on_cmap, names="value")

    def _on_status(_):
        if status_w.value:
            _add_status_overlay()
        else:
            _clear_status_overlay()
        fig.canvas.draw_idle()

    if status_w is not None:
        status_w.observe(_on_status, names="value")

    def _on_click(event):
        if event.inaxes is not ax_map or event.xdata is None or event.ydata is None:
            return
        col = int(round(event.xdata))
        row = int(round(event.ydata))
        if not (0 <= row < ny and 0 <= col < nx):
            return
        state["row"], state["col"] = row, col
        _draw_marker()
        _draw_curves(row, col)
        with readout:
            readout.clear_output()
            print(_param_text(row, col))

    fig.canvas.mpl_connect("button_press_event", _on_click)
    # expose the handler for headless testing (call it with a mock event that
    # has .inaxes / .xdata / .ydata) without going through the mpl event stack
    fig._starling_on_click = _on_click

    _draw_map()
    for ax in curve_axes:
        ax.set_xlabel("motor")
        ax.set_ylabel("counts")
    curve_axes[0].set_title("click a pixel on the map", fontsize=9)
    fig.tight_layout()

    ctrl_row = [cmap_w] + ([status_w] if status_w is not None else [])
    container = widgets.VBox([widgets.HBox(ctrl_row), readout])
    display(container)
    return container
