"""Interactive notebook widgets.

``roi_picker``    — stream a z-sum preview per scan without loading the full
                    dataset; drag or type to pick an ROI before loading.
``denoise_widget`` — non-destructive noise-reduction preview; only commits to
                    the dataset when Apply is pressed.

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


def denoise_widget(dset, bg_n=5, bg_mode="mean", hot_sigma=5.0, roi_threshold=0.05):
    """Interactive denoise preview for a DataSet (JupyterLab, %matplotlib widget).

    Sliders/inputs control the background frame count and mode, the hot-pixel
    sigma and the ROI threshold. Every change recomputes a cheap preview and
    redraws the before/after z-sum, a representative frame and a signal
    histogram, with the proposed ROI box overlaid. The dataset is **not**
    modified until *Apply* is pressed; re-opening the widget always previews from
    the original data.

    Args:
        dset: a ``starling.DataSet`` (uses ``dset.data``).
        bg_n, bg_mode, hot_sigma, roi_threshold: initial control values.

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
    bg_mode_w = widgets.Dropdown(options=["mean", "median"], value=bg_mode, description="bg mode")
    hot_w = widgets.FloatSlider(value=hot_sigma, min=2.0, max=15.0, step=0.5, description="hot σ")
    roi_w = widgets.FloatSlider(value=roi_threshold, min=0.0, max=0.5, step=0.01, description="roi thr")
    apply_btn = widgets.Button(description="Apply", button_style="success")
    status = widgets.Output()

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.2))

    def render(*_):
        pv = preprocess.preview(
            raw, bg_n=bg_n_w.value, bg_mode=bg_mode_w.value,
            hot_sigma=hot_w.value, roi_threshold=roi_w.value,
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
        axes[2].imshow(np.log1p(pv["proc_frame"]), cmap="hot", origin="lower")
        axes[2].set_title("brightest frame (denoised)")
        counts, edges = pv["hist"]
        axes[3].plot(0.5 * (edges[:-1] + edges[1:]), counts, lw=1.2)
        axes[3].set_yscale("log")
        axes[3].set_title("signal histogram")
        fig.canvas.draw_idle()

    def on_apply(_):
        with status:
            status.clear_output()
            bg = preprocess.estimate_background(raw, n_lowest=bg_n_w.value, mode=bg_mode_w.value)
            data = preprocess.subtracted(raw, bg)
            preprocess.remove_hot_pixels(data, n_sigma=hot_w.value)
            roi = preprocess.auto_roi(data, threshold_rel=roi_w.value)
            r1, r2, c1, c2 = roi
            dset.data = np.ascontiguousarray(data[r1:r2, c1:c2])
            dset.roi = roi
            print(
                f"Applied: bg(n={bg_n_w.value}, {bg_mode_w.value}), "
                f"hot σ={hot_w.value}, roi={roi} -> data {dset.data.shape}"
            )

    for w in (bg_n_w, bg_mode_w, hot_w, roi_w):
        w.observe(render, names="value")
    apply_btn.on_click(on_apply)
    render()

    controls = widgets.HBox([bg_n_w, bg_mode_w, hot_w, roi_w, apply_btn])
    box = widgets.VBox([controls, status])
    display(box)
    return box
