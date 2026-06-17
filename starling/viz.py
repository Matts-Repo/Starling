"""Interactive, non-destructive noise-reduction preview (Section 8).

``denoise_widget(dset)`` builds an ipywidgets panel that re-renders a
before/after view as you move the sliders, and only commits the chosen settings
to the dataset when you press **Apply**. Constants live in the notebook / the
slider defaults, not in the library (per the constants decision).

ipywidgets and matplotlib are imported lazily so importing ``starling`` never
requires them.
"""

import numpy as np

from . import preprocess


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
