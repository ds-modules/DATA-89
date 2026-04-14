"""
Joint distribution table → 3D histogram for independent Beta(α,β) marginals on [0,1]².
Used by joint_distribution_demo.ipynb.
"""

from __future__ import annotations

import numpy as np
import ipywidgets as widgets
import plotly.graph_objects as go
from IPython.display import clear_output, display
from scipy import stats


def _bin_edges_unit_interval(delta: float) -> np.ndarray:
    """Edges 0 = e0 < e1 < … < eN = 1 with each step at most delta (last bin may be shorter)."""
    delta = float(delta)
    if delta <= 0:
        raise ValueError("delta must be positive")
    edges = [0.0]
    while edges[-1] < 1.0 - 1e-15:
        edges.append(min(1.0, edges[-1] + delta))
    return np.array(edges, dtype=float)


def _snap_to_grid(v: float, delta: float) -> float:
    delta = float(delta)
    if delta <= 0:
        return 0.0
    k = int(round(v / delta))
    return float(np.clip(k * delta, 0.0, 1.0))


def _cell_prob_independent_beta(
    x_lo: float,
    x_hi: float,
    y_lo: float,
    y_hi: float,
    dist_x: stats.rv_continuous,
    dist_y: stats.rv_continuous,
) -> float:
    px = dist_x.cdf(x_hi) - dist_x.cdf(x_lo)
    py = dist_y.cdf(y_hi) - dist_y.cdf(y_lo)
    return float(px * py)


def _rectangle_prob(dist_x: stats.rv_continuous, dist_y: stats.rv_continuous, a, b, c, d) -> float:
    return float((dist_x.cdf(b) - dist_x.cdf(a)) * (dist_y.cdf(d) - dist_y.cdf(c)))


def _append_box_mesh(
    vertices: list,
    i_list: list,
    j_list: list,
    k_list: list,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    z0: float,
    z1: float,
) -> None:
    b = len(vertices)
    vertices.extend(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ]
    )
    faces = [
        (b + 0, b + 1, b + 2),
        (b + 0, b + 2, b + 3),
        (b + 4, b + 5, b + 6),
        (b + 4, b + 6, b + 7),
        (b + 0, b + 1, b + 5),
        (b + 0, b + 5, b + 4),
        (b + 1, b + 2, b + 6),
        (b + 1, b + 6, b + 5),
        (b + 2, b + 3, b + 7),
        (b + 2, b + 7, b + 6),
        (b + 3, b + 0, b + 4),
        (b + 3, b + 4, b + 7),
    ]
    for i, j, k in faces:
        i_list.append(i)
        j_list.append(j)
        k_list.append(k)


def _build_batched_mesh3d(
    cells: list[tuple[float, float, float, float, float, float]],
) -> tuple[list, list, list, list, list, list, list] | None:
    """
    cells: list of (x0, x1, y0, y1, prob, z_height) — z_height already set per mode.
    Returns (x, y, z, i, j, k, intensity) for Mesh3d, or None if empty.
    """
    if not cells:
        return None

    vertices: list = []
    i_list: list[int] = []
    j_list: list[int] = []
    k_list: list[int] = []
    intensity: list[float] = []

    for x0, x1, y0, y1, prob, z_top in cells:
        z0 = 0.0
        _append_box_mesh(vertices, i_list, j_list, k_list, x0, x1, y0, y1, z0, z_top)
        # Color by bar height (same on all 8 vertices of the box)
        h = float(z_top)
        intensity.extend([h] * 8)

    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    zs = [v[2] for v in vertices]

    return xs, ys, zs, i_list, j_list, k_list, intensity


class JointDistributionVisualizer:
    def __init__(self):
        self.delta_slider = widgets.FloatSlider(
            value=0.1,
            min=0.02,
            max=0.25,
            step=0.01,
            continuous_update=False,
            description="Δx = Δy",
            readout_format=".2f",
        )
        self.ax_slider = widgets.FloatSlider(
            value=2.0, min=0.3, max=8.0, step=0.1, description="α for X", readout_format=".1f"
        )
        self.bx_slider = widgets.FloatSlider(
            value=2.0, min=0.3, max=8.0, step=0.1, description="β for X", readout_format=".1f"
        )
        self.ay_slider = widgets.FloatSlider(
            value=2.0, min=0.3, max=8.0, step=0.1, description="α for Y", readout_format=".1f"
        )
        self.by_slider = widgets.FloatSlider(
            value=2.0, min=0.3, max=8.0, step=0.1, description="β for Y", readout_format=".1f"
        )
        self.mode_toggle = widgets.ToggleButtons(
            options=[
                ("P (height)", "chance"),
                ("P / area (height)", "density"),
            ],
            description="Vertical axis",
        )
        self.a_slider = widgets.FloatSlider(
            value=0.0, min=0.0, max=1.0, step=0.1, description="a (X lower)", readout_format=".2f"
        )
        self.b_slider = widgets.FloatSlider(
            value=1.0, min=0.0, max=1.0, step=0.1, description="b (X upper)", readout_format=".2f"
        )
        self.c_slider = widgets.FloatSlider(
            value=0.0, min=0.0, max=1.0, step=0.1, description="c (Y lower)", readout_format=".2f"
        )
        self.d_slider = widgets.FloatSlider(
            value=1.0, min=0.0, max=1.0, step=0.1, description="d (Y upper)", readout_format=".2f"
        )
        self.summary_box = widgets.HTML()
        self.fig = go.FigureWidget()
        self._did_set_initial_camera = False
        self._last_camera = None

        def _on_relayout(change, _self=self):
            # Keep track of the *current* camera whenever the user rotates/zooms
            # or whenever the view buttons set a new camera.
            try:
                keys = change or {}
                if any(str(k).startswith("scene.camera") for k in keys.keys()):
                    _self._last_camera = _self.fig.layout.scene.camera.to_plotly_json()
            except Exception:
                # If we can't parse camera (e.g., during figure init), just skip.
                return

        self.fig.on_relayout(_on_relayout)
        self.plot_output = widgets.Output(
            layout=widgets.Layout(
                width="100%",
                max_width="980px",
                min_width="480px",
                overflow="visible",
            )
        )

        self.delta_slider.observe(self._on_delta_change, names="value")
        for w in (
            self.ax_slider,
            self.bx_slider,
            self.ay_slider,
            self.by_slider,
            self.mode_toggle,
            self.a_slider,
            self.b_slider,
            self.c_slider,
            self.d_slider,
        ):
            w.observe(self._redraw, names="value")

        self._on_delta_change({"owner": self.delta_slider, "new": self.delta_slider.value})

    def _on_delta_change(self, change):
        d = float(change["new"])
        d = max(0.02, min(0.25, d))
        for s in (self.a_slider, self.b_slider, self.c_slider, self.d_slider):
            s.step = d
            s.min = 0.0
            s.max = 1.0
            s.value = _snap_to_grid(float(s.value), d)
        self._enforce_order()
        self._redraw()

    def _enforce_order(self):
        d = float(self.delta_slider.value)
        a = _snap_to_grid(float(self.a_slider.value), d)
        b = _snap_to_grid(float(self.b_slider.value), d)
        c = _snap_to_grid(float(self.c_slider.value), d)
        dd = _snap_to_grid(float(self.d_slider.value), d)
        if a > b:
            a, b = b, a
        if c > dd:
            c, dd = dd, c
        self.a_slider.unobserve(self._redraw, names="value")
        self.b_slider.unobserve(self._redraw, names="value")
        self.c_slider.unobserve(self._redraw, names="value")
        self.d_slider.unobserve(self._redraw, names="value")
        self.a_slider.value = a
        self.b_slider.value = b
        self.c_slider.value = c
        self.d_slider.value = dd
        self.a_slider.observe(self._redraw, names="value")
        self.b_slider.observe(self._redraw, names="value")
        self.c_slider.observe(self._redraw, names="value")
        self.d_slider.observe(self._redraw, names="value")

    def _redraw(self, *_):
        self._enforce_order()
        delta = float(self.delta_slider.value)
        a = float(self.a_slider.value)
        b = float(self.b_slider.value)
        c = float(self.c_slider.value)
        d = float(self.d_slider.value)

        dist_x = stats.beta(self.ax_slider.value, self.bx_slider.value)
        dist_y = stats.beta(self.ay_slider.value, self.by_slider.value)

        edges = _bin_edges_unit_interval(delta)
        nx = len(edges) - 1
        mode = self.mode_toggle.value

        inside_cells: list[tuple[float, float, float, float, float, float]] = []
        outside_cells: list[tuple[float, float, float, float, float, float]] = []

        max_z = 0.0
        max_prob = 0.0

        for i in range(nx):
            x0, x1 = edges[i], edges[i + 1]
            for j in range(nx):
                y0, y1 = edges[j], edges[j + 1]
                prob = _cell_prob_independent_beta(x0, x1, y0, y1, dist_x, dist_y)
                area = max((x1 - x0) * (y1 - y0), 1e-15)
                if mode == "chance":
                    z_top = prob
                else:
                    z_top = prob / area
                max_z = max(max_z, z_top)
                max_prob = max(max_prob, prob)

                inside = (x0 >= a - 1e-12) and (x1 <= b + 1e-12) and (y0 >= c - 1e-12) and (y1 <= d + 1e-12)
                tup = (x0, x1, y0, y1, prob, z_top)
                if inside:
                    inside_cells.append(tup)
                else:
                    outside_cells.append(tup)

        prob_rect = _rectangle_prob(dist_x, dist_y, a, b, c, d)

        traces = []
        z_title = (
            "Height = P(cell)" if mode == "chance" else "Height = P(cell) / (cell area)"
        )
        cbar_title = "Bar height" if mode == "chance" else "Bar height (prob / area)"

        inside_mesh = _build_batched_mesh3d(inside_cells)
        outside_mesh = _build_batched_mesh3d(outside_cells)

        if outside_mesh is not None:
            xs, ys, zs, ii, jj, kk, _ = outside_mesh
            n = len(xs)
            grey_rgb = "rgb(150,150,150)"
            traces.append(
                go.Mesh3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    i=ii,
                    j=jj,
                    k=kk,
                    vertexcolor=[grey_rgb] * n,
                    name="Outside selection",
                    showlegend=False,
                    opacity=0.38,
                    lighting=dict(ambient=0.9, diffuse=0.5, specular=0.1),
                )
            )

        if inside_mesh is not None:
            xs, ys, zs, ii, jj, kk, intensity = inside_mesh
            traces.append(
                go.Mesh3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    i=ii,
                    j=jj,
                    k=kk,
                    intensity=intensity,
                    colorscale="Viridis",
                    cmin=0,
                    cmax=max(max_z, 1e-12),
                    colorbar=dict(
                        title=cbar_title,
                        x=1.02,
                        xanchor="left",
                        xpad=28,
                        len=0.78,
                        y=0.5,
                        yanchor="middle",
                        thickness=18,
                    ),
                    lighting=dict(ambient=0.65, diffuse=0.85, specular=0.4),
                    name="Selected region",
                    showlegend=False,
                )
            )

        if inside_mesh is None and outside_mesh is not None:
            traces.append(
                go.Scatter3d(
                    x=[0.5],
                    y=[0.5],
                    z=[0.0],
                    mode="markers",
                    marker=dict(
                        size=6,
                        color=[max_z],
                        colorscale="Viridis",
                        cmin=0,
                        cmax=max(max_z, 1e-12),
                        showscale=True,
                        colorbar=dict(
                            title=cbar_title,
                            x=1.02,
                            xanchor="left",
                            xpad=28,
                            len=0.78,
                            y=0.5,
                            yanchor="middle",
                            thickness=18,
                        ),
                        opacity=0.0,
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        if not traces:
            traces.append(
                go.Scatter3d(x=[0], y=[0], z=[0], mode="markers", marker=dict(size=2, opacity=0))
            )

        z_max = max(max_z * 1.08, 1e-6)
        z_center = float(0.45 * z_max)
        # Perspective: lower eye z vs diagonal distance so bars read as 3D (not bird's-eye).
        # Pull back so the full [0,1]² footprint stays in frame.
        cam_perspective = dict(
            eye=dict(x=2.95, y=-2.95, z=1.38),
            center=dict(x=0.5, y=0.5, z=z_center),
            up=dict(x=0, y=0, z=1),
        )
        cam_top = dict(
            eye=dict(x=0.0, y=0.0, z=3.85),
            center=dict(x=0.5, y=0.5, z=0.0),
            up=dict(x=0, y=1, z=0),
        )

        layout_update = dict(
            title=dict(
                text="Joint distribution on [0,1]² (independent Beta marginals)",
                x=0.5,
                xanchor="center",
                y=0.97,
                yanchor="top",
            ),
            template="plotly_white",
            autosize=True,
            height=700,
            margin=dict(l=150, r=160, t=78, b=150),
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    x=0.5,
                    y=1.005,
                    xanchor="center",
                    yanchor="bottom",
                    showactive=False,
                    buttons=[
                        dict(
                            label="View from above (heatmap)",
                            method="relayout",
                            args=[{"scene.camera": cam_top}],
                        ),
                        dict(
                            label="3D perspective",
                            method="relayout",
                            args=[{"scene.camera": cam_perspective}],
                        ),
                    ],
                )
            ],
            scene=dict(
                # Slightly narrower x-domain so the 3D scene sits left of the colorbar with more gap.
                domain=dict(x=[0.07, 0.84], y=[0.10, 0.90]),
                xaxis=dict(
                    title="X",
                    range=[-0.05, 1.05],
                    showbackground=True,
                    gridcolor="rgba(0,0,0,0.12)",
                ),
                yaxis=dict(
                    title="Y",
                    range=[-0.05, 1.05],
                    showbackground=True,
                    gridcolor="rgba(0,0,0,0.12)",
                ),
                zaxis=dict(
                    title=z_title,
                    range=[-0.02 * z_max if z_max > 0 else 0, z_max * 1.06],
                    showbackground=True,
                    gridcolor="rgba(0,0,0,0.12)",
                ),
                aspectmode="manual",
                # Larger z ratio so bar heights are visible in perspective (was too flat).
                aspectratio=dict(x=1, y=1, z=0.95 if z_max > 0 else 1),
                camera=cam_perspective,
            ),
            # Keep the user's current rotation/zoom when sliders change.
            uirevision="joint_dist_v1",
        )

        summary = (
            f"<b>P(X ∈ [a, b], Y ∈ [c, d])</b> with independent marginals "
            f"X ~ Beta({self.ax_slider.value:.2f}, {self.bx_slider.value:.2f}), "
            f"Y ~ Beta({self.ay_slider.value:.2f}, {self.by_slider.value:.2f}).<br>"
            f"<span style='font-size:18px'>P = <b>{prob_rect:.6f}</b></span><br>"
            f"<span style='color:#555'>Sliders a, b, c, d are multiples of Δ = {delta:.4f}. "
            f"Max cell probability ≈ {max_prob:.5f}.</span>"
        )

        self.summary_box.value = summary

        # Update the existing FigureWidget so the camera angle stays where the student left it.
        camera_to_keep = self._last_camera
        if camera_to_keep is None:
            try:
                camera_to_keep = self.fig.layout.scene.camera.to_plotly_json()
            except Exception:
                camera_to_keep = None

        with self.fig.batch_update():
            # Replacing `data` can reset view in some notebook frontends; we reapply the camera after.
            self.fig.data = traces
            self.fig.layout.update(layout_update)
            if not self._did_set_initial_camera:
                self.fig.layout.scene.camera = cam_perspective
                self._did_set_initial_camera = True

            # Always restore the user's last camera (also preserves heatmap vs 3D perspective).
            if camera_to_keep is not None:
                self.fig.layout.scene.camera = camera_to_keep
                self._last_camera = camera_to_keep

        with self.plot_output:
            clear_output(wait=True)
            display(widgets.HTML(summary))
            display(
                widgets.HTML(
                    "<style>"
                    ".plotly-graph-div { margin-left: auto !important; margin-right: auto !important; }"
                    "</style>"
                )
            )
            fig.show(
                config={
                    "responsive": True,
                    "displaylogo": False,
                    "scrollZoom": True,
                }
            )

    def display(self):
        intro = widgets.HTML(
            "<tition [0,1] with width <b>Δ</b> (last bin may be shorter if 1/Δ is not an integer). "
            "Each bar height is the cell probability, or probability divided by cell area (Riemann sum for the joint density). "
            "Rotate the plot or use <b>View from above</b> to read the table as a heatmap; the colorbar matches bar height.</p>"
        )
        controls = widgets.VBox(
            [
                intro,
                self.delta_slider,
                widgets.HTML("<b>Beta parameters (marginals on [0,1])</b>"),
                widgets.HBox([self.ax_slider, self.bx_slider]),
                widgets.HBox([self.ay_slider, self.by_slider]),
                self.mode_toggle,
                widgets.HTML("<b>Rectangle [a,b] × [c,d] (values snap to multiples of Δ)</b>"),
                widgets.HBox([self.a_slider, self.b_slider]),
                widgets.HBox([self.c_slider, self.d_slider]),
            ],
            layout=widgets.Layout(
                flex="0 0 auto",
                width="440px",
                max_width="440px",
                min_width="280px",
            ),
        )
        plot_col = widgets.VBox(
            [self.plot_output],
            layout=widgets.Layout(
                flex="1 1 0%",
                min_width="500px",
                width="auto",
                align_items="center",
                padding="0 8px 0 16px",
            ),
        )
        row = widgets.HBox(
            [controls, plot_col],
            layout=widgets.Layout(width="100%", align_items="flex-start"),
        )
        display(row)


def run_joint_distribution_demo():
    viz = JointDistributionVisualizer()
    viz.display()
    return viz
