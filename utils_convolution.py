"""Interactive visualization: convolution of two independent continuous random variables."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import ipywidgets as widgets
from IPython.display import display
from scipy import stats
from scipy.integrate import quad


COLOR_X = "#2E86AB"
COLOR_Y = "#E94F37"

DIST_NAMES = ("uniform", "exponential", "pareto", "beta", "gamma", "normal")

# Default parameter specs: (name, default)
_PARAM_SPECS: dict[str, list[tuple[str, float]]] = {
    "uniform": [("low", 0.0), ("high", 1.0)],
    "exponential": [("scale", 1.0)],
    "pareto": [("b", 2.0), ("scale", 1.0)],
    "beta": [("a", 2.0), ("b", 2.0)],
    "gamma": [("a", 2.0), ("scale", 1.0)],
    "normal": [("loc", 0.0), ("scale", 1.0)],
}


def _make_float_text(name: str, default: float) -> widgets.FloatText:
    return widgets.FloatText(
        value=default,
        description=f"{name}:",
        layout=widgets.Layout(width="200px"),
        style={"description_width": "72px"},
    )


def build_dist(kind: str, params: dict[str, float]):
    """Return a frozen scipy continuous distribution."""
    k = kind.lower()
    if k == "uniform":
        lo, hi = float(params["low"]), float(params["high"])
        if hi <= lo:
            hi = lo + 1e-6
        return stats.uniform(loc=lo, scale=hi - lo)
    if k == "exponential":
        sc = max(float(params["scale"]), 1e-12)
        return stats.expon(scale=sc)
    if k == "pareto":
        b = max(float(params["b"]), 1e-6)
        sc = max(float(params["scale"]), 1e-12)
        return stats.pareto(b, loc=0.0, scale=sc)
    if k == "beta":
        a = max(float(params["a"]), 1e-8)
        b_ = max(float(params["b"]), 1e-8)
        return stats.beta(a, b_)
    if k == "gamma":
        a = max(float(params["a"]), 1e-8)
        sc = max(float(params["scale"]), 1e-12)
        return stats.gamma(a, scale=sc)
    if k == "normal":
        loc = float(params["loc"])
        sc = max(float(params["scale"]), 1e-12)
        return stats.norm(loc=loc, scale=sc)
    raise ValueError(f"Unknown distribution: {kind}")


def _finite_support(dist, eps: float = 1e-10) -> tuple[float, float]:
    lo, hi = float(dist.ppf(eps)), float(dist.ppf(1.0 - eps))
    if not np.isfinite(lo):
        lo = float(dist.ppf(1e-4))
    if not np.isfinite(hi):
        hi = float(dist.ppf(1.0 - 1e-4))
    if lo >= hi:
        mid = float(dist.mean()) if np.isfinite(dist.mean()) else 0.0
        lo, hi = mid - 5.0, mid + 5.0
    return lo, hi


def _integration_interval(dist_x, dist_y, s: float) -> tuple[float, float] | None:
    x_lo, x_hi = _finite_support(dist_x)
    y_lo, y_hi = _finite_support(dist_y)
    t_lo = max(x_lo, s - y_hi)
    t_hi = min(x_hi, s - y_lo)
    if t_lo >= t_hi:
        return None
    return t_lo, t_hi


def convolution_value(dist_x, dist_y, s: float) -> float:
    """f_{X+Y}(s) = ∫ f_X(t) f_Y(s-t) dt for independent X,Y."""
    iv = _integration_interval(dist_x, dist_y, s)
    if iv is None:
        return 0.0
    t_lo, t_hi = iv

    def integrand(t: float) -> float:
        return float(dist_x.pdf(t) * dist_y.pdf(s - t))

    try:
        val, err = quad(integrand, t_lo, t_hi, limit=200)
        if not np.isfinite(val):
            return 0.0
        return float(val)
    except Exception:
        return 0.0


def _sum_support_range(dist_x, dist_y) -> tuple[float, float]:
    try:
        sx = dist_x.rvs(size=6000, random_state=42)
        sy = dist_y.rvs(size=6000, random_state=42)
        sm = sx + sy
        lo, hi = float(np.quantile(sm, 0.002)), float(np.quantile(sm, 0.998))
        pad = 0.06 * (hi - lo + 1e-9)
        return lo - pad, hi + pad
    except Exception:
        xl0, xh0 = _finite_support(dist_x)
        yl0, yh0 = _finite_support(dist_y)
        return xl0 + yl0, xh0 + yh0


class ConvolutionVisualization:
    """Marginal densities, joint density, slider s, product / convolution overlays."""

    def __init__(self):
        self.x_kind = widgets.Dropdown(options=DIST_NAMES, value="normal", description="X:")
        self.y_kind = widgets.Dropdown(options=DIST_NAMES, value="normal", description="Y:")

        self._x_param_box = widgets.HBox([])
        self._y_param_box = widgets.HBox([])
        self._x_widgets: dict[str, widgets.FloatText] = {}
        self._y_widgets: dict[str, widgets.FloatText] = {}

        self._build_param_widgets("x")
        self._build_param_widgets("y")

        self.x_kind.observe(lambda *_: self._on_kind_change("x"), names="value")
        self.y_kind.observe(lambda *_: self._on_kind_change("y"), names="value")

        self.s_slider = widgets.FloatSlider(
            description="s",
            value=0.0,
            min=-12.0,
            max=12.0,
            step=0.02,
            readout_format=".3f",
            continuous_update=True,
            layout=widgets.Layout(width="420px"),
        )

        self.plot_product = widgets.Checkbox(value=False, description="Plot product", indent=False)
        self.compute_conv = widgets.Checkbox(value=False, description="Compute convolution", indent=False)
        self.reveal_toggle = widgets.ToggleButton(value=False, description="Reveal convolution", tooltip="Show f_S on bottom panel")

        self.save_btn = widgets.Button(description="Save convolution value", layout=widgets.Layout(width="200px"))

        self.conv_readout = widgets.HTML(value="")

        self.saved_points: list[tuple[float, float]] = []

        self._curve_cache: dict | None = None
        self._curve_sig = None
        self._slider_sig = None

        self.out = widgets.Output()

        self.save_btn.on_click(self._on_save)
        for w in [
            self.s_slider,
            self.plot_product,
            self.compute_conv,
            self.reveal_toggle,
            self.x_kind,
            self.y_kind,
            *self._all_param_widgets(),
        ]:
            w.observe(self._render, names="value")

    def _all_param_widgets(self):
        for d in (self._x_widgets, self._y_widgets):
            for w in d.values():
                yield w

    def _on_kind_change(self, which: str) -> None:
        self._build_param_widgets(which)
        self._render()

    def _build_param_widgets(self, which: str) -> None:
        kind_w = self.x_kind if which == "x" else self.y_kind
        box_attr = "_x_param_box" if which == "x" else "_y_param_box"
        dict_attr = "_x_widgets" if which == "x" else "_y_widgets"

        old = getattr(self, dict_attr)
        for w in old.values():
            try:
                w.unobserve(self._render, names="value")
            except Exception:
                pass

        specs = _PARAM_SPECS[kind_w.value]
        newd: dict[str, widgets.FloatText] = {}
        children = []
        for name, default in specs:
            ft = _make_float_text(name, default)
            newd[name] = ft
            children.append(ft)
            ft.observe(self._render, names="value")

        setattr(self, dict_attr, newd)
        getattr(self, box_attr).children = tuple(children)

    def _update_s_slider_range(self) -> None:
        try:
            dx = build_dist(self.x_kind.value, self._read_params("x"))
            dy = build_dist(self.y_kind.value, self._read_params("y"))
            lo, hi = _sum_support_range(dx, dy)
            span = hi - lo
            pad = 0.12 * span
            self.s_slider.min = float(lo - pad)
            self.s_slider.max = float(hi + pad)
            step = max(span / 400.0, 0.01)
            self.s_slider.step = float(step)
            mid = 0.5 * (self.s_slider.min + self.s_slider.max)
            if not (self.s_slider.min <= self.s_slider.value <= self.s_slider.max):
                self.s_slider.value = float(mid)
        except Exception:
            pass

    def _read_params(self, which: str) -> dict[str, float]:
        d = self._x_widgets if which == "x" else self._y_widgets
        return {k: float(w.value) for k, w in d.items()}

    def _dist_sig(self):
        return (
            self.x_kind.value,
            self.y_kind.value,
            tuple(sorted(self._read_params("x").items())),
            tuple(sorted(self._read_params("y").items())),
        )

    def _maybe_update_slider_bounds(self) -> None:
        sig = self._dist_sig()
        if self._slider_sig == sig:
            return
        self._slider_sig = sig
        self._curve_cache = None
        self._curve_sig = None
        self._update_s_slider_range()

    def _on_save(self, *_):
        dx = build_dist(self.x_kind.value, self._read_params("x"))
        dy = build_dist(self.y_kind.value, self._read_params("y"))
        s = float(self.s_slider.value)
        v = convolution_value(dx, dy, s)
        self.saved_points.append((s, v))
        self._render()

    def _ensure_curve_cache(self, dx, dy) -> tuple[np.ndarray, np.ndarray]:
        sig = (
            self.x_kind.value,
            self.y_kind.value,
            tuple(sorted(self._read_params("x").items())),
            tuple(sorted(self._read_params("y").items())),
        )
        if self._curve_cache is not None and self._curve_sig == sig:
            return self._curve_cache

        s_lo, s_hi = _sum_support_range(dx, dy)
        n = 320
        sg = np.linspace(s_lo, s_hi, n)
        fs = np.array([convolution_value(dx, dy, float(t)) for t in sg])
        self._curve_cache = (sg, fs)
        self._curve_sig = sig
        return sg, fs

    def _main_x_bounds(self, dx, dy, s: float) -> tuple[float, float]:
        x_lo, x_hi = _finite_support(dx)
        y_lo, y_hi = _finite_support(dy)
        k_lo, k_hi = s - y_hi, s - y_lo
        lo = min(x_lo, k_lo)
        hi = max(x_hi, k_hi)
        if hi - lo < 1e-8:
            mid = 0.5 * (lo + hi)
            lo, hi = mid - 1.0, mid + 1.0
        pad = 0.03 * (hi - lo)
        return lo - pad, hi + pad

    def _joint_bounds(self, dx, dy) -> tuple[tuple[float, float], tuple[float, float]]:
        x_lo, x_hi = _finite_support(dx)
        y_lo, y_hi = _finite_support(dy)
        x_span = max(x_hi - x_lo, 1e-8)
        y_span = max(y_hi - y_lo, 1e-8)

        # Keep the panel readable when one support is much narrower than the other.
        min_ratio = 0.45
        if x_span < min_ratio * y_span:
            cx = 0.5 * (x_lo + x_hi)
            x_span = min_ratio * y_span
            x_lo, x_hi = cx - 0.5 * x_span, cx + 0.5 * x_span
        elif y_span < min_ratio * x_span:
            cy = 0.5 * (y_lo + y_hi)
            y_span = min_ratio * x_span
            y_lo, y_hi = cy - 0.5 * y_span, cy + 0.5 * y_span

        x_pad = 0.06 * x_span
        y_pad = 0.06 * y_span
        return (x_lo - x_pad, x_hi + x_pad), (y_lo - y_pad, y_hi + y_pad)

    def _render(self, *_):
        self._maybe_update_slider_bounds()
        dx = build_dist(self.x_kind.value, self._read_params("x"))
        dy = build_dist(self.y_kind.value, self._read_params("y"))
        s = float(self.s_slider.value)

        x_lo, x_hi = self._main_x_bounds(dx, dy, s)
        xs = np.linspace(x_lo, x_hi, 900)
        fx = dx.pdf(xs)
        fy_at_x = dy.pdf(xs)
        fy_shift = dy.pdf(s - xs)

        prod = fx * fy_shift
        cval = convolution_value(dx, dy, s)

        (jx_lo, jx_hi), (jy_lo, jy_hi) = self._joint_bounds(dx, dy)
        gx = np.linspace(jx_lo, jx_hi, 160)
        gy = np.linspace(jy_lo, jy_hi, 160)
        X, Y = np.meshgrid(gx, gy)
        Z = dx.pdf(X) * dy.pdf(Y)

        sg, fs = self._ensure_curve_cache(dx, dy)

        with self.out:
            self.out.clear_output(wait=True)
            fig = plt.figure(figsize=(13.5, 8.2), constrained_layout=False)
            gs = GridSpec(
                2,
                2,
                figure=fig,
                width_ratios=[2.35, 1.05],
                height_ratios=[1.45, 1.0],
                wspace=0.34,
                hspace=0.38,
            )
            ax_main = fig.add_subplot(gs[0, 0])
            ax_joint = fig.add_subplot(gs[0, 1])
            ax_conv = fig.add_subplot(gs[1, 0])

            ax_main.plot(xs, fx, color=COLOR_X, lw=2.4, label=r"$f_X(x)$")
            ax_main.plot(xs, fy_at_x, color=COLOR_Y, lw=2.0, ls="-", alpha=0.85, label=r"$f_Y(x)$")
            ax_main.plot(xs, fy_shift, color=COLOR_Y, lw=2.2, ls="--", label=r"$f_Y(s-x)$")

            if self.plot_product.value:
                ax_main.plot(xs, prod, color="#6A4C93", lw=2.0, label=r"$f_X(x)\,f_Y(s-x)$")

            if self.compute_conv.value:
                ax_main.fill_between(xs, 0.0, prod, color="#6A4C93", alpha=0.28)

            ymax = max(
                float(np.nanmax(fx)),
                float(np.nanmax(fy_at_x)),
                float(np.nanmax(fy_shift)),
            )
            if self.plot_product.value or self.compute_conv.value:
                ymax = max(ymax, float(np.nanmax(prod)))
            ax_main.set_ylim(0.0, ymax * 1.12 + 1e-12)

            ax_main.set_xlim(x_lo, x_hi)
            ax_main.set_xlabel(r"$x$")
            ax_main.set_ylabel("density")
            ax_main.set_title(r"Marginal densities and kernel $f_Y(s-x)$")
            ax_main.legend(loc="upper right", fontsize=9)

            cf = ax_joint.contourf(X, Y, Z, levels=28, cmap="viridis", alpha=0.95)
            ax_joint.contour(X, Y, Z, levels=10, colors="k", linewidths=0.35, alpha=0.55)
            fig.colorbar(cf, ax=ax_joint, fraction=0.046, pad=0.04, label=r"$f_X(x)\,f_Y(y)$")
            xs_line = np.array([jx_lo, jx_hi])
            ax_joint.plot(xs_line, s - xs_line, color="#FFDD57", lw=2.2, ls="-", label=r"$y=s-x$")
            ax_joint.set_xlim(jx_lo, jx_hi)
            ax_joint.set_ylim(jy_lo, jy_hi)
            ax_joint.set_aspect("auto")
            ax_joint.set_box_aspect(1.0)
            ax_joint.set_xlabel(r"$x$")
            ax_joint.set_ylabel(r"$y$")
            ax_joint.set_title("Joint density (independent)")
            ax_joint.legend(loc="upper right", fontsize=9)

            ax_conv.set_xlim(sg.min(), sg.max())
            y_candidates = [float(np.nanmax(fs)), float(cval)]
            for _, vv in self.saved_points:
                y_candidates.append(float(vv))
            y2max = max(y_candidates) * 1.12 + 1e-12
            ax_conv.set_ylim(0.0, y2max)

            if self.reveal_toggle.value:
                ax_conv.plot(sg, fs, color="#111111", lw=2.0, label=r"$f_S$ (numerical)")

            if self.compute_conv.value:
                ax_conv.scatter([s], [cval], color="#C73E1D", s=55, zorder=6, label=r"$(s,\,f_S(s))$")

            for (ss, vv) in self.saved_points:
                ax_conv.scatter([ss], [vv], color="#2ECC71", s=42, zorder=5, marker="s", edgecolors="#1B5E20", linewidths=0.6)

            ax_conv.set_xlabel(r"$s$")
            ax_conv.set_ylabel(r"$f_{X+Y}(s)$")
            ax_conv.set_title("Convolution / sum density")
            ax_conv.legend(loc="upper right", fontsize=9)
            ax_conv.grid(True, alpha=0.25)

            fig.suptitle(
                r"Convolution: independent $X$ and $Y$ — compare $f_X$, $f_Y$, and $f_Y(s-x)$",
                y=0.98,
                fontsize=12,
            )
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()

        if self.compute_conv.value:
            self.conv_readout.value = (
                f"<b>Shaded area / convolution at <i>s</i> = {s:.6g}:</b> {cval:.8g}"
            )
        else:
            self.conv_readout.value = ""


    def display(self) -> None:
        controls_x = widgets.VBox(
            [
                widgets.HTML("<b>Distribution for <i>X</i></b>"),
                widgets.HBox([self.x_kind, self._x_param_box]),
            ]
        )
        controls_y = widgets.VBox(
            [
                widgets.HTML("<b>Distribution for <i>Y</i></b>"),
                widgets.HBox([self.y_kind, self._y_param_box]),
            ]
        )
        top = widgets.HBox(
            [controls_x, controls_y],
            layout=widgets.Layout(gap="32px", flex_wrap="wrap"),
        )
        row = widgets.HBox(
            [
                self.s_slider,
                self.plot_product,
                self.compute_conv,
                self.reveal_toggle,
                self.save_btn,
            ],
            layout=widgets.Layout(gap="16px", flex_wrap="wrap", align_items="center"),
        )
        ui = widgets.VBox(
            [
                widgets.HTML(
                    "<b>Convolution explorer</b> — choose <i>X</i> and <i>Y</i>, move <i>s</i>, "
                    "and optionally show the product integral and the sum density."
                ),
                top,
                row,
                self.conv_readout,
                self.out,
            ]
        )
        display(ui)
        self._render()


def show_convolution() -> ConvolutionVisualization:
    """Interactive convolution plots (ipywidgets + matplotlib)."""
    viz = ConvolutionVisualization()
    viz.display()
    return viz
