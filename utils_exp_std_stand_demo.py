"""Interactive expected value, spread, and standardization demo."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from scipy import stats


DIST_OPTIONS = ("uniform", "exponential", "pareto", "beta", "gamma", "normal")
TEXTBOOK_DIST_OPTIONS = ("uniform", "exponential", "normal")

# Plot elements — buttons use the same accent (border) with a light matching fill.
COLOR_HIST = "#4C78A8"
COLOR_DENSITY = "#222222"
COLOR_MEAN = "#D62728"
COLOR_MEDIAN = "#9467BD"
COLOR_MAD = "#F2B134"
COLOR_SD = "#59A14F"

_BTN_LAYOUT = {"border_radius": "6px", "padding": "0 10px"}
_BTN_MEAN_FILL = "#FAD4D4"
_BTN_MEDIAN_FILL = "#E8DCF5"
_BTN_MAD_FILL = "#FFF0CC"
_BTN_SD_FILL = "#D8EED9"
_BTN_STD_FILL = "#D4E4F4"


def _accent_button(description: str, accent_hex: str, fill_hex: str) -> widgets.Button:
    return widgets.Button(
        description=description,
        style=widgets.ButtonStyle(button_color=fill_hex),
        layout=widgets.Layout(border=f"2px solid {accent_hex}", **_BTN_LAYOUT),
    )

DIST_PARAM_SPECS: dict[str, list[tuple[str, float]]] = {
    "uniform": [("low", 0.0), ("high", 1.0)],
    "exponential": [("scale", 1.0)],
    "pareto": [("shape", 2.5), ("scale", 1.0)],
    "beta": [("alpha", 2.0), ("beta", 2.0)],
    "gamma": [("shape", 2.0), ("scale", 1.0)],
    "normal": [("mean", 0.0), ("std", 1.0)],
}


def _build_dist(name: str, params: dict[str, float]):
    if name == "uniform":
        low = float(params["low"])
        high = float(params["high"])
        if high <= low:
            high = low + 1e-6
        return stats.uniform(loc=low, scale=high - low)
    if name == "exponential":
        scale = max(float(params["scale"]), 1e-12)
        return stats.expon(scale=scale)
    if name == "pareto":
        shape = max(float(params["shape"]), 1e-8)
        scale = max(float(params["scale"]), 1e-8)
        return stats.pareto(shape, loc=0.0, scale=scale)
    if name == "beta":
        alpha = max(float(params["alpha"]), 1e-8)
        beta = max(float(params["beta"]), 1e-8)
        return stats.beta(alpha, beta)
    if name == "gamma":
        shape = max(float(params["shape"]), 1e-8)
        scale = max(float(params["scale"]), 1e-8)
        return stats.gamma(shape, scale=scale)
    if name == "normal":
        mean = float(params["mean"])
        std = max(float(params["std"]), 1e-8)
        return stats.norm(loc=mean, scale=std)
    raise ValueError(f"Unknown distribution {name!r}")


class ExpStdStandDemo:
    def __init__(self, textbook: bool = False, centrality_only: bool = False):
        self.textbook = textbook
        self.centrality_only = centrality_only
        self._show_mean = False
        self._show_median = False
        self._show_mad = False
        self._show_sd = False
        self._standardized = False

        options = TEXTBOOK_DIST_OPTIONS if textbook else DIST_OPTIONS
        self.dist_dropdown = widgets.Dropdown(
            options=options,
            value=options[0],
            description="Distribution:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="240px"),
        )

        self.param_widgets: dict[str, widgets.FloatText] = {}
        self.param_box = widgets.HBox([])
        self._build_param_widgets()
        self.dist_dropdown.observe(self._on_dist_change, names="value")

        self.reveal_mean_button = _accent_button("Reveal Expected Value (μ)", COLOR_MEAN, _BTN_MEAN_FILL)
        self.reveal_median_button = _accent_button("Reveal Median", COLOR_MEDIAN, _BTN_MEDIAN_FILL)
        self.reveal_mad_button = _accent_button("Reveal MAD", COLOR_MAD, _BTN_MAD_FILL)
        self.reveal_sd_button = _accent_button("Reveal SD (σ)", COLOR_SD, _BTN_SD_FILL)
        self.standardize_button = _accent_button(
            "Standardize",
            COLOR_HIST,
            _BTN_STD_FILL,
        )

        self.reveal_mean_button.on_click(self._toggle_mean)
        self.reveal_median_button.on_click(self._toggle_median)
        self.reveal_mad_button.on_click(self._toggle_mad)
        self.reveal_sd_button.on_click(self._toggle_sd)
        self.standardize_button.on_click(self._toggle_standardize)

        if centrality_only:
            # Per request: only allow reveal buttons (3) and (4).
            self.reveal_mean_button.disabled = True
            self.reveal_median_button.disabled = True

        self.info_html = widgets.HTML()
        self.output = widgets.Output()

        for widget in self.param_widgets.values():
            widget.observe(self._render, names="value")

    def _build_param_widgets(self) -> None:
        for widget in self.param_widgets.values():
            try:
                widget.unobserve(self._render, names="value")
            except Exception:
                pass

        specs = DIST_PARAM_SPECS[self.dist_dropdown.value]
        self.param_widgets = {}
        children = []
        for name, default in specs:
            field = widgets.FloatText(
                value=default,
                description=f"{name}:",
                style={"description_width": "70px"},
                layout=widgets.Layout(width="170px"),
            )
            field.observe(self._render, names="value")
            self.param_widgets[name] = field
            children.append(field)
        self.param_box.children = tuple(children)

    def _on_dist_change(self, *_args) -> None:
        self._show_mean = False
        self._show_median = False
        self._show_mad = False
        self._show_sd = False
        self._standardized = False
        self._build_param_widgets()
        self._render()

    def _params(self) -> dict[str, float]:
        return {k: float(v.value) for k, v in self.param_widgets.items()}

    def _toggle_mean(self, _btn) -> None:
        self._show_mean = not self._show_mean
        self._render()

    def _toggle_median(self, _btn) -> None:
        self._show_median = not self._show_median
        self._render()

    def _toggle_mad(self, _btn) -> None:
        self._show_mad = not self._show_mad
        self._render()

    def _toggle_sd(self, _btn) -> None:
        self._show_sd = not self._show_sd
        self._render()

    def _toggle_standardize(self, _btn) -> None:
        self._standardized = not self._standardized
        self._render()

    def _display_range(self, dist) -> tuple[float, float]:
        lo = float(dist.ppf(0.001))
        hi = float(dist.ppf(0.999))
        if not np.isfinite(lo):
            lo = float(dist.ppf(0.01))
        if not np.isfinite(hi):
            hi = float(dist.ppf(0.99))
        if not np.isfinite(lo):
            lo = -5.0
        if not np.isfinite(hi):
            hi = 5.0
        if hi <= lo:
            mid = float(dist.mean()) if np.isfinite(dist.mean()) else 0.0
            lo, hi = mid - 5.0, mid + 5.0
        return lo, hi

    def _render(self, *_args) -> None:
        dist = _build_dist(self.dist_dropdown.value, self._params())
        mu = float(dist.mean())
        median = float(dist.median())
        sd = float(dist.std())
        mad = float(np.mean(np.abs(dist.rvs(size=60000, random_state=17) - mu)))

        lo, hi = self._display_range(dist)
        xs = np.linspace(lo, hi, 900)
        pdf = dist.pdf(xs)

        samples = dist.rvs(size=8000, random_state=123)
        # Always draw histogram and density in original x; otherwise density=True
        # rescales bar heights when bin widths change under z = (x - μ) / σ.
        hist_x = samples
        line_x = xs
        mean_x = mu
        median_x = median
        mad_band = (mu - mad, mu + mad)
        sd_band = (mu - sd, mu + sd)

        xlabel = r"$x$"
        xticks = None
        xticklabels = None
        if self._standardized and sd > 1e-12:
            # Same plot as before; only tick positions/labels show μ + kσ in σ-units.
            ks = np.arange(-3, 4, dtype=float)
            xticks = (mu + ks * sd).tolist()
            xticklabels = [
                r"$-3 \sigma$",
                r"$-2 \sigma$",
                r"$-1 \sigma$",
                r"$\mu$",
                r"$1 \sigma$",
                r"$2 \sigma$",
                r"$3 \sigma$",
            ]
            xlabel = r"$x$ (tick marks at $\mu + k\sigma$)"
            pairs = [(t, lab) for t, lab in zip(xticks, xticklabels, strict=True) if lo - 1e-9 <= t <= hi + 1e-9]
            if pairs:
                xticks, xticklabels = zip(*pairs)
                xticks, xticklabels = list(xticks), list(xticklabels)
            else:
                xticks, xticklabels = None, None

        with self.output:
            self.output.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(hist_x, bins=45, density=True, alpha=0.35, color=COLOR_HIST, edgecolor="white")
            ax.plot(line_x, pdf, color=COLOR_DENSITY, lw=2.0, label="Density")

            if self._show_mad:
                ax.axvspan(mad_band[0], mad_band[1], color=COLOR_MAD, alpha=0.25, label="μ ± MAD")
            if self._show_sd:
                ax.axvspan(sd_band[0], sd_band[1], color=COLOR_SD, alpha=0.2, label="μ ± SD")
            if self._show_mean:
                ax.axvline(mean_x, color=COLOR_MEAN, lw=2.2, ls="--", label="Expected value")
            if self._show_median:
                ax.axvline(median_x, color=COLOR_MEDIAN, lw=2.2, ls="-.", label="Median")

            ax.set_title(f"Probability histogram: {self.dist_dropdown.value}")
            ax.set_ylabel("Density")
            ax.set_xlabel(xlabel)
            ax.set_xlim(lo, hi)
            if xticks is not None:
                ax.set_xticks(xticks)
                ax.set_xticklabels(xticklabels)
            ax.grid(alpha=0.2)
            ax.legend(loc="upper right")
            plt.show()

        lines = []
        if self._show_mean:
            lines.append(f"Expected value (μ): <b>{mu:.2f}</b>")
        if self._show_median:
            lines.append(f"Median: <b>{median:.2f}</b>")
        if self._show_mad:
            lines.append(f"MAD: <b>{mad:.2f}</b>")
        if self._show_sd:
            lines.append(f"SD (σ): <b>{sd:.2f}</b>")
        if not lines:
            lines.append("Use the reveal buttons to display centrality/spread values.")
        self.info_html.value = "<br/>".join(lines)

    def display(self) -> None:
        row1 = widgets.HBox([self.dist_dropdown, self.param_box], layout=widgets.Layout(gap="10px", flex_wrap="wrap"))
        row2 = widgets.HBox(
            [
                self.reveal_mean_button,
                self.reveal_median_button,
                self.reveal_mad_button,
                self.reveal_sd_button,
                self.standardize_button,
            ],
            layout=widgets.Layout(gap="8px", flex_wrap="wrap"),
        )
        ui = widgets.VBox([row1, row2, self.info_html, self.output])
        display(ui)
        self._render()


def show_exp_std_stand_demo(textbook: bool = False, centrality_only: bool = False) -> ExpStdStandDemo:
    demo = ExpStdStandDemo(textbook=textbook, centrality_only=centrality_only)
    demo.display()
    return demo
