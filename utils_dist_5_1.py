"""
Alternate distribution plotter for Data 89 §5.1 (Tails and Rare Events).
Adds: Student-t (continuous), Power law (discrete), and log-scale toggles for axes.
"""

import numpy as np
import ipywidgets as widgets
import plotly.graph_objects as go
from IPython.display import display, clear_output
from scipy import stats

# Import base implementation and extend
from utils_dist import (
    sample_distribution as _sample_dist_base,
    compute_pdf_pmf as _compute_pdf_pmf_base,
    compute_true_probability as _compute_true_probability_base,
    DistributionProbabilityVisualization,
)

# For power law with exponent a <= 1, infinite sum diverges; use truncated support 1..N
POWER_LAW_N = 5000


def _power_law_pmf(x_int, a):
    """PMF for power law: P(X=x) ∝ x^{-a}. For a>1 use Zipf; for a<=1 use truncated on 1..POWER_LAW_N."""
    x_int = np.asarray(x_int, dtype=int)
    if a > 1:
        return stats.zipf.pmf(x_int, a)
    # Truncated: support 1..POWER_LAW_N
    z = np.sum(np.arange(1, POWER_LAW_N + 1, dtype=float) ** (-a))
    return np.where(
        (x_int >= 1) & (x_int <= POWER_LAW_N),
        (x_int.astype(float) ** (-a)) / z,
        0.0,
    )


def _power_law_cdf(x_int, a):
    """CDF for power law (integer x)."""
    x_int = int(np.round(x_int))
    if x_int < 1:
        return 0.0
    if a > 1:
        return stats.zipf.cdf(x_int, a)
    k = np.arange(1, min(x_int, POWER_LAW_N) + 1, dtype=float)
    z = np.sum(np.arange(1, POWER_LAW_N + 1, dtype=float) ** (-a))
    return np.sum(k ** (-a)) / z


def sample_distribution(dist_type, dist_category, n_samples, **params):
    """Sample from the specified distribution (includes Student-t and Power law)."""
    if dist_category == "Continuous" and dist_type == "Student-t":
        df = params.get("df", 3.0)
        return stats.t.rvs(df, size=n_samples)
    if dist_category == "Discrete" and dist_type == "Power law":
        a = params.get("a", 2.0)
        if a > 1:
            return stats.zipf.rvs(a, size=n_samples)
        # Truncated power law: sample by inverse CDF
        k = np.arange(1, POWER_LAW_N + 1, dtype=float)
        probs = (k ** (-a)) / np.sum(k ** (-a))
        return np.random.choice(k.astype(int), size=n_samples, p=probs)
    return _sample_dist_base(dist_type, dist_category, n_samples, **params)


def compute_pdf_pmf(x_values, dist_type, dist_category, **params):
    """Compute PDF (continuous) or PMF (discrete), including Student-t and Power law."""
    if dist_category == "Continuous" and dist_type == "Student-t":
        df = params.get("df", 3.0)
        return stats.t.pdf(x_values, df)
    if dist_category == "Discrete" and dist_type == "Power law":
        a = params.get("a", 2.0)
        return _power_law_pmf(np.round(x_values).astype(int), a)
    return _compute_pdf_pmf_base(x_values, dist_type, dist_category, **params)


def compute_true_probability(dist_type, dist_category, prob_type, bound1, bound2, **params):
    """Compute true probability; supports Student-t and Power law."""
    if dist_category == "Continuous" and dist_type == "Student-t":
        df = params.get("df", 3.0)
        dist = stats.t(df)
        if prob_type == "of outcome":
            return 0.0
        if prob_type == "under upper bound":
            prob = dist.cdf(bound2)
            return 1.0 if abs(prob - 1.0) < 1e-10 else prob
        if prob_type == "above lower bound":
            prob = 1 - dist.cdf(bound1)
            return 1.0 if abs(prob - 1.0) < 1e-10 else prob
        if prob_type == "in interval":
            prob = dist.cdf(bound2) - dist.cdf(bound1)
            return max(0.0, min(1.0, prob))
    if dist_category == "Discrete" and dist_type == "Power law":
        a = params.get("a", 2.0)
        if prob_type == "of outcome":
            return float(_power_law_pmf(np.array([int(np.round(bound1))]), a)[0])
        if prob_type == "under upper bound":
            return _power_law_cdf(int(np.round(bound2)), a)
        if prob_type == "above lower bound":
            return 1 - _power_law_cdf(int(np.round(bound1)) - 1, a)
        if prob_type == "in interval":
            return _power_law_cdf(int(np.round(bound2)), a) - _power_law_cdf(int(np.round(bound1)) - 1, a)
    return _compute_true_probability_base(
        dist_type, dist_category, prob_type, bound1, bound2, **params
    )


class DistributionProbabilityVisualization51(DistributionProbabilityVisualization):
    """
    Extended distribution explorer: Student-t (continuous), Power law (discrete),
    and log-scale toggles for x and y axes (x log only when RV is nonnegative).
    """

    def __init__(self):
        super().__init__()
        # Add new distributions
        self.continuous_dists = list(self.continuous_dists) + ["Student-t"]
        self.discrete_dists = list(self.discrete_dists) + ["Power law"]

        # Student-t: degrees of freedom
        self.param_widgets["Student-t"] = [
            widgets.FloatSlider(
                value=3.0,
                min=0.5,
                max=20,
                step=0.5,
                description="df:",
                style={"description_width": "initial"},
            )
        ]
        # Power law: exponent a in [0.5, 5]; for a<=1 use truncated support
        self.param_widgets["Power law"] = [
            widgets.FloatSlider(
                value=2.0,
                min=0.5,
                max=5,
                step=0.1,
                description="Exponent (a):",
                style={"description_width": "initial"},
            )
        ]

        # So discrete dropdown shows Power law from the start (options were set in base before we added it)
        self.dist_dropdown.options = self.discrete_dists

        # Make sure new parameter sliders trigger updates (base callbacks were
        # attached before these widgets existed).
        for w in self.param_widgets["Student-t"]:
            w.observe(self._on_param_change, names="value")
        for w in self.param_widgets["Power law"]:
            w.observe(self._on_param_change, names="value")

        # Log scale state
        self.log_x = False
        self.log_y = False
        self.log_x_button = widgets.ToggleButton(
            value=False,
            description="Log scale (x-axis)",
            tooltip="Only available when the random variable is nonnegative.",
            style={"button_width": "initial"},
        )
        self.log_y_button = widgets.ToggleButton(
            value=False,
            description="Log scale (y-axis)",
            style={"button_width": "initial"},
        )
        self.log_x_button.observe(self._on_log_x_change, names="value")
        self.log_y_button.observe(self._on_log_y_change, names="value")

        # Enable/disable log-x based on whether RV is nonnegative
        self._update_log_x_enabled()

    def _is_nonnegative_rv(self):
        """True if the current distribution has nonnegative support (so x log scale is allowed)."""
        dist_type = self.dist_dropdown.value
        dist_category = self.category_dropdown.value
        if dist_category == "Discrete":
            # All discrete options here have nonnegative support
            return True
        # Continuous
        if dist_type in ("Exponential", "Pareto", "Beta", "Gamma"):
            return True
        if dist_type == "Uniform":
            low = self.param_widgets["Uniform"][0].value
            return low >= 0
        # Normal, Student-t can be negative
        return False

    def _update_log_x_enabled(self):
        can_log_x = self._is_nonnegative_rv()
        self.log_x_button.disabled = not can_log_x
        if not can_log_x and self.log_x:
            self.log_x = False
            self.log_x_button.value = False

    def _on_log_x_change(self, change):
        self.log_x = change["new"]
        self._update_plot()

    def _on_log_y_change(self, change):
        self.log_y = change["new"]
        self._update_plot()

    def _on_category_change(self, change):
        super()._on_category_change(change)
        self._update_log_x_enabled()
        # Update dropdown options when switching category so both lists include our new dists
        if change["new"] == "Continuous":
            self.dist_dropdown.options = self.continuous_dists
        else:
            self.dist_dropdown.options = self.discrete_dists
        self._update_plot()

    def _on_dist_change(self, change):
        super()._on_dist_change(change)
        self._update_log_x_enabled()
        if self.dist_dropdown.value == "Power law":
            self._update_bound_sliders()
        self._update_plot()

    def _on_param_change(self, change):
        self._update_log_x_enabled()
        self._update_plot()

    def _get_params_dict(self):
        d = super()._get_params_dict()
        if self.dist_dropdown.value == "Student-t":
            d["df"] = self.param_widgets["Student-t"][0].value
        if self.dist_dropdown.value == "Power law":
            d["a"] = self.param_widgets["Power law"][0].value
        return d

    def _get_x_axis_range(self):
        dist_type = self.dist_dropdown.value
        dist_category = self.category_dropdown.value
        if dist_type == "Power law":
            x_min = 1
            x_max = 50
            return x_min, x_max
        if dist_type == "Student-t" and len(self.samples) == 0:
            return -5, 5
        return super()._get_x_axis_range()

    def _update_plot(self, change=None):
        """
        Show only the theoretical PDF/PMF for the selected distribution (no samples, no probability panel).
        """
        log_x = self.log_x and self._is_nonnegative_rv()
        log_y = self.log_y

        with self.plot_output:
            clear_output(wait=True)
            dist_type = self.dist_dropdown.value
            dist_category = self.category_dropdown.value
            params = self._get_params_dict()
            fig = go.Figure()
            x_min, x_max = self._get_x_axis_range()

            if dist_category == "Discrete":
                x_range = np.arange(int(x_min), int(x_max) + 1)
            else:
                x_range = np.linspace(x_min, x_max, 500)

            pdf_pmf_values = compute_pdf_pmf(x_range, dist_type, dist_category, **params)

            if dist_category == "Discrete":
                fig.add_trace(
                    go.Bar(
                        x=x_range,
                        y=pdf_pmf_values,
                        name="PMF",
                        marker=dict(color="rgba(255,165,0,0.8)", line=dict(color="orange", width=2)),
                        showlegend=True,
                        opacity=0.8,
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=pdf_pmf_values,
                        mode="lines",
                        name="PDF",
                        line=dict(color="orange", width=3),
                        showlegend=True,
                    )
                )

            fig.update_xaxes(title_text="x")
            fig.update_yaxes(title_text="Density")
            if log_x:
                fig.update_xaxes(type="log")
                fig.update_xaxes(range=[max(0.01, x_min), x_max])
            if log_y:
                fig.update_yaxes(type="log")
                fig.update_yaxes(rangemode="tozero")
            fig.update_layout(
                height=600, showlegend=True, title="PDF/PMF for Selected Distribution"
            )
            fig.show()

    def display(self):
        """Control column: type, distribution, params, log-scale toggles only (no probability panel)."""
        self._update_param_widgets()
        self._update_plot()
        log_row = widgets.HBox(
            [self.log_x_button, self.log_y_button],
            layout=widgets.Layout(flex_wrap="wrap"),
        )
        controls = widgets.VBox(
            [
                self.category_dropdown,
                self.dist_dropdown,
                self.param_container,
                widgets.HTML("<b>Axes:</b>"),
                log_row,
            ]
        )
        display(widgets.HBox([controls, self.plot_output]))


def run_distribution_explorer_51():
    """Create and display the §5.1 alternate distribution visualization."""
    viz = DistributionProbabilityVisualization51()
    viz.display()
    return viz
