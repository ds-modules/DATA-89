"""
Utility functions and visualization for the Law of Large Numbers demo.
Used by large_numbers.ipynb
"""

import numpy as np
import ipywidgets as widgets
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display, clear_output
from scipy import stats
import time

from utils_dist import sample_distribution, compute_pdf_pmf


def compute_true_expected_value(dist_type, dist_category, **params):
    """Return the theoretical expected value E[X] for the distribution."""
    if dist_category == "Continuous":
        if dist_type == "Uniform":
            low = params.get('low', 0)
            high = params.get('high', 1)
            return stats.uniform(loc=low, scale=high - low).mean()
        elif dist_type == "Exponential":
            scale = params.get('scale', 1)
            return stats.expon(scale=scale).mean()
        elif dist_type == "Pareto":
            shape = params.get('shape', 2.0)
            scale = params.get('scale', 1.0)
            if shape <= 1:
                return np.nan  # infinite mean
            return stats.pareto(shape, loc=0, scale=scale).mean()
        elif dist_type == "Beta":
            alpha = params.get('alpha', 2)
            beta = params.get('beta', 2)
            return stats.beta(alpha, beta).mean()
        elif dist_type == "Gamma":
            shape = params.get('shape', 2)
            scale = params.get('scale', 1)
            return stats.gamma(shape, scale=scale).mean()
        elif dist_type == "Normal":
            mean = params.get('mean', 0)
            std = params.get('std', 1)
            return stats.norm(mean, std).mean()
        else:
            return stats.norm(0, 1).mean()
    else:  # Discrete
        if dist_type == "Bernoulli":
            p = params.get('p', 0.5)
            return stats.bernoulli(p).mean()
        elif dist_type == "Geometric":
            p = params.get('p', 0.5)
            return stats.geom(p).mean()
        elif dist_type == "Binomial":
            n = params.get('n', 10)
            p = params.get('p', 0.5)
            return stats.binom(n, p).mean()
        elif dist_type == "Poisson":
            lam = params.get('lam', 5)
            return stats.poisson(lam).mean()
        elif dist_type == "Hypergeometric":
            ngood = params.get('ngood', 10)
            nbad = params.get('nbad', 10)
            nsample = params.get('nsample', 10)
            return stats.hypergeom(ngood + nbad, ngood, nsample).mean()
        else:
            return stats.bernoulli(0.5).mean()


def determine_batch_size(sample_index):
    """Batch size for animation: more samples per update later."""
    if sample_index < 50:
        return 5
    elif sample_index < 200:
        return 20
    elif sample_index < 500:
        return 50
    else:
        return 100


class LargeNumbersVisualization:
    """Interactive Law of Large Numbers: running average and running histogram with window."""

    MAX_SAMPLES = 1000

    def __init__(self):
        self.samples = np.array([])
        self.plot_output = widgets.Output()

        self.continuous_dists = ["Uniform", "Exponential", "Pareto", "Beta", "Gamma", "Normal"]
        self.discrete_dists = ["Bernoulli", "Geometric", "Binomial", "Poisson", "Hypergeometric"]

        self._create_widgets()
        self._setup_callbacks()

    def _create_widgets(self):
        self.category_dropdown = widgets.Dropdown(
            options=["Discrete", "Continuous"],
            value="Discrete",
            description="Type:",
            style={'description_width': 'initial'}
        )
        self.dist_dropdown = widgets.Dropdown(
            options=self.discrete_dists,
            value="Bernoulli",
            description="Distribution:",
            style={'description_width': 'initial'}
        )

        self.param_widgets = {}
        self.param_widgets['Normal'] = [
            widgets.FloatSlider(value=0, min=-5, max=5, step=0.1, description='Mean:', style={'description_width': 'initial'}),
            widgets.FloatSlider(value=1, min=0.1, max=3, step=0.1, description='Std:', style={'description_width': 'initial'})
        ]
        self.param_widgets['Exponential'] = [
            widgets.FloatSlider(value=1, min=0.1, max=5, step=0.1, description='Scale:', style={'description_width': 'initial'})
        ]
        self.param_widgets['Beta'] = [
            widgets.FloatSlider(value=2, min=0.5, max=10, step=0.1, description='Alpha:', style={'description_width': 'initial'}),
            widgets.FloatSlider(value=2, min=0.5, max=10, step=0.1, description='Beta:', style={'description_width': 'initial'})
        ]
        self.param_widgets['Gamma'] = [
            widgets.FloatSlider(value=2, min=0.5, max=10, step=0.1, description='Shape:', style={'description_width': 'initial'}),
            widgets.FloatSlider(value=1, min=0.1, max=5, step=0.1, description='Scale:', style={'description_width': 'initial'})
        ]
        self.param_widgets['Uniform'] = [
            widgets.FloatSlider(value=0, min=-5, max=5, step=0.1, description='Low:', style={'description_width': 'initial'}),
            widgets.FloatSlider(value=1, min=-5, max=5, step=0.1, description='High:', style={'description_width': 'initial'})
        ]
        self.param_widgets['Pareto'] = [
            widgets.FloatSlider(value=2.0, min=1.1, max=5, step=0.1, description='Shape (α):', style={'description_width': 'initial'}),
            widgets.FloatSlider(value=1.0, min=0.1, max=5, step=0.1, description='Scale (xₘ):', style={'description_width': 'initial'})
        ]
        self.param_widgets['Poisson'] = [
            widgets.FloatSlider(value=2, min=0.5, max=20, step=0.1, description='Lambda:', style={'description_width': 'initial'})
        ]
        self.param_widgets['Binomial'] = [
            widgets.IntSlider(value=10, min=1, max=50, step=1, description='n:', style={'description_width': 'initial'}),
            widgets.FloatSlider(value=0.5, min=0.1, max=0.9, step=0.05, description='p:', style={'description_width': 'initial'})
        ]
        self.param_widgets['Bernoulli'] = [
            widgets.FloatSlider(value=0.5, min=0.1, max=0.9, step=0.05, description='p:', style={'description_width': 'initial'})
        ]
        self.param_widgets['Geometric'] = [
            widgets.FloatSlider(value=0.5, min=0.1, max=0.9, step=0.05, description='p:', style={'description_width': 'initial'})
        ]
        self.param_widgets['Hypergeometric'] = [
            widgets.IntSlider(value=10, min=1, max=50, step=1, description='ngood:', style={'description_width': 'initial'}),
            widgets.IntSlider(value=10, min=1, max=50, step=1, description='nbad:', style={'description_width': 'initial'}),
            widgets.IntSlider(value=10, min=1, max=50, step=1, description='nsample:', style={'description_width': 'initial'})
        ]

        self.param_container = widgets.VBox([])

        self.sample_button = widgets.Button(description="Sample", button_style='success')
        self.window_slider = widgets.FloatSlider(
            value=0.5, min=0.01, max=2.0, step=0.05,
            description="Window half-width:",
            style={'description_width': 'initial'}
        )
        self.status_html = widgets.HTML(value="Click Sample to draw up to 1,000 samples.")

    def _setup_callbacks(self):
        self.category_dropdown.observe(self._on_category_change, names='value')
        self.dist_dropdown.observe(self._on_dist_change, names='value')
        self.sample_button.on_click(self._on_sample_clicked)
        self.window_slider.observe(self._on_window_change, names='value')
        for wl in self.param_widgets.values():
            for w in wl:
                w.observe(self._on_param_change, names='value')

    def _on_category_change(self, change):
        if change['new'] == "Continuous":
            self.dist_dropdown.options = self.continuous_dists
            self.dist_dropdown.value = "Uniform"
        else:
            self.dist_dropdown.options = self.discrete_dists
            self.dist_dropdown.value = "Bernoulli"
        self._update_param_widgets()
        self.samples = np.array([])
        self._redraw()

    def _on_dist_change(self, change):
        self._update_param_widgets()
        self.samples = np.array([])
        self._redraw()

    def _on_param_change(self, change):
        if len(self.samples) > 0:
            self._redraw()

    def _on_window_change(self, change):
        if len(self.samples) > 0:
            self._redraw()

    def _update_param_widgets(self):
        dist = self.dist_dropdown.value
        if dist in self.param_widgets:
            self.param_container.children = tuple(self.param_widgets[dist])

    def _get_params_dict(self):
        dist_type = self.dist_dropdown.value
        params = {}
        if dist_type in self.param_widgets:
            wl = self.param_widgets[dist_type]
            if dist_type == "Uniform":
                params['low'], params['high'] = wl[0].value, wl[1].value
            elif dist_type == "Exponential":
                params['scale'] = wl[0].value
            elif dist_type == "Pareto":
                params['shape'], params['scale'] = wl[0].value, wl[1].value
            elif dist_type in ["Beta", "Gamma", "Normal"]:
                params['alpha' if dist_type == "Beta" else 'shape' if dist_type == "Gamma" else 'mean'] = wl[0].value
                params['beta' if dist_type == "Beta" else 'scale' if dist_type in ("Gamma", "Exponential") else 'std'] = wl[1].value
            elif dist_type == "Bernoulli":
                params['p'] = wl[0].value
            elif dist_type == "Geometric":
                params['p'] = wl[0].value
            elif dist_type == "Binomial":
                params['n'], params['p'] = wl[0].value, wl[1].value
            elif dist_type == "Poisson":
                params['lam'] = wl[0].value
            elif dist_type == "Hypergeometric":
                params['ngood'], params['nbad'], params['nsample'] = wl[0].value, wl[1].value, wl[2].value
        return params

    def _on_sample_clicked(self, button):
        dist_type = self.dist_dropdown.value
        dist_category = self.category_dropdown.value
        params = self._get_params_dict()
        n_total = min(self.MAX_SAMPLES, 1000)

        all_samples = sample_distribution(dist_type, dist_category, n_total, **params)
        self.sample_button.disabled = True
        self.status_html.value = "Sampling..."

        sample_index = 0
        batch_count = 0
        while sample_index < n_total:
            batch_size = determine_batch_size(sample_index)
            end_index = min(sample_index + batch_size, n_total)
            self.samples = all_samples[:end_index]

            if sample_index == 0:
                self._update_param_widgets()

            should_update = (sample_index < 100) or (batch_count % 2 == 0)
            if should_update:
                self._redraw()
            self.status_html.value = f"Generated {end_index} / {n_total} samples"

            delay = batch_size / 500.0
            time.sleep(delay)
            sample_index = end_index
            batch_count += 1

        self.samples = all_samples
        self._redraw()
        self.sample_button.disabled = False
        self.status_html.value = f"Complete! Generated {n_total} samples."

    def _redraw(self):
        with self.plot_output:
            clear_output(wait=True)
            if len(self.samples) == 0:
                fig = make_subplots(rows=1, cols=2, subplot_titles=("Running average", "Running histogram"),
                                    horizontal_spacing=0.12)
                fig.update_layout(height=500, showlegend=True)
                fig.update_xaxes(title_text="Sample number", row=1, col=1)
                fig.update_yaxes(title_text="Running average", row=1, col=1)
                fig.update_xaxes(title_text="x", row=1, col=2)
                fig.update_yaxes(title_text="Density", row=1, col=2)
                fig.show()
                return

            dist_type = self.dist_dropdown.value
            dist_category = self.category_dropdown.value
            params = self._get_params_dict()
            true_mean = compute_true_expected_value(dist_type, dist_category, **params)
            if np.isnan(true_mean):
                true_mean = 0.0
            half = self.window_slider.value
            low_win = true_mean - half
            high_win = true_mean + half

            n = len(self.samples)
            indices = np.arange(1, n + 1, dtype=float)
            running_avg = np.cumsum(self.samples) / indices
            in_window = (running_avg >= low_win) & (running_avg <= high_win)

            x_min, x_max = self._get_value_range()
            y_range = [min(x_min, true_mean - half - 0.5), max(x_max, true_mean + half + 0.5)]

            fig = make_subplots(rows=1, cols=2, subplot_titles=("Running average", "Running histogram"),
                                horizontal_spacing=0.12,
                                column_widths=[0.5, 0.5])

            # Left: window band
            fig.add_trace(
                go.Scatter(
                    x=[0, n + 10], y=[low_win, low_win],
                    mode='lines', line=dict(color='rgba(0,200,0,0.4)', width=1, dash='dot'),
                    fill='none', showlegend=False
                ), row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=[0, n + 10], y=[high_win, high_win],
                    mode='lines', line=dict(color='rgba(0,200,0,0.4)', width=1, dash='dot'),
                    fill='tonexty', fillcolor='rgba(0,200,0,0.15)', showlegend=False
                ), row=1, col=1
            )
            # Left: true mean line
            fig.add_trace(
                go.Scatter(
                    x=[0, n + 10], y=[true_mean, true_mean],
                    mode='lines', name='True E[X]', line=dict(color='orange', width=2, dash='dash')
                ), row=1, col=1
            )
            # Left: running average line
            fig.add_trace(
                go.Scatter(x=indices, y=running_avg, mode='lines', name='Running average',
                           line=dict(color='#1f77b4', width=1.5)), row=1, col=1
            )
            # Left: scatter along trace - green if in window, red if outside
            in_idx = indices[in_window]
            out_idx = indices[~in_window]
            in_avg = running_avg[in_window]
            out_avg = running_avg[~in_window]
            if len(in_idx) > 0:
                fig.add_trace(
                    go.Scatter(x=in_idx, y=in_avg, mode='markers', name='In window',
                               marker=dict(size=4, color='green', opacity=0.8)), row=1, col=1
                )
            if len(out_idx) > 0:
                fig.add_trace(
                    go.Scatter(x=out_idx, y=out_avg, mode='markers', name='Out of window',
                               marker=dict(size=4, color='red', opacity=0.8)), row=1, col=1
                )
            # Front marker (current) - larger
            fig.add_trace(
                go.Scatter(x=[indices[-1]], y=[running_avg[-1]], mode='markers',
                           marker=dict(size=12, color='green' if in_window[-1] else 'red',
                                       line=dict(width=2, color='black')), showlegend=False),
                row=1, col=1
            )

            # Right: value range for histogram and PDF
            if dist_category == "Discrete":
                x_hist = np.arange(int(x_min), int(x_max) + 1)
            else:
                x_hist = np.linspace(x_min, x_max, 300)
            pdf_vals = compute_pdf_pmf(x_hist, dist_type, dist_category, **params)

            # Right: window band (vertical band in value space)
            fig.add_trace(
                go.Scatter(x=[low_win, low_win], y=[0, np.nanmax(pdf_vals) * 1.2 if np.any(np.isfinite(pdf_vals)) else 1],
                           mode='lines', line=dict(color='rgba(0,200,0,0.4)', width=1, dash='dot'),
                           fill='none', showlegend=False), row=1, col=2
            )
            fig.add_trace(
                go.Scatter(x=[high_win, high_win], y=[0, np.nanmax(pdf_vals) * 1.2 if np.any(np.isfinite(pdf_vals)) else 1],
                           mode='lines', line=dict(color='rgba(0,200,0,0.4)', width=1, dash='dot'),
                           fill='tonexty', fillcolor='rgba(0,200,0,0.15)', showlegend=False), row=1, col=2
            )
            # Right: true PDF/PMF (behind)
            if dist_category == "Discrete":
                fig.add_trace(
                    go.Bar(x=x_hist, y=pdf_vals, name='True PMF', marker=dict(color='rgba(255,165,0,0.5)', line=dict(color='orange')),
                           showlegend=True), row=1, col=2
                )
            else:
                fig.add_trace(
                    go.Scatter(x=x_hist, y=pdf_vals, mode='lines', name='True PDF',
                               line=dict(color='orange', width=2)), row=1, col=2
                )
            # Right: true mean line
            y_max_hist = np.nanmax(pdf_vals) * 1.15 if np.any(np.isfinite(pdf_vals)) else 1
            fig.add_trace(
                go.Scatter(x=[true_mean, true_mean], y=[0, y_max_hist],
                           mode='lines', name='True E[X]', line=dict(color='orange', width=2, dash='dash'),
                           showlegend=False), row=1, col=2
            )
            # Right: running histogram
            if dist_category == "Discrete":
                unique_vals, counts = np.unique(self.samples, return_counts=True)
                counts = counts / len(self.samples)
                fig.add_trace(
                    go.Bar(x=unique_vals, y=counts, name='Sample histogram',
                           marker=dict(color='rgba(70,130,180,0.7)', line=dict(color='navy')),
                           width=0.4), row=1, col=2
                )
            else:
                counts, bin_edges = np.histogram(self.samples, bins=40, range=(x_min, x_max), density=True)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                fig.add_trace(
                    go.Bar(x=bin_centers, y=counts, name='Sample histogram',
                           marker=dict(color='rgba(70,130,180,0.7)', line=dict(color='navy')),
                           width=(bin_edges[1] - bin_edges[0]) * 0.9), row=1, col=2
                )
            # Right: empirical mean line - green if in window, red otherwise
            current_mean = running_avg[-1]
            mean_color = 'green' if (low_win <= current_mean <= high_win) else 'red'
            fig.add_trace(
                go.Scatter(x=[current_mean, current_mean], y=[0, y_max_hist],
                           mode='lines', name='Running average',
                           line=dict(color=mean_color, width=3)), row=1, col=2
            )

            fig.update_xaxes(title_text="Sample number", range=[0, n + 5], row=1, col=1)
            fig.update_yaxes(title_text="Running average", range=y_range, row=1, col=1)
            fig.update_xaxes(title_text="x", range=y_range, row=1, col=2)
            fig.update_yaxes(title_text="Density", row=1, col=2)
            fig.update_layout(height=500, showlegend=True)
            fig.show()

    def _get_value_range(self):
        dist_type = self.dist_dropdown.value
        dist_category = self.category_dropdown.value
        params = self._get_params_dict()
        true_mean = compute_true_expected_value(dist_type, dist_category, **params)
        if np.isnan(true_mean):
            true_mean = 0.0

        if dist_type == "Bernoulli":
            return 0.0, 1.0
        if dist_type == "Poisson":
            lam = params.get('lam', 5)
            return 0, max(15, int(lam * 3))
        if dist_type == "Binomial":
            n = params.get('n', 10)
            return 0, int(n)
        if dist_type == "Uniform":
            low, high = params.get('low', 0), params.get('high', 1)
            return low - 0.2, high + 0.2
        if dist_type == "Geometric":
            if len(self.samples) > 0:
                return 0, max(15, int(np.max(self.samples)) + 2)
            return 0, 15
        if dist_type == "Hypergeometric":
            if len(self.samples) > 0:
                return 0, max(20, int(np.max(self.samples)) + 2)
            return 0, 20
        if dist_type == "Pareto":
            scale = params.get('scale', 1.0)
            if len(self.samples) > 0:
                return scale, max(scale + 5, float(np.max(self.samples)) + 0.5)
            return scale, scale + 5
        if len(self.samples) > 0:
            x_min = float(np.min(self.samples))
            x_max = float(np.max(self.samples))
            pad = max(0.5, (x_max - x_min) * 0.1)
            return x_min - pad, x_max + pad
        return true_mean - 2, true_mean + 2

    def display(self):
        self._update_param_widgets()
        with self.plot_output:
            clear_output(wait=True)
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Running average", "Running histogram"),
                                horizontal_spacing=0.12)
            fig.update_layout(height=500, showlegend=True)
            fig.update_xaxes(title_text="Sample number", row=1, col=1)
            fig.update_yaxes(title_text="Running average", row=1, col=1)
            fig.update_xaxes(title_text="x", row=1, col=2)
            fig.update_yaxes(title_text="Density", row=1, col=2)
            fig.show()

        controls = widgets.VBox([
            self.category_dropdown,
            self.dist_dropdown,
            self.param_container,
            self.sample_button,
            self.window_slider,
            self.status_html
        ])
        display(widgets.HBox([controls, self.plot_output]))


def run_large_numbers_explorer():
    """Create and display the Law of Large Numbers visualization."""
    viz = LargeNumbersVisualization()
    viz.display()
    return viz
