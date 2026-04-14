"""
Joint Distribution Table to Surface Visualizer

Visualizes joint distribution of two independent beta-distributed random variables
over [0,1]×[0,1] as a 3D histogram with interactive controls.
"""

import numpy as np
import plotly.graph_objects as go
from scipy.special import beta as beta_function
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore')


class JointDistributionVisualizer:
    """Interactive visualizer for joint distributions."""
    
    def __init__(self):
        self.delta_x = 0.1
        self.alpha_x = 2.0
        self.beta_x = 2.0
        self.alpha_y = 2.0
        self.beta_y = 2.0
        self.normalize_by_area = False
        self.a = 0.0
        self.b = 1.0
        self.c = 0.0
        self.d = 1.0
        self.current_fig = None
        self.view_mode = "3D perspective"  # "3D perspective", "Birds-eye", or "Heatmap"
        self.output_widget = widgets.Output()
        # Keep a consistent light gradient across heatmap + 3D views.
        # Plotly named scales: https://plotly.com/python/builtin-colorscales/
        self.colorscale = "YlGnBu"
        
    def beta_pdf(self, x, alpha, beta):
        """Evaluate beta PDF at x."""
        # Handle boundary cases
        x = np.clip(x, 1e-10, 1 - 1e-10)
        return (x**(alpha-1) * (1-x)**(beta-1)) / beta_function(alpha, beta)
    
    def compute_probabilities(self):
        """Compute joint probability table."""
        n_bins = int(1.0 / self.delta_x)
        probs = np.zeros((n_bins, n_bins))
        
        for i in range(n_bins):
            for j in range(n_bins):
                # Bin edges
                x_low = i * self.delta_x
                x_high = (i + 1) * self.delta_x
                y_low = j * self.delta_x
                y_high = (j + 1) * self.delta_x
                
                # Midpoints for evaluation (approximate via midpoint rule)
                x_mid = (x_low + x_high) / 2
                y_mid = (y_low + y_high) / 2
                
                # Probability ≈ pdf(x_mid) * pdf(y_mid) * delta_x^2
                px = self.beta_pdf(x_mid, self.alpha_x, self.beta_x)
                py = self.beta_pdf(y_mid, self.alpha_y, self.beta_y)
                
                prob = px * py * (self.delta_x ** 2)
                probs[i, j] = prob
        
        # Normalize to ensure sum ≈ 1
        probs = probs / np.sum(probs)
        
        return probs
    
    def get_bar_heights(self, probs):
        """Get bar heights (normalized by area if needed)."""
        if self.normalize_by_area:
            # probability / area = probability / delta_x^2
            heights = probs / (self.delta_x ** 2)
        else:
            heights = probs
        return heights
    
    def snap_to_grid(self, value):
        """Snap value to nearest multiple of delta_x."""
        return round(value / self.delta_x) * self.delta_x
    
    def compute_interval_probability(self, probs):
        """Compute probability of X in [a,b], Y in [c,d]."""
        a_snapped = self.snap_to_grid(self.a)
        b_snapped = self.snap_to_grid(self.b)
        c_snapped = self.snap_to_grid(self.c)
        d_snapped = self.snap_to_grid(self.d)
        
        # Ensure valid interval
        if a_snapped >= b_snapped:
            b_snapped = a_snapped + self.delta_x
        if c_snapped >= d_snapped:
            d_snapped = c_snapped + self.delta_x
        
        i_min = int(np.round(a_snapped / self.delta_x))
        i_max = int(np.round(b_snapped / self.delta_x))
        j_min = int(np.round(c_snapped / self.delta_x))
        j_max = int(np.round(d_snapped / self.delta_x))
        
        # Clamp to valid range
        n_bins = probs.shape[0]
        i_min = max(0, min(i_min, n_bins - 1))
        i_max = max(0, min(i_max, n_bins))
        j_min = max(0, min(j_min, n_bins - 1))
        j_max = max(0, min(j_max, n_bins))
        
        prob = np.sum(probs[i_min:i_max, j_min:j_max])
        
        return prob, (i_min, i_max, j_min, j_max), (a_snapped, b_snapped, c_snapped, d_snapped)
    
    def create_heatmap_figure(self, probs):
        """Create 2D heatmap view."""
        n_bins = probs.shape[0]
        heights = self.get_bar_heights(probs)
        zmax = float(np.max(heights)) if np.size(heights) else 1.0
        if zmax <= 0:
            zmax = 1.0
        
        # Compute interval and highlighting
        prob_interval, (i_min, i_max, j_min, j_max), snapped = self.compute_interval_probability(probs)
        
        # Create mask for highlighted cells
        mask = np.zeros_like(heights)
        mask[i_min:i_max, j_min:j_max] = 1
        
        # Create bins for axis labels
        bin_edges = np.arange(0, 1 + self.delta_x, self.delta_x)
        bin_centers = bin_edges[:-1] + self.delta_x / 2
        
        fig = go.Figure()
        
        # Add heatmap
        fig.add_trace(go.Heatmap(
            x=bin_centers,
            y=bin_centers,
            z=heights,
            colorscale=self.colorscale,
            zmin=0.0,
            zmax=zmax,
            colorbar=dict(title="Probability" if not self.normalize_by_area else "Prob/Area"),
            hovertemplate="X: %{x:.3f}<br>Y: %{y:.3f}<br>Value: %{z:.6f}<extra></extra>"
        ))
        
        # Add rectangle showing selected region
        a_snapped, b_snapped, c_snapped, d_snapped = snapped
        fig.add_shape(
            type="rect",
            x0=a_snapped, x1=b_snapped,
            y0=c_snapped, y1=d_snapped,
            line=dict(color="red", width=3),
            fillcolor=None,
            name="Selected region"
        )
        
        fig.update_layout(
            title=dict(
                text=f"Heatmap View: P(X∈[{a_snapped:.2f},{b_snapped:.2f}], Y∈[{c_snapped:.2f},{d_snapped:.2f}]) = {prob_interval:.6f}",
                x=0.5,
                xanchor='center',
                font=dict(size=14)
            ),
            xaxis_title="X",
            yaxis_title="Y",
            width=800,
            height=700,
            yaxis=dict(scaleanchor="x", scaleratio=1)
        )
        
        return fig, prob_interval
    

    def create_figure(self, probs, birds_eye=False):
        """Create 3D histogram figure with interactivity - optimized with Mesh3d."""
        n_bins = probs.shape[0]
        heights = self.get_bar_heights(probs)
        zmax = float(np.max(heights)) if np.size(heights) else 1.0
        if zmax <= 0:
            zmax = 1.0
        
        # Compute interval and highlighting
        prob_interval, (i_min, i_max, j_min, j_max), snapped = self.compute_interval_probability(probs)
        
        fig = go.Figure()
        
        # Build mesh data for all bars - much more efficient than individual traces
        # Make bars look more "natural": less gap + better lighting, less transparency.
        bar_size = self.delta_x * 0.94
        
        # Collect vertices/faces separately for inside vs outside so we can
        # grey+fade the outside region while keeping the SAME colorscale.
        inside_vx, inside_vy, inside_vz, inside_intensity = [], [], [], []
        inside_fi, inside_fj, inside_fk = [], [], []
        outside_vx, outside_vy, outside_vz = [], [], []
        outside_fi, outside_fj, outside_fk = [], [], []
        inside_vertex_idx = 0
        outside_vertex_idx = 0
        
        hover_x = []
        hover_y = []
        hover_z = []
        hover_text = []

        def append_box(target: str, x_center: float, y_center: float, height: float) -> None:
            nonlocal inside_vertex_idx, outside_vertex_idx
            x_off = bar_size / 2
            y_off = bar_size / 2

            box_x = [
                x_center - x_off, x_center + x_off, x_center + x_off, x_center - x_off,
                x_center - x_off, x_center + x_off, x_center + x_off, x_center - x_off,
            ]
            box_y = [
                y_center - y_off, y_center - y_off, y_center + y_off, y_center + y_off,
                y_center - y_off, y_center - y_off, y_center + y_off, y_center + y_off,
            ]
            box_z = [0.0, 0.0, 0.0, 0.0, height, height, height, height]

            if target == "inside":
                base = inside_vertex_idx
                inside_vx.extend(box_x); inside_vy.extend(box_y); inside_vz.extend(box_z)
                inside_intensity.extend([float(height)] * 8)
                fi, fj, fk = inside_fi, inside_fj, inside_fk
                inside_vertex_idx += 8
            else:
                base = outside_vertex_idx
                outside_vx.extend(box_x); outside_vy.extend(box_y); outside_vz.extend(box_z)
                fi, fj, fk = outside_fi, outside_fj, outside_fk
                outside_vertex_idx += 8

            # 12 triangles (top, bottom, 4 sides) like a solid box
            fi.extend([base+0, base+0, base+4, base+4, base+0, base+0, base+1, base+1, base+2, base+2, base+3, base+3])
            fj.extend([base+1, base+2, base+5, base+6, base+1, base+5, base+2, base+6, base+3, base+7, base+0, base+4])
            fk.extend([base+2, base+3, base+6, base+7, base+5, base+4, base+6, base+5, base+7, base+6, base+4, base+7])

        for i in range(n_bins):
            for j in range(n_bins):
                x_center = i * self.delta_x + self.delta_x/2
                y_center = j * self.delta_x + self.delta_x/2
                height = heights[i, j]
                
                in_rectangle = (i_min <= i < i_max) and (j_min <= j < j_max)

                append_box("inside" if in_rectangle else "outside", x_center, y_center, float(height))

                # Add a hoverable point at the top center of the bar.
                # Mesh3d hover is limited; this gives students exact (x,y,height) readouts.
                hover_x.append(x_center)
                hover_y.append(y_center)
                hover_z.append(float(height))
                hover_text.append(
                    f"x={x_center:.3f}<br>y={y_center:.3f}<br>height={float(height):.6g}"
                )
        
        # Outside region: grey + faded (no colorscale)
        if outside_vx:
            fig.add_trace(
                go.Mesh3d(
                    x=outside_vx,
                    y=outside_vy,
                    z=outside_vz,
                    i=outside_fi,
                    j=outside_fj,
                    k=outside_fk,
                    color="rgb(200,200,200)",
                    opacity=0.20,
                    hoverinfo="skip",
                    showlegend=False,
                    lighting=dict(ambient=0.85, diffuse=0.55, specular=0.05, roughness=0.9),
                    flatshading=True,
                )
            )

        # Inside region: same colorscale as heatmap + colorbar
        if inside_vx:
            fig.add_trace(
                go.Mesh3d(
                    x=inside_vx,
                    y=inside_vy,
                    z=inside_vz,
                    i=inside_fi,
                    j=inside_fj,
                    k=inside_fk,
                    intensity=inside_intensity,
                    colorscale=self.colorscale,
                    cmin=0.0,
                    cmax=zmax,
                    showscale=True,
                    colorbar=dict(
                        title="Probability" if not self.normalize_by_area else "Prob/Area",
                        x=1.02,
                        xanchor="left",
                        len=0.85,
                    ),
                    opacity=0.98,
                    hoverinfo="skip",
                    showlegend=False,
                    lighting=dict(ambient=0.55, diffuse=0.85, specular=0.25, roughness=0.4, fresnel=0.1),
                    flatshading=True,
                )
            )

        # Hover overlay points
        fig.add_trace(
            go.Scatter3d(
                x=hover_x,
                y=hover_y,
                z=hover_z,
                mode="markers",
                # Make hovering easy in both 3D and birds-eye:
                # a tiny-but-not-zero alpha marker captures hover events reliably.
                # Bigger marker radius = easier hit-testing (less pixel-perfect).
                marker=dict(size=14, color="rgba(0,0,0,0.03)"),
                hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>height=%{z:.6g}<extra></extra>",
                showlegend=False,
            )
        )
        
        # Draw rectangle on the plane z=0
        a_snapped, b_snapped, c_snapped, d_snapped = snapped
        rect_x = [a_snapped, b_snapped, b_snapped, a_snapped, a_snapped]
        rect_y = [c_snapped, c_snapped, d_snapped, d_snapped, c_snapped]
        rect_z = [0, 0, 0, 0, 0]
        
        fig.add_trace(go.Scatter3d(
            x=rect_x, y=rect_y, z=rect_z,
            mode='lines',
            line=dict(color='red', width=4),
            name="Rectangle [a,b]×[c,d]",
            hoverinfo='skip'
        ))
        
        # Set camera based on view mode
        if birds_eye:
            camera = dict(
                eye=dict(x=0, y=0, z=2.5),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=1, z=0)
            )
        else:
            camera = dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"Joint Distribution: P(X∈[{a_snapped:.2f},{b_snapped:.2f}], Y∈[{c_snapped:.2f},{d_snapped:.2f}]) = {prob_interval:.6f}",
                x=0.5,
                xanchor='center',
                font=dict(size=14)
            ),
            scene=dict(
                xaxis=dict(title="X", range=[-0.05, 1.05]),
                yaxis=dict(title="Y", range=[-0.05, 1.05]),
                zaxis=dict(title="Probability" if not self.normalize_by_area else "Probability/Area"),
                camera=camera,
            ),
            width=900,
            height=700,
            showlegend=True,
            hovermode='closest'
        )
        
        return fig, prob_interval

    
    def update_visualization(self, *args):
        """Update the visualization based on current parameters."""
        probs = self.compute_probabilities()
        
        if self.view_mode == "Heatmap":
            fig, prob = self.create_heatmap_figure(probs)
        elif self.view_mode == "Birds-eye":
            fig, prob = self.create_figure(probs, birds_eye=True)
        else:  # 3D perspective
            fig, prob = self.create_figure(probs, birds_eye=False)
        
        # Clear previous output and display new figure
        self.current_fig = fig
        with self.output_widget:
            clear_output(wait=True)
            display(fig)
    
    def create_controls(self):
        """Create interactive control widgets."""
        # Delta x slider
        delta_slider = widgets.FloatSlider(
            value=self.delta_x,
            min=0.02,
            max=0.5,
            step=0.02,
            description='Δx:',
            style={'description_width': '50px'},
            layout=widgets.Layout(width='600px')
        )
        
        # Beta parameters for X
        alpha_x_slider = widgets.FloatSlider(
            value=self.alpha_x,
            min=0.5,
            max=5,
            step=0.1,
            description='α(X):',
            style={'description_width': '50px'},
            layout=widgets.Layout(width='300px')
        )
        
        beta_x_slider = widgets.FloatSlider(
            value=self.beta_x,
            min=0.5,
            max=5,
            step=0.1,
            description='β(X):',
            style={'description_width': '50px'},
            layout=widgets.Layout(width='300px')
        )
        
        # Beta parameters for Y
        alpha_y_slider = widgets.FloatSlider(
            value=self.alpha_y,
            min=0.5,
            max=5,
            step=0.1,
            description='α(Y):',
            style={'description_width': '50px'},
            layout=widgets.Layout(width='300px')
        )
        
        beta_y_slider = widgets.FloatSlider(
            value=self.beta_y,
            min=0.5,
            max=5,
            step=0.1,
            description='β(Y):',
            style={'description_width': '50px'},
            layout=widgets.Layout(width='300px')
        )
        
        # Normalize by area toggle
        normalize_toggle = widgets.Checkbox(
            value=self.normalize_by_area,
            description='Normalize by area (density)',
            indent=False
        )
        
        # View mode toggle
        view_button_3d = widgets.Button(description='3D Perspective', button_style='info')
        view_button_birds_eye = widgets.Button(description='Birds-eye', button_style='')
        view_button_heatmap = widgets.Button(description='Heatmap', button_style='')
        
        def on_view_3d(b):
            self.view_mode = "3D perspective"
            view_button_3d.button_style = 'info'
            view_button_birds_eye.button_style = ''
            view_button_heatmap.button_style = ''
            self.update_visualization()
        
        def on_view_birds_eye(b):
            self.view_mode = "Birds-eye"
            view_button_3d.button_style = ''
            view_button_birds_eye.button_style = 'info'
            view_button_heatmap.button_style = ''
            self.update_visualization()
        
        def on_view_heatmap(b):
            self.view_mode = "Heatmap"
            view_button_3d.button_style = ''
            view_button_birds_eye.button_style = ''
            view_button_heatmap.button_style = 'info'
            self.update_visualization()
        
        view_button_3d.on_click(on_view_3d)
        view_button_birds_eye.on_click(on_view_birds_eye)
        view_button_heatmap.on_click(on_view_heatmap)
        
        # Rectangle bounds - snap to multiples of delta_x
        a_slider = widgets.FloatSlider(
            value=self.a,
            min=0,
            max=1,
            step=0.01,
            description='a:',
            style={'description_width': '50px'},
            layout=widgets.Layout(width='400px')
        )
        
        b_slider = widgets.FloatSlider(
            value=self.b,
            min=0,
            max=1,
            step=0.01,
            description='b:',
            style={'description_width': '50px'},
            layout=widgets.Layout(width='400px')
        )
        
        c_slider = widgets.FloatSlider(
            value=self.c,
            min=0,
            max=1,
            step=0.01,
            description='c:',
            style={'description_width': '50px'},
            layout=widgets.Layout(width='400px')
        )
        
        d_slider = widgets.FloatSlider(
            value=self.d,
            min=0,
            max=1,
            step=0.01,
            description='d:',
            style={'description_width': '50px'},
            layout=widgets.Layout(width='400px')
        )
        
        # Update function
        def on_change(change):
            self.delta_x = delta_slider.value
            self.alpha_x = alpha_x_slider.value
            self.beta_x = beta_x_slider.value
            self.alpha_y = alpha_y_slider.value
            self.beta_y = beta_y_slider.value
            self.normalize_by_area = normalize_toggle.value
            
            # Snap sliders to grid
            self.a = self.snap_to_grid(a_slider.value)
            self.b = self.snap_to_grid(b_slider.value)
            self.c = self.snap_to_grid(c_slider.value)
            self.d = self.snap_to_grid(d_slider.value)
            
            # Update slider displays to show snapped values
            a_slider.value = self.a
            b_slider.value = self.b
            c_slider.value = self.c
            d_slider.value = self.d
            
            self.update_visualization()
        
        # Attach handlers
        delta_slider.observe(on_change, names='value')
        alpha_x_slider.observe(on_change, names='value')
        beta_x_slider.observe(on_change, names='value')
        alpha_y_slider.observe(on_change, names='value')
        beta_y_slider.observe(on_change, names='value')
        normalize_toggle.observe(on_change, names='value')
        a_slider.observe(on_change, names='value')
        b_slider.observe(on_change, names='value')
        c_slider.observe(on_change, names='value')
        d_slider.observe(on_change, names='value')
        
        # Create layout
        controls = widgets.VBox([
            widgets.HTML("<b>View Mode</b>"),
            widgets.HBox([view_button_3d, view_button_birds_eye, view_button_heatmap]),
            widgets.HTML("<b>Bin Width</b>"),
            delta_slider,
            widgets.HTML("<b>X Distribution: Beta(α, β)</b>"),
            widgets.HBox([alpha_x_slider, beta_x_slider]),
            widgets.HTML("<b>Y Distribution: Beta(α, β)</b>"),
            widgets.HBox([alpha_y_slider, beta_y_slider]),
            widgets.HTML("<b>Vertical Axis</b>"),
            normalize_toggle,
            widgets.HTML("<b>Rectangle [a,b] × [c,d]</b>"),
            widgets.HTML("<i>Values snap to multiples of Δx</i>"),
            a_slider,
            b_slider,
            c_slider,
            d_slider,
        ])
        
        return controls


def run_joint_distribution_demo():
    """Run the interactive joint distribution visualizer."""
    visualizer = JointDistributionVisualizer()
    controls = visualizer.create_controls()
    # Controls left, plot right.
    display(
        widgets.HBox(
            [
                controls,
                widgets.Box([visualizer.output_widget], layout=widgets.Layout(flex="1 1 auto", min_width="700px")),
            ],
            layout=widgets.Layout(width="100%", align_items="flex-start", gap="18px"),
        )
    )
    visualizer.update_visualization()


if __name__ == "__main__":
    run_joint_distribution_demo()
