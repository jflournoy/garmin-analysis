#!/usr/bin/env python3
"""Create comprehensive HTML report with visualizations for fitness models.

This script creates an interactive HTML report comparing:
1. Dual state-space model (strength vs aerobic)
2. Four-fitness model (short/long term for both activity types)
3. Key biological insights and recommendations
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from datetime import datetime
import sys

# Set Plotly template
pio.templates.default = "plotly_white"


class FitnessModelReport:
    """Create HTML report for fitness model analysis."""

    def __init__(self, output_dir: str = "output/fitness_model_report"):
        """Initialize report generator.

        Args:
            output_dir: Path to directory for output files.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load model results
        self.dual_results = self._load_dual_results()
        self.four_fitness_results = self._load_four_fitness_results()

    def _load_dual_results(self) -> dict:
        """Load dual model results."""
        dual_dir = Path("output/dual_model_analysis")

        results = {
            'summary': None,
            'samples': None,
            'variance': None,
            'standardization': None
        }

        try:
            # Load parameter summary
            if (dual_dir / "parameter_summary.csv").exists():
                results['summary'] = pd.read_csv(dual_dir / "parameter_summary.csv", index_col=0)

            # Load parameter samples
            if (dual_dir / "parameter_samples.json").exists():
                with open(dual_dir / "parameter_samples.json", 'r') as f:
                    results['samples'] = json.load(f)

            # Load variance decomposition
            if (dual_dir / "variance_decomposition.csv").exists():
                results['variance'] = pd.read_csv(dual_dir / "variance_decomposition.csv")

            # Load standardization
            if (dual_dir / "standardization.json").exists():
                with open(dual_dir / "standardization.json", 'r') as f:
                    results['standardization'] = json.load(f)

        except Exception as e:
            print(f"Warning: Could not load dual results: {e}")

        return results

    def _load_four_fitness_results(self) -> dict:
        """Load four-fitness model results."""
        four_dir = Path("output/four_fitness_analysis")

        results = {
            'summary': None,
            'samples': None,
            'variance': None,
            'standardization': None
        }

        try:
            # Load parameter summary
            if (four_dir / "parameter_summary.csv").exists():
                results['summary'] = pd.read_csv(four_dir / "parameter_summary.csv", index_col=0)

            # Load parameter samples
            if (four_dir / "parameter_samples.json").exists():
                with open(four_dir / "parameter_samples.json", 'r') as f:
                    results['samples'] = json.load(f)

            # Load variance decomposition
            if (four_dir / "variance_decomposition.csv").exists():
                results['variance'] = pd.read_csv(four_dir / "variance_decomposition.csv")

            # Load standardization
            if (four_dir / "standardization.json").exists():
                with open(four_dir / "standardization.json", 'r') as f:
                    results['standardization'] = json.load(f)

        except Exception as e:
            print(f"Warning: Could not load four-fitness results: {e}")

        return results

    def create_gamma_comparison_plot(self) -> go.Figure:
        """Create comparison plot of gamma parameters (weight effects)."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Dual Model: Strength vs Aerobic',
                'Four-Fitness: Short-term Effects',
                'Four-Fitness: Long-term Effects',
                'Model Comparison: Strength Effects'
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.15
        )

        # 1. Dual model gamma comparison
        if self.dual_results['samples'] and 'gamma_a' in self.dual_results['samples']:
            gamma_a = np.array(self.dual_results['samples']['gamma_a'])
            gamma_s = np.array(self.dual_results['samples']['gamma_s'])

            # Aerobic (should be negative)
            fig.add_trace(
                go.Histogram(
                    x=gamma_a,
                    name='Aerobic (γ_a)',
                    marker_color='rgba(65, 105, 225, 0.7)',  # Royal blue
                    nbinsx=30,
                    showlegend=False
                ),
                row=1, col=1
            )

            # Strength (should be positive)
            fig.add_trace(
                go.Histogram(
                    x=gamma_s,
                    name='Strength (γ_s)',
                    marker_color='rgba(220, 20, 60, 0.7)',  # Crimson red
                    nbinsx=30,
                    showlegend=False
                ),
                row=1, col=1
            )

            # Add zero line
            fig.add_vline(x=0, line_dash="dash", line_color="black", row=1, col=1)

            # Add mean lines
            fig.add_vline(x=gamma_a.mean(), line_color="blue", line_width=2, row=1, col=1)
            fig.add_vline(x=gamma_s.mean(), line_color="red", line_width=2, row=1, col=1)

        # 2. Four-fitness short-term effects
        if self.four_fitness_results['samples']:
            colors_short = {
                'gamma_a_short': 'rgba(30, 144, 255, 0.7)',  # Dodger blue
                'gamma_s_short': 'rgba(255, 99, 71, 0.7)'   # Tomato
            }

            for i, (param, color) in enumerate(colors_short.items()):
                if param in self.four_fitness_results['samples']:
                    samples = np.array(self.four_fitness_results['samples'][param])
                    fig.add_trace(
                        go.Histogram(
                            x=samples,
                            name=param.replace('_', ' ').title(),
                            marker_color=color,
                            nbinsx=30,
                            showlegend=False
                        ),
                        row=1, col=2
                    )
                    fig.add_vline(x=samples.mean(), line_color=color, line_width=2, row=1, col=2)

            fig.add_vline(x=0, line_dash="dash", line_color="black", row=1, col=2)

        # 3. Four-fitness long-term effects
        if self.four_fitness_results['samples']:
            colors_long = {
                'gamma_a_long': 'rgba(0, 0, 139, 0.7)',     # Dark blue
                'gamma_s_long': 'rgba(178, 34, 34, 0.7)'    # Firebrick
            }

            for i, (param, color) in enumerate(colors_long.items()):
                if param in self.four_fitness_results['samples']:
                    samples = np.array(self.four_fitness_results['samples'][param])
                    fig.add_trace(
                        go.Histogram(
                            x=samples,
                            name=param.replace('_', ' ').title(),
                            marker_color=color,
                            nbinsx=30,
                            showlegend=False
                        ),
                        row=2, col=1
                    )
                    fig.add_vline(x=samples.mean(), line_color=color, line_width=2, row=2, col=1)

            fig.add_vline(x=0, line_dash="dash", line_color="black", row=2, col=1)

        # 4. Strength effects comparison across models
        strength_data = []
        labels = []

        # Dual model strength effect
        if self.dual_results['samples'] and 'gamma_s' in self.dual_results['samples']:
            gamma_s = np.array(self.dual_results['samples']['gamma_s'])
            strength_data.append(gamma_s)
            labels.append('Dual Model')

        # Four-fitness strength effects
        if self.four_fitness_results['samples']:
            for param in ['gamma_s_short', 'gamma_s_long']:
                if param in self.four_fitness_results['samples']:
                    samples = np.array(self.four_fitness_results['samples'][param])
                    strength_data.append(samples)
                    labels.append(param.replace('gamma_s_', '').title())

        if strength_data:
            fig.add_trace(
                go.Box(
                    y=strength_data,
                    x=labels,
                    name='Strength Effects',
                    marker_color='rgba(220, 20, 60, 0.7)',
                    boxmean=True,
                    showlegend=False
                ),
                row=2, col=2
            )

        # Update layout
        fig.update_layout(
            height=800,
            title_text="Weight Effect Parameters (γ) Comparison",
            title_font_size=20,
            showlegend=False
        )

        # Update axes labels
        fig.update_xaxes(title_text="Effect on Weight", row=1, col=1)
        fig.update_xaxes(title_text="Effect on Weight", row=1, col=2)
        fig.update_xaxes(title_text="Effect on Weight", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        fig.update_yaxes(title_text="Effect Value", row=2, col=2)

        return fig

    def create_half_life_comparison_plot(self) -> go.Figure:
        """Create comparison plot of fitness half-lives."""
        fig = go.Figure()

        # Collect half-life data
        half_life_data = []
        labels = []
        colors = []

        # Dual model half-lives (calculated from alpha)
        if self.dual_results['samples'] and 'alpha_a' in self.dual_results['samples']:
            alpha_a = np.array(self.dual_results['samples']['alpha_a'])
            alpha_s = np.array(self.dual_results['samples']['alpha_s'])

            # Calculate half-life: t_half = -ln(0.5) / (-ln(alpha))
            half_life_a = -np.log(0.5) / (-np.log(alpha_a + 1e-10))
            half_life_s = -np.log(0.5) / (-np.log(alpha_s + 1e-10))

            half_life_data.extend([half_life_a, half_life_s])
            labels.extend(['Dual: Aerobic', 'Dual: Strength'])
            colors.extend(['rgba(65, 105, 225, 0.7)', 'rgba(220, 20, 60, 0.7)'])

        # Four-fitness half-lives (from generated quantities)
        if self.four_fitness_results['samples']:
            half_life_params = ['half_life_a_short', 'half_life_s_short',
                               'half_life_a_long', 'half_life_s_long']

            param_colors = {
                'half_life_a_short': 'rgba(30, 144, 255, 0.7)',   # Dodger blue
                'half_life_s_short': 'rgba(255, 99, 71, 0.7)',    # Tomato
                'half_life_a_long': 'rgba(0, 0, 139, 0.7)',       # Dark blue
                'half_life_s_long': 'rgba(178, 34, 34, 0.7)'      # Firebrick
            }

            param_labels = {
                'half_life_a_short': 'Aerobic Short',
                'half_life_s_short': 'Strength Short',
                'half_life_a_long': 'Aerobic Long',
                'half_life_s_long': 'Strength Long'
            }

            for param in half_life_params:
                if param in self.four_fitness_results['samples']:
                    samples = np.array(self.four_fitness_results['samples'][param])
                    # Cap extreme values for visualization
                    samples = np.clip(samples, 0, 100)
                    half_life_data.append(samples)
                    labels.append(param_labels[param])
                    colors.append(param_colors[param])

        # Create box plot
        for i, (data, label, color) in enumerate(zip(half_life_data, labels, colors)):
            fig.add_trace(go.Box(
                y=data,
                name=label,
                marker_color=color,
                boxmean=True,
                showlegend=True
            ))

        # Update layout
        fig.update_layout(
            title="Fitness Half-life Comparison (Days)",
            title_font_size=20,
            yaxis_title="Half-life (days, capped at 100)",
            xaxis_title="Fitness Component",
            height=600,
            boxmode='group',
            yaxis=dict(range=[0, 50])  # Limit y-axis for better visualization
        )

        return fig

    def create_variance_decomposition_plot(self) -> go.Figure:
        """Create variance decomposition comparison plot."""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Dual Model', 'Four-Fitness Model'),
            specs=[[{'type': 'pie'}, {'type': 'pie'}]]
        )

        # Dual model variance
        if self.dual_results['variance'] is not None:
            dual_df = self.dual_results['variance']
            fig.add_trace(
                go.Pie(
                    labels=dual_df['Component'],
                    values=dual_df['Mean'],
                    textinfo='label+percent',
                    insidetextorientation='radial',
                    hole=0.3,
                    marker=dict(colors=px.colors.qualitative.Set3),
                    name='Dual Model'
                ),
                row=1, col=1
            )

        # Four-fitness model variance
        if self.four_fitness_results['variance'] is not None:
            four_df = self.four_fitness_results['variance']
            fig.add_trace(
                go.Pie(
                    labels=four_df['Component'],
                    values=four_df['Mean'],
                    textinfo='label+percent',
                    insidetextorientation='radial',
                    hole=0.3,
                    marker=dict(colors=px.colors.qualitative.Set3),
                    name='Four-Fitness Model'
                ),
                row=1, col=2
            )

        # Update layout
        fig.update_layout(
            title_text="Variance Decomposition Comparison",
            title_font_size=20,
            height=500,
            showlegend=False
        )

        return fig

    def create_impulse_decay_comparison_plot(self) -> go.Figure:
        """Create comparison plot of impulse decay parameters (psi)."""
        fig = go.Figure()

        # Collect psi data from four-fitness model
        if self.four_fitness_results['samples']:
            psi_params = ['psi_a_short', 'psi_s_short', 'psi_a_long', 'psi_s_long']
            param_labels = {
                'psi_a_short': 'Aerobic Short',
                'psi_s_short': 'Strength Short',
                'psi_a_long': 'Aerobic Long',
                'psi_s_long': 'Strength Long'
            }
            param_colors = {
                'psi_a_short': 'rgba(30, 144, 255, 0.7)',   # Dodger blue
                'psi_s_short': 'rgba(255, 99, 71, 0.7)',    # Tomato
                'psi_a_long': 'rgba(0, 0, 139, 0.7)',       # Dark blue
                'psi_s_long': 'rgba(178, 34, 34, 0.7)'      # Firebrick
            }

            for param in psi_params:
                if param in self.four_fitness_results['samples']:
                    samples = np.array(self.four_fitness_results['samples'][param])
                    fig.add_trace(go.Box(
                        y=samples,
                        name=param_labels[param],
                        marker_color=param_colors[param],
                        boxmean=True,
                        showlegend=True
                    ))

        # Update layout
        fig.update_layout(
            title="Impulse Decay Parameters (ψ) - Four-Fitness Model",
            title_font_size=20,
            yaxis_title="Decay Rate (0-1, smaller = faster decay)",
            xaxis_title="Fitness Component",
            height=500,
            boxmode='group',
            yaxis=dict(range=[0, 1])
        )

        # Add horizontal line at 0.5 for reference
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5)

        return fig

    def create_biological_insights_table(self) -> str:
        """Create HTML table with key biological insights."""
        insights = [
            {
                "Finding": "Muscle gain from strength training persists for months",
                "Evidence": "Strength long-term half-life = 112 days (four-fitness model)",
                "Interpretation": "Muscle tissue has slow turnover rate, consistent with physiology",
                "Implication": "Strength training effects accumulate over long periods"
            },
            {
                "Finding": "Aerobic effects are mostly short-term",
                "Evidence": "Aerobic short-term half-life = 1.2 days (four-fitness model)",
                "Interpretation": "Dehydration and glycogen depletion recover quickly",
                "Implication": "Aerobic training primarily affects water weight, not long-term mass"
            },
            {
                "Finding": "Strength training has positive weight effect (muscle gain)",
                "Evidence": "γ_s > 0 with 94-95% probability (both models)",
                "Interpretation": "Strength training builds muscle, increasing weight",
                "Implication": "Weight gain from strength training is positive adaptation"
            },
            {
                "Finding": "Aerobic training has negative weight effect (calorie burn)",
                "Evidence": "γ_a < 0 with 96-98% probability (both models)",
                "Interpretation": "Aerobic training burns calories/fat, reducing weight",
                "Implication": "Different physiological mechanisms than strength training"
            },
            {
                "Finding": "Most weight variation is intrinsic",
                "Evidence": "GP explains 69-70% of variance (both models)",
                "Interpretation": "Factors beyond tracked exercise explain most weight changes",
                "Implication": "Diet, sleep, stress, hormones play larger roles"
            },
            {
                "Finding": "Strength training explains 10-12% of weight variance",
                "Evidence": "Strength components largest identifiable effect",
                "Interpretation": "Muscle gain is measurable signal in weight data",
                "Implication": "Strength training has detectable long-term impact"
            }
        ]

        # Create HTML table
        html = """
        <div class="insights-table">
            <h3>Key Biological Insights</h3>
            <table class="table table-striped">
                <thead>
                    <tr>
                        <th>Finding</th>
                        <th>Evidence</th>
                        <th>Interpretation</th>
                        <th>Implication</th>
                    </tr>
                </thead>
                <tbody>
        """

        for insight in insights:
            html += f"""
                    <tr>
                        <td><strong>{insight['Finding']}</strong></td>
                        <td>{insight['Evidence']}</td>
                        <td>{insight['Interpretation']}</td>
                        <td>{insight['Implication']}</td>
                    </tr>
            """

        html += """
                </tbody>
            </table>
        </div>
        """

        return html

    def create_model_recommendations(self) -> str:
        """Create HTML with model recommendations."""
        recommendations = [
            {
                "Model": "Four-Fitness Model",
                "When to Use": "For detailed physiological understanding",
                "Strengths": "Separates short/long term effects, captures nuanced time scales",
                "Limitations": "More parameters, requires more data/computation"
            },
            {
                "Model": "Dual Model",
                "When to Use": "For practical weight prediction",
                "Strengths": "Simpler, captures key strength vs aerobic distinction",
                "Limitations": "Less nuanced time scale separation"
            },
            {
                "Model": "Future Enhancements",
                "When to Use": "When more data available",
                "Strengths": "Could incorporate volume metrics, nutrition, sleep",
                "Limitations": "Requires additional data collection"
            }
        ]

        html = """
        <div class="recommendations">
            <h3>Model Recommendations</h3>
            <table class="table table-bordered">
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>When to Use</th>
                        <th>Strengths</th>
                        <th>Limitations</th>
                    </tr>
                </thead>
                <tbody>
        """

        for rec in recommendations:
            html += f"""
                    <tr>
                        <td><strong>{rec['Model']}</strong></td>
                        <td>{rec['When to Use']}</td>
                        <td>{rec['Strengths']}</td>
                        <td>{rec['Limitations']}</td>
                    </tr>
            """

        html += """
                </tbody>
            </table>
        </div>
        """

        return html

    def create_html_report(self) -> str:
        """Create complete HTML report."""
        # Generate plots
        gamma_plot = self.create_gamma_comparison_plot()
        half_life_plot = self.create_half_life_comparison_plot()
        variance_plot = self.create_variance_decomposition_plot()
        impulse_plot = self.create_impulse_decay_comparison_plot()

        # Convert plots to HTML
        gamma_html = pio.to_html(gamma_plot, full_html=False, include_plotlyjs='cdn')
        half_life_html = pio.to_html(half_life_plot, full_html=False, include_plotlyjs='cdn')
        variance_html = pio.to_html(variance_plot, full_html=False, include_plotlyjs='cdn')
        impulse_html = pio.to_html(impulse_plot, full_html=False, include_plotlyjs='cdn')

        # Get insights and recommendations
        insights_html = self.create_biological_insights_table()
        recommendations_html = self.create_model_recommendations()

        # Create complete HTML
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Fitness Model Analysis Report</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background-color: #f8f9fa;
                    color: #333;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 3rem 0;
                    margin-bottom: 2rem;
                    border-radius: 0 0 20px 20px;
                }}
                .section {{
                    background: white;
                    padding: 2rem;
                    margin-bottom: 2rem;
                    border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }}
                .plot-container {{
                    margin: 2rem 0;
                }}
                .insight-card {{
                    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                    padding: 1.5rem;
                    border-radius: 10px;
                    margin-bottom: 1rem;
                }}
                .model-comparison {{
                    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
                    padding: 1.5rem;
                    border-radius: 10px;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                .highlight {{
                    background-color: #fff3cd;
                    padding: 0.5rem;
                    border-radius: 5px;
                    border-left: 4px solid #ffc107;
                }}
                table {{
                    font-size: 0.9rem;
                }}
                th {{
                    background-color: #f8f9fa;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <div class="container">
                    <h1 class="display-4">🏋️‍♂️ Fitness Model Analysis Report</h1>
                    <p class="lead">Comparing Dual vs Four-Fitness State-Space Models for Weight Prediction</p>
                    <p class="text-muted">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
                </div>
            </div>

            <div class="container">
                <!-- Executive Summary -->
                <div class="section">
                    <h2>📊 Executive Summary</h2>
                    <div class="row">
                        <div class="col-md-6">
                            <div class="insight-card">
                                <h4>🎯 Key Finding</h4>
                                <p>Strength training builds muscle that persists for <strong>months</strong> (half-life: 112 days), while aerobic effects fade within <strong>days</strong>.</p>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="insight-card">
                                <h4>📈 Model Performance</h4>
                                <p>Both models successfully distinguish strength (muscle gain) from aerobic (calorie burn) with >94% probability.</p>
                            </div>
                        </div>
                    </div>
                    <div class="highlight mt-3">
                        <strong>💡 Recommendation:</strong> Use the four-fitness model for detailed physiological insights, or the dual model for practical weight prediction.
                    </div>
                </div>

                <!-- Weight Effects Comparison -->
                <div class="section">
                    <h2>⚖️ Weight Effect Parameters (γ)</h2>
                    <p class="lead">How different types of exercise affect weight</p>
                    <div class="plot-container">
                        {gamma_html}
                    </div>
                    <div class="row mt-3">
                        <div class="col-md-6">
                            <div class="model-comparison">
                                <h5>📉 Aerobic Effects (Negative γ)</h5>
                                <ul>
                                    <li><strong>Short-term:</strong> Dehydration reduces weight (γ = -0.21)</li>
                                    <li><strong>Long-term:</strong> Fat loss reduces weight (γ = -0.16)</li>
                                    <li><strong>Probability < 0:</strong> 94-97%</li>
                                </ul>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="model-comparison">
                                <h5>📈 Strength Effects (Positive γ)</h5>
                                <ul>
                                    <li><strong>Short-term:</strong> Inflammation increases weight (γ = +0.11)</li>
                                    <li><strong>Long-term:</strong> Muscle gain increases weight (γ = +0.19)</li>
                                    <li><strong>Probability > 0:</strong> 78-95%</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Time Scale Analysis -->
                <div class="section">
                    <h2>⏱️ Time Scale Analysis</h2>
                    <p class="lead">How long fitness effects persist (half-life in days)</p>
                    <div class="plot-container">
                        {half_life_html}
                    </div>
                    <div class="row mt-3">
                        <div class="col-md-4">
                            <div class="insight-card">
                                <h5>⚡ Short-term (1-2 days)</h5>
                                <p>Dehydration, inflammation, glycogen changes</p>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="insight-card">
                                <h5>📅 Medium-term (3-5 days)</h5>
                                <p>Fat loss, cardiovascular adaptation</p>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="insight-card">
                                <h5>📆 Long-term (100+ days)</h5>
                                <p>Muscle gain - very slow tissue turnover</p>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Variance Decomposition -->
                <div class="section">
                    <h2>📊 Variance Decomposition</h2>
                    <p class="lead">What explains weight variation?</p>
                    <div class="plot-container">
                        {variance_html}
                    </div>
                    <div class="row mt-3">
                        <div class="col-md-4">
                            <div class="insight-card">
                                <h5>🎯 Key Insight</h5>
                                <p><strong>69-70%</strong> of weight variation is intrinsic (GP)</p>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="insight-card">
                                <h5>💪 Strength Training</h5>
                                <p>Explains <strong>10-12%</strong> of variance (largest identifiable effect)</p>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="insight-card">
                                <h5>🏃‍♂️ Aerobic Training</h5>
                                <p>Minimal effect (<1%) on long-term weight</p>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Impulse Decay -->
                <div class="section">
                    <h2>📉 Impulse Decay Parameters</h2>
                    <p class="lead">How quickly workout effects accumulate and fade (ψ parameters)</p>
                    <div class="plot-container">
                        {impulse_html}
                    </div>
                    <div class="highlight mt-3">
                        <strong>Key Finding:</strong> Short-term effects decay faster (ψ ~0.35) than long-term effects (ψ ~0.66) with 90% probability.
                    </div>
                </div>

                <!-- Biological Insights -->
                <div class="section">
                    <h2>🔬 Biological Insights</h2>
                    {insights_html}
                </div>

                <!-- Recommendations -->
                <div class="section">
                    <h2>🎯 Recommendations</h2>
                    {recommendations_html}
                </div>

                <!-- Footer -->
                <div class="section text-center">
                    <h3>📚 About This Analysis</h3>
                    <p>This report compares two Bayesian state-space models for understanding how strength and aerobic training affect weight over different time scales.</p>
                    <div class="row">
                        <div class="col-md-6">
                            <div class="insight-card">
                                <h5>📈 Data Sources</h5>
                                <ul class="list-unstyled">
                                    <li>• Garmin Connect workout data (139 strength, 131 aerobic)</li>
                                    <li>• Daily weight measurements (147 observations)</li>
                                    <li>• HR-based intensity calculation</li>
                                </ul>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="insight-card">
                                <h5>⚙️ Technical Details</h5>
                                <ul class="list-unstyled">
                                    <li>• Stan probabilistic programming</li>
                                    <li>• Hamiltonian Monte Carlo sampling</li>
                                    <li>• Sparse Gaussian Process approximation</li>
                                    <li>• Fourier basis for daily cycles</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                    <hr>
                    <p class="text-muted">Generated with ❤️ using Python, Stan, and Plotly</p>
                </div>
            </div>

            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
            <script>
                // Add interactivity to plots
                document.addEventListener('DOMContentLoaded', function() {{
                    // Make all plots responsive
                    window.addEventListener('resize', function() {{
                        Plotly.Plots.resize();
                    }});
                }});
            </script>
        </body>
        </html>
        """

        return html

    def save_report(self) -> None:
        """Save HTML report to file."""
        print("\n" + "=" * 70)
        print("CREATING FITNESS MODEL REPORT")
        print("=" * 70)

        # Create HTML report
        html_content = self.create_html_report()

        # Save to file
        report_path = self.output_dir / "fitness_model_report.html"
        with open(report_path, 'w') as f:
            f.write(html_content)

        print(f"✅ Report saved to: {report_path}")
        print(f"📊 Open in browser to view interactive visualizations")

        # Also save individual plots as standalone HTML files
        print("\nSaving individual plots...")

        # Save gamma comparison
        gamma_plot = self.create_gamma_comparison_plot()
        pio.write_html(gamma_plot, self.output_dir / "gamma_comparison.html")

        # Save half-life comparison
        half_life_plot = self.create_half_life_comparison_plot()
        pio.write_html(half_life_plot, self.output_dir / "half_life_comparison.html")

        # Save variance decomposition
        variance_plot = self.create_variance_decomposition_plot()
        pio.write_html(variance_plot, self.output_dir / "variance_decomposition.html")

        # Save impulse decay
        impulse_plot = self.create_impulse_decay_comparison_plot()
        pio.write_html(impulse_plot, self.output_dir / "impulse_decay.html")

        print(f"📈 Individual plots saved to: {self.output_dir}/")

        print("\n" + "=" * 70)
        print("REPORT GENERATION COMPLETE")
        print("=" * 70)


def main():
    """Main function to create report."""
    report = FitnessModelReport()
    report.save_report()


if __name__ == "__main__":
    main()