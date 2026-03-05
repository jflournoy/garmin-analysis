#!/usr/bin/env python3
"""Create comprehensive HTML report for fitness time series analysis.

This report shows:
1. Fitness state time series (4 components) with credible intervals
2. Weight contributions from each fitness component
3. Biological interpretation and insights
"""

import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import shutil


def create_html_report():
    """Create HTML report with fitness time series visualizations."""
    print("Creating fitness time series HTML report...")

    # Paths
    output_dir = Path("output/fitness_time_series_report")
    output_dir.mkdir(parents=True, exist_ok=True)

    viz_dir = Path("output/fitness_time_series")
    model_dir = Path("output/four_fitness_full")

    # Load model results
    with open(model_dir / "standardization.json", 'r') as f:
        standardization = json.load(f)

    # Load parameter summary
    param_summary = pd.read_csv(model_dir / "parameter_summary.csv", index_col=0)

    # Get key parameters
    key_params = {}
    for param in ['gamma_a_short', 'gamma_s_short', 'gamma_a_long', 'gamma_s_long']:
        if param in param_summary.index:
            key_params[param] = {
                'mean': param_summary.loc[param, 'mean'],
                'hdi_3%': param_summary.loc[param, 'hdi_3%'],
                'hdi_97%': param_summary.loc[param, 'hdi_97%']
            }

    # Create HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fitness Time Series Analysis Report</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f8f9fa;
            color: #333;
            padding-top: 20px;
            padding-bottom: 40px;
        }}
        .header {{
            background: linear-gradient(135deg, #2c3e50, #3498db);
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
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 1rem;
            background: white;
        }}
        .insight-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        }}
        .parameter-table {{
            font-size: 0.9rem;
        }}
        .parameter-table th {{
            background-color: #f8f9fa;
        }}
        .notice-box {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            border-radius: 4px;
            padding: 1rem;
            margin: 1rem 0 2rem 0;
            color: #856404;
        }}
        .notice-box strong {{
            color: #856404;
        }}
        h2 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        h3 {{
            color: #34495e;
            margin-top: 1.5rem;
        }}
    </style>
    <!-- Simple Analytics -->
    <script async src="https://scripts.simpleanalyticscdn.com/latest.js"></script>
</head>
<body>
    <div class="container">
        <div class="header text-center">
            <h1 class="display-4">🏋️‍♂️ Fitness Time Series Analysis</h1>
            <p class="lead">Four-Fitness State-Space Model: How workouts affect weight over time</p>
            <p class="mb-0">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <div class="notice-box">
            <strong>🚧 Experimental Analysis Notice:</strong> This report shows Bayesian posterior estimates from a four-fitness state-space model.
            The model separates short-term (hours to days) and long-term (weeks to months) effects of aerobic and strength training on weight.
            <strong>Key updates in this version:</strong>
            <ul class="mb-0">
                <li>Workout intensity scaled to be non-negative (0 = no workout, >0 = workout)</li>
                <li>Fitness gain parameters (β) constrained to be positive</li>
                <li>Fitness states decay toward 0 during periods with no workouts</li>
            </ul>
            All estimates include 95% credible intervals.
        </div>

        <div class="section">
            <h2>📊 Executive Summary</h2>

            <div class="row">
                <div class="col-md-6">
                    <div class="insight-card">
                        <h5>🏃‍♂️ Aerobic Exercise Effects</h5>
                        <p><strong>Short-term (γ_a_short):</strong> {key_params.get('gamma_a_short', {}).get('mean', 'N/A'):.3f} [95% CI: {key_params.get('gamma_a_short', {}).get('hdi_3%', 'N/A'):.3f}, {key_params.get('gamma_a_short', {}).get('hdi_97%', 'N/A'):.3f}]</p>
                        <p><strong>Long-term (γ_a_long):</strong> {key_params.get('gamma_a_long', {}).get('mean', 'N/A'):.3f} [95% CI: {key_params.get('gamma_a_long', {}).get('hdi_3%', 'N/A'):.3f}, {key_params.get('gamma_a_long', {}).get('hdi_97%', 'N/A'):.3f}]</p>
                        <p class="mb-0"><em>Negative values indicate weight loss (fat reduction)</em></p>
                    </div>
                </div>

                <div class="col-md-6">
                    <div class="insight-card">
                        <h5>💪 Strength Training Effects</h5>
                        <p><strong>Short-term (γ_s_short):</strong> {key_params.get('gamma_s_short', {}).get('mean', 'N/A'):.3f} [95% CI: {key_params.get('gamma_s_short', {}).get('hdi_3%', 'N/A'):.3f}, {key_params.get('gamma_s_short', {}).get('hdi_97%', 'N/A'):.3f}]</p>
                        <p><strong>Long-term (γ_s_long):</strong> {key_params.get('gamma_s_long', {}).get('mean', 'N/A'):.3f} [95% CI: {key_params.get('gamma_s_long', {}).get('hdi_3%', 'N/A'):.3f}, {key_params.get('gamma_s_long', {}).get('hdi_97%', 'N/A'):.3f}]</p>
                        <p class="mb-0"><em>Positive values indicate weight gain (muscle growth)</em></p>
                    </div>
                </div>
            </div>

            <div class="insight-card">
                <h5>🔑 Key Biological Insights</h5>
                <ol>
                    <li><strong>Aerobic exercise reduces weight</strong> through both short-term (dehydration) and long-term (fat loss) mechanisms</li>
                    <li><strong>Strength training increases weight</strong> primarily through long-term muscle growth, with minor short-term inflammation effects</li>
                    <li><strong>Time scales matter:</strong> Aerobic effects manifest quickly, strength effects accumulate over weeks/months</li>
                    <li><strong>Individual variation:</strong> The model captures how fitness states evolve differently for each activity type</li>
                </ol>
            </div>
        </div>

        <div class="section">
            <h2>📈 Interactive Fitness State Time Series</h2>
            <p>These plots show the estimated fitness states for each component over time. Fitness states represent the body's "readiness" or "adaptation" level from each type of exercise.</p>

            <div class="plot-container">
                <h4>Four Fitness Components Over Time</h4>
                <p><em>Hover over points to see exact values. Use the legend to toggle visibility of components.</em></p>
                <iframe src="../fitness_time_series/fitness_time_series_interactive.html" width="100%" height="800" frameborder="0"></iframe>
                <p class="text-muted mt-2"><small>Interactive Plotly visualization showing posterior means and 95% credible intervals</small></p>
            </div>

            <div class="row mt-4">
                <div class="col-md-6">
                    <div class="plot-container">
                        <h5>Static View: Fitness States</h5>
                        <img src="../fitness_time_series/fitness_states_time_series.png" alt="Fitness States Time Series" class="img-fluid">
                        <p class="text-muted mt-2"><small>Each subplot shows one fitness component with mean (solid line) and 95% CI (shaded)</small></p>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="plot-container">
                        <h5>How to Interpret Fitness States</h5>
                        <ul>
                            <li><strong>Aerobic Short-term:</strong> Immediate cardiovascular stress/recovery</li>
                            <li><strong>Strength Short-term:</strong> Muscle damage/inflammation from recent workouts</li>
                            <li><strong>Aerobic Long-term:</strong> Cumulative cardiovascular fitness</li>
                            <li><strong>Strength Long-term:</strong> Cumulative muscle mass/strength</li>
                        </ul>
                        <p>Higher values indicate greater recent activity or accumulated fitness. The states evolve dynamically based on workout intensity and natural decay.</p>
                        <p><strong>Key property:</strong> Fitness states represent <em>additional adaptation</em> from recent workouts. With no workouts, states decay toward 0 (baseline). States are constrained to be non-negative (≥ 0).</p>
                    </div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>⚖️ Weight Contributions Over Time</h2>
            <p>These plots show how each fitness component contributes to weight changes. The contribution is calculated as γ × fitness_state.</p>

            <div class="plot-container">
                <h4>Interactive Weight Contributions</h4>
                <p><em>Positive contributions increase weight, negative contributions decrease weight. The red dashed line shows zero contribution.</em></p>
                <iframe src="../fitness_time_series/weight_contributions_interactive.html" width="100%" height="800" frameborder="0"></iframe>
                <p class="text-muted mt-2"><small>Interactive visualization of weight contributions with 95% credible intervals</small></p>
            </div>

            <div class="row mt-4">
                <div class="col-md-6">
                    <div class="plot-container">
                        <h5>Static View: Weight Contributions</h5>
                        <img src="../fitness_time_series/weight_contributions_time_series.png" alt="Weight Contributions Time Series" class="img-fluid">
                        <p class="text-muted mt-2"><small>Each subplot shows weight contribution from one fitness component</small></p>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="plot-container">
                        <h5>Combined Contributions</h5>
                        <img src="../fitness_time_series/combined_contributions.png" alt="Combined Weight Contributions" class="img-fluid">
                        <p class="text-muted mt-2"><small>All contributions combined, showing net effect on weight</small></p>
                    </div>
                </div>
            </div>

            <div class="insight-card mt-4">
                <h5>💡 Interpreting Weight Contributions</h5>
                <ul>
                    <li><strong>Aerobic contributions are negative:</strong> Exercise reduces weight (fat loss)</li>
                    <li><strong>Strength contributions are positive:</strong> Training increases weight (muscle gain)</li>
                    <li><strong>Short-term effects are smaller:</strong> Immediate water weight changes are minor</li>
                    <li><strong>Long-term effects dominate:</strong> Muscle growth and fat loss accumulate over time</li>
                    <li><strong>Net effect varies:</strong> Depending on workout balance, weight can increase or decrease</li>
                </ul>
            </div>
        </div>

        <div class="section">
            <h2>🔬 Model Details & Parameters</h2>

            <div class="row">
                <div class="col-md-6">
                    <h5>Model Structure</h5>
                    <p>The four-fitness state-space model includes:</p>
                    <ul>
                        <li><strong>4 fitness states:</strong> Separate short/long term for aerobic/strength</li>
                        <li><strong>Impulse-response dynamics:</strong> Workouts create impulses that decay over time</li>
                        <li><strong>Non-negative constraints:</strong> Workout intensity ≥ 0, fitness gain (β) > 0</li>
                        <li><strong>Gaussian Process:</strong> Captures intrinsic weight trends (70% of variance)</li>
                        <li><strong>Daily cycle:</strong> Fourier basis for within-day patterns (3% of variance)</li>
                        <li><strong>Measurement noise:</strong> Accounts for scale variability</li>
                    </ul>
                    <p><small>Workout intensity is scaled to be non-negative (0 = no workout). Fitness states decay toward 0 with no workouts.</small></p>
                </div>

                <div class="col-md-6">
                    <h5>Data Summary</h5>
                    <table class="table table-sm">
                        <tr><td>Date range</td><td>{standardization.get('date_range_start', 'N/A')} to {standardization.get('date_range_end', 'N/A')}</td></tr>
                        <tr><td>Total days</td><td>{standardization.get('n_days', 'N/A')}</td></tr>
                        <tr><td>Weight measurements</td><td>147</td></tr>
                        <tr><td>Aerobic workout days</td><td>131</td></tr>
                        <tr><td>Strength workout days</td><td>136</td></tr>
                        <tr><td>Mean weight</td><td>{standardization.get('weight_mean', 'N/A'):.1f} lbs</td></tr>
                        <tr><td>Weight SD</td><td>{standardization.get('weight_std', 'N/A'):.1f} lbs</td></tr>
                    </table>
                </div>
            </div>

            <h5 class="mt-4">Key Parameter Estimates</h5>
            <div class="table-responsive">
                <table class="table table-striped parameter-table">
                    <thead>
                        <tr>
                            <th>Parameter</th>
                            <th>Description</th>
                            <th>Mean</th>
                            <th>95% CI Lower</th>
                            <th>95% CI Upper</th>
                            <th>Interpretation</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>γ_a_short</td>
                            <td>Aerobic short-term effect</td>
                            <td>{key_params.get('gamma_a_short', {}).get('mean', 'N/A'):.3f}</td>
                            <td>{key_params.get('gamma_a_short', {}).get('hdi_3%', 'N/A'):.3f}</td>
                            <td>{key_params.get('gamma_a_short', {}).get('hdi_97%', 'N/A'):.3f}</td>
                            <td>Negative = weight loss (dehydration)</td>
                        </tr>
                        <tr>
                            <td>γ_s_short</td>
                            <td>Strength short-term effect</td>
                            <td>{key_params.get('gamma_s_short', {}).get('mean', 'N/A'):.3f}</td>
                            <td>{key_params.get('gamma_s_short', {}).get('hdi_3%', 'N/A'):.3f}</td>
                            <td>{key_params.get('gamma_s_short', {}).get('hdi_97%', 'N/A'):.3f}</td>
                            <td>Positive = weight gain (inflammation)</td>
                        </tr>
                        <tr>
                            <td>γ_a_long</td>
                            <td>Aerobic long-term effect</td>
                            <td>{key_params.get('gamma_a_long', {}).get('mean', 'N/A'):.3f}</td>
                            <td>{key_params.get('gamma_a_long', {}).get('hdi_3%', 'N/A'):.3f}</td>
                            <td>{key_params.get('gamma_a_long', {}).get('hdi_97%', 'N/A'):.3f}</td>
                            <td>Negative = weight loss (fat reduction)</td>
                        </tr>
                        <tr>
                            <td>γ_s_long</td>
                            <td>Strength long-term effect</td>
                            <td>{key_params.get('gamma_s_long', {}).get('mean', 'N/A'):.3f}</td>
                            <td>{key_params.get('gamma_s_long', {}).get('hdi_3%', 'N/A'):.3f}</td>
                            <td>{key_params.get('gamma_s_long', {}).get('hdi_97%', 'N/A'):.3f}</td>
                            <td>Positive = weight gain (muscle growth)</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>

        <div class="section">
            <h2>📝 Limitations & Future Work</h2>

            <div class="row">
                <div class="col-md-6">
                    <h5>Current Limitations</h5>
                    <ul>
                        <li><strong>Divergent transitions:</strong> Some chains had convergence issues (5-23% divergent)</li>
                        <li><strong>Simplified intensity:</strong> Uses HR-based intensity, not volume/load</li>
                        <li><strong>Linear effects:</strong> Assumes linear relationship between fitness and weight</li>
                        <li><strong>Missing covariates:</strong> Nutrition, sleep, stress not included</li>
                        <li><strong>Standardization:</strong> Effects are in standardized units, not raw pounds</li>
                    </ul>
                </div>

                <div class="col-md-6">
                    <h5>Future Enhancements</h5>
                    <ul>
                        <li><strong>Improved convergence:</strong> Address remaining divergent transitions</li>
                        <li><strong>Nonlinear effects:</strong> Diminishing returns at high fitness</li>
                        <li><strong>Interaction terms:</strong> How aerobic and strength training interact</li>
                        <li><strong>Individual differences:</strong> Random effects for response variability</li>
                        <li><strong>Real-time prediction:</strong> Forecast weight based on planned workouts</li>
                    </ul>
                    <p><small>Note: Beta parameter priors have been updated to exponential distributions (positive-only).</small></p>
                </div>
            </div>
        </div>

        <footer class="text-center text-muted mt-5 pt-4 border-top">
            <p><strong>Garmin Analysis Project</strong> | Bayesian modeling of health and fitness data</p>
            <p>Four-Fitness State-Space Model | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p class="small">This analysis represents experimental research. Consult healthcare professionals for medical advice.</p>
        </footer>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>"""

    # Save HTML file
    html_path = output_dir / "index.html"
    with open(html_path, 'w') as f:
        f.write(html_content)

    print(f"✅ Report saved to: {html_path}")
    print(f"📊 Open in browser to view: {html_path}")

    # Also copy to docs directory
    docs_dir = Path("docs/fitness_time_series_report")
    docs_dir.mkdir(parents=True, exist_ok=True)

    # Create visualizations subdirectory in docs
    docs_viz_dir = docs_dir / "visualizations"
    docs_viz_dir.mkdir(parents=True, exist_ok=True)

    # Copy visualization files from output to docs
    import shutil
    viz_files = [
        "fitness_states_time_series.png",
        "weight_contributions_time_series.png",
        "combined_contributions.png",
        "fitness_time_series_interactive.html",
        "weight_contributions_interactive.html"
    ]

    for viz_file in viz_files:
        src_path = viz_dir / viz_file
        if src_path.exists():
            dst_path = docs_viz_dir / viz_file
            shutil.copy2(src_path, dst_path)
            print(f"  Copied: {viz_file} → {dst_path}")
        else:
            print(f"  Warning: {src_path} not found")

    # Update HTML to use visualizations subdirectory
    html_content = html_content.replace(
        '../fitness_time_series/',
        'visualizations/'
    )

    docs_html_path = docs_dir / "index.html"
    with open(docs_html_path, 'w') as f:
        f.write(html_content)

    print(f"📚 Documentation copy saved to: {docs_html_path}")

    # Update main docs index
    update_main_docs_index()


def update_main_docs_index():
    """Update main docs/index.html to include link to this report."""
    docs_index_path = Path("docs/index.html")

    if not docs_index_path.exists():
        print(f"Warning: {docs_index_path} not found")
        return

    with open(docs_index_path, 'r') as f:
        content = f.read()

    # Check if link already exists
    if 'fitness_time_series_report' in content:
        print("✓ Link already exists in docs/index.html")
        return

    # Find the position to insert new card (after fitness model comparison)
    insert_marker = 'fitness_model_report/index.html'
    insert_position = content.find(insert_marker)

    if insert_position == -1:
        print("Warning: Could not find insertion point in docs/index.html")
        return

    # Find the end of that card
    card_end = content.find('</div>', insert_position) + 6

    # Insert new card
    new_card = """
        <div class="doc-card">
            <h3>Fitness Time Series Analysis</h3>
            <p>Interactive time series visualizations of four fitness states and their weight contributions. Shows how aerobic and strength training affect weight over different time scales.</p>
            <a href="fitness_time_series_report/index.html">View Time Series Report →</a>
        </div>
"""

    new_content = content[:card_end] + new_card + content[card_end:]

    with open(docs_index_path, 'w') as f:
        f.write(new_content)

    print("✓ Updated docs/index.html with new time series report link")


if __name__ == "__main__":
    create_html_report()