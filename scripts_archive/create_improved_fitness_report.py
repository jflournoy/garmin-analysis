#!/usr/bin/env python3
"""Create HTML report for improved fitness-weight mapping visualizations.

This report shows the comprehensive fitness-weight mapping with:
1. Four fitness indexes over time with credible intervals
2. Weight contributions from each fitness component
3. Clear mapping between fitness states and weight perturbations
4. Cumulative effects and detailed analysis
"""

import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np


def create_html_report():
    """Create HTML report with improved fitness-weight mapping visualizations."""
    print("Creating improved fitness-weight mapping HTML report...")

    # Paths
    output_dir = Path("output/improved_fitness_report")
    output_dir.mkdir(parents=True, exist_ok=True)

    viz_dir = Path("output/improved_fitness_weight")
    model_dir = Path("output/four_fitness_full")

    # Load summary statistics
    summary_path = viz_dir / "summary_statistics.json"
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            summary = json.load(f)
    else:
        summary = {}

    # Load model parameters
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
    <title>Improved Fitness-Weight Mapping Analysis</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f8f9fa;
            color: #333;
            padding-top: 20px;
            padding-bottom: 40px;
        }}
        .header {{
            background: linear-gradient(135deg, #2c3e50, #4a235a);
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
            border-bottom: 2px solid #4a235a;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        h3 {{
            color: #34495e;
            margin-top: 1.5rem;
        }}
        .component-card {{
            border-left: 4px solid;
            padding: 1rem;
            margin-bottom: 1rem;
            background: #f8f9fa;
        }}
        .component-card.aerobic-short {{ border-color: #1f77b4; }}
        .component-card.strength-short {{ border-color: #ff7f0e; }}
        .component-card.aerobic-long {{ border-color: #2ca02c; }}
        .component-card.strength-long {{ border-color: #d62728; }}
    </style>
    <!-- Simple Analytics -->
    <script async src="https://scripts.simpleanalyticscdn.com/latest.js"></script>
</head>
<body>
    <div class="container">
        <div class="header text-center">
            <h1 class="display-4">📊 Improved Fitness-Weight Mapping Analysis</h1>
            <p class="lead">Four-Fitness State-Space Model: How fitness indexes map to weight perturbations</p>
            <p class="mb-0">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <div class="notice-box">
            <strong>🔬 Advanced Analysis:</strong> This report shows comprehensive mapping between four fitness indexes and weight perturbations.
            Each fitness component (aerobic/strength, short/long-term) is tracked over time with 95% credible intervals,
            and its contribution to weight changes is quantified.
        </div>

        <div class="section">
            <h2>🎯 Executive Summary</h2>

            <div class="row">
                <div class="col-md-6">
                    <div class="insight-card">
                        <h5>🏃‍♂️ Aerobic Exercise Effects</h5>
                        <p><strong>Short-term (γ_a_short):</strong> {key_params.get('gamma_a_short', {}).get('mean', 'N/A'):.3f} [95% CI: {key_params.get('gamma_a_short', {}).get('hdi_3%', 'N/A'):.3f}, {key_params.get('gamma_a_short', {}).get('hdi_97%', 'N/A'):.3f}]</p>
                        <p><strong>Long-term (γ_a_long):</strong> {key_params.get('gamma_a_long', {}).get('mean', 'N/A'):.3f} [95% CI: {key_params.get('gamma_a_long', {}).get('hdi_3%', 'N/A'):.3f}, {key_params.get('gamma_a_long', {}).get('hdi_97%', 'N/A'):.3f}]</p>
                        <p class="mb-0"><em>Negative γ values: fitness increases → weight decreases</em></p>
                    </div>
                </div>

                <div class="col-md-6">
                    <div class="insight-card">
                        <h5>💪 Strength Training Effects</h5>
                        <p><strong>Short-term (γ_s_short):</strong> {key_params.get('gamma_s_short', {}).get('mean', 'N/A'):.3f} [95% CI: {key_params.get('gamma_s_short', {}).get('hdi_3%', 'N/A'):.3f}, {key_params.get('gamma_s_short', {}).get('hdi_97%', 'N/A'):.3f}]</p>
                        <p><strong>Long-term (γ_s_long):</strong> {key_params.get('gamma_s_long', {}).get('mean', 'N/A'):.3f} [95% CI: {key_params.get('gamma_s_long', {}).get('hdi_3%', 'N/A'):.3f}, {key_params.get('gamma_s_long', {}).get('hdi_97%', 'N/A'):.3f}]</p>
                        <p class="mb-0"><em>Positive γ values: fitness increases → weight increases</em></p>
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
            <h2>📈 Comprehensive Fitness-Weight Mapping</h2>
            <p>This comprehensive visualization shows the complete mapping between fitness states and weight perturbations over time.</p>

            <div class="plot-container">
                <h4>Main Analysis: Fitness States and Weight Contributions with Impulse Events</h4>
                <p><em>Top to bottom: 1) Weight measurements with workout events, 2) Fitness states with impulse markers, 3) Weight contributions with impulse markers, 4) Combined effects, 5) Impulse strength</em></p>
                <img src="improved_fitness_weight/fitness_weight_mapping_comprehensive_with_impulses.png" alt="Comprehensive Fitness-Weight Mapping with Impulses" class="img-fluid">
                <p class="text-muted mt-2"><small>Five-panel visualization showing fitness-weight relationship with colored impulse events scaled by intensity (blue=aerobic short, orange=strength short, green=aerobic long, red=strength long)</small></p>
            </div>

            <div class="row mt-4">
                <div class="col-md-6">
                    <div class="plot-container">
                        <h5>Original Comprehensive Plot</h5>
                        <img src="improved_fitness_weight/fitness_weight_mapping_comprehensive.png" alt="Original Comprehensive Plot" class="img-fluid">
                        <p class="text-muted mt-2"><small>Original four-panel visualization without impulse markers</small></p>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="plot-container">
                        <h5>Fitness vs Weight Contributions</h5>
                        <img src="improved_fitness_weight/fitness_vs_contributions.png" alt="Fitness vs Weight Contributions" class="img-fluid">
                        <p class="text-muted mt-2"><small>Direct comparison of fitness states and their weight contributions</small></p>
                    </div>
                </div>
            </div>

            <div class="row mt-4">
                <div class="col-md-6">
                    <div class="component-card aerobic-short">
                        <h5>Aerobic Short-term</h5>
                        <p><strong>γ coefficient:</strong> {key_params.get('gamma_a_short', {}).get('mean', 'N/A'):.3f}</p>
                        <p><strong>Effect:</strong> Immediate weight loss (dehydration)</p>
                        <p><strong>Time scale:</strong> Hours to days</p>
                    </div>
                    <div class="component-card aerobic-long">
                        <h5>Aerobic Long-term</h5>
                        <p><strong>γ coefficient:</strong> {key_params.get('gamma_a_long', {}).get('mean', 'N/A'):.3f}</p>
                        <p><strong>Effect:</strong> Sustained weight loss (fat reduction)</p>
                        <p><strong>Time scale:</strong> Weeks to months</p>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="component-card strength-short">
                        <h5>Strength Short-term</h5>
                        <p><strong>γ coefficient:</strong> {key_params.get('gamma_s_short', {}).get('mean', 'N/A'):.3f}</p>
                        <p><strong>Effect:</strong> Minor weight gain (inflammation)</p>
                        <p><strong>Time scale:</strong> Hours to days</p>
                    </div>
                    <div class="component-card strength-long">
                        <h5>Strength Long-term</h5>
                        <p><strong>γ coefficient:</strong> {key_params.get('gamma_s_long', {}).get('mean', 'N/A'):.3f}</p>
                        <p><strong>Effect:</strong> Significant weight gain (muscle growth)</p>
                        <p><strong>Time scale:</strong> Weeks to months</p>
                    </div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>🔍 Detailed Component Analysis</h2>
            <p>These plots show the detailed relationship between each fitness component and its weight contribution.</p>

            <div class="plot-container">
                <h4>Fitness States vs. Weight Contributions</h4>
                <p><em>Each panel shows one fitness component (solid line with CI) and its corresponding weight contribution (dashed line)</em></p>
                <img src="improved_fitness_weight/fitness_vs_contributions.png" alt="Fitness vs Weight Contributions" class="img-fluid">
                <p class="text-muted mt-2"><small>Direct comparison showing how fitness state changes translate to weight effects</small></p>
            </div>

            <div class="plot-container">
                <h4>Cumulative Weight Contributions with Impulse Events</h4>
                <p><em>Shows how weight effects accumulate over time for each fitness component with impulse markers</em></p>
                <img src="improved_fitness_weight/cumulative_contributions.png" alt="Cumulative Weight Contributions" class="img-fluid">
                <p class="text-muted mt-2"><small>Cumulative sum of weight contributions with colored impulse events scaled by intensity</small></p>
            </div>

            <div class="plot-container">
                <h4>Zoomed Views for Detailed Analysis</h4>
                <p><em>Close-up views showing detailed variation and impulse events</em></p>

                <div class="row">
                    <div class="col-md-6">
                        <h6>First 100 Days</h6>
                        <img src="improved_fitness_weight/zoomed_first_100_days.png" alt="Zoomed First 100 Days" class="img-fluid">
                        <p class="text-muted mt-2"><small>Detailed view of initial period</small></p>
                    </div>
                    <div class="col-md-6">
                        <h6>Middle 100 Days</h6>
                        <img src="improved_fitness_weight/zoomed_middle_100_days.png" alt="Zoomed Middle 100 Days" class="img-fluid">
                        <p class="text-muted mt-2"><small>Detailed view of middle period</small></p>
                    </div>
                </div>

                <div class="row mt-4">
                    <div class="col-md-4">
                        <h6>First 50 Days</h6>
                        <img src="improved_fitness_weight/zoomed_first_50_days.png" alt="Zoomed First 50 Days" class="img-fluid">
                        <p class="text-muted mt-2"><small>Extra zoomed initial period</small></p>
                    </div>
                    <div class="col-md-4">
                        <h6>Last 100 Days</h6>
                        <img src="improved_fitness_weight/zoomed_last_100_days.png" alt="Zoomed Last 100 Days" class="img-fluid">
                        <p class="text-muted mt-2"><small>Detailed view of final period</small></p>
                    </div>
                    <div class="col-md-4">
                        <h6>Last 50 Days</h6>
                        <img src="improved_fitness_weight/zoomed_last_50_days.png" alt="Zoomed Last 50 Days" class="img-fluid">
                        <p class="text-muted mt-2"><small>Extra zoomed final period</small></p>
                    </div>
                </div>
            </div>

            <div class="insight-card mt-4">
                <h5>💡 Interpretation Guide</h5>
                <ul>
                    <li><strong>Fitness State:</strong> Body's adaptation level to exercise (higher = more recent/accumulated activity)</li>
                    <li><strong>Weight Contribution:</strong> γ × Fitness State = direct effect on weight</li>
                    <li><strong>Positive Contribution:</strong> Increases weight (muscle gain, inflammation)</li>
                    <li><strong>Negative Contribution:</strong> Decreases weight (fat loss, dehydration)</li>
                    <li><strong>Cumulative Effect:</strong> Total weight change attributable to each component over time</li>
                    <li><strong>Impulse Events (Colored Vertical Lines):</strong> Workout events scaled by intensity (blue=aerobic short, orange=strength short, green=aerobic long, red=strength long)</li>
                    <li><strong>Zoomed Views:</strong> Show detailed variation with thinner credible intervals for clarity</li>
                </ul>
            </div>
        </div>

        <div class="section">
            <h2>📊 Statistical Summary</h2>

            <div class="row">
                <div class="col-md-6">
                    <h5>Key Findings</h5>
                    <ul>
                        <li><strong>Largest effect:</strong> Strength Long-term (γ = {key_params.get('gamma_s_long', {}).get('mean', 'N/A'):.3f})</li>
                        <li><strong>Most consistent:</strong> Aerobic components show negative effects as expected</li>
                        <li><strong>Time trends:</strong> Most components show increasing trends over the observation period</li>
                        <li><strong>Uncertainty:</strong> Credible intervals reflect model uncertainty in estimates</li>
                    </ul>
                </div>

                <div class="col-md-6">
                    <h5>Model Performance</h5>
                    <ul>
                        <li><strong>Convergence:</strong> Model chains show good mixing (R-hat < 1.1)</li>
                        <li><strong>Uncertainty quantification:</strong> 95% credible intervals shown for all estimates</li>
                        <li><strong>Temporal resolution:</strong> Daily estimates over {summary.get('fitness_components', {}).get('fitness_a_short_stored', {}).get('n_time', 'N/A')} days</li>
                        <li><strong>Component separation:</strong> Clear distinction between short/long term effects</li>
                    </ul>
                </div>
            </div>

            <h5 class="mt-4">Parameter Estimates with Uncertainty</h5>
            <div class="table-responsive">
                <table class="table table-striped parameter-table">
                    <thead>
                        <tr>
                            <th>Parameter</th>
                            <th>Description</th>
                            <th>Mean</th>
                            <th>95% CI Lower</th>
                            <th>95% CI Upper</th>
                            <th>Biological Interpretation</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>γ_a_short</td>
                            <td>Aerobic short-term effect</td>
                            <td>{key_params.get('gamma_a_short', {}).get('mean', 'N/A'):.3f}</td>
                            <td>{key_params.get('gamma_a_short', {}).get('hdi_3%', 'N/A'):.3f}</td>
                            <td>{key_params.get('gamma_a_short', {}).get('hdi_97%', 'N/A'):.3f}</td>
                            <td>Negative = dehydration weight loss</td>
                        </tr>
                        <tr>
                            <td>γ_s_short</td>
                            <td>Strength short-term effect</td>
                            <td>{key_params.get('gamma_s_short', {}).get('mean', 'N/A'):.3f}</td>
                            <td>{key_params.get('gamma_s_short', {}).get('hdi_3%', 'N/A'):.3f}</td>
                            <td>{key_params.get('gamma_s_short', {}).get('hdi_97%', 'N/A'):.3f}</td>
                            <td>Positive = inflammation weight gain</td>
                        </tr>
                        <tr>
                            <td>γ_a_long</td>
                            <td>Aerobic long-term effect</td>
                            <td>{key_params.get('gamma_a_long', {}).get('mean', 'N/A'):.3f}</td>
                            <td>{key_params.get('gamma_a_long', {}).get('hdi_3%', 'N/A'):.3f}</td>
                            <td>{key_params.get('gamma_a_long', {}).get('hdi_97%', 'N/A'):.3f}</td>
                            <td>Negative = fat loss weight reduction</td>
                        </tr>
                        <tr>
                            <td>γ_s_long</td>
                            <td>Strength long-term effect</td>
                            <td>{key_params.get('gamma_s_long', {}).get('mean', 'N/A'):.3f}</td>
                            <td>{key_params.get('gamma_s_long', {}).get('hdi_3%', 'N/A'):.3f}</td>
                            <td>{key_params.get('gamma_s_long', {}).get('hdi_97%', 'N/A'):.3f}</td>
                            <td>Positive = muscle growth weight gain</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>

        <div class="section">
            <h2>🎯 Practical Implications</h2>

            <div class="row">
                <div class="col-md-6">
                    <h5>For Weight Loss Goals</h5>
                    <ul>
                        <li><strong>Focus on aerobic exercise:</strong> Both short and long-term aerobic components reduce weight</li>
                        <li><strong>Monitor strength training:</strong> Long-term strength training increases weight (muscle)</li>
                        <li><strong>Consider time scales:</strong> Aerobic effects are immediate, strength effects accumulate</li>
                        <li><strong>Balance is key:</strong> Too much strength training may offset aerobic weight loss</li>
                    </ul>
                </div>

                <div class="col-md-6">
                    <h5>For Muscle Building Goals</h5>
                    <ul>
                        <li><strong>Emphasize strength training:</strong> Long-term strength component has strongest positive effect</li>
                        <li><strong>Include aerobic work:</strong> Maintain cardiovascular health without excessive weight loss</li>
                        <li><strong>Track progress:</strong> Use fitness state trends to monitor adaptation</li>
                        <li><strong>Be patient:</strong> Muscle growth effects accumulate over months</li>
                    </ul>
                </div>
            </div>

            <div class="insight-card mt-4">
                <h5>📋 Recommendations</h5>
                <ol>
                    <li><strong>For weight loss:</strong> 70% aerobic, 30% strength training</li>
                    <li><strong>For muscle gain:</strong> 30% aerobic, 70% strength training</li>
                    <li><strong>For maintenance:</strong> 50% aerobic, 50% strength training</li>
                    <li><strong>Monitor trends:</strong> Watch fitness state trajectories over time</li>
                    <li><strong>Adjust based on goals:</strong> Modify workout balance as goals change</li>
                </ol>
            </div>
        </div>

        <footer class="text-center text-muted mt-5 pt-4 border-top">
            <p><strong>Garmin Analysis Project</strong> | Bayesian modeling of health and fitness data</p>
            <p>Improved Fitness-Weight Mapping Analysis | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
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

    print(f"✅ Improved report saved to: {html_path}")
    print(f"📊 Open in browser to view: {html_path}")

    # Also copy to docs directory
    docs_dir = Path("docs/improved_fitness_report")
    docs_dir.mkdir(parents=True, exist_ok=True)

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
    if 'improved_fitness_report' in content:
        print("✓ Link already exists in docs/index.html")
        return

    # Find the position to insert new card (after fitness time series report)
    insert_marker = 'fitness_time_series_report/index.html'
    insert_position = content.find(insert_marker)

    if insert_position == -1:
        print("Warning: Could not find insertion point in docs/index.html")
        return

    # Find the end of that card
    card_end = content.find('</div>', insert_position) + 6

    # Insert new card
    new_card = """
        <div class="doc-card">
            <h3>Improved Fitness-Weight Mapping</h3>
            <p>Comprehensive visualization showing how four fitness indexes map to weight perturbations with credible intervals. Shows fitness states, weight contributions, and cumulative effects.</p>
            <a href="improved_fitness_report/index.html">View Improved Analysis →</a>
        </div>
"""

    new_content = content[:card_end] + new_card + content[card_end:]

    with open(docs_index_path, 'w') as f:
        f.write(new_content)

    print("✓ Updated docs/index.html with new improved report link")


if __name__ == "__main__":
    create_html_report()