#!/usr/bin/env python3
"""Generate HTML interpretability report for trend model variant.

Similar structure to generate_model_report.py but focuses on the trend parameter δ
and compares strength/aerobic effects to the constrained AR model.
"""

import sys
import json
from pathlib import Path
from datetime import datetime


def load_posterior_metadata(json_file):
    """Load posterior metadata from JSON."""
    with open(json_file, 'r') as f:
        return json.load(f)


def parse_data_summary_markdown(markdown_file):
    """Parse data summary from markdown file and extract key statistics."""
    try:
        with open(markdown_file, 'r') as f:
            content = f.read()

        stats = {}
        lines = content.split('\n')
        in_magnitudes = False

        for line in lines:
            if 'Component Magnitudes' in line:
                in_magnitudes = True
                continue

            if in_magnitudes and line.strip().startswith('-'):
                parts = line.strip().split(':')
                if len(parts) == 2:
                    key = parts[0].replace('-', '').replace('**', '').strip()
                    try:
                        value = float(parts[1].strip())
                        stats[key.lower()] = value
                    except:
                        pass

            if in_magnitudes and line.strip().startswith('##'):
                in_magnitudes = False

        return stats
    except Exception as e:
        print(f"Warning: Could not parse data summary markdown: {e}")
        return {}


def generate_html_report(metadata, data_summary_stats=None):
    """Generate HTML report for trend model from metadata."""

    today = datetime.now().strftime("%Y-%m-%d")

    if data_summary_stats is None:
        data_summary_stats = {}

    posteriors = metadata.get('posterior', {})
    diagnostics = metadata.get('diagnostics', {})

    # Build posterior table rows for trend model parameters
    posterior_rows = []
    params = ['delta', 'rho', 'sigma_epsilon', 'gamma_s', 'gamma_a', 'weight_intercept', 'nu',
              'sigma_w', 'sigma_fourier', 'beta_s', 'beta_a',
              'alpha_d_s', 'alpha_m_s', 'alpha_d_a', 'alpha_m_a']

    for param in params:
        if param in posteriors:
            p = posteriors[param]
            mean = p['mean']
            lower = p['lower_ci']
            upper = p['upper_ci']
            rhat = p.get('rhat', None)
            ess_bulk = p.get('ess_bulk', None)

            rhat_str = f"{rhat:.3f}" if rhat else "—"
            ess_str = f"{int(ess_bulk)}" if ess_bulk else "—"

            posterior_rows.append(f"""
        <tr>
            <td><code>{param}</code></td>
            <td>{mean:8.4f}</td>
            <td>{lower:8.4f}</td>
            <td>{upper:8.4f}</td>
            <td>{rhat_str}</td>
            <td>{ess_str}</td>
        </tr>""")

    posterior_table = "\n".join(posterior_rows)

    # Extract trend parameter for interpretation
    delta_mean = posteriors.get('delta', {}).get('mean', 0)
    delta_lower = posteriors.get('delta', {}).get('lower_ci', 0)
    delta_upper = posteriors.get('delta', {}).get('upper_ci', 0)

    # Compute change in lbs over study period (924 days)
    num_days = 924
    trend_lbs = delta_mean * num_days
    trend_lower_lbs = delta_lower * num_days
    trend_upper_lbs = delta_upper * num_days

    # Model diagnostics
    n_chains = diagnostics.get('num_chains', '?')
    n_sampling = diagnostics.get('num_draws_sampling', '?')
    n_warmup = diagnostics.get('num_draws_warmup', '?')

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trend Model Variant: Interpretability Report</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f7fa;
            padding: 20px;
        }}
        .header {{
            background: linear-gradient(135deg, #2c3e50, #e74c3c);
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
        .equation-box {{
            background: #f8f9fa;
            border-left: 4px solid #e74c3c;
            padding: 1.5rem;
            margin: 1.5rem 0;
            border-radius: 5px;
            font-family: 'Cambria', 'Times New Roman', serif;
        }}
        .parameter-table {{
            font-size: 0.85rem;
        }}
        .parameter-table th {{
            background-color: #f8f9fa;
        }}
        h2 {{
            color: #2c3e50;
            border-bottom: 2px solid #e74c3c;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        h3 {{
            color: #34495e;
            margin-top: 1.5rem;
        }}
        .interpretation-note {{
            background: #fef5e7;
            border-left: 4px solid #f39c12;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 4px;
        }}
        .comparison-box {{
            background: #ecf0f1;
            border-left: 4px solid #3498db;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 4px;
        }}
        .badge {{
            display: inline-block;
            padding: 0.5rem 0.75rem;
            border-radius: 4px;
            font-size: 0.85rem;
            font-weight: 600;
            margin-left: 0.5rem;
        }}
        .badge-good {{
            background: #d4edda;
            color: #155724;
        }}
        .badge-warning {{
            background: #fff3cd;
            color: #856404;
        }}
        .badge-neutral {{
            background: #e2e3e5;
            color: #383d41;
        }}
        .toc {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 2rem;
        }}
        .toc h3 {{
            margin-top: 0;
        }}
        .toc ol {{
            margin-bottom: 0;
        }}
    </style>
    <!-- Simple Analytics -->
    <script async src="https://scripts.simpleanalyticscdn.com/latest.js"></script>
</head>
<body>
    <div class="header">
        <div style="max-width: 1000px; margin: 0 auto; padding: 0 20px;">
            <h1>Trend Model Variant: Interpretability Report</h1>
            <div class="subtitle" style="opacity: 0.95; margin-top: 0.5rem;">
                Testing for secular weight trends using linear δ parameter
            </div>
        </div>
    </div>

    <div style="max-width: 1000px; margin: 0 auto;">
        <div class="section">
            <h2>Overview</h2>
            <p>This report analyzes a model variant that adds an explicit linear trend parameter <code>δ</code> to test whether
            weight changes over the study period show a secular trend independent of fitness effects. Comparing this model to
            the constrained AR(1) model helps determine whether strength/aerobic effects are robust or confounded with trends.</p>

            <div class="interpretation-note">
                <strong>Key Question:</strong> Does adding a trend parameter change estimates of γ_s and γ_a? If yes,
                the fitness effects were partially confounded with the trend.
            </div>
        </div>

        <div class="section">
            <h2>Model Specification</h2>

            <h3>Trend Model Equation</h3>
            <div class="equation-box">
                <p>$$weight[t] \\sim \\text{{Student-t}}(\\nu, \\mu[t], \\sigma_w)$$</p>
                <p>$$\\mu[t] = weight\\_intercept + \\delta \\cdot \\frac{{t - 0.5D}}{{D}} + \\gamma_s \\cdot strength\\_fitness[t] + \\gamma_a \\cdot aerobic\\_fitness[t] + f\\_{{daily}}[t] + \\epsilon[t]$$</p>
                <p>where:
                <ul>
                    <li><strong>δ:</strong> Linear trend parameter (per standardized time unit)</li>
                    <li><strong>Time centering:</strong> (t - 0.5D) / D maps to [-0.5, 0.5] to reduce correlation with intercept</li>
                    <li>All other components identical to base model</li>
                </ul>
                </p>
            </div>

            <h3>Prior for Trend Parameter</h3>
            <p><code>delta ~ normal(0, 1.0)</code></p>
            <p><em>Weakly informative:</em> 1 std unit of δ corresponds to ~{trend_lbs:.2f} lbs change over full study period (924 days).
            If δ = ±0.01, that's {delta_mean * 924 / 100:.2f} lbs per 924 days.</p>
        </div>

        <div class="section">
            <h2>Posterior Parameters</h2>

            <p>Posterior means and 95% credible intervals. <strong>Focus on δ and comparison of γ_s, γ_a to base model.</strong></p>

            <table class="table table-striped parameter-table">
                <thead>
                    <tr>
                        <th>Parameter</th>
                        <th>Mean</th>
                        <th>2.5% CI</th>
                        <th>97.5% CI</th>
                        <th>R-hat</th>
                        <th>ESS (bulk)</th>
                    </tr>
                </thead>
                <tbody>
{posterior_table}
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>Trend Parameter Interpretation</h2>

            <h3>Estimated Secular Trend (δ)</h3>
            <p><strong>Point estimate:</strong> δ = {delta_mean:.6f} (95% CI: [{delta_lower:.6f}, {delta_upper:.6f}])</p>

            <p><strong>Implied weight change over study period:</strong></p>
            <ul>
                <li><strong>Mean:</strong> {trend_lbs:.3f} lbs over {num_days} days</li>
                <li><strong>95% CI:</strong> [{trend_lower_lbs:.3f}, {trend_upper_lbs:.3f}] lbs</li>
                <li><strong>Per year:</strong> {trend_lbs / (num_days / 365.25):.3f} lbs/year (95% CI: [{trend_lower_lbs / (num_days / 365.25):.3f}, {trend_upper_lbs / (num_days / 365.25):.3f}])</li>
            </ul>

            <div class="interpretation-note">
                <strong>Interpretation:</strong>
                <ul>
                    <li><strong>If CI includes 0:</strong> No clear evidence of secular trend; fitness effects are likely genuine</li>
                    <li><strong>If δ > 0 and significant:</strong> Upward weight trend exists; check if γ_s shrank compared to base model</li>
                    <li><strong>If δ < 0 and significant:</strong> Downward weight trend exists; opposite interpretation</li>
                </ul>
            </div>
        </div>

        <div class="section">
            <h2>Comparison to Base Model</h2>

            <p>To assess whether adding the trend parameter changes fitness effect estimates, compare:</p>

            <div class="comparison-box">
                <table class="table table-sm" style="margin-bottom: 0;">
                    <thead>
                        <tr>
                            <th>Parameter</th>
                            <th>Base Model</th>
                            <th>Trend Model</th>
                            <th>Change</th>
                            <th>Interpretation</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><code>γ_s</code></td>
                            <td>~0.143</td>
                            <td>See table above</td>
                            <td>↓ = confounding</td>
                            <td>Strength effect less/more confounded with trend</td>
                        </tr>
                        <tr>
                            <td><code>γ_a</code></td>
                            <td>~-0.086</td>
                            <td>See table above</td>
                            <td>Shift = trend impact</td>
                            <td>Aerobic effect independent of trend</td>
                        </tr>
                        <tr>
                            <td><code>δ</code></td>
                            <td>N/A</td>
                            <td>See above</td>
                            <td>—</td>
                            <td>Estimated secular weight change</td>
                        </tr>
                    </tbody>
                </table>
            </div>

            <p><strong>Model Comparison (LOO-CV):</strong> If elpd difference is &lt;1 SE, both models are equivalent. If trend model has better
            predictive performance, the trend parameter provides real value for prediction.</p>
        </div>

        <div class="section">
            <h2>Diagnostics</h2>

            <p>MCMC Sampling Configuration:</p>
            <ul>
                <li>Number of chains: {n_chains}</li>
                <li>Sampling iterations: {n_sampling}</li>
                <li>Warmup iterations: {n_warmup}</li>
                <li>Total post-warmup: {n_sampling * n_chains if isinstance(n_sampling, int) else '?'} draws</li>
            </ul>

            <p><strong>Convergence:</strong> Check R-hat values above. All should be &lt;1.01 for good mixing.</p>
        </div>

        <div class="section">
            <h2>Key Takeaways</h2>

            <ul>
                <li><strong>Trend is significant:</strong> Weight has a secular trend independent of fitness activities.
                    The base model may have partially absorbed this trend into γ_s.</li>
                <li><strong>Trend is negligible:</strong> No clear evidence of secular weight change. Fitness effects in base model
                    are not confounded with trend and represent genuine signal.</li>
                <li><strong>γ_s changes substantially:</strong> Strength effect was confounded with trend; trend model is preferred
                    for interpretation of true training effects.</li>
                <li><strong>γ_s is stable:</strong> Strength effect is robust across models; trend parameter provides minor additional
                    flexibility but doesn't improve understanding.</li>
            </ul>
        </div>

        <div class="section">
            <h2>Next Steps</h2>

            <ul>
                <li>Run formal model comparison (LOO-CV ELPD difference with SE)</li>
                <li>If trend model preferred: re-interpret γ_s from trend model as the "true" training effect</li>
                <li>Consider non-linear trend (quadratic) if δ-based linear trend doesn't fit well</li>
                <li>Investigate whether trend correlates with external factors (age progression, seasonal patterns)</li>
            </ul>
        </div>

        <footer style="text-align: center; margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #ddd; color: #666; font-size: 0.9rem;">
            <p><strong>Trend Model Report</strong> | Generated: {today}</p>
            <p>Model: <code>weight_state_space_training_decay_aerobic_symmetric_student_ar_spline_constrained_trend.stan</code></p>
        </footer>
    </div>
</body>
</html>"""

    return html


def main():
    """Main entry point."""
    # For now, define output paths - in actual use would accept command-line args
    # or search for metadata in a specific directory

    metadata_file = Path(__file__).parent.parent.parent / 'docs' / 'component_predictions' / 'posterior_metadata_trend.json'

    if not metadata_file.exists():
        print(f"Error: {metadata_file} not found.")
        print("Please run Stan model with trend parameter first and extract metadata.")
        sys.exit(1)

    print(f"Loading trend model metadata from {metadata_file}...")
    metadata = load_posterior_metadata(metadata_file)

    # Try to load data summary markdown
    data_summary_file = Path(__file__).parent.parent.parent / 'docs' / 'component_predictions' / 'prediction_summary_lbs.md'
    data_summary_stats = {}
    if data_summary_file.exists():
        print(f"Loading data summary from {data_summary_file}...")
        data_summary_stats = parse_data_summary_markdown(data_summary_file)

    print("Generating trend model HTML report...")
    html_content = generate_html_report(metadata, data_summary_stats)

    # Save to file
    output_dir = Path(__file__).parent.parent.parent / 'docs' / 'trend_model_report'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'index.html'

    with open(output_file, 'w') as f:
        f.write(html_content)

    print(f"✓ Successfully generated trend model report")
    print(f"  Output: {output_file}")
    print(f"  View at: file://{output_file}")


if __name__ == '__main__':
    main()
