#!/usr/bin/env python3
"""Create simple educational report for four-fitness state-space model.

Simplified version without f-string issues.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import textwrap


def load_model_results():
    """Load four-fitness model results."""
    model_dir = Path("output/four_fitness_analysis")

    results = {}

    try:
        if (model_dir / "parameter_summary.csv").exists():
            results['summary'] = pd.read_csv(
                model_dir / "parameter_summary.csv", index_col=0)

        if (model_dir / "variance_decomposition.csv").exists():
            results['variance'] = pd.read_csv(
                model_dir / "variance_decomposition.csv")

    except Exception as e:
        print(f"Warning: Could not load model results: {e}")

    return results


def create_parameter_table(summary):
    """Create HTML table of key parameters."""
    if summary is None:
        return "<p>Parameter summary not available.</p>"

    # Select key parameters
    key_params = [
        'psi_a_short', 'psi_s_short', 'psi_a_long', 'psi_s_long',
        'alpha_a_short', 'alpha_s_short', 'alpha_a_long', 'alpha_s_long',
        'beta_a_short', 'beta_s_short', 'beta_a_long', 'beta_s_long',
        'gamma_a_short', 'gamma_s_short', 'gamma_a_long', 'gamma_s_long',
        'sigma_w', 'alpha_gp', 'rho_gp'
    ]

    html = []
    html.append('<table class="table table-striped parameter-table">')
    html.append('    <thead>')
    html.append('        <tr>')
    html.append('            <th>Parameter</th>')
    html.append('            <th>Mean</th>')
    html.append('            <th>Std</th>')
    html.append('            <th>HDI 2.5%</th>')
    html.append('            <th>HDI 97.5%</th>')
    html.append('        </tr>')
    html.append('    </thead>')
    html.append('    <tbody>')

    for param in key_params:
        if param in summary.index:
            row = summary.loc[param]
            html.append(f'        <tr>')
            html.append(f'            <td><code>{param}</code></td>')
            html.append(f'            <td>{row["mean"]:.3f}</td>')
            html.append(f'            <td>{row["sd"]:.3f}</td>')
            html.append(f'            <td>{row["hdi_3%"]:.3f}</td>')
            html.append(f'            <td>{row["hdi_97%"]:.3f}</td>')
            html.append(f'        </tr>')

    html.append('    </tbody>')
    html.append('</table>')

    return '\n'.join(html)


def create_equations_section():
    """Create mathematical equations section."""
    equations = []

    equations.append('<div class="section" id="mathematical-foundations">')
    equations.append('    <h2>1. Mathematical Foundations</h2>')

    equations.append('    <h3>1.1 Complete State-Space System</h3>')
    equations.append('    <div class="equation-box">')
    equations.append('        \\[')
    equations.append('        \\begin{align*}')
    equations.append('        \\text{Impulse dynamics:} & \\\\')
    equations.append('        I_i[t] &= \\psi_i I_i[t-1] + X_i[t] \\\\')
    equations.append('        \\\\')
    equations.append('        \\text{Fitness dynamics:} & \\\\')
    equations.append('        F_i[t] &= \\alpha_i F_i[t-1] + \\beta_i I_i[t-1] \\\\')
    equations.append('        \\\\')
    equations.append('        \\text{Observation equation:} & \\\\')
    equations.append('        W[t] &= \\sum_{i} \\gamma_i F_i[t] + \\text{GP}(t) + f_{\\text{daily}}(t) + \\epsilon_w[t]')
    equations.append('        \\end{align*}')
    equations.append('        \\]')
    equations.append('        for $i \\in \\{a\\_short, s\\_short, a\\_long, s\\_long\\}$.')
    equations.append('    </div>')

    equations.append('    <h3>1.2 Key Mathematical Properties</h3>')
    equations.append('    <div class="equation-box">')
    equations.append('        <strong>Impulse accumulation:</strong>')
    equations.append('        \\[')
    equations.append('        I_i[t] = \\sum_{k=0}^{t} \\psi_i^k X_i[t-k]')
    equations.append('        \\]')
    equations.append('        Shows exponential decay of past intensities.')
    equations.append('    </div>')

    equations.append('    <div class="equation-box">')
    equations.append('        <strong>Fitness accumulation:</strong>')
    equations.append('        \\[')
    equations.append('        F_i[t] = \\sum_{k=1}^{t} \\alpha_i^{k-1} \\beta_i I_i[t-k]')
    equations.append('        \\]')
    equations.append('        Shows how impulses translate to fitness.')
    equations.append('    </div>')

    equations.append('    <div class="equation-box">')
    equations.append('        <strong>Complete weight prediction:</strong>')
    equations.append('        \\[')
    equations.append('        \\hat{W}[t] = \\sum_{i} \\gamma_i \\sum_{k=1}^{t} \\alpha_i^{k-1} \\beta_i \\sum_{j=0}^{t-k} \\psi_i^j X_i[t-k-j]')
    equations.append('        \\]')
    equations.append('        Shows the full cascade from activity to weight.')
    equations.append('    </div>')

    equations.append('</div>')

    return '\n'.join(equations)


def create_process_interactions():
    """Create process interactions section."""
    interactions = []

    interactions.append('<div class="section" id="process-interactions">')
    interactions.append('    <h2>2. Process Interactions</h2>')

    interactions.append('    <h3>2.1 System Diagram</h3>')
    interactions.append('    <div class="mermaid">')
    interactions.append('    graph TD')
    interactions.append('        A[Aerobic Intensity] --> B[Impulse: ψ_a_short]')
    interactions.append('        A --> C[Impulse: ψ_a_long]')
    interactions.append('        B --> D[Fitness: α_a_short, β_a_short]')
    interactions.append('        C --> E[Fitness: α_a_long, β_a_long]')
    interactions.append('        ')
    interactions.append('        F[Strength Intensity] --> G[Impulse: ψ_s_short]')
    interactions.append('        F --> H[Impulse: ψ_s_long]')
    interactions.append('        G --> I[Fitness: α_s_short, β_s_short]')
    interactions.append('        H --> J[Fitness: α_s_long, β_s_long]')
    interactions.append('        ')
    interactions.append('        D --> K[Weight: γ_a_short]')
    interactions.append('        E --> L[Weight: γ_a_long]')
    interactions.append('        I --> M[Weight: γ_s_short]')
    interactions.append('        J --> N[Weight: γ_s_long]')
    interactions.append('        ')
    interactions.append('        K --> O[Σ Weight Effects]')
    interactions.append('        L --> O')
    interactions.append('        M --> O')
    interactions.append('        N --> O')
    interactions.append('        ')
    interactions.append('        P[GP Trend] --> O')
    interactions.append('        Q[Daily Cycle] --> O')
    interactions.append('        O --> R[Observed Weight]')
    interactions.append('        S[Observation Noise] --> R')
    interactions.append('    </div>')

    interactions.append('    <h3>2.2 Time Scale Separation</h3>')
    interactions.append('    <div class="equation-box">')
    interactions.append('        <strong>Short-term half-life:</strong>')
    interactions.append('        \\[')
    interactions.append('        t_{1/2}^{\\text{short}} = -\\frac{\\log(2)}{\\log(\\psi_{\\text{short}})}')
    interactions.append('        \\]')
    interactions.append('        Typically hours to days.')
    interactions.append('    </div>')

    interactions.append('    <div class="equation-box">')
    interactions.append('        <strong>Long-term half-life:</strong>')
    interactions.append('        \\[')
    interactions.append('        t_{1/2}^{\\text{long}} = -\\frac{\\log(2)}{\\log(\\alpha_{\\text{long}} \\psi_{\\text{long}})}')
    interactions.append('        \\]')
    interactions.append('        Typically weeks to months.')
    interactions.append('    </div>')

    interactions.append('</div>')

    return '\n'.join(interactions)


def create_posterior_predictions():
    """Create posterior predictions section."""
    predictions = []

    predictions.append('<div class="section" id="posterior-predictions">')
    predictions.append('    <h2>3. Posterior Predictions</h2>')

    predictions.append('    <h3>3.1 Single Workout Response</h3>')
    predictions.append('    <div class="equation-box">')
    predictions.append('        \\[')
    predictions.append('        \\Delta W[t] = \\gamma \\beta \\alpha^{t} \\psi^{t} X[0]')
    predictions.append('        \\]')
    predictions.append('        Weight change at time $t$ from single workout at time 0.')
    predictions.append('    </div>')

    predictions.append('    <h3>3.2 Multiple Workout Superposition</h3>')
    predictions.append('    <div class="equation-box">')
    predictions.append('        \\[')
    predictions.append('        \\Delta W[t] = \\sum_{s=0}^{t} \\gamma \\beta \\alpha^{t-s} \\psi^{t-s} X[s]')
    predictions.append('        \\]')
    predictions.append('        Linear superposition of workouts at different times.')
    predictions.append('    </div>')

    predictions.append('    <h3>3.3 Uncertainty Propagation</h3>')
    predictions.append('    <div class="equation-box">')
    predictions.append('        \\[')
    predictions.append('        \\text{Var}(\\hat{W}[t]) = \\sum_{i} \\gamma_i^2 \\text{Var}(F_i[t]) + \\text{Var}(\\text{GP}(t)) + \\text{Var}(f_{\\text{daily}}(t)) + \\sigma_w^2')
    predictions.append('        \\]')
    predictions.append('        Shows how parameter uncertainty affects prediction uncertainty.')
    predictions.append('    </div>')

    predictions.append('</div>')

    return '\n'.join(predictions)


def create_html_report():
    """Create the complete HTML report."""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results = load_model_results()

    html_parts = []

    # Header
    html_parts.append(f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Four-Fitness State-Space Model: Technical Walkthrough</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
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
        .equation-box {{
            background: #f8f9fa;
            border-left: 4px solid #3498db;
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
        .mermaid {{
            background: white;
            padding: 1.5rem;
            border-radius: 5px;
            border: 1px solid #dee2e6;
            margin: 1.5rem 0;
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
            <h1 class="display-4">Four-Fitness State-Space Model</h1>
            <p class="lead">Technical Walkthrough: Mathematics and Predictions</p>
            <p class="mb-0">Generated: {current_time}</p>
        </div>''')

    # Table of Contents
    html_parts.append('''        <!-- Table of Contents -->
        <div class="section">
            <h2>📋 Table of Contents</h2>
            <ol>
                <li><a href="#parameter-estimates">Parameter Estimates</a></li>
                <li><a href="#mathematical-foundations">Mathematical Foundations</a></li>
                <li><a href="#process-interactions">Process Interactions</a></li>
                <li><a href="#posterior-predictions">Posterior Predictions</a></li>
            </ol>
        </div>''')

    # Parameter Estimates
    html_parts.append(f'''        <!-- Parameter Estimates -->
        <div class="section" id="parameter-estimates">
            <h2>1. Parameter Estimates</h2>
            <p>Posterior means and 95% highest density intervals from MCMC sampling:</p>
            {create_parameter_table(results.get('summary'))}
        </div>''')

    # Mathematical Foundations
    html_parts.append(create_equations_section())

    # Process Interactions
    html_parts.append(create_process_interactions())

    # Posterior Predictions
    html_parts.append(create_posterior_predictions())

    # Footer
    html_parts.append(f'''        <!-- Footer -->
        <div class="section text-center text-muted">
            <hr>
            <p>Four-Fitness State-Space Model Technical Walkthrough</p>
            <p>Generated on {current_time} | Model: weight_state_space_four_fitness.stan</p>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/mermaid@10.6.1/dist/mermaid.min.js"></script>
    <script>mermaid.initialize({{startOnLoad: true}});</script>

    <script>
        // Initialize MathJax
        MathJax = {{
            tex: {{
                inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
                displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
            }},
            svg: {{
                fontCache: 'global'
            }}
        }};
    </script>
</body>
</html>''')

    return '\n'.join(html_parts)


def main():
    """Main function."""
    print("Creating four-fitness educational report...")

    # Create output directory
    output_dir = Path("docs/four_fitness_educational")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate and save HTML
    html_content = create_html_report()

    output_file = output_dir / "index.html"
    with open(output_file, 'w') as f:
        f.write(html_content)

    print(f"Report saved to: {output_file}")

    # Also save equations as text
    equations_file = output_dir / "equations.txt"
    with open(equations_file, 'w') as f:
        f.write("FOUR-FITNESS STATE-SPACE MODEL EQUATIONS\n")
        f.write("=" * 50 + "\n\n")
        f.write("Impulse dynamics:\n")
        f.write("I_i[t] = ψ_i I_i[t-1] + X_i[t]\n\n")
        f.write("Fitness dynamics:\n")
        f.write("F_i[t] = α_i F_i[t-1] + β_i I_i[t-1]\n\n")
        f.write("Observation equation:\n")
        f.write("W[t] = Σ_i γ_i F_i[t] + GP(t) + f_daily(t) + ε_w[t]\n\n")
        f.write("Complete prediction:\n")
        f.write("Ŵ[t] = Σ_i γ_i Σ_{k=1}^t α_i^{k-1} β_i Σ_{j=0}^{t-k} ψ_i^j X_i[t-k-j]\n")

    print(f"Equations saved to: {equations_file}")

    print("\n" + "=" * 70)
    print("EDUCATIONAL REPORT CREATED SUCCESSFULLY")
    print("=" * 70)
    print("The report includes:")
    print("1. Parameter estimates with credible intervals")
    print("2. Mathematical foundations with LaTeX equations")
    print("3. Process interaction diagrams")
    print("4. Posterior prediction equations")
    print("\nOpen docs/four_fitness_educational/index.html in your browser")


if __name__ == "__main__":
    main()