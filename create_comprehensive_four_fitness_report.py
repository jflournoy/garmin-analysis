#!/usr/bin/env python3
"""Create comprehensive educational report for four-fitness state-space model.

This report provides a complete walkthrough of:
1. Mathematical formulation with LaTeX equations
2. Stan code implementation details
3. Python data processing pipeline
4. Step-by-step posterior predictions
5. Process interactions and component contributions
6. Parameter sensitivity analysis
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
import plotly.io as pio
import textwrap
import sys

# Set Plotly template
pio.templates.default = "plotly_white"


class ComprehensiveFourFitnessReport:
    """Create comprehensive educational HTML report for four-fitness model."""

    def __init__(self, output_dir: str = "docs/four_fitness_comprehensive"):
        """Initialize report generator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load model results if available
        self.model_results = self._load_model_results()

    def _load_model_results(self) -> dict:
        """Load four-fitness model results if available."""
        model_dir = Path("output/four_fitness_analysis")
        results = {'summary': None, 'variance': None}

        try:
            # Load parameter summary
            if (model_dir / "parameter_summary.csv").exists():
                results['summary'] = pd.read_csv(
                    model_dir / "parameter_summary.csv", index_col=0)

            # Load variance decomposition
            if (model_dir / "variance_decomposition.csv").exists():
                results['variance'] = pd.read_csv(
                    model_dir / "variance_decomposition.csv")
        except Exception as e:
            print(f"Note: Could not load model results: {e}")

        return results

    def create_mathematical_foundations(self) -> str:
        """Create mathematical foundations section."""
        return r"""
        <div class="section" id="mathematical-foundations">
            <h2>1. Mathematical Foundations</h2>

            <h3>1.1 State-Space Formulation</h3>
            <div class="equation-box">
                <strong>Complete system for component \(i \in \{a\_short, s\_short, a\_long, s\_long\}\):</strong><br><br>
                \[
                \begin{aligned}
                \text{Impulse dynamics:} & \quad I_i[t] = \psi_i I_i[t-1] + X_i[t] \\
                \text{Fitness dynamics:} & \quad F_i[t] = \alpha_i F_i[t-1] + \beta_i I_i[t-1] \\
                \text{Observation equation:} & \quad W[t] = \sum_i \gamma_i F_i[t] + \text{GP}(t) + f_{\text{daily}}(t) + \epsilon_w[t]
                \end{aligned}
                \]
            </div>

            <h3>1.2 Parameter Interpretation</h3>
            <div class="equation-box">
                <strong>Key parameters:</strong>
                <ul>
                    <li>\(\psi_i \in [0,1]\): Impulse decay (short-term: ~0.3-0.5, long-term: ~0.7-0.9)</li>
                    <li>\(\alpha_i \in [0,1]\): Fitness decay (short-term: ~0.5, long-term: ~0.8-0.9)</li>
                    <li>\(\beta_i > 0\): Fitness gain per impulse unit</li>
                    <li>\(\gamma_i\): Weight effect (negative for fat loss, positive for muscle gain)</li>
                </ul>
            </div>

            <h3>1.3 Time Scale Separation</h3>
            <div class="equation-box">
                <strong>Short-term effects (hours to days):</strong>
                \[
                \text{Half-life} = -\frac{\log(2)}{\log(\psi_{\text{short}})} \ \text{days}
                \]
                <strong>Long-term effects (weeks to months):</strong>
                \[
                \text{Effective half-life} = -\frac{\log(2)}{\log(\alpha_{\text{long}} \psi_{\text{long}})} \ \text{days}
                \]
            </div>
        </div>
        """

    def create_stan_implementation(self) -> str:
        """Create Stan implementation section."""
        # Try to read Stan file
        stan_content = ""
        try:
            with open("stan/weight_state_space_four_fitness.stan", "r") as f:
                stan_content = f.read()
        except:
            stan_content = "// Stan file not found"

        # Extract key parts
        data_block = self._extract_block(stan_content, "data {", "}")
        params_block = self._extract_block(stan_content, "parameters {", "}")

        return f"""
        <div class="section" id="stan-implementation">
            <h2>2. Stan Implementation</h2>

            <h3>2.1 Data Block - Model Inputs</h3>
            <div class="code-block">
{textwrap.indent(data_block if data_block else "// Data block not available", "                ")}
            </div>
            <p><strong>Key inputs:</strong> Daily activity intensities (aerobic/strength), weight observations, time indices, hour of day, Fourier basis, sparse GP configuration.</p>

            <h3>2.2 Parameters Block - Unknown Quantities</h3>
            <div class="code-block">
{textwrap.indent(params_block if params_block else "// Parameters block not available", "                ")}
            </div>
            <p><strong>16 core parameters:</strong> 4 ψ (impulse decay), 4 α (fitness decay), 4 β (fitness gain), 4 γ (weight effects).</p>

            <h3>2.3 State Computations</h3>
            <div class="equation-box">
                <strong>Impulse state recursion:</strong>
                \[
                I_i[t] = \\psi_i I_i[t-1] + X_i[t]
                \]
                <strong>Fitness state recursion:</strong>
                \[
                F_i[t] = \\alpha_i F_i[t-1] + \\beta_i I_i[t-1]
                \]
                Implemented in Stan's <code>transformed parameters</code> block.
            </div>

            <h3>2.4 Priors and Likelihood</h3>
            <div class="equation-box">
                <strong>Informative priors based on physiological knowledge:</strong>
                <ul>
                    <li>Short-term ψ ~ Beta(3,5) (favors ~0.375, faster decay)</li>
                    <li>Long-term ψ ~ Beta(5,2) (favors ~0.714, slower decay)</li>
                    <li>γ_a_short ~ Normal(-0.3, 0.15) (aerobic reduces weight short-term)</li>
                    <li>γ_s_long ~ Normal(0.3, 0.15) (strength increases weight long-term)</li>
                </ul>
            </div>
        </div>
        """

    def _extract_block(self, content: str, start_marker: str, end_char: str = "}") -> str:
        """Extract a code block from Stan file."""
        start = content.find(start_marker)
        if start == -1:
            return ""

        # Find matching closing brace
        brace_count = 0
        for i in range(start, len(content)):
            if content[i] == '{':
                brace_count += 1
            elif content[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    return content[start:i+1]
        return ""

    def create_python_pipeline(self) -> str:
        """Create Python data pipeline section."""
        return """
        <div class="section" id="python-pipeline">
            <h2>3. Python Data Pipeline</h2>

            <h3>3.1 Data Loading and Preparation</h3>
            <div class="equation-box">
                <strong>Key steps:</strong>
                <ol>
                    <li><strong>Load Garmin data:</strong> Weight measurements, activity intensities</li>
                    <li><strong>Aggregate to daily:</strong> Sum intensities, average weights</li>
                    <li><strong>Handle missing values:</strong> Interpolation or carry-forward</li>
                    <li><strong>Standardize:</strong> Zero mean, unit variance for stable sampling</li>
                </ol>
            </div>

            <h3>3.2 Stan Data Preparation</h3>
            <div class="equation-box">
                <strong>Data structures for Stan:</strong>
                <ul>
                    <li><strong>Time indices:</strong> Map weight observations to days</li>
                    <li><strong>Fourier basis:</strong> Create sin/cos terms for daily cycles</li>
                    <li><strong>Sparse GP:</strong> Inducing points for computational efficiency</li>
                    <li><strong>Activity matrices:</strong> Separate aerobic and strength intensities</li>
                </ul>
            </div>

            <h3>3.3 Model Fitting with CmdStanPy</h3>
            <div class="code-block">
# Compile and fit model
model = CmdStanModel(stan_file="stan/weight_state_space_four_fitness.stan")

fit = model.sample(
    data=stan_data,
    chains=4,
    iter_warmup=500,
    iter_sampling=500,
    adapt_delta=0.95,
    max_treedepth=12,
    show_progress=True
)

# Convert to ArviZ InferenceData for analysis
idata = az.from_cmdstanpy(
    posterior=fit,
    posterior_predictive='y_weight_rep',
    log_likelihood='log_lik_weight'
)
            </div>
        </div>
        """

    def create_parameter_estimates(self) -> str:
        """Create parameter estimates section."""
        if self.model_results['summary'] is None:
            return """
            <div class="section" id="parameter-estimates">
                <h2>4. Parameter Estimates</h2>
                <p>Model results not available. Run the analysis first.</p>
            </div>
            """

        summary = self.model_results['summary']

        # Create HTML table
        table_html = """
        <table class="table table-striped parameter-table">
            <thead>
                <tr>
                    <th>Parameter</th>
                    <th>Description</th>
                    <th>Mean</th>
                    <th>Std</th>
                    <th>HDI 2.5%</th>
                    <th>HDI 97.5%</th>
                </tr>
            </thead>
            <tbody>
        """

        param_descriptions = {
            'psi_a_short': 'Aerobic short-term impulse decay',
            'psi_s_short': 'Strength short-term impulse decay',
            'psi_a_long': 'Aerobic long-term impulse decay',
            'psi_s_long': 'Strength long-term impulse decay',
            'alpha_a_short': 'Aerobic short-term fitness decay',
            'alpha_s_short': 'Strength short-term fitness decay',
            'alpha_a_long': 'Aerobic long-term fitness decay',
            'alpha_s_long': 'Strength long-term fitness decay',
            'beta_a_short': 'Aerobic short-term fitness gain',
            'beta_s_short': 'Strength short-term fitness gain',
            'beta_a_long': 'Aerobic long-term fitness gain',
            'beta_s_long': 'Strength long-term fitness gain',
            'gamma_a_short': 'Weight effect: aerobic short-term',
            'gamma_s_short': 'Weight effect: strength short-term',
            'gamma_a_long': 'Weight effect: aerobic long-term',
            'gamma_s_long': 'Weight effect: strength long-term',
        }

        for param, desc in param_descriptions.items():
            if param in summary.index:
                row = summary.loc[param]
                table_html += f"""
                <tr>
                    <td><code>{param}</code></td>
                    <td>{desc}</td>
                    <td>{row['mean']:.3f}</td>
                    <td>{row['sd']:.3f}</td>
                    <td>{row.get('hdi_3%', row.get('hdi_2.5%', 0)):.3f}</td>
                    <td>{row.get('hdi_97%', row.get('hdi_97.5%', 0)):.3f}</td>
                </tr>
                """

        table_html += """
            </tbody>
        </table>
        """

        return f"""
        <div class="section" id="parameter-estimates">
            <h2>4. Parameter Estimates</h2>
            <p>Posterior means and 95% highest density intervals from MCMC sampling:</p>
            {table_html}
        </div>
        """

    def create_variance_decomposition_plot(self) -> str:
        """Create variance decomposition plot if data is available."""
        if self.model_results['variance'] is None:
            return "<p>Variance decomposition data not available. Run the analysis first.</p>"

        # Create a simple HTML table instead of Plotly for simplicity
        var_df = self.model_results['variance']

        table_html = """
        <table class="table table-striped">
            <thead>
                <tr>
                    <th>Component</th>
                    <th>Mean Variance</th>
                    <th>Std</th>
                </tr>
            </thead>
            <tbody>
        """

        for _, row in var_df.iterrows():
            table_html += f"""
                <tr>
                    <td>{row['Component']}</td>
                    <td>{row['Mean']:.3f}</td>
                    <td>{row.get('Std', 0):.3f}</td>
                </tr>
            """

        table_html += """
            </tbody>
        </table>
        """

        return f"""
        <div class="equation-box">
            <strong>Variance Decomposition:</strong>
            <p>Proportion of weight variance explained by each model component:</p>
            {table_html}
            <p><strong>Interpretation:</strong> Shows how much of the weight variability is explained by fitness effects vs other components (GP trend, daily cycles, noise).</p>
        </div>
        """

    def create_step_by_step_predictions(self) -> str:
        """Create step-by-step predictions section."""
        return r"""
        <div class="section" id="step-by-step-predictions">
            <h2>5. Step-by-Step Predictions</h2>

            <h3>5.1 Single Workout Propagation</h3>
            <div class="equation-box">
                <strong>Short-term response to single aerobic workout:</strong>
                \[
                \Delta W[t] = \gamma_{\text{a\_short}} \beta_{\text{a\_short}} \psi_{\text{a\_short}}^t X[0]
                \]
                <strong>Interpretation:</strong> Immediate weight loss (dehydration) that decays over 1-2 days.
            </div>

            <div class="equation-box">
                <strong>Long-term response to single strength workout:</strong>
                \[
                \Delta W[t] = \gamma_{\text{s\_long}} \beta_{\text{s\_long}} (\alpha_{\text{s\_long}} \psi_{\text{s\_long}})^t X[0]
                \]
                <strong>Interpretation:</strong> Gradual muscle gain that accumulates over weeks.
            </div>

            <h3>5.2 Multiple Workout Superposition</h3>
            <div class="equation-box">
                <strong>Cumulative effect of workout history:</strong>
                \[
                W[t] = \sum_{s=0}^t \sum_i \gamma_i \beta_i (\alpha_i \psi_i)^{t-s} X_i[s]
                \]
                <strong>Interpretation:</strong> Current weight depends on entire history of workouts, with exponential decay of past contributions.
            </div>

            <h3>5.3 Interaction Effects</h3>
            <div class="equation-box">
                <strong>Aerobic-strength interactions:</strong>
                <ul>
                    <li><strong>Short-term:</strong> Aerobic dehydration vs strength inflammation</li>
                    <li><strong>Long-term:</strong> Aerobic fat loss vs strength muscle gain</li>
                    <li><strong>Net effect:</strong> Sum of all 4 components determines weight change</li>
                </ul>
            </div>
        </div>
        """

    def create_parameter_sensitivity(self) -> str:
        """Create parameter sensitivity analysis section."""
        return """
        <div class="section" id="parameter-sensitivity">
            <h2>6. Parameter Sensitivity Analysis</h2>

            <h3>6.1 Impulse Decay (ψ) Sensitivity</h3>
            <div class="equation-box">
                <strong>Effect on workout persistence:</strong>
                <ul>
                    <li>\(\psi = 0.3\): Effects last ~1 day (half-life = 0.5 days)</li>
                    <li>\(\psi = 0.7\): Effects last ~2 days (half-life = 1.9 days)</li>
                    <li>\(\psi = 0.9\): Effects last ~7 days (half-life = 6.6 days)</li>
                </ul>
                \[
                \\text{{Half-life}} = -\\frac{{\\log(2)}}{{\\log(\\psi)}} \\ \\text{{days}}
                \]
            </div>

            <h3>6.2 Weight Effect (γ) Sensitivity</h3>
            <div class="equation-box">
                <strong>Effect on weight response:</strong>
                <ul>
                    <li>\(\gamma = -0.5\): Strong fat loss effect</li>
                    <li>\(\gamma = -0.1\): Weak fat loss effect</li>
                    <li>\(\gamma = 0.3\): Moderate muscle gain effect</li>
                    <li>\(\gamma = 0.1\): Weak muscle gain effect</li>
                </ul>
                <strong>Interpretation:</strong> Magnitude determines how much fitness changes affect weight.
            </div>

            <h3>6.3 Biological Interpretation</h3>
            <div class="equation-box">
                <strong>Expected parameter patterns:</strong>
                <ul>
                    <li><strong>γ_a_short &lt; 0:</strong> Aerobic exercise causes dehydration</li>
                    <li><strong>γ_s_short &gt; 0:</strong> Strength training causes inflammation</li>
                    <li><strong>γ_a_long &lt; 0:</strong> Aerobic training promotes fat loss</li>
                    <li><strong>γ_s_long &gt; 0:</strong> Strength training builds muscle</li>
                    <li><strong>ψ_short &lt; ψ_long:</strong> Short-term effects decay faster</li>
                    <li><strong>α_long ≈ 0.9:</strong> Long-term fitness persists for months</li>
                </ul>
            </div>
        </div>
        """

    def create_html_report(self) -> str:
        """Create the complete HTML report."""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Build HTML
        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Four-Fitness State-Space Model: Comprehensive Educational Report</title>
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
        .code-block {{
            background: #2c3e50;
            color: #ecf0f1;
            padding: 1.5rem;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            font-size: 0.9rem;
            overflow-x: auto;
            margin: 1.5rem 0;
        }}
        .parameter-table {{
            font-size: 0.85rem;
        }}
        .parameter-table th {{
            background-color: #f8f9fa;
            position: sticky;
            top: 0;
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
        .notice {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 4px;
        }}
    </style>
    <!-- Simple Analytics -->
    <script async src="https://scripts.simpleanalyticscdn.com/latest.js"></script>
</head>
<body>
    <div class="container">
        <div class="header text-center">
            <h1 class="display-4">Four-Fitness State-Space Model</h1>
            <p class="lead">Comprehensive Educational Walkthrough</p>
            <p class="mb-0">Generated: {current_time}</p>
        </div>

        <div class="notice">
            <strong>Educational Report:</strong> This report provides a complete technical walkthrough of the four-fitness state-space model, including mathematical foundations, code implementation, and step-by-step predictions.
        </div>

        <!-- Table of Contents -->
        <div class="section">
            <h2>📋 Table of Contents</h2>
            <ol>
                <li><a href="#mathematical-foundations">Mathematical Foundations</a></li>
                <li><a href="#stan-implementation">Stan Implementation</a></li>
                <li><a href="#python-pipeline">Python Data Pipeline</a></li>
                <li><a href="#parameter-estimates">Parameter Estimates</a></li>
                <li><a href="#variance-decomposition">Variance Decomposition</a></li>
                <li><a href="#step-by-step-predictions">Step-by-Step Predictions</a></li>
                <li><a href="#parameter-sensitivity">Parameter Sensitivity Analysis</a></li>
            </ol>
        </div>

        {self.create_mathematical_foundations()}
        {self.create_stan_implementation()}
        {self.create_python_pipeline()}
        {self.create_parameter_estimates()}

        <!-- Variance Decomposition -->
        <div class="section" id="variance-decomposition">
            <h2>5. Variance Decomposition</h2>
            {self.create_variance_decomposition_plot()}
        </div>

        {self.create_step_by_step_predictions()}
        {self.create_parameter_sensitivity()}

        <!-- Footer -->
        <div class="section text-center text-muted">
            <hr>
            <p>Four-Fitness State-Space Model: Comprehensive Educational Walkthrough</p>
            <p>Generated on {current_time} | Model: weight_state_space_four_fitness.stan</p>
        </div>
    </div>

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
</html>'''

        return html

    def save_report(self) -> None:
        """Save HTML report to file."""
        print("Creating comprehensive four-fitness educational report...")

        html_content = self.create_html_report()

        output_file = self.output_dir / "index.html"
        with open(output_file, 'w') as f:
            f.write(html_content)

        print(f"Report saved to: {output_file}")
        print("\n" + "=" * 70)
        print("COMPREHENSIVE EDUCATIONAL REPORT CREATED SUCCESSFULLY")
        print("=" * 70)
        print("The report includes:")
        print("1. Mathematical foundations with LaTeX equations")
        print("2. Stan code implementation details")
        print("3. Python data processing pipeline")
        print("4. Parameter estimates with credible intervals")
        print("5. Step-by-step prediction equations")
        print("6. Parameter sensitivity analysis")
        print(f"\nOpen {output_file} in your browser")


def main():
    """Main function."""
    report = ComprehensiveFourFitnessReport()
    report.save_report()


if __name__ == "__main__":
    main()