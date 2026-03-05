#!/usr/bin/env python3
"""Create comprehensive educational report for four-fitness state-space model.

This report provides a technical walkthrough of:
1. Mathematical formulation of the state-space model
2. Stan code implementation details
3. Python data processing pipeline
4. Posterior predictions showing data influence
5. Process interactions and component contributions

NO biological interpretations or conclusions - purely technical/mathematical.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import sys
import textwrap

# Set Plotly template
pio.templates.default = "plotly_white"


class FourFitnessEducationalReport:
    """Create educational HTML report for four-fitness state-space model."""

    def __init__(self, output_dir: str = "docs/four_fitness_educational"):
        """Initialize report generator.

        Args:
            output_dir: Path to directory for output files.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load model results
        self.model_results = self._load_model_results()
        self.standardization = self._load_standardization()

        # Mathematical equations
        self.equations = self._create_equations()

        # Stan code snippets
        self.stan_code = self._extract_stan_code()

        # Python code snippets
        self.python_code = self._extract_python_code()

        # Step-by-step predictions
        self.predictions = self.create_step_by_step_predictions()

        # Parameter sensitivity
        self.sensitivity_analysis = self.create_parameter_sensitivity_analysis()

    def _load_model_results(self) -> dict:
        """Load four-fitness model results."""
        model_dir = Path("output/four_fitness_analysis")

        results = {
            'summary': None,
            'samples': None,
            'variance': None,
            'fitness_states': None
        }

        try:
            # Load parameter summary
            if (model_dir / "parameter_summary.csv").exists():
                results['summary'] = pd.read_csv(
                    model_dir / "parameter_summary.csv", index_col=0)

            # Load parameter samples
            if (model_dir / "parameter_samples.json").exists():
                with open(model_dir / "parameter_samples.json", 'r') as f:
                    results['samples'] = json.load(f)

            # Load variance decomposition
            if (model_dir / "variance_decomposition.csv").exists():
                results['variance'] = pd.read_csv(
                    model_dir / "variance_decomposition.csv")

            # Try to load fitness states if available
            states_path = Path("output/four_fitness_full/fitness_states.csv")
            if states_path.exists():
                results['fitness_states'] = pd.read_csv(states_path)

        except Exception as e:
            print(f"Warning: Could not load some model results: {e}")

        return results

    def _load_standardization(self) -> dict:
        """Load standardization information."""
        model_dir = Path("output/four_fitness_analysis")
        standardization = {}

        try:
            if (model_dir / "standardization.json").exists():
                with open(model_dir / "standardization.json", 'r') as f:
                    standardization = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load standardization: {e}")

        return standardization

    def _create_equations(self) -> dict:
        """Create mathematical equations for the report."""
        return {
            'impulse_dynamics': r"""
            \text{Impulse dynamics for component } i \in \{a\_short, s\_short, a\_long, s\_long\}:
            \[
            I_i[t] = \psi_i \cdot I_i[t-1] + X_i[t]
            \]
            where:
            \begin{itemize}
            \item $I_i[t]$: Impulse state at time $t$
            \item $\psi_i \in [0,1]$: Impulse decay parameter
            \item $X_i[t]$: Activity intensity (standardized)
            \end{itemize}
            """,

            'fitness_dynamics': r"""
            \text{Fitness dynamics:}
            \[
            F_i[t] = \alpha_i \cdot F_i[t-1] + \beta_i \cdot I_i[t-1]
            \]
            where:
            \begin{itemize}
            \item $F_i[t]$: Fitness state at time $t$
            \item $\alpha_i \in [0,1]$: Fitness decay parameter
            \item $\beta_i > 0$: Fitness gain parameter
            \end{itemize}
            """,

            'observation_equation': r"""
            \text{Weight observation equation:}
            \[
            W[t] = \sum_{i} \gamma_i \cdot F_i[t] + \text{GP}(t) + f_{\text{daily}}(t) + \epsilon_w[t]
            \]
            where:
            \begin{itemize}
            \item $W[t]$: Observed weight (standardized)
            \item $\gamma_i$: Weight effect parameter
            \item $\text{GP}(t)$: Gaussian Process for intrinsic variations
            \item $f_{\text{daily}}(t)$: Fourier basis for daily cycles
            \item $\epsilon_w[t] \sim \mathcal{N}(0, \sigma_w^2)$: Observation noise
            \end{itemize}
            """,

            'complete_system': r"""
            \text{Complete state-space system:}
            \begin{align*}
            \text{State equations:} & \\
            I_i[t] &= \psi_i I_i[t-1] + X_i[t] \\
            F_i[t] &= \alpha_i F_i[t-1] + \beta_i I_i[t-1] \\
            & \\
            \text{Observation equation:} & \\
            W[t] &= \sum_{i} \gamma_i F_i[t] + \text{GP}(t) + f_{\text{daily}}(t) + \epsilon_w[t]
            \end{align*}
            for $i \in \{a\_short, s\_short, a\_long, s\_long\}$.
            """
        }

    def _extract_stan_code(self) -> dict:
        """Extract key Stan code snippets from the model file."""
        stan_file = Path("stan/weight_state_space_four_fitness.stan")

        if not stan_file.exists():
            return {"error": "Stan file not found"}

        try:
            with open(stan_file, 'r') as f:
                content = f.read()

            # Extract key sections
            sections = {}

            # Find data block with better boundary detection
            data_start = content.find("data {")
            if data_start != -1:
                # Find matching closing brace
                brace_count = 0
                data_end = data_start
                for i in range(data_start, len(content)):
                    if content[i] == '{':
                        brace_count += 1
                    elif content[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            data_end = i + 1
                            break
                sections['data_block'] = content[data_start:data_end]

            # Find parameters block
            params_start = content.find("parameters {")
            if params_start != -1:
                brace_count = 0
                params_end = params_start
                for i in range(params_start, len(content)):
                    if content[i] == '{':
                        brace_count += 1
                    elif content[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            params_end = i + 1
                            break
                sections['parameters_block'] = content[params_start:params_end]

            # Find transformed parameters block
            trans_start = content.find("transformed parameters {")
            if trans_start != -1:
                brace_count = 0
                trans_end = trans_start
                for i in range(trans_start, len(content)):
                    if content[i] == '{':
                        brace_count += 1
                    elif content[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            trans_end = i + 1
                            break
                trans_content = content[trans_start:trans_end]

                # Extract impulse computation
                impulse_start = trans_content.find("// Compute impulse states")
                if impulse_start != -1:
                    impulse_end = trans_content.find("\n  // Compute fitness states", impulse_start)
                    if impulse_end == -1:
                        impulse_end = trans_content.find("\n  // GP covariance", impulse_start)
                    if impulse_end != -1:
                        sections['impulse_computation'] = trans_content[impulse_start:impulse_end]

                # Extract fitness computation
                fitness_start = trans_content.find("// Compute fitness states")
                if fitness_start != -1:
                    fitness_end = trans_content.find("\n  // GP covariance", fitness_start)
                    if fitness_end != -1:
                        sections['fitness_computation'] = trans_content[fitness_start:fitness_end]

                # Extract GP computation
                gp_start = trans_content.find("// GP covariance")
                if gp_start != -1:
                    gp_end = trans_content.find("\n  // Daily spline", gp_start)
                    if gp_end != -1:
                        sections['gp_computation'] = trans_content[gp_start:gp_end]

                # Extract daily spline computation
                daily_start = trans_content.find("// Daily spline")
                if daily_start != -1:
                    daily_end = trans_content.find("\n  // Weight prediction", daily_start)
                    if daily_end != -1:
                        sections['daily_spline'] = trans_content[daily_start:daily_end]

            # Find model block
            model_start = content.find("model {")
            if model_start != -1:
                brace_count = 0
                model_end = model_start
                for i in range(model_start, len(content)):
                    if content[i] == '{':
                        brace_count += 1
                    elif content[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            model_end = i + 1
                            break
                sections['model_block'] = content[model_start:model_end]

            # Find generated quantities block
            gen_start = content.find("generated quantities {")
            if gen_start != -1:
                brace_count = 0
                gen_end = gen_start
                for i in range(gen_start, len(content)):
                    if content[i] == '{':
                        brace_count += 1
                    elif content[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            gen_end = i + 1
                            break
                sections['generated_quantities'] = content[gen_start:gen_end]

            return sections

        except Exception as e:
            return {"error": f"Failed to extract Stan code: {e}"}

    def _extract_python_code(self) -> dict:
        """Extract key Python code snippets from analysis script."""
        python_file = Path("analyze_four_fitness_model.py")

        if not python_file.exists():
            return {"error": "Python file not found"}

        try:
            with open(python_file, 'r') as f:
                content = f.read()

            sections = {}

            # Find data preparation method
            data_start = content.find("def load_and_prepare_data(self)")
            if data_start != -1:
                # Find the end of the method (next method or end of class)
                next_method = content.find("\n    def ", data_start + 1)
                class_end = content.find("\n\nclass ", data_start + 1)
                if next_method != -1 and (class_end == -1 or next_method < class_end):
                    sections['data_preparation'] = content[data_start:next_method]
                elif class_end != -1:
                    sections['data_preparation'] = content[data_start:class_end]

            # Find Stan data preparation
            stan_data_start = content.find("# Prepare Stan data")
            if stan_data_start != -1:
                # Find the next major section
                next_section = content.find("\n        # Save standardization", stan_data_start)
                if next_section != -1:
                    sections['stan_data_prep'] = content[stan_data_start:next_section]

            # Find model fitting method
            fit_start = content.find("def fit_model(self)")
            if fit_start != -1:
                next_method = content.find("\n    def ", fit_start + 1)
                class_end = content.find("\n\nclass ", fit_start + 1)
                if next_method != -1 and (class_end == -1 or next_method < class_end):
                    sections['model_fitting'] = content[fit_start:next_method]
                elif class_end != -1:
                    sections['model_fitting'] = content[fit_start:class_end]

            # Find analysis method
            analyze_start = content.find("def analyze_results(self)")
            if analyze_start != -1:
                next_method = content.find("\n    def ", analyze_start + 1)
                class_end = content.find("\n\nclass ", analyze_start + 1)
                if next_method != -1 and (class_end == -1 or next_method < class_end):
                    sections['results_analysis'] = content[analyze_start:next_method]
                elif class_end != -1:
                    sections['results_analysis'] = content[analyze_start:class_end]

            return sections

        except Exception as e:
            return {"error": f"Failed to extract Python code: {e}"}

    def create_parameter_table(self) -> str:
        """Create HTML table of key parameters."""
        if self.model_results['summary'] is None:
            return "<p>Parameter summary not available.</p>"

        summary = self.model_results['summary']

        # Select key parameters
        key_params = [
            'psi_a_short', 'psi_s_short', 'psi_a_long', 'psi_s_long',
            'alpha_a_short', 'alpha_s_short', 'alpha_a_long', 'alpha_s_long',
            'beta_a_short', 'beta_s_short', 'beta_a_long', 'beta_s_long',
            'gamma_a_short', 'gamma_s_short', 'gamma_a_long', 'gamma_s_long',
            'sigma_w', 'alpha_gp', 'rho_gp'
        ]

        html = """
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
            'sigma_w': 'Weight observation noise',
            'alpha_gp': 'GP marginal standard deviation',
            'rho_gp': 'GP length scale'
        }

        for param in key_params:
            if param in summary.index:
                row = summary.loc[param]
                desc = param_descriptions.get(param, param)

                html += f"""
                <tr>
                    <td><code>{param}</code></td>
                    <td>{desc}</td>
                    <td>{row['mean']:.3f}</td>
                    <td>{row['sd']:.3f}</td>
                    <td>{row['hdi_3%']:.3f}</td>
                    <td>{row['hdi_97%']:.3f}</td>
                </tr>
                """

        html += """
            </tbody>
        </table>
        """

        return html

    def create_variance_decomposition_plot(self) -> str:
        """Create Plotly figure for variance decomposition."""
        if self.model_results['variance'] is None:
            return "<p>Variance decomposition data not available.</p>"

        var_df = self.model_results['variance']

        fig = go.Figure(data=[
            go.Bar(
                x=var_df['Component'],
                y=var_df['Mean'],
                error_y=dict(type='data', array=var_df['Std'], visible=True),
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                             '#9467bd', '#8c564b', '#e377c2']
            )
        ])

        fig.update_layout(
            title='Variance Decomposition of Weight',
            xaxis_title='Component',
            yaxis_title='Proportion of Variance',
            yaxis=dict(range=[0, 1]),
            template='plotly_white',
            height=400
        )

        return pio.to_html(fig, full_html=False, include_plotlyjs=False)

    def create_process_interaction_diagram(self) -> str:
        """Create mermaid.js diagram showing process interactions."""
        diagram = """
        <div class="mermaid">
        graph TD
            A[Aerobic Intensity] --> B[Impulse: ψ_a_short]
            A --> C[Impulse: ψ_a_long]
            B --> D[Fitness: α_a_short, β_a_short]
            C --> E[Fitness: α_a_long, β_a_long]

            F[Strength Intensity] --> G[Impulse: ψ_s_short]
            F --> H[Impulse: ψ_s_long]
            G --> I[Fitness: α_s_short, β_s_short]
            H --> J[Fitness: α_s_long, β_s_long]

            D --> K[Weight: γ_a_short]
            E --> L[Weight: γ_a_long]
            I --> M[Weight: γ_s_short]
            J --> N[Weight: γ_s_long]

            K --> O[Σ Weight Effects]
            L --> O
            M --> O
            N --> O

            P[GP Trend] --> O
            Q[Daily Cycle] --> O
            O --> R[Observed Weight]
            S[Observation Noise] --> R
        </div>

        <script src="https://cdn.jsdelivr.net/npm/mermaid@10.6.1/dist/mermaid.min.js"></script>
        <script>mermaid.initialize({startOnLoad: true});</script>
        """

        return diagram

    def create_step_by_step_predictions(self) -> dict:
        """Create step-by-step prediction explanations."""
        return {
            'single_workout_short': r"""
            \textbf{Single workout - short-term response:}
            \[
            \Delta W_{\text{a\_short}}[t] = \gamma_{\text{a\_short}} \beta_{\text{a\_short}} \psi_{\text{a\_short}}^{t} X_{\text{aerobic}}[0]
            \]
            \textbf{Half-life:} \( t_{1/2} = -\log(2)/\log(\psi_{\text{a\_short}}) \) days

            \textbf{Interpretation:} A single aerobic workout creates an impulse that decays exponentially.
            The fitness gain (\(\beta\)) converts this to fitness, which affects weight (\(\gamma\)).
            Short-term effects decay quickly (hours to days).
            """,

            'single_workout_long': r"""
            \textbf{Single workout - long-term response:}
            \[
            \Delta W_{\text{a\_long}}[t] = \gamma_{\text{a\_long}} \beta_{\text{a\_long}} (\alpha_{\text{a\_long}} \psi_{\text{a\_long}})^{t} X_{\text{aerobic}}[0]
            \]
            \textbf{Effective half-life:} \( t_{1/2} = -\log(2)/\log(\alpha_{\text{a\_long}} \psi_{\text{a\_long}}) \) days

            \textbf{Interpretation:} Long-term effects involve both impulse decay (\(\psi\)) and fitness decay (\(\alpha\)).
            This creates slower decay (weeks to months) representing physiological adaptations.
            """,

            'multiple_workouts': r"""
            \textbf{Multiple workout superposition:}
            \[
            \Delta W[t] = \sum_{s=0}^{t} \sum_{i} \gamma_i \beta_i \alpha_i^{t-s} \psi_i^{t-s} X_i[s]
            \]
            where \(i \in \{\text{a\_short}, \text{s\_short}, \text{a\_long}, \text{s\_long}\}\)

            \textbf{Interpretation:} Each workout contributes to current weight based on:
            1. Time since workout (\(t-s\))
            2. Activity type (aerobic vs strength)
            3. Time scale (short vs long term)
            4. Parameter values (\(\gamma, \beta, \alpha, \psi\))
            """,

            'data_influence': r"""
            \textbf{How data influences predictions:}
            Each weight observation provides information about:
            \begin{itemize}
            \item \textbf{Current fitness states:} \(F_i[t]\) for all 4 components
            \item \textbf{Recent activities:} \(X_i[t-k]\) for small \(k\) (via impulse states)
            \item \textbf{Long-term trends:} Through GP component \(\text{GP}(t)\)
            \item \textbf{Daily patterns:} Through Fourier component \(f_{\text{daily}}(t)\)
            \item \textbf{Parameter values:} All \(\gamma_i, \beta_i, \alpha_i, \psi_i\)
            \end{itemize}
            """,

            'uncertainty_propagation': r"""
            \textbf{Uncertainty propagation:}
            \[
            \text{Var}(\hat{W}[t]) = \sum_{i} \gamma_i^2 \text{Var}(F_i[t]) + \text{Var}(\text{GP}(t)) + \text{Var}(f_{\text{daily}}(t)) + \sigma_w^2
            \]
            \textbf{Components:}
            \begin{itemize}
            \item \textbf{Fitness uncertainty:} From parameter uncertainty in \(\alpha_i, \beta_i, \psi_i\)
            \item \textbf{GP uncertainty:} From \(\alpha_{\text{gp}}, \rho_{\text{gp}}\)
            \item \textbf{Daily uncertainty:} From Fourier coefficients
            \item \textbf{Observation noise:} \(\sigma_w^2\)
            \end{itemize}
            """
        }

    def create_parameter_sensitivity_analysis(self) -> str:
        """Create parameter sensitivity analysis section."""
        return """
        <h3>Parameter Sensitivity Analysis</h3>

        <div class="equation-box">
            <strong>1. Impulse decay (ψ) sensitivity:</strong>
            <ul>
                <li><strong>ψ ≈ 0:</strong> No persistence - workouts only affect current day</li>
                <li><strong>ψ ≈ 0.5:</strong> Moderate persistence - effects last ~1-2 days</li>
                <li><strong>ψ ≈ 0.9:</strong> High persistence - effects last weeks</li>
            </ul>
            <strong>Half-life formula:</strong> \( t_{1/2} = -\log(2)/\log(\psi) \) days
        </div>

        <div class="equation-box">
            <strong>2. Fitness decay (α) sensitivity:</strong>
            <ul>
                <li><strong>α ≈ 0:</strong> Fitness resets daily - no cumulative training effect</li>
                <li><strong>α ≈ 0.5:</strong> Moderate fitness retention</li>
                <li><strong>α ≈ 0.95:</strong> High fitness retention - training effects accumulate over months</li>
            </ul>
        </div>

        <div class="equation-box">
            <strong>3. Weight effect (γ) sensitivity:</strong>
            <ul>
                <li><strong>γ &lt; 0:</strong> Fitness reduces weight (fat loss, dehydration)</li>
                <strong>γ &gt; 0:</strong> Fitness increases weight (muscle gain, inflammation)</li>
                <li><strong>|γ| large:</strong> Strong weight response to fitness changes</li>
                <li><strong>|γ| small:</strong> Weak weight response to fitness changes</li>
            </ul>
        </div>

        <div class="equation-box">
            <strong>4. Interaction effects:</strong>
            <ul>
                <li><strong>Short-term strength (γ_s_short &gt; 0):</strong> Inflammation/water retention</li>
                <li><strong>Short-term aerobic (γ_a_short &lt; 0):</strong> Dehydration/water loss</li>
                <li><strong>Long-term strength (γ_s_long &gt; 0):</strong> Muscle gain</li>
                <li><strong>Long-term aerobic (γ_a_long &lt; 0):</strong> Fat loss</li>
            </ul>
        </div>
        """

    def create_html_report(self) -> str:
        """Create the complete HTML report."""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html = r'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Four-Fitness State-Space Model: Technical Walkthrough</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f8f9fa;
            color: #333;
            padding-top: 20px;
            padding-bottom: 40px;
        }
        .header {
            background: linear-gradient(135deg, #2c3e50, #3498db);
            color: white;
            padding: 3rem 0;
            margin-bottom: 2rem;
            border-radius: 0 0 20px 20px;
        }
        .section {
            background: white;
            padding: 2rem;
            margin-bottom: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .equation-box {
            background: #f8f9fa;
            border-left: 4px solid #3498db;
            padding: 1.5rem;
            margin: 1.5rem 0;
            border-radius: 5px;
            font-family: 'Cambria', 'Times New Roman', serif;
        }
        .code-block {
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
            <p class="lead">Technical Walkthrough: Mathematics, Code, and Predictions</p>
            <p class="mb-0">Generated: {current_time}</p>
        </div>

        <div class="notice">
            <strong>Technical Report Notice:</strong> This report provides a purely technical walkthrough of the four-fitness state-space model.
            It includes mathematical formulations, code implementations, and computational results without biological interpretations or conclusions.
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
                <li><a href="#process-interactions">Process Interactions</a></li>
                <li><a href="#step-by-step-predictions">Step-by-Step Predictions</a></li>
                <li><a href="#parameter-sensitivity">Parameter Sensitivity Analysis</a></li>
            </ol>
        </div>

        <!-- Mathematical Foundations -->
        <div class="section" id="mathematical-foundations">
            <h2>1. Mathematical Foundations</h2>

            <h3>1.1 State-Space Formulation</h3>
            <div class="equation-box">
                {self.equations['complete_system']}
            </div>

            <h3>1.2 Impulse Dynamics</h3>
            <div class="equation-box">
                {self.equations['impulse_dynamics']}
            </div>

            <h3>1.3 Fitness Dynamics</h3>
            <div class="equation-box">
                {self.equations['fitness_dynamics']}
            </div>

            <h3>1.4 Observation Equation</h3>
            <div class="equation-box">
                {self.equations['observation_equation']}
            </div>

            <h3>1.5 Gaussian Process Component</h3>
            <div class="equation-box">
                \[
                \text{GP}(t) \sim \mathcal{GP}(0, k(t, t'))
                \]
                \[
                k(t, t') = \alpha_{\text{gp}}^2 \exp\left(-\frac{(t - t')^2}{2\rho_{\text{gp}}^2}\right)
                \]
                where:
                <ul>
                <li>$\alpha_{\text{gp}}$: Marginal standard deviation</li>
                <li>$\rho_{\text{gp}}$: Length scale</li>
                </ul>
            </div>

            <h3>1.6 Daily Cycle (Fourier Basis)</h3>
            <div class="equation-box">
                \[
                f_{\text{daily}}(t) = \sum_{k=1}^{K} \left[a_k \sin(2\pi k \cdot \text{hour}/24) + b_k \cos(2\pi k \cdot \text{hour}/24)\right]
                \]
                where $K$ is the number of Fourier harmonics.
            </div>
        </div>

        <!-- Stan Implementation -->
        <div class="section" id="stan-implementation">
            <h2>2. Stan Implementation</h2>

            <h3>2.1 Data Block - Model Inputs</h3>
            <div class="code-block">
{textwrap.indent(self.stan_code.get('data_block', '// Data block not available'), '                ')}
            </div>
            <p><strong>Key inputs:</strong> Daily activity intensities, weight observations, time indices, hour of day, Fourier harmonics, sparse GP configuration.</p>

            <h3>2.2 Parameters Block - Unknown Quantities</h3>
            <div class="code-block">
{textwrap.indent(self.stan_code.get('parameters_block', '// Parameters block not available'), '                ')}
            </div>
            <p><strong>16 core parameters:</strong> 4 ψ (impulse decay), 4 α (fitness decay), 4 β (fitness gain), 4 γ (weight effects).</p>

            <h3>2.3 Transformed Parameters - State Computations</h3>
            <p><strong>Impulse state computation:</strong> Converts activity intensities to decaying impulses.</p>
            <div class="code-block">
{textwrap.indent(self.stan_code.get('impulse_computation', '// Impulse computation not available'), '                ')}
            </div>

            <p><strong>Fitness state computation:</strong> Converts impulses to fitness states.</p>
            <div class="code-block">
{textwrap.indent(self.stan_code.get('fitness_computation', '// Fitness computation not available'), '                ')}
            </div>

            <p><strong>GP computation:</strong> Gaussian Process for intrinsic variations.</p>
            <div class="code-block">
{textwrap.indent(self.stan_code.get('gp_computation', '// GP computation not available'), '                ')}
            </div>

            <p><strong>Daily spline computation:</strong> Fourier basis for circadian rhythms.</p>
            <div class="code-block">
{textwrap.indent(self.stan_code.get('daily_spline', '// Daily spline computation not available'), '                ')}
            </div>

            <h3>2.4 Model Block - Priors and Likelihood</h3>
            <div class="code-block">
{textwrap.indent(self.stan_code.get('model_block', '// Model block not available'), '                ')}
            </div>
            <p><strong>Prior design:</strong> Informative priors based on physiological knowledge (short-term vs long-term, aerobic vs strength).</p>

            <h3>2.5 Generated Quantities - Predictions and Diagnostics</h3>
            <div class="code-block">
{textwrap.indent(self.stan_code.get('generated_quantities', '// Generated quantities block not available'), '                ')}
            </div>
            <p><strong>Outputs:</strong> Posterior predictions, log-likelihood, replicated data for model checking.</p>
        </div>

        <!-- Python Data Pipeline -->
        <div class="section" id="python-pipeline">
            <h2>3. Python Data Pipeline</h2>

            <h3>3.1 Data Preparation Method</h3>
            <p><strong>Key steps:</strong> Load Garmin data, aggregate to daily time series, handle missing values, standardize variables.</p>
            <div class="code-block">
{textwrap.indent(self.python_code.get('data_preparation', '# Data preparation code not available'), '                ')}
            </div>

            <h3>3.2 Stan Data Preparation</h3>
            <p><strong>Key steps:</strong> Create time indices, prepare Fourier basis, set up sparse GP, standardize data.</p>
            <div class="code-block">
{textwrap.indent(self.python_code.get('stan_data_prep', '# Stan data preparation code not available'), '                ')}
            </div>

            <h3>3.3 Model Fitting with CmdStanPy</h3>
            <div class="code-block">
{textwrap.indent(self.python_code.get('model_fitting', '# Model fitting code not available'), '                ')}
            </div>

            <h3>3.4 Results Analysis</h3>
            <div class="code-block">
{textwrap.indent(self.python_code.get('results_analysis', '# Results analysis code not available'), '                ')}
            </div>
        </div>

        <!-- Parameter Estimates -->
        <div class="section" id="parameter-estimates">
            <h2>4. Parameter Estimates</h2>
            <p>Posterior means and 95% highest density intervals from MCMC sampling:</p>
            {self.create_parameter_table()}
        </div>

        <!-- Variance Decomposition -->
        <div class="section" id="variance-decomposition">
            <h2>5. Variance Decomposition</h2>
            <p>Proportion of weight variance explained by each model component:</p>
            <div id="variance-plot">
                {self.create_variance_decomposition_plot()}
            </div>
        </div>

        <!-- Process Interactions -->
        <div class="section" id="process-interactions">
            <h2>6. Process Interactions</h2>
            <p>Diagram showing how activity intensities propagate through the system:</p>
            {self.create_process_interaction_diagram()}

            <h3>6.1 Key Interaction Patterns</h3>
            <div class="equation-box">
                <strong>Impulse accumulation:</strong>
                \[
                I_i[t] = \sum_{k=0}^{t} \psi_i^k X_i[t-k]
                \]
                Shows how past intensities decay exponentially.
            </div>

            <div class="equation-box">
                <strong>Fitness accumulation:</strong>
                \[
                F_i[t] = \sum_{k=1}^{t} \alpha_i^{k-1} \beta_i I_i[t-k]
                \]
                Shows how impulses translate to fitness with additional decay.
            </div>

            <div class="equation-box">
                <strong>Weight prediction:</strong>
                \[
                \hat{W}[t] = \sum_{i} \gamma_i \sum_{k=1}^{t} \alpha_i^{k-1} \beta_i \sum_{j=0}^{t-k} \psi_i^j X_i[t-k-j]
                \]
                Complete expression showing how activity intensities affect weight through the cascade of processes.
            </div>
        </div>

        <!-- Step-by-Step Predictions -->
        <div class="section" id="step-by-step-predictions">
            <h2>7. Step-by-Step Predictions</h2>

            <h3>7.1 Single Workout Analysis</h3>
            <div class="equation-box">
                {self.predictions['single_workout_short']}
            </div>

            <div class="equation-box">
                {self.predictions['single_workout_long']}
            </div>

            <h3>7.2 Multiple Workout Interactions</h3>
            <div class="equation-box">
                {self.predictions['multiple_workouts']}
            </div>

            <h3>7.3 Data Influence on Predictions</h3>
            <div class="equation-box">
                {self.predictions['data_influence']}
            </div>

            <h3>7.4 Uncertainty Propagation</h3>
            <div class="equation-box">
                {self.predictions['uncertainty_propagation']}
            </div>
        </div>

        <!-- Parameter Sensitivity Analysis -->
        <div class="section" id="parameter-sensitivity">
            <h2>8. Parameter Sensitivity Analysis</h2>
            {self.sensitivity_analysis}
        </div>

        <!-- Footer -->
        <div class="section text-center text-muted">
            <hr>
            <p>Four-Fitness State-Space Model: Comprehensive Educational Walkthrough</p>
            <p>Generated on {current_time} | Model: weight_state_space_four_fitness.stan</p>
            <p>Sections: 1. Mathematical Foundations, 2. Stan Implementation, 3. Python Pipeline, 4. Parameter Estimates,<br>
            5. Variance Decomposition, 6. Process Interactions, 7. Step-by-Step Predictions, 8. Parameter Sensitivity</p>
        </div>
    </div>

    <script>
        // Initialize MathJax
        MathJax = {{
            tex: {{
                inlineMath: [['$', '$'], ['\\(', '\\)']],
                displayMath: [['$$', '$$'], ['\\[', '\\]']]
            }},
            svg: {{
                fontCache: 'global'
            }}
        }};
    </script>
</body>
</html>
"""

        return html

    def save_report(self) -> None:
        """Save HTML report to file."""
        print("Creating four-fitness educational report...")

        html_content = self.create_html_report()

        output_file = self.output_dir / "index.html"
        with open(output_file, 'w') as f:
            f.write(html_content)

        print(f"Report saved to: {output_file}")

        # Also save a plain text version of key equations
        equations_file = self.output_dir / "equations.txt"
        with open(equations_file, 'w') as f:
            f.write("FOUR-FITNESS STATE-SPACE MODEL EQUATIONS\n")
            f.write("=" * 50 + "\n\n")
            for key, eq in self.equations.items():
                f.write(f"{key.upper().replace('_', ' ')}:\n")
                f.write(eq + "\n\n")

        print(f"Equations saved to: {equations_file}")


def main():
    """Main function."""
    report = FourFitnessEducationalReport()
    report.save_report()

    print("\n" + "=" * 70)
    print("EDUCATIONAL REPORT CREATED SUCCESSFULLY")
    print("=" * 70)
    print("The report includes:")
    print("1. Mathematical foundations with LaTeX equations")
    print("2. Stan code implementation details")
    print("3. Python data processing pipeline")
    print("4. Parameter estimates with credible intervals")
    print("5. Variance decomposition visualization")
    print("6. Process interaction diagrams")
    print("7. Posterior prediction equations")
    print("\nOpen docs/four_fitness_educational/index.html in your browser")


if __name__ == "__main__":
    main()