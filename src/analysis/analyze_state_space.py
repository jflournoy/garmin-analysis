#!/usr/bin/env python3
"""Analyze state-space model for weight prediction using workout intensity.

This script runs the state-space model on real Garmin data and creates
comprehensive visualizations and summary reports.

Usage:
    python -m src.analysis.analyze_state_space --output-dir output/state_space_analysis
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import arviz as az

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.weight import load_weight_data
from src.data.intensity import load_intensity_data
from src.models.fit_weight import fit_state_space_model
from src.models.plot_cyclic import plot_state_space_expectations


class StateSpaceAnalyzer:
    """Analyze state-space model for weight prediction."""

    def __init__(
        self,
        output_dir: str = "output/state_space_analysis",
        data_dir: str = "data",
        activity_types: List[str] = None,
        chains: int = 4,
        iter_warmup: int = 500,
        iter_sampling: int = 500,
        adapt_delta: float = 0.95,
        max_treedepth: int = 12,
        use_sparse: bool = True,
        n_inducing_points: int = 50,
    ):
        """Initialize analyzer with configuration.

        Args:
            output_dir: Path to directory for output files.
            data_dir: Path to data directory containing Garmin export data.
            activity_types: List of activity types to include for intensity calculation.
                           If None, includes ['strength_training', 'walking', 'cycling'].
            chains: Number of MCMC chains.
            iter_warmup: Warmup iterations per chain.
            iter_sampling: Sampling iterations per chain.
            adapt_delta: Target acceptance probability for NUTS (default: 0.95).
            max_treedepth: Maximum tree depth for NUTS (default: 12).
            use_sparse: Whether to use sparse GP approximation.
            n_inducing_points: Number of inducing points for sparse GP.
        """
        self.output_dir = Path(output_dir)
        self.data_dir = Path(data_dir)

        if activity_types is None:
            self.activity_types = ['strength_training', 'walking', 'cycling']
        else:
            self.activity_types = activity_types

        self.chains = chains
        self.iter_warmup = iter_warmup
        self.iter_sampling = iter_sampling
        self.adapt_delta = adapt_delta
        self.max_treedepth = max_treedepth
        self.use_sparse = use_sparse
        self.n_inducing_points = n_inducing_points

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Store loaded data and results
        self.df_weight = None
        self.df_intensity = None
        self.fit = None
        self.idata = None
        self.stan_data = None
        self.results = {}

    def load_data(self) -> None:
        """Load weight and intensity data."""
        print("\n" + "=" * 70)
        print("LOADING DATA")
        print("=" * 70)

        print("Loading weight data...")
        self.df_weight = load_weight_data(self.data_dir)
        print(f"  Loaded {len(self.df_weight)} weight measurements")
        print(f"  Date range: {self.df_weight['timestamp'].min()} to {self.df_weight['timestamp'].max()}")

        print("Loading workout intensity data...")
        self.df_intensity = load_intensity_data(
            data_dir=self.data_dir,
            activity_types=self.activity_types,
            max_hr=185.0,
            intensity_col="intensity",
        )
        print(f"  Loaded {len(self.df_intensity)} days of intensity data")
        print(f"  Date range: {self.df_intensity['date'].min()} to {self.df_intensity['date'].max()}")
        print(f"  Non-zero intensity days: {(self.df_intensity['intensity'] > 0).sum()}")

        # Save data summary
        data_summary = {
            'n_weight_obs': len(self.df_weight),
            'n_intensity_days': len(self.df_intensity),
            'n_nonzero_intensity': int((self.df_intensity['intensity'] > 0).sum()),
            'weight_date_min': str(self.df_weight['timestamp'].min()),
            'weight_date_max': str(self.df_weight['timestamp'].max()),
            'intensity_date_min': str(self.df_intensity['date'].min()),
            'intensity_date_max': str(self.df_intensity['date'].max()),
            'mean_intensity': float(self.df_intensity['intensity'].mean()),
            'std_intensity': float(self.df_intensity['intensity'].std()),
            'mean_weight': float(self.df_weight['weight_lbs'].mean()),
            'std_weight': float(self.df_weight['weight_lbs'].std()),
        }

        with open(self.output_dir / "data_summary.json", 'w') as f:
            json.dump(data_summary, f, indent=2)

        print(f"\nData summary saved to: {self.output_dir / 'data_summary.json'}")

    def fit_model(self) -> None:
        """Fit state-space model to data."""
        print("\n" + "=" * 70)
        print("FITTING STATE-SPACE MODEL")
        print("=" * 70)

        print(f"Configuration:")
        print(f"  Chains: {self.chains}")
        print(f"  Warmup iterations: {self.iter_warmup}")
        print(f"  Sampling iterations: {self.iter_sampling}")
        print(f"  Use sparse GP: {self.use_sparse}")
        print(f"  Inducing points: {self.n_inducing_points}")
        print(f"  Activity types: {self.activity_types}")

        # Fit the model
        self.fit, self.idata, self.df_weight, self.df_intensity, self.stan_data = fit_state_space_model(
            data_dir=self.data_dir,
            df_weight=self.df_weight,
            df_intensity=self.df_intensity,
            chains=self.chains,
            iter_warmup=self.iter_warmup,
            iter_sampling=self.iter_sampling,
            adapt_delta=self.adapt_delta,
            max_treedepth=self.max_treedepth,
            activity_types=self.activity_types,
            use_sparse=self.use_sparse,
            n_inducing_points=self.n_inducing_points,
            cache=True,
            force_refit=True,
        )

        print("\nModel fitting completed!")
        print(f"  Number of samples: {self.fit.num_draws_sampling}")
        print(f"  Number of chains: {self.fit.chains}")

        # Save model results
        self._save_model_results()

    def _save_model_results(self) -> None:
        """Save model results to files."""
        print("\nSaving model results...")

        # Save summary statistics
        summary = az.summary(self.idata, var_names=['alpha', 'beta', 'gamma', 'sigma_w', 'alpha_gp', 'rho_gp'])
        summary.to_csv(self.output_dir / "parameter_summary.csv")
        print(f"  Parameter summary saved to: {self.output_dir / 'parameter_summary.csv'}")

        # Save fit diagnostics
        # CmdStanPy uses divergences (list per chain), max_treedepths (list per chain), step_size (list per chain)
        divergences = getattr(self.fit, 'divergences', None)
        max_treedepths = getattr(self.fit, 'max_treedepths', None)
        step_size = getattr(self.fit, 'step_size', None)

        # Convert numpy arrays to Python lists for JSON serialization
        def _to_serializable(obj):
            if obj is None:
                return None
            if hasattr(obj, 'tolist'):  # numpy array
                return obj.tolist()
            if isinstance(obj, (list, tuple)):
                return [_to_serializable(item) for item in obj]
            return obj

        divergences_serial = _to_serializable(divergences)
        max_treedepths_serial = _to_serializable(max_treedepths)
        step_size_serial = _to_serializable(step_size)

        # Compute totals if lists exist
        total_divergences = None
        if divergences_serial is not None:
            if isinstance(divergences_serial, (list, tuple)):
                total_divergences = sum(divergences_serial)
            else:
                total_divergences = divergences_serial

        max_treedepth_val = None
        if max_treedepths_serial is not None:
            if isinstance(max_treedepths_serial, (list, tuple)):
                max_treedepth_val = max(max_treedepths_serial) if max_treedepths_serial else None
            else:
                max_treedepth_val = max_treedepths_serial

        step_size_val = None
        if step_size_serial is not None:
            if isinstance(step_size_serial, (list, tuple)):
                step_size_val = step_size_serial[0] if step_size_serial else None  # Assume same across chains
            else:
                step_size_val = step_size_serial

        diagnostics = {
            'n_chains': self.fit.chains,
            'n_warmup': self.iter_warmup,
            'n_sampling': self.iter_sampling,
            'total_samples': self.fit.num_draws_sampling,
            'divergent_transitions': total_divergences,
            'max_treedepth': max_treedepth_val,
            'stepsize': step_size_val,
            'divergences_per_chain': divergences_serial,
            'max_treedepths_per_chain': max_treedepths_serial,
            'step_size_per_chain': step_size_serial,
        }

        with open(self.output_dir / "diagnostics.json", 'w') as f:
            json.dump(diagnostics, f, indent=2)

        print(f"  Diagnostics saved to: {self.output_dir / 'diagnostics.json'}")

    def analyze_parameters(self) -> None:
        """Analyze model parameters and extract key findings."""
        print("\n" + "=" * 70)
        print("ANALYZING MODEL PARAMETERS")
        print("=" * 70)

        # Extract posterior samples
        posterior = self.idata.posterior

        # Compute key statistics
        alpha_samples = posterior['alpha'].values.flatten()
        beta_samples = posterior['beta'].values.flatten()
        gamma_samples = posterior['gamma'].values.flatten()

        # Convert gamma to original units if available
        if 'gamma_original_units' in posterior:
            gamma_original_samples = posterior['gamma_original_units'].values.flatten()
            gamma_for_analysis = gamma_original_samples
            gamma_unit = "lbs per unit fitness"
        else:
            gamma_for_analysis = gamma_samples
            gamma_unit = "standardized units"

        # Compute statistics
        results = {
            'alpha': {
                'mean': float(np.mean(alpha_samples)),
                'std': float(np.std(alpha_samples)),
                '2.5%': float(np.percentile(alpha_samples, 2.5)),
                '50%': float(np.percentile(alpha_samples, 50)),
                '97.5%': float(np.percentile(alpha_samples, 97.5)),
                'interpretation': 'Fitness persistence (close to 1 = high persistence)',
            },
            'beta': {
                'mean': float(np.mean(beta_samples)),
                'std': float(np.std(beta_samples)),
                '2.5%': float(np.percentile(beta_samples, 2.5)),
                '50%': float(np.percentile(beta_samples, 50)),
                '97.5%': float(np.percentile(beta_samples, 97.5)),
                'interpretation': 'Fitness gain per unit workout intensity',
            },
            'gamma': {
                'mean': float(np.mean(gamma_for_analysis)),
                'std': float(np.std(gamma_for_analysis)),
                '2.5%': float(np.percentile(gamma_for_analysis, 2.5)),
                '50%': float(np.percentile(gamma_for_analysis, 50)),
                '97.5%': float(np.percentile(gamma_for_analysis, 97.5)),
                'unit': gamma_unit,
                'interpretation': 'Weight effect per unit fitness (negative = fitness reduces weight)',
            },
            'sigma_w': {
                'mean': float(np.mean(posterior['sigma_w'].values.flatten())),
                'std': float(np.std(posterior['sigma_w'].values.flatten())),
                'interpretation': 'Weight observation noise',
            },
        }

        # Determine if gamma is significantly negative
        gamma_negative_prob = np.mean(gamma_for_analysis < 0)
        results['gamma_negative_probability'] = float(gamma_negative_prob)

        if gamma_negative_prob > 0.95:
            results['gamma_interpretation'] = 'Strong evidence that fitness reduces weight'
        elif gamma_negative_prob > 0.9:
            results['gamma_interpretation'] = 'Moderate evidence that fitness reduces weight'
        elif gamma_negative_prob > 0.8:
            results['gamma_interpretation'] = 'Weak evidence that fitness reduces weight'
        else:
            results['gamma_interpretation'] = 'No clear evidence that fitness affects weight'

        # Determine if beta is significantly positive
        beta_positive_prob = np.mean(beta_samples > 0)
        results['beta_positive_probability'] = float(beta_positive_prob)

        if beta_positive_prob > 0.95:
            results['beta_interpretation'] = 'Strong evidence that workouts increase fitness'
        elif beta_positive_prob > 0.9:
            results['beta_interpretation'] = 'Moderate evidence that workouts increase fitness'
        elif beta_positive_prob > 0.8:
            results['beta_interpretation'] = 'Weak evidence that workouts increase fitness'
        else:
            results['beta_interpretation'] = 'No clear evidence that workouts affect fitness'

        # Save results
        self.results = results
        with open(self.output_dir / "analysis_results.json", 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Analysis results saved to: {self.output_dir / 'analysis_results.json'}")

        # Print summary
        print("\nKey Findings:")
        print(f"  1. Fitness persistence (α): {results['alpha']['mean']:.3f} ({results['alpha']['2.5%']:.3f} to {results['alpha']['97.5%']:.3f})")
        print(f"  2. Fitness gain per intensity (β): {results['beta']['mean']:.3f} ({results['beta']['2.5%']:.3f} to {results['beta']['97.5%']:.3f})")
        print(f"     {results['beta_interpretation']}")
        print(f"  3. Weight effect per fitness (γ): {results['gamma']['mean']:.3f} {gamma_unit}")
        print(f"     ({results['gamma']['2.5%']:.3f} to {results['gamma']['97.5%']:.3f})")
        print(f"     {results['gamma_interpretation']}")
        print(f"  4. Probability γ < 0: {gamma_negative_prob:.1%}")
        print(f"  5. Probability β > 0: {beta_positive_prob:.1%}")

    def create_visualizations(self) -> None:
        """Create visualizations of model results."""
        print("\n" + "=" * 70)
        print("CREATING VISUALIZATIONS")
        print("=" * 70)

        # Create visualizations directory
        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)

        # Set plot style
        plt.style.use('seaborn-v0_8-whitegrid')

        # 1. Trace plots for key parameters
        print("Creating trace plots...")
        param_names = ['alpha', 'beta', 'gamma', 'sigma_w', 'alpha_gp']
        valid_params = [p for p in param_names if p in self.idata.posterior]

        if valid_params:
            # Use ArviZ's built-in trace plot with compact layout
            fig = az.plot_trace(self.idata, var_names=valid_params, compact=True, figsize=(12, 3 * len(valid_params)))
            plt.tight_layout()
            plt.savefig(viz_dir / "trace_plots.png", dpi=150, bbox_inches='tight')
            plt.close()

        # 2. Posterior distributions of key parameters
        print("Creating posterior distribution plots...")
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        key_params = [('alpha', 'Fitness persistence (α)'),
                      ('beta', 'Fitness gain per intensity (β)'),
                      ('gamma', 'Weight effect per fitness (γ)'),
                      ('sigma_w', 'Weight observation noise (σ_w)')]

        for i, (param, title) in enumerate(key_params):
            if param in self.idata.posterior:
                ax = axes[i]
                samples = self.idata.posterior[param].values.flatten()

                # Plot histogram
                ax.hist(samples, bins=50, alpha=0.7, density=True, edgecolor='black')

                # Add vertical lines for credible interval
                ci_low = np.percentile(samples, 2.5)
                ci_high = np.percentile(samples, 97.5)
                mean_val = np.mean(samples)

                ax.axvline(ci_low, color='red', linestyle='--', alpha=0.7, label='95% CI')
                ax.axvline(ci_high, color='red', linestyle='--', alpha=0.7)
                ax.axvline(mean_val, color='green', linestyle='-', alpha=0.7, label='Mean')

                # Add kernel density estimate
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(samples)
                x_range = np.linspace(samples.min(), samples.max(), 1000)
                ax.plot(x_range, kde(x_range), 'b-', alpha=0.5)

                ax.set_xlabel('Parameter value')
                ax.set_ylabel('Density')
                ax.set_title(title)
                ax.legend()
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(viz_dir / "posterior_distributions.png", dpi=150, bbox_inches='tight')
        plt.close()

        # 3. Parameter relationships scatter plot
        print("Creating parameter relationship plots...")
        if 'alpha' in self.idata.posterior and 'beta' in self.idata.posterior and 'gamma' in self.idata.posterior:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Alpha vs Beta
            alpha_samples = self.idata.posterior['alpha'].values.flatten()
            beta_samples = self.idata.posterior['beta'].values.flatten()

            # Subsample for scatter plot
            n_samples = len(alpha_samples)
            subsample_idx = np.random.choice(n_samples, min(1000, n_samples), replace=False)

            axes[0].scatter(alpha_samples[subsample_idx], beta_samples[subsample_idx],
                           alpha=0.5, s=10)
            axes[0].set_xlabel('Fitness persistence (α)')
            axes[0].set_ylabel('Fitness gain per intensity (β)')
            axes[0].set_title('α vs β relationship')
            axes[0].grid(True, alpha=0.3)

            # Beta vs Gamma
            gamma_samples = self.idata.posterior['gamma'].values.flatten()
            axes[1].scatter(beta_samples[subsample_idx], gamma_samples[subsample_idx],
                           alpha=0.5, s=10)
            axes[1].set_xlabel('Fitness gain per intensity (β)')
            axes[1].set_ylabel('Weight effect per fitness (γ)')
            axes[1].set_title('β vs γ relationship')
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(viz_dir / "parameter_relationships.png", dpi=150, bbox_inches='tight')
            plt.close()

        # 4. Data visualization
        print("Creating data visualization...")
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Plot weight data
        ax1 = axes[0]
        ax1.plot(self.df_weight['timestamp'], self.df_weight['weight_lbs'], 'b-', alpha=0.7, label='Weight')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Weight (lbs)')
        ax1.set_title('Weight Measurements Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

        # Plot intensity data
        ax2 = axes[1]
        ax2.bar(self.df_intensity['date'], self.df_intensity['intensity'],
                width=1.0, alpha=0.7, color='orange', label='Workout Intensity')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Workout Intensity')
        ax2.set_title('Daily Workout Intensity')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        plt.savefig(viz_dir / "data_overview.png", dpi=150, bbox_inches='tight')
        plt.close()

        # 5. State-space model expectations
        print("Creating state-space expectations visualization...")
        fig = plot_state_space_expectations(
            idata=self.idata,
            df_weight=self.df_weight,
            df_intensity=self.df_intensity,
            stan_data=self.stan_data,
            model_name="State-Space Fitness Model",
            output_path=viz_dir / "state_space_expectations.png",
            show_ci=True,
        )
        plt.close(fig)

        print(f"Visualizations saved to: {viz_dir}/")

    def generate_html_report(self) -> None:
        """Generate HTML report with results and visualizations."""
        print("\n" + "=" * 70)
        print("GENERATING HTML REPORT")
        print("=" * 70)

        report_path = self.output_dir / "analysis_report.html"

        # Load analysis results if not already loaded
        if not self.results:
            results_path = self.output_dir / "analysis_results.json"
            if results_path.exists():
                with open(results_path, 'r') as f:
                    self.results = json.load(f)

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>State-Space Model Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; border-bottom: 2px solid #333; padding-bottom: 10px; }}
                h2 {{ color: #555; margin-top: 30px; }}
                h3 {{ color: #777; margin-top: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; font-weight: bold; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .summary {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                .finding {{ background-color: #f0f8f0; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                .image {{ max-width: 100%; height: auto; margin: 20px 0; border: 1px solid #ddd; padding: 10px; }}
                .param-table {{ width: 80%; margin: 20px auto; }}
            </style>
        </head>
        <body>
            <h1>State-Space Model Analysis Report</h1>
            <p>Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Output directory: {self.output_dir}</p>

            <div class="summary">
                <h2>Analysis Summary</h2>
                <p>This report presents results from a state-space model that tracks how workout intensity affects
                a latent fitness state, which in turn affects weight measurements.</p>
                <p><strong>Model structure:</strong></p>
                <ul>
                    <li><strong>Fitness evolution:</strong> fitness[t] = α·fitness[t-1] + β·intensity[t-1] + ε_f[t]</li>
                    <li><strong>Weight observation:</strong> weight[t] = baseline + γ·fitness[day(t)] + GP(t) + ε_w[t]</li>
                </ul>
                <p><strong>Data:</strong> {len(self.df_weight)} weight measurements, {len(self.df_intensity)} days of intensity data</p>
                <p><strong>Model settings:</strong> {self.chains} chains, {self.iter_warmup} warmup + {self.iter_sampling} sampling iterations</p>
            </div>
        """

        # Add key findings if available
        if self.results:
            html_content += """
            <div class="finding">
                <h2>Key Findings</h2>
                <table class="param-table">
                    <tr>
                        <th>Parameter</th>
                        <th>Mean (95% CI)</th>
                        <th>Interpretation</th>
                    </tr>
            """

            params_to_show = ['alpha', 'beta', 'gamma']
            for param in params_to_show:
                if param in self.results:
                    result = self.results[param]
                    ci_text = f"{result['mean']:.3f} ({result['2.5%']:.3f} to {result['97.5%']:.3f})"
                    html_content += f"""
                    <tr>
                        <td><strong>{param}</strong></td>
                        <td>{ci_text}</td>
                        <td>{result['interpretation']}</td>
                    </tr>
                    """

            # Add probability summaries
            if 'gamma_negative_probability' in self.results:
                gamma_prob = self.results['gamma_negative_probability']
                beta_prob = self.results['beta_positive_probability']

                html_content += f"""
                <tr>
                    <td><strong>Probability γ &lt; 0</strong></td>
                    <td>{gamma_prob:.1%}</td>
                    <td>{self.results.get('gamma_interpretation', '')}</td>
                </tr>
                <tr>
                    <td><strong>Probability β &gt; 0</strong></td>
                    <td>{beta_prob:.1%}</td>
                    <td>{self.results.get('beta_interpretation', '')}</td>
                </tr>
                """

            html_content += """
                </table>
            </div>
            """

        # Add model diagnostics if available
        diagnostics_path = self.output_dir / "diagnostics.json"
        if diagnostics_path.exists():
            try:
                with open(diagnostics_path, 'r') as f:
                    diagnostics = json.load(f)

                html_content += """
            <div class="summary">
                <h2>Model Diagnostics</h2>
                <table class="param-table">
                    <tr>
                        <th>Diagnostic</th>
                        <th>Value</th>
                        <th>Interpretation</th>
                    </tr>
                """

                # Add diagnostic rows
                diag_rows = [
                    ('divergent_transitions', 'Divergent transitions', 'Number of sampler transitions that diverged. Should be 0.'),
                    ('max_treedepth', 'Max treedepth', 'Maximum tree depth encountered. Should be less than max_treedepth setting.'),
                    ('stepsize', 'Step size', 'Step size used by Hamiltonian Monte Carlo.'),
                    ('n_chains', 'Chains', 'Number of MCMC chains.'),
                    ('n_warmup', 'Warmup iterations', 'Warmup iterations per chain.'),
                    ('n_sampling', 'Sampling iterations', 'Sampling iterations per chain.'),
                ]

                for key, label, interpretation in diag_rows:
                    if key in diagnostics:
                        value = diagnostics[key]
                        if isinstance(value, (int, float)):
                            display_value = f"{value:g}"
                        else:
                            display_value = str(value)

                        html_content += f"""
                    <tr>
                        <td><strong>{label}</strong></td>
                        <td>{display_value}</td>
                        <td>{interpretation}</td>
                    </tr>
                        """

                html_content += """
                </table>
            </div>
                """
            except Exception as e:
                html_content += f"""
            <div class="summary">
                <p>Error loading diagnostics: {e}</p>
            </div>
                """

        # Add visualizations
        viz_dir = self.output_dir / "visualizations"
        if viz_dir.exists():
            html_content += "<h2>Visualizations</h2>"

            viz_files = [
                ("data_overview.png", "Data Overview: Weight measurements and workout intensity over time"),
                ("trace_plots.png", "Trace Plots: MCMC convergence diagnostics for key parameters"),
                ("posterior_distributions.png", "Posterior Distributions: Parameter estimates with credible intervals"),
                ("parameter_relationships.png", "Parameter Relationships: Scatter plots showing correlations between parameters"),
                ("state_space_expectations.png", "State-Space Expectations: Fitness and weight predictions with data overlays"),
            ]

            for img_file, description in viz_files:
                img_path = viz_dir / img_file
                if img_path.exists():
                    html_content += f"""
                    <h3>{description}</h3>
                    <img class="image" src="{img_path.relative_to(self.output_dir)}" alt="{description}">
                    """

        # Add interpretation guide
        html_content += """
            <h2>Interpretation Guide</h2>
            <div class="summary">
                <ul>
                    <li><strong>α (fitness persistence)</strong>: Close to 1 means fitness state persists strongly day-to-day.
                        Lower values mean fitness decays quickly without workouts.</li>
                    <li><strong>β (fitness gain per intensity)</strong>: Positive values mean workouts increase fitness state.
                        Larger values mean more fitness gain per unit workout intensity.</li>
                    <li><strong>γ (weight effect per fitness)</strong>: Negative values mean higher fitness reduces weight
                        (likely through increased metabolism or muscle gain replacing fat). Positive values would mean
                        fitness increases weight (muscle mass gain).</li>
                    <li><strong>σ_f (fitness process noise)</strong>: Variability in fitness state not explained by workouts.</li>
                    <li><strong>σ_w (weight observation noise)</strong>: Measurement error and short-term weight fluctuations.</li>
                </ul>
                <p><strong>Key insight:</strong> This model separates the cumulative effect of workouts (through fitness state)
                from immediate effects. A negative γ with positive β suggests workouts build fitness, which in turn helps
                reduce weight over time.</p>
            </div>

            <footer>
                <p>Report generated by analyze_state_space.py</p>
                <p>Model: State-space fitness model (stan/weight_state_space.stan)</p>
            </footer>
        </body>
        </html>
        """

        with open(report_path, 'w') as f:
            f.write(html_content)

        print(f"Generated HTML report: {report_path}")

    def run_analysis(self) -> None:
        """Run complete analysis pipeline."""
        print("=" * 70)
        print("STATE-SPACE MODEL ANALYSIS")
        print("=" * 70)
        print(f"Output directory: {self.output_dir}")
        print("=" * 70)

        # Step 1: Load data
        self.load_data()

        # Step 2: Fit model
        self.fit_model()

        # Step 3: Analyze parameters
        self.analyze_parameters()

        # Step 4: Create visualizations
        self.create_visualizations()

        # Step 5: Generate HTML report
        self.generate_html_report()

        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)
        print(f"Results saved to: {self.output_dir}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze state-space model for weight prediction using workout intensity",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/state_space_analysis",
        help="Directory for output files"
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing Garmin export data"
    )

    parser.add_argument(
        "--activity-types",
        type=str,
        default="strength_training,walking,cycling",
        help="Comma-separated list of activity types to include"
    )

    parser.add_argument(
        "--chains",
        type=int,
        default=4,
        help="Number of MCMC chains"
    )

    parser.add_argument(
        "--iter-warmup",
        type=int,
        default=500,
        help="Warmup iterations per chain"
    )

    parser.add_argument(
        "--iter-sampling",
        type=int,
        default=500,
        help="Sampling iterations per chain"
    )

    parser.add_argument(
        "--adapt-delta",
        type=float,
        default=0.95,
        help="Target acceptance probability for NUTS"
    )

    parser.add_argument(
        "--max-treedepth",
        type=int,
        default=12,
        help="Maximum tree depth for NUTS"
    )

    parser.add_argument(
        "--no-sparse",
        action="store_true",
        help="Disable sparse GP approximation"
    )

    parser.add_argument(
        "--n-inducing-points",
        type=int,
        default=50,
        help="Number of inducing points for sparse GP"
    )

    args = parser.parse_args()

    # Parse activity types
    activity_types = [at.strip() for at in args.activity_types.split(',')]

    # Create and run analyzer
    analyzer = StateSpaceAnalyzer(
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        activity_types=activity_types,
        chains=args.chains,
        iter_warmup=args.iter_warmup,
        iter_sampling=args.iter_sampling,
        use_sparse=not args.no_sparse,
        n_inducing_points=args.n_inducing_points,
    )

    analyzer.run_analysis()


if __name__ == "__main__":
    main()