#!/usr/bin/env python3
"""Analyze impulse-response state-space model for weight prediction using workout intensity.

This script runs the impulse-response state-space model on real Garmin data and creates
comprehensive visualizations and summary reports.

Usage:
    python -m src.analysis.analyze_state_space_impulse --output-dir output/state_space_impulse_analysis
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
from src.models.fit_weight import fit_state_space_model_impulse
from src.models.plot_cyclic import plot_state_space_expectations


class ImpulseStateSpaceAnalyzer:
    """Analyze impulse-response state-space model for weight prediction."""

    def __init__(
        self,
        output_dir: str = "output/state_space_impulse_analysis",
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
        """Fit impulse-response state-space model to data."""
        print("\n" + "=" * 70)
        print("FITTING IMPULSE-RESPONSE STATE-SPACE MODEL")
        print("=" * 70)

        print(f"Configuration:")
        print(f"  Chains: {self.chains}")
        print(f"  Warmup iterations: {self.iter_warmup}")
        print(f"  Sampling iterations: {self.iter_sampling}")
        print(f"  Use sparse GP: {self.use_sparse}")
        print(f"  Inducing points: {self.n_inducing_points}")
        print(f"  Activity types: {self.activity_types}")

        # Fit the impulse-response model
        self.fit, self.idata, self.df_weight, self.df_intensity, self.stan_data = fit_state_space_model_impulse(
            data_dir=self.data_dir,
            df_weight=self.df_weight,
            df_intensity=self.df_intensity,
            output_dir=self.output_dir,
            chains=self.chains,
            iter_warmup=self.iter_warmup,
            iter_sampling=self.iter_sampling,
            adapt_delta=self.adapt_delta,
            max_treedepth=self.max_treedepth,
            use_sparse=self.use_sparse,
            n_inducing_points=self.n_inducing_points,
            cache=False,  # Disable cache for fresh runs
        )

        print(f"\nModel fitting complete!")
        print(f"  Parameters: {list(self.idata.posterior.data_vars)}")
        print(f"  Number of samples: {len(self.idata.posterior.draw) * len(self.idata.posterior.chain)}")

        # Check for divergent transitions
        if hasattr(self.fit, 'diagnose'):
            print("\nChecking diagnostics...")
            print(self.fit.diagnose())

    def analyze_parameters(self) -> None:
        """Analyze model parameters and save results."""
        print("\n" + "=" * 70)
        print("ANALYZING PARAMETERS")
        print("=" * 70)

        # Extract key parameters
        posterior = self.idata.posterior

        # For impulse-response model: alpha, psi, beta, gamma
        params_to_analyze = ['alpha', 'psi', 'beta', 'gamma', 'sigma_w', 'alpha_gp', 'rho_gp']
        available_params = [p for p in params_to_analyze if p in posterior.data_vars]

        self.results = {}
        for param in available_params:
            samples = posterior[param].values.flatten()
            self.results[param] = {
                'mean': float(np.mean(samples)),
                'std': float(np.std(samples)),
                '2.5%': float(np.percentile(samples, 2.5)),
                '50%': float(np.percentile(samples, 50)),
                '97.5%': float(np.percentile(samples, 97.5)),
            }

            # Add interpretations
            if param == 'alpha':
                self.results[param]['interpretation'] = "Fitness persistence (close to 1 = high persistence)"
            elif param == 'psi':
                self.results[param]['interpretation'] = "Impulse decay rate (close to 1 = slow decay)"
            elif param == 'beta':
                self.results[param]['interpretation'] = "Fitness gain per unit workout intensity"
            elif param == 'gamma':
                self.results[param]['interpretation'] = "Weight effect per unit fitness (negative = fitness reduces weight)"
                self.results[param]['unit'] = "standardized units"
            elif param == 'sigma_w':
                self.results[param]['interpretation'] = "Weight observation noise"
            elif param == 'alpha_gp':
                self.results[param]['interpretation'] = "GP marginal standard deviation"
            elif param == 'rho_gp':
                self.results[param]['interpretation'] = "GP length scale"

        # Calculate probabilities of interest
        if 'gamma' in posterior.data_vars:
            gamma_samples = posterior['gamma'].values.flatten()
            self.results['gamma_negative_probability'] = float(np.mean(gamma_samples < 0))
            if self.results['gamma_negative_probability'] > 0.95:
                self.results['gamma_interpretation'] = "Strong evidence that fitness reduces weight"
            elif self.results['gamma_negative_probability'] > 0.8:
                self.results['gamma_interpretation'] = "Moderate evidence that fitness reduces weight"
            else:
                self.results['gamma_interpretation'] = "Weak or no evidence that fitness reduces weight"

        if 'beta' in posterior.data_vars:
            beta_samples = posterior['beta'].values.flatten()
            self.results['beta_positive_probability'] = float(np.mean(beta_samples > 0))
            if self.results['beta_positive_probability'] > 0.95:
                self.results['beta_interpretation'] = "Strong evidence that workouts increase fitness"
            elif self.results['beta_positive_probability'] > 0.8:
                self.results['beta_interpretation'] = "Moderate evidence that workouts increase fitness"
            else:
                self.results['beta_interpretation'] = "Weak or no evidence that workouts increase fitness"

        if 'psi' in posterior.data_vars:
            psi_samples = posterior['psi'].values.flatten()
            self.results['psi_high_probability'] = float(np.mean(psi_samples > 0.5))
            self.results['psi_interpretation'] = f"Impulse persistence: {self.results['psi_high_probability']:.1%} probability > 0.5"

        # Save results
        with open(self.output_dir / "analysis_results.json", 'w') as f:
            json.dump(self.results, f, indent=2)

        # Save parameter summary as CSV
        summary_rows = []
        for param, stats_dict in self.results.items():
            # Check if this is a parameter dictionary (has 'mean' key)
            if isinstance(stats_dict, dict) and 'mean' in stats_dict:
                summary_rows.append({
                    'parameter': param,
                    'mean': stats_dict['mean'],
                    'std': stats_dict['std'],
                    '2.5%': stats_dict['2.5%'],
                    '50%': stats_dict['50%'],
                    '97.5%': stats_dict['97.5%'],
                    'interpretation': stats_dict.get('interpretation', ''),
                })

        if summary_rows:
            df_summary = pd.DataFrame(summary_rows)
            df_summary.to_csv(self.output_dir / "parameter_summary.csv", index=False)

        print(f"Parameter analysis saved to:")
        print(f"  {self.output_dir / 'analysis_results.json'}")
        print(f"  {self.output_dir / 'parameter_summary.csv'}")

        # Print key results
        print("\nKEY RESULTS:")
        if 'alpha' in self.results:
            alpha = self.results['alpha']
            print(f"  α (fitness persistence): {alpha['mean']:.3f} [{alpha['2.5%']:.3f}, {alpha['97.5%']:.3f}]")

        if 'psi' in self.results:
            psi = self.results['psi']
            print(f"  ψ (impulse decay): {psi['mean']:.3f} [{psi['2.5%']:.3f}, {psi['97.5%']:.3f}]")

        if 'beta' in self.results:
            beta = self.results['beta']
            print(f"  β (fitness gain per intensity): {beta['mean']:.3f} [{beta['2.5%']:.3f}, {beta['97.5%']:.3f}]")
            if 'beta_positive_probability' in self.results:
                print(f"    Probability β > 0: {self.results['beta_positive_probability']:.1%}")

        if 'gamma' in self.results:
            gamma = self.results['gamma']
            print(f"  γ (weight effect per fitness): {gamma['mean']:.3f} [{gamma['2.5%']:.3f}, {gamma['97.5%']:.3f}]")
            if 'gamma_negative_probability' in self.results:
                print(f"    Probability γ < 0: {self.results['gamma_negative_probability']:.1%}")

    def create_visualizations(self) -> None:
        """Create visualizations of model results."""
        print("\n" + "=" * 70)
        print("CREATING VISUALIZATIONS")
        print("=" * 70)

        # Create visualizations directory
        vis_dir = self.output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)

        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")

        # 1. Data overview
        print("Creating data overview plot...")
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Weight data
        ax = axes[0]
        ax.scatter(self.df_weight['timestamp'], self.df_weight['weight_lbs'],
                  alpha=0.6, s=20, label='Weight measurements')
        ax.set_ylabel('Weight (lbs)')
        ax.set_title('Weight Measurements Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Intensity data
        ax = axes[1]
        ax.bar(self.df_intensity['date'], self.df_intensity['intensity'],
               width=0.8, alpha=0.7, label='Workout intensity')
        ax.set_xlabel('Date')
        ax.set_ylabel('Intensity (standardized)')
        ax.set_title('Daily Workout Intensity')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(vis_dir / "data_overview.png", dpi=150, bbox_inches='tight')
        plt.close()

        # 2. Posterior distributions of key parameters
        print("Creating posterior distributions plot...")
        key_params = ['alpha', 'psi', 'beta', 'gamma', 'sigma_w']
        available_params = [p for p in key_params if p in self.idata.posterior.data_vars]

        n_params = len(available_params)
        if n_params > 0:
            # Determine grid layout: 1 row if <= 3 params, 2 rows otherwise
            if n_params <= 3:
                n_rows, n_cols = 1, n_params
            else:
                n_rows = 2
                n_cols = (n_params + 1) // 2  # ceil division

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 5*n_rows))

            # Flatten axes array for easy iteration
            if n_rows == 1 and n_cols == 1:
                axes = np.array([axes])
            elif n_rows == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()

            for i, param in enumerate(available_params):
                ax = axes[i]
                samples = self.idata.posterior[param].values.flatten()

                # Histogram
                ax.hist(samples, bins=50, density=True, alpha=0.7,
                       color='steelblue', edgecolor='black')

                # Add mean and credible interval
                mean_val = np.mean(samples)
                ci_low = np.percentile(samples, 2.5)
                ci_high = np.percentile(samples, 97.5)

                ax.axvline(mean_val, color='red', linestyle='-', linewidth=2, label=f'Mean: {mean_val:.3f}')
                ax.axvline(ci_low, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
                ax.axvline(ci_high, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

                ax.set_xlabel(param)
                ax.set_ylabel('Density')
                ax.set_title(f'{param} Posterior Distribution')
                ax.legend()
                ax.grid(True, alpha=0.3)

            # Hide any unused subplots
            for i in range(n_params, len(axes)):
                axes[i].axis('off')

            plt.tight_layout()
            plt.savefig(vis_dir / "posterior_distributions.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Created posterior distributions plot with {n_params} parameters")

        # 3. Parameter relationships (scatter plots)
        print("Creating parameter relationships plot...")

        # Determine which parameters are available
        available_params = []
        for param in ['alpha', 'psi', 'beta', 'gamma']:
            if param in self.idata.posterior.data_vars:
                available_params.append(param)

        if len(available_params) >= 2:
            # Create subplots for interesting relationships
            relationships = []

            if 'alpha' in available_params and 'beta' in available_params:
                relationships.append(('alpha', 'beta', 'α (fitness persistence)', 'β (fitness gain per impulse)', 'α vs β'))
            if 'alpha' in available_params and 'psi' in available_params:
                relationships.append(('alpha', 'psi', 'α (fitness persistence)', 'ψ (impulse decay)', 'α vs ψ'))
            if 'beta' in available_params and 'gamma' in available_params:
                relationships.append(('beta', 'gamma', 'β (fitness gain per impulse)', 'γ (weight effect per fitness)', 'β vs γ'))
            if 'psi' in available_params and 'gamma' in available_params:
                relationships.append(('psi', 'gamma', 'ψ (impulse decay)', 'γ (weight effect per fitness)', 'ψ vs γ'))

            if relationships:
                n_plots = len(relationships)
                # Arrange in grid: up to 4 plots in 2x2
                if n_plots <= 2:
                    n_rows, n_cols = 1, n_plots
                    figsize = (6 * n_cols, 5)
                else:
                    n_rows, n_cols = 2, 2
                    figsize = (12, 10)

                fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
                if n_rows == 1 and n_cols == 1:
                    axes = np.array([axes])
                elif n_rows == 1:
                    axes = axes.flatten()
                else:
                    axes = axes.flatten()

                for i, (x_param, y_param, x_label, y_label, title) in enumerate(relationships):
                    if i >= len(axes):
                        break
                    ax = axes[i]
                    x_samples = self.idata.posterior[x_param].values.flatten()[:1000]
                    y_samples = self.idata.posterior[y_param].values.flatten()[:1000]
                    ax.scatter(x_samples, y_samples, alpha=0.3, s=10)
                    ax.set_xlabel(x_label)
                    ax.set_ylabel(y_label)
                    ax.set_title(title)
                    ax.grid(True, alpha=0.3)

                # Hide any unused subplots
                for i in range(len(relationships), len(axes)):
                    axes[i].axis('off')

                plt.tight_layout()
                plt.savefig(vis_dir / "parameter_relationships.png", dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  Created parameter relationships plot with {len(relationships)} relationships")
            else:
                print("  No parameter relationships available for plotting")
        else:
            print("  Insufficient parameters available for relationship plots")

        # 4. State-space expectations plot
        print("Creating state-space expectations plot...")
        try:
            # Use the existing plot_state_space_expectations function
            fig = plot_state_space_expectations(
                idata=self.idata,
                df_weight=self.df_weight,
                df_intensity=self.df_intensity,
                stan_data=self.stan_data,
                model_name="Impulse-Response State-Space Model",
                output_path=vis_dir / "state_space_expectations.png",
                show_ci=True,
            )
            if fig is not None:
                # Figure already saved by function with output_path
                plt.close(fig)
                print("  State-space expectations plot created successfully")
            else:
                print("  Warning: plot_state_space_expectations returned None")
        except Exception as e:
            print(f"  Warning: Could not create state-space expectations plot: {e}")

        # 5. Trace plots (for diagnostics)
        print("Creating trace plots...")
        key_params_trace = ['alpha', 'psi', 'beta', 'gamma', 'sigma_w']
        available_trace = [p for p in key_params_trace if p in self.idata.posterior.data_vars]

        if available_trace:
            try:
                fig, axes = plt.subplots(len(available_trace), 2, figsize=(12, 3*len(available_trace)))
                if len(available_trace) == 1:
                    axes = axes.reshape(1, 2)

                for i, param in enumerate(available_trace):
                    # Trace plot
                    ax = axes[i, 0]
                    for chain in range(self.chains):
                        chain_data = self.idata.posterior[param].isel(chain=chain).values
                        ax.plot(chain_data, alpha=0.7, label=f'Chain {chain+1}')
                    ax.set_xlabel('Iteration')
                    ax.set_ylabel(param)
                    ax.set_title(f'{param} Trace Plot')
                    ax.legend(fontsize='small')
                    ax.grid(True, alpha=0.3)

                    # Density plot
                    ax = axes[i, 1]
                    for chain in range(self.chains):
                        chain_data = self.idata.posterior[param].isel(chain=chain).values
                        sns.kdeplot(chain_data, ax=ax, label=f'Chain {chain+1}')
                    ax.set_xlabel(param)
                    ax.set_ylabel('Density')
                    ax.set_title(f'{param} Density by Chain')
                    ax.legend(fontsize='small')
                    ax.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(vis_dir / "trace_plots.png", dpi=150, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"  Warning: Could not create trace plots: {e}")

        print(f"\nVisualizations saved to: {vis_dir}")

    def generate_html_report(self) -> None:
        """Generate HTML report with results and visualizations."""
        print("\n" + "=" * 70)
        print("GENERATING HTML REPORT")
        print("=" * 70)

        # Read analysis results
        with open(self.output_dir / "analysis_results.json", 'r') as f:
            results = json.load(f)

        # Create HTML report
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Impulse-Response State-Space Model Analysis</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f7fa;
        }}
        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        h1, h2, h3 {{
            color: #2c3e50;
            margin-top: 1.5em;
        }}
        h1 {{
            color: white;
            margin: 0;
            font-size: 2.5rem;
        }}
        .subtitle {{
            font-size: 1.2rem;
            opacity: 0.9;
            margin-top: 0.5rem;
        }}
        .section {{
            background: white;
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: 600;
        }}
        tr:hover {{
            background-color: #f8f9fa;
        }}
        .visualization {{
            margin: 2rem 0;
            text-align: center;
        }}
        .visualization img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        .caption {{
            font-style: italic;
            color: #666;
            margin-top: 0.5rem;
            text-align: center;
        }}
        .key-result {{
            background: #e8f4fd;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            border-left: 4px solid #3498db;
        }}
        .highlight {{
            color: #e74c3c;
            font-weight: bold;
        }}
        .date {{
            color: #95a5a6;
            font-size: 0.9rem;
            margin-top: 0.5rem;
        }}
        .nav-links {{
            display: flex;
            gap: 1rem;
            margin-bottom: 2rem;
            flex-wrap: wrap;
        }}
        .nav-links a {{
            background: #3498db;
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            text-decoration: none;
            transition: background 0.3s;
        }}
        .nav-links a:hover {{
            background: #2980b9;
        }}
    </style>
</head>
<body>
    <header>
        <h1>Impulse-Response State-Space Model Analysis</h1>
        <div class="subtitle">Bayesian Analysis of Workout Intensity Effects on Weight Through Latent Fitness State</div>
        <div class="date">Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    </header>

    <div class="nav-links">
        <a href="#overview">Overview</a>
        <a href="#results">Key Results</a>
        <a href="#visualizations">Visualizations</a>
        <a href="#interpretation">Interpretation</a>
    </div>

    <section id="overview" class="section">
        <h2>Analysis Overview</h2>
        <p>This report presents results from an impulse-response state-space model analyzing the relationship between workout intensity and weight through a latent fitness state.</p>

        <h3>Model Structure</h3>
        <p>The impulse-response state-space model extends the basic state-space model by adding an impulse accumulation process:</p>
        <ul>
            <li><strong>Impulse state:</strong> impulse[t] = ψ·impulse[t-1] + intensity[t]</li>
            <li><strong>Fitness state:</strong> fitness[t] = α·fitness[t-1] + β·impulse[t-1] + ε_f[t]</li>
            <li><strong>Weight observation:</strong> weight[t] = baseline + γ·fitness[day(t)] + GP(t) + ε_w[t]</li>
        </ul>

        <h3>Data Summary</h3>
"""

        # Add data summary if available
        data_summary_path = self.output_dir / "data_summary.json"
        if data_summary_path.exists():
            with open(data_summary_path, 'r') as f:
                data_summary = json.load(f)

            html += f"""
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Weight measurements</td>
                <td>{data_summary['n_weight_obs']:,}</td>
            </tr>
            <tr>
                <td>Days with intensity data</td>
                <td>{data_summary['n_intensity_days']:,}</td>
            </tr>
            <tr>
                <td>Days with non-zero intensity</td>
                <td>{data_summary['n_nonzero_intensity']:,}</td>
            </tr>
            <tr>
                <td>Mean intensity</td>
                <td>{data_summary['mean_intensity']:.3f}</td>
            </tr>
            <tr>
                <td>Mean weight</td>
                <td>{data_summary['mean_weight']:.1f} lbs</td>
            </tr>
        </table>
"""

        html += """
    </section>

    <section id="results" class="section">
        <h2>Key Results</h2>
"""

        # Add parameter results
        if results:
            html += """
        <h3>Parameter Estimates</h3>
        <table>
            <tr>
                <th>Parameter</th>
                <th>Mean</th>
                <th>95% Credible Interval</th>
                <th>Interpretation</th>
            </tr>
"""

            param_order = ['alpha', 'psi', 'beta', 'gamma', 'sigma_w', 'alpha_gp', 'rho_gp']
            for param in param_order:
                if param in results and 'mean' in results[param]:
                    ci = f"[{results[param]['2.5%']:.3f}, {results[param]['97.5%']:.3f}]"
                    interpretation = results[param].get('interpretation', '')
                    html += f"""
            <tr>
                <td>{param}</td>
                <td>{results[param]['mean']:.3f}</td>
                <td>{ci}</td>
                <td>{interpretation}</td>
            </tr>
"""

            html += """
        </table>
"""

            # Add probability results
            html += """
        <div class="key-result">
            <h4>Key Probabilities</h4>
"""

            if 'beta_positive_probability' in results:
                prob = results['beta_positive_probability']
                interpretation = results.get('beta_interpretation', '')
                html += f"""
            <p><strong>Probability β > 0 (workouts increase fitness):</strong> {prob:.1%} - {interpretation}</p>
"""

            if 'gamma_negative_probability' in results:
                prob = results['gamma_negative_probability']
                interpretation = results.get('gamma_interpretation', '')
                html += f"""
            <p><strong>Probability γ < 0 (fitness reduces weight):</strong> {prob:.1%} - {interpretation}</p>
"""

            if 'psi_high_probability' in results:
                prob = results['psi_high_probability']
                interpretation = results.get('psi_interpretation', '')
                html += f"""
            <p><strong>Probability ψ > 0.5 (slow impulse decay):</strong> {prob:.1%} - {interpretation}</p>
"""

            html += """
        </div>
"""

        html += """
    </section>

    <section id="visualizations" class="section">
        <h2>Visualizations</h2>

        <div class="visualization">
            <h3>Data Overview</h3>
            <img src="visualizations/data_overview.png" alt="Data overview">
            <div class="caption">Top: Weight measurements over time. Bottom: Daily workout intensity.</div>
        </div>

        <div class="visualization">
            <h3>Posterior Distributions</h3>
            <img src="visualizations/posterior_distributions.png" alt="Posterior distributions">
            <div class="caption">Posterior distributions for key parameters. Dashed red lines show 95% credible intervals.</div>
        </div>

        <div class="visualization">
            <h3>Parameter Relationships</h3>
            <img src="visualizations/parameter_relationships.png" alt="Parameter relationships">
            <div class="caption">Scatter plots showing relationships between parameters.</div>
        </div>

        <div class="visualization">
            <h3>State-Space Expectations</h3>
            <img src="visualizations/state_space_expectations.png" alt="State-space expectations">
            <div class="caption">Two-panel visualization showing latent fitness state evolution and weight predictions.</div>
        </div>

        <div class="visualization">
            <h3>Trace Plots</h3>
            <img src="visualizations/trace_plots.png" alt="Trace plots">
            <div class="caption">Trace plots (left) and density plots (right) for parameter convergence diagnostics.</div>
        </div>
    </section>

    <section id="interpretation" class="section">
        <h2>Interpretation</h2>

        <h3>Key Insights</h3>
        <ul>
            <li><strong>Fitness persistence (α):</strong> Values close to 1 indicate that fitness state persists strongly from day to day.</li>
            <li><strong>Impulse decay (ψ):</strong> Controls how quickly workout intensity accumulates and decays in the impulse state.</li>
            <li><strong>Workout to fitness (β):</strong> Positive values indicate that workouts increase the latent fitness state.</li>
            <li><strong>Fitness to weight (γ):</strong> Negative values indicate that higher fitness leads to lower weight.</li>
        </ul>

        <h3>Model Performance</h3>
        <p>The impulse-response model provides a more physiologically plausible representation of how workouts affect fitness and weight over time compared to basic state-space models.</p>

        <h3>Limitations</h3>
        <ul>
            <li><strong>Confounding factors:</strong> Diet, sleep, stress, and other lifestyle factors not included</li>
            <li><strong>Measurement error:</strong> Weight fluctuations from hydration, digestion, timing</li>
            <li><strong>Model complexity:</strong> Requires careful prior specification and convergence checking</li>
        </ul>
    </section>

    <footer class="section">
        <p><strong>Analysis generated using:</strong> Stan, CmdStanPy, ArviZ, and custom Python analysis scripts.</p>
        <p><strong>Data source:</strong> Personal Garmin health data export.</p>
    </footer>
</body>
</html>
"""

        # Write HTML file
        html_path = self.output_dir / "analysis_report.html"
        with open(html_path, 'w') as f:
            f.write(html)

        print(f"HTML report saved to: {html_path}")

    def run_analysis(self) -> None:
        """Run complete analysis pipeline."""
        print("=" * 70)
        print("IMPULSE-RESPONSE STATE-SPACE MODEL ANALYSIS")
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
        description="Analyze impulse-response state-space model for weight prediction using workout intensity",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/state_space_impulse_analysis",
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
    analyzer = ImpulseStateSpaceAnalyzer(
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        activity_types=activity_types,
        chains=args.chains,
        iter_warmup=args.iter_warmup,
        iter_sampling=args.iter_sampling,
        adapt_delta=args.adapt_delta,
        max_treedepth=args.max_treedepth,
        use_sparse=not args.no_sparse,
        n_inducing_points=args.n_inducing_points,
    )

    analyzer.run_analysis()


if __name__ == "__main__":
    main()