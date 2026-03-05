#!/usr/bin/env python3
"""Create comprehensive visualizations for enhanced sensitivity model.

This script generates a full set of visualizations showing:
1. Time series predictions with component breakdown
2. Fitness state time series (4 components)
3. Component contributions over time
4. Activity intensity vs fitness states
5. Convergence diagnostics
6. Parameter distributions
"""

import sys
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.plot_cyclic import (
    plot_state_space_spline_decomposition,
    plot_state_space_expectations,
    plot_state_space_component_details,
    plot_spline_daily_pattern,
)

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_enhanced_model_data():
    """Load enhanced sensitivity model data."""
    model_dir = Path("output/enhanced_sensitivity")

    if not model_dir.exists():
        print(f"❌ Enhanced model directory not found: {model_dir}")
        print("Please run the enhanced sensitivity model first:")
        print("  uv run python run_enhanced_sensitivity.py")
        return None

    try:
        # Load pickle data
        pickle_path = model_dir / "inference_data.pkl"
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)

        print(f"✓ Loaded enhanced model data")
        print(f"  InferenceData shape: {data['idata'].posterior.dims}")
        print(f"  Weight observations: {len(data['df_weight'])}")
        print(f"  Days: {len(data['df_daily'])}")
        print(f"  Prediction points: {len(data['pred_timestamps'])}")

        return data

    except Exception as e:
        print(f"❌ Error loading enhanced model data: {e}")
        return None


def create_component_time_series_plots(data, output_dir):
    """Create comprehensive time series plots of model components."""
    print("\nCreating component time series plots...")

    idata = data['idata']
    df_weight = data['df_weight']
    df_daily = data['df_daily']
    stan_data = data['stan_data']
    standardization = data['standardization']
    pred_timestamps = data['pred_timestamps']

    weight_std = standardization['weight_std']

    # Create output directory
    viz_dir = output_dir / "component_time_series"
    viz_dir.mkdir(exist_ok=True)

    # 1. State-space spline decomposition
    print("  1. Creating state-space spline decomposition...")
    try:
        fig = plot_state_space_spline_decomposition(
            idata=idata,
            df_weight=df_weight,
            df_intensity=df_daily,
            stan_data=stan_data,
            model_name="Enhanced Sensitivity Model",
            output_path=viz_dir / "spline_decomposition.png",
            show_ci=True,
        )
        plt.close(fig)
        print("    ✓ Created spline decomposition")
    except Exception as e:
        print(f"    ⚠️  Error creating spline decomposition: {e}")

    # 2. State-space expectations
    print("  2. Creating state-space expectations...")
    try:
        fig = plot_state_space_expectations(
            idata=idata,
            df_weight=df_weight,
            df_intensity=df_daily,
            stan_data=stan_data,
            model_name="Enhanced Sensitivity Model",
            output_path=viz_dir / "state_space_expectations.png",
            show_ci=True,
        )
        plt.close(fig)
        print("    ✓ Created state-space expectations")
    except Exception as e:
        print(f"    ⚠️  Error creating state-space expectations: {e}")

    # 3. Component details
    print("  3. Creating component details...")
    try:
        fig = plot_state_space_component_details(
            idata=idata,
            df_weight=df_weight,
            df_intensity=df_daily,
            stan_data=stan_data,
            model_name="Enhanced Sensitivity Model",
            output_path=viz_dir / "component_details.png",
            show_ci=True,
        )
        plt.close(fig)
        print("    ✓ Created component details")
    except Exception as e:
        print(f"    ⚠️  Error creating component details: {e}")

    # 4. Fitness state time series
    print("  4. Creating fitness state time series...")
    try:
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))

        fitness_states = {
            'Aerobic Short': 'fitness_a_short_stored',
            'Strength Short': 'fitness_s_short_stored',
            'Aerobic Long': 'fitness_a_long_stored',
            'Strength Long': 'fitness_s_long_stored'
        }

        for idx, (name, var_name) in enumerate(fitness_states.items()):
            ax = axes[idx]

            if var_name in idata.posterior:
                samples = idata.posterior[var_name].values
                mean = samples.mean(axis=(0, 1))
                lower = np.percentile(samples, 3, axis=(0, 1))
                upper = np.percentile(samples, 97, axis=(0, 1))
                days = np.arange(1, len(mean) + 1)

                ax.plot(days, mean, 'b-', linewidth=2, label='Mean')
                ax.fill_between(days, lower, upper, alpha=0.3, color='blue', label='94% CI')

                if idx < 2:
                    intensity = df_daily['aerobic_intensity_std'].values if idx == 0 else df_daily['strength_intensity_std'].values
                    ax2 = ax.twinx()
                    ax2.fill_between(days, 0, intensity, alpha=0.2, color='orange', label='Activity Intensity')
                    ax2.set_ylabel('Intensity (std)', color='orange')
                    ax2.tick_params(axis='y', labelcolor='orange')

                ax.set_xlabel('Day')
                ax.set_ylabel('Fitness State (std)')
                ax.set_title(f'{name} Fitness State')
                ax.legend(loc='upper left')
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(viz_dir / "fitness_state_time_series.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("    ✓ Created fitness state time series")
    except Exception as e:
        print(f"    ⚠️  Error creating fitness state time series: {e}")

    # 5. Component contributions over time
    print("  5. Creating component contributions...")
    try:
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        ax = axes[0]

        components = {
            'GP': idata.posterior['f_gp_pred'].mean(dim=['chain', 'draw']).values * weight_std,
            'Daily': idata.posterior['f_daily_pred'].mean(dim=['chain', 'draw']).values * weight_std,
            'Aerobic Short': idata.posterior['fitness_contrib_a_short'].mean(dim=['chain', 'draw']).values * weight_std,
            'Strength Short': idata.posterior['fitness_contrib_s_short'].mean(dim=['chain', 'draw']).values * weight_std,
            'Aerobic Long': idata.posterior['fitness_contrib_a_long'].mean(dim=['chain', 'draw']).values * weight_std,
            'Strength Long': idata.posterior['fitness_contrib_s_long'].mean(dim=['chain', 'draw']).values * weight_std,
        }

        bottom = np.zeros_like(pred_timestamps, dtype=float)
        colors = plt.cm.Set3(np.linspace(0, 1, len(components)))

        for (label, values), color in zip(components.items(), colors):
            ax.fill_between(pred_timestamps, bottom, bottom + values,
                          alpha=0.6, label=label, color=color)
            bottom += values

        ax.set_xlabel('Date')
        ax.set_ylabel('Component Contribution (lbs)')
        ax.set_title('Enhanced Model: Component Contributions Over Time')
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        total = np.sum([comp for comp in components.values()], axis=0)
        relative = {label: values / total * 100 for label, values in components.items()}

        bottom = np.zeros_like(pred_timestamps, dtype=float)
        for (label, values), color in zip(relative.items(), colors):
            ax.fill_between(pred_timestamps, bottom, bottom + values,
                          alpha=0.6, label=label, color=color)
            bottom += values

        ax.set_xlabel('Date')
        ax.set_ylabel('Relative Contribution (%)')
        ax.set_title('Relative Component Contributions')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(viz_dir / "component_contributions.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("    ✓ Created component contributions")
    except Exception as e:
        print(f"    ⚠️  Error creating component contributions: {e}")

    return viz_dir


def create_convergence_diagnostics(data, output_dir):
    """Create convergence diagnostic plots."""
    print("\nCreating convergence diagnostics...")

    idata = data['idata']
    viz_dir = output_dir / "convergence_diagnostics"
    viz_dir.mkdir(exist_ok=True)

    # 1. Trace plots for key parameters
    print("  1. Creating trace plots...")
    try:
        key_params = ['gamma_a_short', 'gamma_s_short', 'gamma_a_long', 'gamma_s_long',
                     'alpha_gp', 'rho_gp', 'sigma_w']
        existing_params = [p for p in key_params if p in idata.posterior]

        if existing_params:
            fig, axes = plt.subplots(len(existing_params), 2, figsize=(12, 3 * len(existing_params)))

            for idx, param in enumerate(existing_params):
                ax_trace = axes[idx, 0] if len(existing_params) > 1 else axes[0]
                az.plot_trace(idata, var_names=[param], axes=[ax_trace, ax_trace], show=False)
                ax_trace.set_title(f'{param} trace', fontsize=10)

                ax_acf = axes[idx, 1] if len(existing_params) > 1 else axes[1]
                az.plot_autocorr(idata, var_names=[param], ax=ax_acf, show=False)
                ax_acf.set_title(f'{param} autocorrelation', fontsize=10)

            plt.tight_layout()
            plt.savefig(viz_dir / "trace_plots.png", dpi=150, bbox_inches='tight')
            plt.close()
            print("    ✓ Created trace plots")
    except Exception as e:
        print(f"    ⚠️  Error creating trace plots: {e}")

    # 2. R-hat distribution
    print("  2. Creating R-hat distribution...")
    try:
        summary = az.summary(idata)

        if 'r_hat' in summary.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            rhat_values = summary['r_hat'].dropna()
            ax.hist(rhat_values, bins=30, alpha=0.7, color='steelblue', edgecolor='black')

            ax.axvline(x=1.0, color='green', linestyle='--', linewidth=2, label='Perfect (1.0)')
            ax.axvline(x=1.01, color='orange', linestyle='--', linewidth=2, label='Good (<1.01)')
            ax.axvline(x=1.1, color='red', linestyle='--', linewidth=2, label='Acceptable (<1.1)')

            mean_rhat = rhat_values.mean()
            max_rhat = rhat_values.max()
            n_params = len(rhat_values)
            n_good = (rhat_values <= 1.01).sum()
            n_acceptable = (rhat_values <= 1.1).sum()

            stats_text = f'Parameters: {n_params}\nMean R-hat: {mean_rhat:.4f}\nMax R-hat: {max_rhat:.4f}\nGood (≤1.01): {n_good}/{n_params}\nAcceptable (≤1.1): {n_acceptable}/{n_params}'

            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            ax.set_xlabel('R-hat value')
            ax.set_ylabel('Number of parameters')
            ax.set_title('Enhanced Model: R-hat Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(viz_dir / "rhat_distribution.png", dpi=150, bbox_inches='tight')
            plt.close()
            print("    ✓ Created R-hat distribution")
    except Exception as e:
        print(f"    ⚠️  Error creating R-hat distribution: {e}")

    return viz_dir


def create_parameter_distributions(data, output_dir):
    """Create parameter distribution plots."""
    print("\nCreating parameter distributions...")

    idata = data['idata']
    viz_dir = output_dir / "parameter_distributions"
    viz_dir.mkdir(exist_ok=True)

    # Forest plots for parameter groups
    param_groups = {
        'Weight Effects': ['gamma_a_short', 'gamma_s_short', 'gamma_a_long', 'gamma_s_long'],
        'GP Parameters': ['alpha_gp', 'rho_gp'],
        'Measurement Noise': ['sigma_w'],
    }

    for group_name, params in param_groups.items():
        existing_params = [p for p in params if p in idata.posterior]
        if not existing_params:
            continue

        try:
            fig, ax = plt.subplots(figsize=(10, max(4, len(existing_params) * 0.5)))
            az.plot_forest(idata, var_names=existing_params, combined=True, ax=ax, show=False)
            ax.set_title(f'Enhanced Model: {group_name}')
            ax.grid(True, alpha=0.3)

            filename = f"forest_{group_name.lower().replace(' ', '_')}.png"
            plt.tight_layout()
            plt.savefig(viz_dir / filename, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"    ✓ Created {group_name} forest plot")
        except Exception as e:
            print(f"    ⚠️  Error creating {group_name} forest plot: {e}")

    return viz_dir


def create_summary_report(output_dir):
    """Create a summary HTML report."""
    print("\nCreating summary report...")

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced Sensitivity Model: Comprehensive Visualizations</title>
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
        .viz-card {{
            background: white;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
            border: 1px solid #dee2e6;
        }}
        .viz-card img {{
            width: 100%;
            height: auto;
            border-radius: 4px;
            margin-bottom: 1rem;
        }}
        h2 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
    </style>
    <!-- Simple Analytics -->
    <script async src="https://scripts.simpleanalyticscdn.com/latest.js"></script>
</head>
<body>
    <div class="container">
        <div class="header text-center">
            <h1 class="display-4">Enhanced Sensitivity Model Visualizations</h1>
            <p class="lead">Comprehensive Time Series and Component Analysis</p>
            <p class="mb-0">Generated: {current_time}</p>
        </div>

        <div class="alert alert-success">
            <h4>🎯 Complete Visualization Suite</h4>
            <p>This report provides comprehensive visualizations of the enhanced sensitivity model with conservative sampling (adapt_delta=0.99).</p>
        </div>

        <div class="section">
            <h2>📈 Component Time Series Analysis</h2>
            <div class="row">
                <div class="col-md-6">
                    <div class="viz-card">
                        <h5>1. Spline Decomposition</h5>
                        <img src="component_time_series/spline_decomposition.png" alt="Spline Decomposition">
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="viz-card">
                        <h5>2. State-Space Expectations</h5>
                        <img src="component_time_series/state_space_expectations.png" alt="State-Space Expectations">
                    </div>
                </div>
            </div>
            <div class="row">
                <div class="col-md-6">
                    <div class="viz-card">
                        <h5>3. Component Details</h5>
                        <img src="component_time_series/component_details.png" alt="Component Details">
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="viz-card">
                        <h5>4. Fitness State Time Series</h5>
                        <img src="component_time_series/fitness_state_time_series.png" alt="Fitness State Time Series">
                    </div>
                </div>
            </div>
            <div class="row">
                <div class="col-md-6">
                    <div class="viz-card">
                        <h5>5. Component Contributions</h5>
                        <img src="component_time_series/component_contributions.png" alt="Component Contributions">
                    </div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>🔬 Convergence Diagnostics</h2>
            <div class="row">
                <div class="col-md-6">
                    <div class="viz-card">
                        <h5>1. Trace Plots</h5>
                        <img src="convergence_diagnostics/trace_plots.png" alt="Trace Plots">
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="viz-card">
                        <h5>2. R-hat Distribution</h5>
                        <img src="convergence_diagnostics/rhat_distribution.png" alt="R-hat Distribution">
                    </div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>📊 Parameter Distributions</h2>
            <div class="row">
                <div class="col-md-6">
                    <div class="viz-card">
                        <h5>Weight Effects</h5>
                        <img src="parameter_distributions/forest_weight_effects.png" alt="Weight Effects">
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="viz-card">
                        <h5>GP Parameters</h5>
                        <img src="parameter_distributions/forest_gp_parameters.png" alt="GP Parameters">
                    </div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>'''

    output_file = output_dir / "index.html"
    with open(output_file, 'w') as f:
        f.write(html_content)

    print(f"✓ Summary report saved to: {output_file}")
    return output_file


def main():
    """Main function."""
    print("=" * 70)
    print("CREATING COMPREHENSIVE VISUALIZATIONS FOR ENHANCED SENSITIVITY MODEL")
    print("=" * 70)

    data = load_enhanced_model_data()
    if data is None:
        return

    output_dir = Path("docs/enhanced_model_visualizations")
    output_dir.mkdir(exist_ok=True)

    print(f"\nOutput directory: {output_dir}")

    create_component_time_series_plots(data, output_dir)
    create_convergence_diagnostics(data, output_dir)
    create_parameter_distributions(data, output_dir)
    create_summary_report(output_dir)

    print("\n" + "=" * 70)
    print("COMPREHENSIVE VISUALIZATIONS CREATED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nVisualizations saved to: {output_dir}/")
    print(f"Summary report: {output_dir}/index.html")


if __name__ == "__main__":
    main()