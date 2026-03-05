#!/usr/bin/env python3
"""Create missing visualizations for enhanced sensitivity model."""

import sys
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from datetime import datetime

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_enhanced_model_data():
    """Load enhanced sensitivity model data."""
    model_dir = Path("output/enhanced_sensitivity")

    if not model_dir.exists():
        print(f"❌ Enhanced model directory not found: {model_dir}")
        return None

    try:
        pickle_path = model_dir / "inference_data.pkl"
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)

        print(f"✓ Loaded enhanced model data")
        return data

    except Exception as e:
        print(f"❌ Error loading enhanced model data: {e}")
        return None


def create_simple_component_plots(data, output_dir):
    """Create simple component visualization plots."""
    print("\nCreating simple component plots...")

    idata = data['idata']
    df_weight = data['df_weight']
    df_daily = data['df_daily']
    standardization = data['standardization']
    pred_timestamps = data['pred_timestamps']

    weight_mean = standardization['weight_mean']
    weight_std = standardization['weight_std']

    # Create output directory
    viz_dir = output_dir / "simple_components"
    viz_dir.mkdir(exist_ok=True)

    # 1. Weight predictions with observations
    print("  1. Creating weight predictions...")
    try:
        fig, ax = plt.subplots(figsize=(14, 6))

        # Get predictions
        y_pred = idata.posterior['y_pred'].values
        y_pred_mean = y_pred.mean(axis=(0, 1)) * weight_std + weight_mean
        y_pred_lower = np.percentile(y_pred, 3, axis=(0, 1)) * weight_std + weight_mean
        y_pred_upper = np.percentile(y_pred, 97, axis=(0, 1)) * weight_std + weight_mean

        # Plot predictions
        ax.plot(pred_timestamps, y_pred_mean, 'b-', linewidth=2, label='Predicted Weight')
        ax.fill_between(pred_timestamps, y_pred_lower, y_pred_upper, alpha=0.3, color='blue', label='94% CI')

        # Plot actual weight observations
        weight_dates = pd.to_datetime(df_weight['date'])
        ax.scatter(weight_dates, df_weight['weight_lbs'], alpha=0.6, s=30, color='red', label='Actual Weight')

        ax.set_xlabel('Date')
        ax.set_ylabel('Weight (lbs)')
        ax.set_title('Enhanced Model: Weight Predictions vs Observations')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(viz_dir / "weight_predictions.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("    ✓ Created weight predictions")
    except Exception as e:
        print(f"    ⚠️  Error creating weight predictions: {e}")

    # 2. Component breakdown
    print("  2. Creating component breakdown...")
    try:
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))

        # Extract components
        components = {
            'GP Trend': ('f_gp_pred', 'GP Trend'),
            'Daily Cycle': ('f_daily_pred', 'Daily Cycle'),
            'Aerobic Short': ('fitness_contrib_a_short', 'Aerobic Short-term'),
            'Strength Short': ('fitness_contrib_s_short', 'Strength Short-term'),
            'Aerobic Long': ('fitness_contrib_a_long', 'Aerobic Long-term'),
            'Strength Long': ('fitness_contrib_s_long', 'Strength Long-term')
        }

        for idx, (var_name, title) in enumerate(components.values()):
            ax = axes[idx // 2, idx % 2]

            if var_name in idata.posterior:
                samples = idata.posterior[var_name].values
                mean = samples.mean(axis=(0, 1)) * weight_std
                lower = np.percentile(samples, 3, axis=(0, 1)) * weight_std
                upper = np.percentile(samples, 97, axis=(0, 1)) * weight_std

                ax.plot(pred_timestamps, mean, 'b-', linewidth=2, label='Mean')
                ax.fill_between(pred_timestamps, lower, upper, alpha=0.3, color='blue', label='94% CI')

                ax.set_xlabel('Date')
                ax.set_ylabel('Contribution (lbs)')
                ax.set_title(title)
                ax.legend(loc='upper left')
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(viz_dir / "component_breakdown.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("    ✓ Created component breakdown")
    except Exception as e:
        print(f"    ⚠️  Error creating component breakdown: {e}")

    # 3. Fitness state evolution
    print("  3. Creating fitness state evolution...")
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        fitness_states = {
            (0, 0): ('fitness_a_short_stored', 'Aerobic Short-term Fitness'),
            (0, 1): ('fitness_s_short_stored', 'Strength Short-term Fitness'),
            (1, 0): ('fitness_a_long_stored', 'Aerobic Long-term Fitness'),
            (1, 1): ('fitness_s_long_stored', 'Strength Long-term Fitness')
        }

        for (row, col), (var_name, title) in fitness_states.items():
            ax = axes[row, col]

            if var_name in idata.posterior:
                samples = idata.posterior[var_name].values
                mean = samples.mean(axis=(0, 1))
                lower = np.percentile(samples, 3, axis=(0, 1))
                upper = np.percentile(samples, 97, axis=(0, 1))

                # Create time index (days)
                days = np.arange(1, len(mean) + 1)

                ax.plot(days, mean, 'b-', linewidth=2, label='Mean')
                ax.fill_between(days, lower, upper, alpha=0.3, color='blue', label='94% CI')

                # Add activity intensity overlay for short-term components
                if 'short' in var_name:
                    activity_type = 'aerobic' if 'a_' in var_name else 'strength'
                    activity_col = f'{activity_type}_intensity_std'

                    if activity_col in df_daily.columns:
                        intensity = df_daily[activity_col].values
                        ax2 = ax.twinx()
                        ax2.fill_between(days, 0, intensity, alpha=0.2, color='orange', label='Activity Intensity')
                        ax2.set_ylabel('Intensity (std)', color='orange')
                        ax2.tick_params(axis='y', labelcolor='orange')

                ax.set_xlabel('Day')
                ax.set_ylabel('Fitness State (std)')
                ax.set_title(title)
                ax.legend(loc='upper left')
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(viz_dir / "fitness_state_evolution.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("    ✓ Created fitness state evolution")
    except Exception as e:
        print(f"    ⚠️  Error creating fitness state evolution: {e}")

    # 4. Variance proportions
    print("  4. Creating variance proportions...")
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Get variance proportions
        var_props = {
            'Aerobic Short': 'prop_variance_a_short',
            'Strength Short': 'prop_variance_s_short',
            'Aerobic Long': 'prop_variance_a_long',
            'Strength Long': 'prop_variance_s_long',
            'GP': 'prop_variance_gp',
            'Daily': 'prop_variance_daily'
        }

        # Left: Bar chart of mean proportions
        ax = axes[0]

        prop_means = {}
        prop_cis = {}
        for label, var_name in var_props.items():
            if var_name in idata.posterior:
                samples = idata.posterior[var_name].values.flatten()
                prop_means[label] = samples.mean() * 100
                prop_cis[label] = (np.percentile(samples, 3) * 100, np.percentile(samples, 97) * 100)

        labels = list(prop_means.keys())
        means = list(prop_means.values())
        ci_lower = [prop_cis[label][0] for label in labels]
        ci_upper = [prop_cis[label][1] for label in labels]

        x_pos = np.arange(len(labels))
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))

        bars = ax.bar(x_pos, means, color=colors, alpha=0.7, edgecolor='black')
        # Convert to numpy arrays for subtraction
        means_arr = np.array(means)
        ci_lower_arr = np.array(ci_lower)
        ci_upper_arr = np.array(ci_upper)

        ax.errorbar(x_pos, means_arr, yerr=[means_arr - ci_lower_arr, ci_upper_arr - means_arr], fmt='none',
                   ecolor='black', capsize=5, capthick=2)

        ax.set_xlabel('Component')
        ax.set_ylabel('Variance Proportion (%)')
        ax.set_title('Component Variance Proportions')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

        # Right: Time series of proportions
        ax = axes[1]

        for label, var_name in var_props.items():
            if var_name in idata.posterior:
                samples = idata.posterior[var_name].values
                mean = samples.mean(axis=(0, 1)) * 100
                ax.plot(pred_timestamps, mean, label=label, linewidth=2)

        ax.set_xlabel('Date')
        ax.set_ylabel('Variance Proportion (%)')
        ax.set_title('Variance Proportions Over Time')
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(viz_dir / "variance_proportions.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("    ✓ Created variance proportions")
    except Exception as e:
        print(f"    ⚠️  Error creating variance proportions: {e}")

    return viz_dir


def create_parameter_summary(data, output_dir):
    """Create parameter summary plots."""
    print("\nCreating parameter summary...")

    idata = data['idata']
    viz_dir = output_dir / "parameter_summary"
    viz_dir.mkdir(exist_ok=True)

    # 1. Key parameter distributions
    print("  1. Creating key parameter distributions...")
    try:
        key_params = {
            'Weight Effects': ['gamma_a_short', 'gamma_s_short', 'gamma_a_long', 'gamma_s_long'],
            'GP Parameters': ['alpha_gp', 'rho_gp'],
            'Noise': ['sigma_w']
        }

        fig, axes = plt.subplots(len(key_params), 1, figsize=(10, 3 * len(key_params)))

        for idx, (group_name, params) in enumerate(key_params.items()):
            ax = axes[idx] if len(key_params) > 1 else axes

            existing_params = [p for p in params if p in idata.posterior]
            if existing_params:
                # Create forest plot
                az.plot_forest(idata, var_names=existing_params, combined=True, ax=ax, show=False)
                ax.set_title(f'{group_name}')
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(viz_dir / "key_parameters.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("    ✓ Created key parameter distributions")
    except Exception as e:
        print(f"    ⚠️  Error creating key parameter distributions: {e}")

    # 2. Half-life distributions
    print("  2. Creating half-life distributions...")
    try:
        half_life_params = ['half_life_a_short', 'half_life_s_short', 'half_life_a_long', 'half_life_s_long']
        existing_half_lives = [p for p in half_life_params if p in idata.posterior]

        if existing_half_lives:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Create box plot of half-lives
            data_list = []
            labels = []
            for param in existing_half_lives:
                samples = idata.posterior[param].values.flatten()
                data_list.append(samples)
                labels.append(param.replace('half_life_', '').replace('_', ' ').title())

            bp = ax.boxplot(data_list, labels=labels, patch_artist=True)

            # Color the boxes
            colors = plt.cm.Set3(np.linspace(0, 1, len(data_list)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)

            ax.set_ylabel('Half-life (days)')
            ax.set_title('Fitness Component Half-lives')
            ax.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            plt.savefig(viz_dir / "half_lives.png", dpi=150, bbox_inches='tight')
            plt.close()
            print("    ✓ Created half-life distributions")
    except Exception as e:
        print(f"    ⚠️  Error creating half-life distributions: {e}")

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
    <title>Enhanced Sensitivity Model: Missing Visualizations</title>
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
            <h1 class="display-4">Enhanced Sensitivity Model: Missing Visualizations</h1>
            <p class="lead">Key Time Series and Component Analysis</p>
            <p class="mb-0">Generated: {current_time}</p>
        </div>

        <div class="alert alert-success">
            <h4>🎯 Essential Visualization Suite</h4>
            <p>This report provides the missing visualizations showing time series and component interactions for the enhanced sensitivity model.</p>
        </div>

        <div class="section">
            <h2>📈 Component Time Series</h2>
            <div class="row">
                <div class="col-md-6">
                    <div class="viz-card">
                        <h5>1. Weight Predictions</h5>
                        <img src="simple_components/weight_predictions.png" alt="Weight Predictions">
                        <p>Model predictions vs actual weight observations with credible intervals.</p>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="viz-card">
                        <h5>2. Component Breakdown</h5>
                        <img src="simple_components/component_breakdown.png" alt="Component Breakdown">
                        <p>Individual contributions of all 6 model components over time.</p>
                    </div>
                </div>
            </div>
            <div class="row">
                <div class="col-md-6">
                    <div class="viz-card">
                        <h5>3. Fitness State Evolution</h5>
                        <img src="simple_components/fitness_state_evolution.png" alt="Fitness State Evolution">
                        <p>Time series of all 4 fitness states with activity intensity overlay.</p>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="viz-card">
                        <h5>4. Variance Proportions</h5>
                        <img src="simple_components/variance_proportions.png" alt="Variance Proportions">
                        <p>Relative importance of each component in explaining weight variance.</p>
                    </div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>📊 Parameter Analysis</h2>
            <div class="row">
                <div class="col-md-6">
                    <div class="viz-card">
                        <h5>1. Key Parameters</h5>
                        <img src="parameter_summary/key_parameters.png" alt="Key Parameters">
                        <p>Posterior distributions of weight effects, GP parameters, and measurement noise.</p>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="viz-card">
                        <h5>2. Half-life Distributions</h5>
                        <img src="parameter_summary/half_lives.png" alt="Half-life Distributions">
                        <p>Half-lives of fitness decay for each component type.</p>
                    </div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>🔗 Related Reports</h2>
            <div class="alert alert-info">
                <h5>Complete Analysis Suite</h5>
                <ul>
                    <li><a href="../enhanced_model_visualizations/index.html">Enhanced Model Visualizations</a> - Comprehensive visualizations</li>
                    <li><a href="../enhanced_sensitivity_report/index.html">Enhanced Sensitivity Report</a> - Detailed analysis report</li>
                    <li><a href="../enhanced_model/index.html">Enhanced Model Overview</a> - Model summary and diagnostics</li>
                </ul>
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
    print("CREATING MISSING VISUALIZATIONS FOR ENHANCED SENSITIVITY MODEL")
    print("=" * 70)

    data = load_enhanced_model_data()
    if data is None:
        return

    output_dir = Path("docs/missing_visualizations")
    output_dir.mkdir(exist_ok=True)

    print(f"\nOutput directory: {output_dir}")

    create_simple_component_plots(data, output_dir)
    create_parameter_summary(data, output_dir)
    create_summary_report(output_dir)

    print("\n" + "=" * 70)
    print("MISSING VISUALIZATIONS CREATED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nVisualizations saved to: {output_dir}/")
    print(f"Summary report: {output_dir}/index.html")


if __name__ == "__main__":
    main()