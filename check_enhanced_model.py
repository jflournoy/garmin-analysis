#!/usr/bin/env python3
"""Check enhanced sensitivity model results and generate summary."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def check_model_results():
    """Check enhanced sensitivity model results."""
    output_dir = Path("output/enhanced_sensitivity")

    if not output_dir.exists():
        print(f"❌ Output directory not found: {output_dir}")
        print("The model is still running or failed to start.")
        return 1

    print(f"\n📁 Checking enhanced sensitivity model results in: {output_dir}")

    # Check what files exist
    files = list(output_dir.glob("*"))
    if not files:
        print("❌ No files found in output directory")
        return 1

    print(f"\n📊 Found {len(files)} files:")
    for file in sorted(files):
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  • {file.name} ({size_mb:.2f} MB)")

    # Check key files
    key_files = {
        'parameter_summary.csv': 'Parameter estimates',
        'key_parameters.json': 'Key parameter values',
        'inference_data.nc': 'Full InferenceData',
        'inference_data.pkl': 'Pickle data',
        'predictions_time_series.png': 'Time series predictions',
        'variance_proportions.png': 'Variance proportions',
        'diagnostic_trace_plots.png': 'Diagnostic plots'
    }

    print(f"\n🔍 Checking key files:")
    missing_files = []
    for file_name, description in key_files.items():
        file_path = output_dir / file_name
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ {file_name}: {description} ({size_mb:.2f} MB)")
        else:
            print(f"  ❌ {file_name}: {description} - MISSING")
            missing_files.append(file_name)

    if missing_files:
        print(f"\n⚠️  Missing {len(missing_files)} key files")
        print("   The model may still be running or encountered an error.")

    # Try to load and display parameter summary
    param_file = output_dir / "parameter_summary.csv"
    if param_file.exists():
        print(f"\n📈 Loading parameter summary...")
        try:
            params = pd.read_csv(param_file, index_col=0)
            print(f"  ✓ Loaded {len(params)} parameters")

            # Display key parameters
            key_params = [
                'gamma_a_short', 'gamma_s_short', 'gamma_a_long', 'gamma_s_long',
                'alpha_gp', 'rho_gp', 'sigma_w',
                'prop_variance_gp', 'prop_variance_daily'
            ]

            print(f"\n🔑 Key parameter estimates:")
            for param in key_params:
                if param in params.index:
                    mean = params.loc[param, 'mean']
                    sd = params.loc[param, 'sd']
                    hdi_low = params.loc[param, 'hdi_3%']
                    hdi_high = params.loc[param, 'hdi_97%']
                    r_hat = params.loc[param, 'r_hat'] if 'r_hat' in params.columns else 'N/A'

                    print(f"  • {param}:")
                    print(f"    Mean: {mean:.4f}, SD: {sd:.4f}")
                    print(f"    94% HDI: [{hdi_low:.4f}, {hdi_high:.4f}]")
                    print(f"    R-hat: {r_hat}")

            # Check convergence
            if 'r_hat' in params.columns:
                high_rhat = params[params['r_hat'] > 1.1]
                if len(high_rhat) > 0:
                    print(f"\n⚠️  {len(high_rhat)} parameters with R-hat > 1.1:")
                    for param in high_rhat.index[:5]:  # Show first 5
                        print(f"    • {param}: R-hat = {high_rhat.loc[param, 'r_hat']:.3f}")
                else:
                    print(f"\n✓ All parameters have R-hat ≤ 1.1 (good convergence)")

        except Exception as e:
            print(f"  ❌ Error loading parameter summary: {e}")

    # Try to load key parameters JSON
    key_params_file = output_dir / "key_parameters.json"
    if key_params_file.exists():
        print(f"\n📋 Loading key parameters JSON...")
        try:
            with open(key_params_file, 'r') as f:
                key_params = json.load(f)

            print(f"  ✓ Loaded {len(key_params)} key parameters")
            print(f"\n🎯 Key parameter values:")
            for param, value in key_params.items():
                print(f"  • {param}: {value:.4f}")

        except Exception as e:
            print(f"  ❌ Error loading key parameters: {e}")

    # Check for images
    image_files = list(output_dir.glob("*.png"))
    if image_files:
        print(f"\n🖼️  Generated {len(image_files)} visualization files:")
        for img_file in sorted(image_files):
            size_kb = img_file.stat().st_size / 1024
            print(f"  • {img_file.name} ({size_kb:.1f} KB)")

    # Check standardization info
    std_file = output_dir / "standardization.json"
    if std_file.exists():
        print(f"\n⚖️  Standardization information:")
        try:
            with open(std_file, 'r') as f:
                std_info = json.load(f)

            print(f"  • Weight: mean={std_info.get('weight_mean', 'N/A'):.2f} lbs, std={std_info.get('weight_std', 'N/A'):.2f} lbs")
            print(f"  • Date range: {std_info.get('date_range_start', 'N/A')} to {std_info.get('date_range_end', 'N/A')}")
            print(f"  • Days: {std_info.get('n_days', 'N/A')}")
            print(f"  • Prediction points: {len(std_info.get('prediction_timestamps', []))}")

        except Exception as e:
            print(f"  ❌ Error loading standardization info: {e}")

    # Generate quick summary
    print(f"\n" + "=" * 70)
    print("ENHANCED SENSITIVITY MODEL - QUICK SUMMARY")
    print("=" * 70)

    if param_file.exists():
        try:
            params = pd.read_csv(param_file, index_col=0)

            # GP alpha constraint check
            if 'alpha_gp' in params.index:
                alpha_gp = params.loc['alpha_gp', 'mean']
                print(f"\n🔬 GP Alpha Constraint Test:")
                print(f"  • alpha_gp = {alpha_gp:.4f}")
                if alpha_gp > 0.5:
                    print(f"  → Exceeds 0.5 constraint (GP needs more flexibility)")
                else:
                    print(f"  → Within 0.5 constraint (constraint may be reasonable)")

            # Weight effects
            weight_params = ['gamma_a_short', 'gamma_s_short', 'gamma_a_long', 'gamma_s_long']
            effect_names = ['Aerobic Short', 'Strength Short', 'Aerobic Long', 'Strength Long']
            expected_signs = ['negative', 'positive', 'negative', 'positive']

            print(f"\n📊 Weight Effects (Physiological Expectations):")
            for param, name, expected in zip(weight_params, effect_names, expected_signs):
                if param in params.index:
                    value = params.loc[param, 'mean']
                    sign = "negative" if value < 0 else "positive"
                    matches = "✓" if (expected == "negative" and value < 0) or (expected == "positive" and value > 0) else "✗"
                    ci_low = params.loc[param, 'hdi_3%']
                    ci_high = params.loc[param, 'hdi_97%']

                    print(f"  • {name}: {value:.4f} ({sign}) [{ci_low:.4f}, {ci_high:.4f}] {matches} expected {expected}")

            # Variance proportions
            print(f"\n📈 Variance Proportions:")
            var_params = ['prop_variance_gp', 'prop_variance_daily',
                         'prop_variance_a_short', 'prop_variance_s_short',
                         'prop_variance_a_long', 'prop_variance_s_long']

            for param in var_params:
                if param in params.index:
                    value = params.loc[param, 'mean']
                    name = param.replace('prop_variance_', '').replace('_', ' ').title()
                    print(f"  • {name}: {value:.3f}")

        except Exception as e:
            print(f"❌ Error generating summary: {e}")

    print(f"\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print(f"1. Review parameter estimates in: {output_dir}/parameter_summary.csv")
    print(f"2. Check visualizations in: {output_dir}/")
    print(f"3. Compare with previous sensitivity analysis")
    print(f"4. Use this model as primary for the report")
    print(f"5. Generate comprehensive analysis report")

    return 0


if __name__ == "__main__":
    sys.exit(check_model_results())