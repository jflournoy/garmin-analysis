#!/usr/bin/env python3
"""Compare sensitivity analysis model with improved_v2 model.

This script compares the results of:
1. Improved_v2 model (physiological priors, alpha_gp < 0.5)
2. Sensitivity model (weaker priors, alpha_gp < 1.0)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_model_results(model_name, output_dir):
    """Load model results from output directory."""
    base_path = Path(output_dir) / model_name

    # Load parameter summary
    param_file = base_path / "parameter_summary.csv"
    if not param_file.exists():
        raise FileNotFoundError(f"Parameter file not found: {param_file}")

    params = pd.read_csv(param_file, index_col=0)

    # Load key parameters if available
    key_params_file = base_path / "key_parameters.json"
    key_params = {}
    if key_params_file.exists():
        with open(key_params_file, 'r') as f:
            key_params = json.load(f)

    return params, key_params


def compare_models():
    """Compare sensitivity analysis with improved_v2 model."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON: SENSITIVITY ANALYSIS vs IMPROVED_V2")
    print("=" * 70)

    # Load both models
    print("\nLoading model results...")

    try:
        sens_params, sens_key = load_model_results("sensitivity_analysis", "output")
        print(f"  ✓ Loaded sensitivity analysis model")
    except Exception as e:
        print(f"  ❌ Failed to load sensitivity analysis: {e}")
        return 1

    try:
        imp_params, imp_key = load_model_results("four_fitness_full", "output")
        print(f"  ✓ Loaded improved_v2 model")
    except Exception as e:
        print(f"  ❌ Failed to load improved_v2 model: {e}")
        return 1

    # Create comparison directory
    output_dir = Path("output/model_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Key parameters to compare
    key_param_names = [
        'gamma_a_short', 'gamma_s_short', 'gamma_a_long', 'gamma_s_long',
        'alpha_gp', 'rho_gp', 'sigma_w'
    ]

    # Create comparison table
    print("\n" + "=" * 70)
    print("KEY PARAMETER COMPARISON")
    print("=" * 70)
    print("\nParameter                Sensitivity    Improved_v2    Difference")
    print("-" * 70)

    comparison_data = []
    for param in key_param_names:
        sens_val = sens_params.loc[param, 'mean'] if param in sens_params.index else sens_key.get(param, np.nan)
        imp_val = imp_params.loc[param, 'mean'] if param in imp_params.index else imp_key.get(param, np.nan)

        if not np.isnan(sens_val) and not np.isnan(imp_val):
            diff = sens_val - imp_val
            comparison_data.append({
                'parameter': param,
                'sensitivity': sens_val,
                'improved_v2': imp_val,
                'difference': diff
            })

            # Format for display
            param_display = param.ljust(25)
            sens_display = f"{sens_val:8.4f}".rjust(12)
            imp_display = f"{imp_val:8.4f}".rjust(12)
            diff_display = f"{diff:8.4f}".rjust(12)

            print(f"{param_display} {sens_display} {imp_display} {diff_display}")

    # Save comparison table
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(output_dir / "parameter_comparison.csv", index=False)
    print(f"\n✓ Comparison table saved to: {output_dir / 'parameter_comparison.csv'}")

    # Create visual comparison
    print("\n" + "=" * 70)
    print("VISUAL COMPARISON")
    print("=" * 70)

    try:
        # 1. Weight effects comparison
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        weight_params = ['gamma_a_short', 'gamma_s_short', 'gamma_a_long', 'gamma_s_long']
        titles = ['Aerobic Short-term', 'Strength Short-term', 'Aerobic Long-term', 'Strength Long-term']

        for idx, (param, title) in enumerate(zip(weight_params, titles)):
            ax = axes[idx//2, idx%2]

            if param in sens_params.index and param in imp_params.index:
                # Get posterior distributions
                sens_mean = sens_params.loc[param, 'mean']
                sens_sd = sens_params.loc[param, 'sd']
                sens_low = sens_params.loc[param, 'hdi_3%']
                sens_high = sens_params.loc[param, 'hdi_97%']

                imp_mean = imp_params.loc[param, 'mean']
                imp_sd = imp_params.loc[param, 'sd']
                imp_low = imp_params.loc[param, 'hdi_3%']
                imp_high = imp_params.loc[param, 'hdi_97%']

                # Create error bars
                x_pos = [0, 1]
                means = [sens_mean, imp_mean]
                errors_low = [sens_mean - sens_low, imp_mean - imp_low]
                errors_high = [sens_high - sens_mean, imp_high - imp_mean]

                ax.errorbar(x_pos, means, yerr=[errors_low, errors_high],
                          fmt='o', capsize=5, capthick=2, markersize=8)

                ax.set_xticks([0, 1])
                ax.set_xticklabels(['Sensitivity', 'Improved_v2'])
                ax.set_ylabel('Effect Size')
                ax.set_title(f'{title} (γ)')
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

                # Add value labels
                for i, (x, y) in enumerate(zip(x_pos, means)):
                    ax.text(x, y + (0.02 if y >= 0 else -0.02), f'{y:.3f}',
                           ha='center', va='bottom' if y >= 0 else 'top', fontsize=9)

        plt.suptitle('Weight Effects: Sensitivity vs Improved_v2', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / "weight_effects_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Weight effects comparison saved")

        # 2. GP parameters comparison
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        gp_params = ['alpha_gp', 'rho_gp']
        gp_titles = ['GP Standard Deviation (α)', 'GP Length Scale (ρ)']

        for idx, (param, title) in enumerate(zip(gp_params, gp_titles)):
            ax = axes[idx]

            if param in sens_params.index and param in imp_params.index:
                sens_mean = sens_params.loc[param, 'mean']
                sens_sd = sens_params.loc[param, 'sd']

                imp_mean = imp_params.loc[param, 'mean']
                imp_sd = imp_params.loc[param, 'sd']

                # Bar plot
                x_pos = [0, 1]
                means = [sens_mean, imp_mean]
                sds = [sens_sd, imp_sd]

                bars = ax.bar(x_pos, means, yerr=sds, capsize=5,
                             color=['skyblue', 'lightcoral'], edgecolor='black')

                ax.set_xticks([0, 1])
                ax.set_xticklabels(['Sensitivity', 'Improved_v2'])
                ax.set_ylabel('Parameter Value')
                ax.set_title(title)

                # Add value labels
                for bar, mean_val in zip(bars, means):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{mean_val:.3f}', ha='center', va='bottom', fontsize=9)

        plt.suptitle('GP Parameters: Sensitivity vs Improved_v2', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / "gp_parameters_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ GP parameters comparison saved")

        # 3. Variance proportions comparison
        fig, ax = plt.subplots(figsize=(10, 6))

        var_params = ['prop_variance_a_short', 'prop_variance_s_short',
                     'prop_variance_a_long', 'prop_variance_s_long',
                     'prop_variance_daily', 'prop_variance_gp']

        var_labels = ['Aerobic Short', 'Strength Short', 'Aerobic Long',
                     'Strength Long', 'Daily Cycle', 'GP']

        sens_means = []
        imp_means = []

        for param in var_params:
            if param in sens_params.index and param in imp_params.index:
                sens_means.append(sens_params.loc[param, 'mean'])
                imp_means.append(imp_params.loc[param, 'mean'])

        x = np.arange(len(var_labels))
        width = 0.35

        ax.bar(x - width/2, sens_means, width, label='Sensitivity', color='skyblue', edgecolor='black')
        ax.bar(x + width/2, imp_means, width, label='Improved_v2', color='lightcoral', edgecolor='black')

        ax.set_xlabel('Variance Component')
        ax.set_ylabel('Proportion of Total Variance')
        ax.set_title('Variance Proportions: Sensitivity vs Improved_v2')
        ax.set_xticks(x)
        ax.set_xticklabels(var_labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "variance_proportions_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Variance proportions comparison saved")

    except Exception as e:
        print(f"  ⚠️  Warning creating visualizations: {e}")

    # Generate summary report
    print("\n" + "=" * 70)
    print("SUMMARY OF FINDINGS")
    print("=" * 70)

    # Check key differences
    print("\n🔬 Key Differences:")

    # 1. Check if alpha_gp exceeds 0.5 in sensitivity model
    alpha_gp_sens = sens_params.loc['alpha_gp', 'mean'] if 'alpha_gp' in sens_params.index else np.nan
    if not np.isnan(alpha_gp_sens):
        if alpha_gp_sens > 0.5:
            print(f"  • alpha_gp in sensitivity model: {alpha_gp_sens:.4f} > 0.5")
            print(f"    → GP needs more flexibility than 0.5 constraint allows")
        else:
            print(f"  • alpha_gp in sensitivity model: {alpha_gp_sens:.4f} ≤ 0.5")
            print(f"    → 0.5 constraint may be reasonable")

    # 2. Check weight effect signs
    print("\n📊 Weight Effect Signs:")
    weight_effects = ['gamma_a_short', 'gamma_s_short', 'gamma_a_long', 'gamma_s_long']
    effect_names = ['Aerobic Short', 'Strength Short', 'Aerobic Long', 'Strength Long']
    expected_signs = ['negative', 'positive', 'negative', 'positive']

    for param, name, expected in zip(weight_effects, effect_names, expected_signs):
        if param in sens_params.index:
            sens_val = sens_params.loc[param, 'mean']
            sign = "negative" if sens_val < 0 else "positive"
            matches = "✓" if (expected == "negative" and sens_val < 0) or (expected == "positive" and sens_val > 0) else "✗"
            print(f"  • {name}: {sens_val:.4f} ({sign}) {matches} expected {expected}")

    # 3. Check variance proportions
    print("\n📈 Variance Proportions:")
    if 'prop_variance_gp' in sens_params.index and 'prop_variance_gp' in imp_params.index:
        sens_gp_var = sens_params.loc['prop_variance_gp', 'mean']
        imp_gp_var = imp_params.loc['prop_variance_gp', 'mean']
        print(f"  • GP variance: {sens_gp_var:.3f} (sensitivity) vs {imp_gp_var:.3f} (improved_v2)")

        if sens_gp_var > imp_gp_var:
            print(f"    → Sensitivity model assigns MORE variance to GP")
        else:
            print(f"    → Sensitivity model assigns LESS variance to GP")

    # 4. Check parameter uncertainties
    print("\n🎯 Parameter Uncertainties:")
    for param in weight_effects:
        if param in sens_params.index and param in imp_params.index:
            sens_sd = sens_params.loc[param, 'sd']
            imp_sd = imp_params.loc[param, 'sd']
            ratio = sens_sd / imp_sd if imp_sd > 0 else np.nan

            if not np.isnan(ratio):
                if ratio > 1.2:
                    print(f"  • {param}: Uncertainty {ratio:.2f}x higher in sensitivity model")
                elif ratio < 0.8:
                    print(f"  • {param}: Uncertainty {ratio:.2f}x lower in sensitivity model")

    # Save summary report
    summary_file = output_dir / "comparison_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("MODEL COMPARISON: SENSITIVITY ANALYSIS vs IMPROVED_V2\n")
        f.write("=" * 60 + "\n\n")

        f.write("KEY FINDINGS:\n")
        f.write("-" * 40 + "\n")

        # Alpha GP constraint
        if not np.isnan(alpha_gp_sens):
            f.write(f"1. GP alpha constraint:\n")
            f.write(f"   • Sensitivity model alpha_gp: {alpha_gp_sens:.4f}\n")
            if alpha_gp_sens > 0.5:
                f.write(f"   → Exceeds 0.5 constraint, suggesting GP needs more flexibility\n")
            else:
                f.write(f"   → Within 0.5 constraint, suggesting constraint may be reasonable\n")
            f.write("\n")

        # Weight effects
        f.write("2. Weight effects (physiological expectations):\n")
        for param, name, expected in zip(weight_effects, effect_names, expected_signs):
            if param in sens_params.index:
                sens_val = sens_params.loc[param, 'mean']
                sign = "negative" if sens_val < 0 else "positive"
                matches = "MATCHES" if (expected == "negative" and sens_val < 0) or (expected == "positive" and sens_val > 0) else "DOES NOT MATCH"
                f.write(f"   • {name}: {sens_val:.4f} ({sign}) - {matches} expected {expected}\n")
        f.write("\n")

        # Model comparison
        f.write("3. Model comparison insights:\n")
        f.write("   • Sensitivity analysis tests how much results depend on priors\n")
        f.write("   • Weaker priors centered at 0 allow data to speak more\n")
        f.write("   • Divergent transitions indicate challenging posterior geometry\n")
        f.write("   • Similar parameter estimates suggest robustness to prior choices\n")
        f.write("   • Different estimates suggest sensitivity to prior assumptions\n")

    print(f"\n✓ Summary report saved to: {summary_file}")

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)
    print(f"\n📊 All results saved to: {output_dir}")
    print(f"📈 Key files:")
    print(f"   • parameter_comparison.csv - Numerical comparison")
    print(f"   • weight_effects_comparison.png - Visual comparison of weight effects")
    print(f"   • gp_parameters_comparison.png - Visual comparison of GP parameters")
    print(f"   • variance_proportions_comparison.png - Variance component comparison")
    print(f"   • comparison_summary.txt - Summary of key findings")

    print(f"\n🔬 Key insights from comparison:")
    print(f"   1. Check if alpha_gp exceeds 0.5 (GP constraint test)")
    print(f"   2. See if weight effects maintain physiological signs")
    print(f"   3. Compare parameter uncertainties between models")
    print(f"   4. Assess variance proportion differences")

    return 0


if __name__ == "__main__":
    sys.exit(compare_models())