#!/usr/bin/env python3
"""Test the heavily constrained AR(1) model."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import cmdstanpy
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.data.weight import load_weight_data
from src.data.intensity import load_intensity_by_activity

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def prepare_test_data():
    """Prepare test data for model."""
    print("Preparing test data...")

    data_dir = "data"
    df_weight = load_weight_data(data_dir)

    # Load intensity data
    df_intensity = load_intensity_by_activity(
        data_dir=data_dir,
        activity_types=['strength_training', 'walking', 'cycling'],
        max_hr=185.0,
    )

    # Create full date range
    all_dates = set()
    all_dates.update(df_weight['timestamp'].dt.date)
    all_dates.update(df_intensity['date'].dt.date)

    min_date = min(all_dates)
    max_date = max(all_dates)
    date_range = pd.date_range(start=min_date, end=max_date, freq='D')
    D = len(date_range)

    # Create daily dataframe
    df_daily = pd.DataFrame({'date': date_range})

    # Merge intensity data
    df_act = df_intensity[['date', 'strength_training', 'walking', 'cycling']].copy()
    df_daily = df_daily.merge(df_act, on='date', how='left')
    df_daily = df_daily.fillna(0)

    # Standardize intensity data
    strength_mean = df_daily['strength_training'].mean()
    strength_std = df_daily['strength_training'].std()
    aerobic_mean = (df_daily['walking'] + df_daily['cycling']).mean()
    aerobic_std = (df_daily['walking'] + df_daily['cycling']).std()

    df_daily['strength_intensity_std'] = (df_daily['strength_training'] - strength_mean) / strength_std
    df_daily['aerobic_intensity_std'] = ((df_daily['walking'] + df_daily['cycling']) - aerobic_mean) / aerobic_std

    # Prepare weight data
    df_weight['date'] = df_weight['timestamp'].dt.date
    # Convert to same type for merge
    df_daily['date_date'] = df_daily['date'].dt.date
    df_weight = df_weight.merge(df_daily[['date_date']].rename(columns={'date_date': 'date'}), on='date', how='inner')

    # Standardize weight
    weight_mean = df_weight['weight_lbs'].mean()
    weight_std = df_weight['weight_lbs'].std()
    df_weight['weight_std'] = (df_weight['weight_lbs'] - weight_mean) / weight_std

    # Create day index mapping
    date_to_idx = {date: i+1 for i, date in enumerate(date_range.date)}
    df_weight['day_idx'] = df_weight['date'].map(date_to_idx)

    # Extract hour of day
    df_weight['hour_of_day'] = df_weight['timestamp'].dt.hour + df_weight['timestamp'].dt.minute / 60.0

    # Prepare Stan data
    # Define hours to predict at (0, 6, 12, 18, 24 hours)
    H = 5
    pred_hours = np.array([0.0, 6.0, 12.0, 18.0, 24.0])
    pred_hours_scaled = pred_hours / 24.0

    stan_data = {
        'D': D,
        'strength_intensity': df_daily['strength_intensity_std'].values,
        'aerobic_intensity': df_daily['aerobic_intensity_std'].values,
        'N_weight': len(df_weight),
        'y_weight': df_weight['weight_std'].values,
        'day_idx': df_weight['day_idx'].values.astype(int),
        'hour_of_day': df_weight['hour_of_day'].values,
        'K': 2,  # 24h and 12h cycles
        'H': H,
        'pred_hours_scaled': pred_hours_scaled
    }

    return stan_data, df_weight, df_daily


def run_constrained_model(stan_data):
    """Run the heavily constrained AR(1) model."""
    print("\nRunning heavily constrained AR(1) model...")

    model_path = "stan/weight_state_space_training_decay_aerobic_symmetric_student_ar_spline_constrained.stan"

    try:
        model = cmdstanpy.CmdStanModel(stan_file=model_path)
        print(f"✓ Model compiled successfully")
    except Exception as e:
        print(f"✗ Model compilation failed: {e}")
        return None

    # Run sampling with conservative settings
    print("Running MCMC sampling...")
    fit = model.sample(
        data=stan_data,
        chains=4,
        parallel_chains=4,
        iter_warmup=500,
        iter_sampling=500,
        seed=12345,
        adapt_delta=0.95,
        max_treedepth=12,
        show_console=False,
        show_progress=True
    )

    return fit


def compare_with_previous(fit_constrained, fit_previous=None):
    """Compare constrained model with previous results."""
    print("\n" + "="*60)
    print("Model Comparison: Constrained vs Previous")
    print("="*60)

    # Extract AR(1) parameters from constrained model
    rho_constrained = fit_constrained.stan_variable('rho')
    sigma_epsilon_constrained = fit_constrained.stan_variable('sigma_epsilon')

    print(f"\nConstrained Model AR(1) Parameters:")
    print(f"  ρ: {np.mean(rho_constrained):.3f} [{np.percentile(rho_constrained, 2.5):.3f}, {np.percentile(rho_constrained, 97.5):.3f}]")
    print(f"  σ_ε: {np.mean(sigma_epsilon_constrained):.3f} [{np.percentile(sigma_epsilon_constrained, 2.5):.3f}, {np.percentile(sigma_epsilon_constrained, 97.5):.3f}]")

    # Extract fitness effects
    gamma_s_constrained = fit_constrained.stan_variable('gamma_s')
    gamma_a_constrained = fit_constrained.stan_variable('gamma_a')

    print(f"\nConstrained Model Fitness Effects:")
    print(f"  γ_s: {np.mean(gamma_s_constrained):.3f} [{np.percentile(gamma_s_constrained, 2.5):.3f}, {np.percentile(gamma_s_constrained, 97.5):.3f}]")
    print(f"  γ_a: {np.mean(gamma_a_constrained):.3f} [{np.percentile(gamma_a_constrained, 2.5):.3f}, {np.percentile(gamma_a_constrained, 97.5):.3f}]")

    if fit_previous is not None:
        # Compare with previous model if provided
        rho_previous = fit_previous.stan_variable('rho')
        sigma_epsilon_previous = fit_previous.stan_variable('sigma_epsilon')

        print(f"\nPrevious Model AR(1) Parameters:")
        print(f"  ρ: {np.mean(rho_previous):.3f} [{np.percentile(rho_previous, 2.5):.3f}, {np.percentile(rho_previous, 97.5):.3f}]")
        print(f"  σ_ε: {np.mean(sigma_epsilon_previous):.3f} [{np.percentile(sigma_epsilon_previous, 2.5):.3f}, {np.percentile(sigma_epsilon_previous, 97.5):.3f}]")

        print(f"\nImprovement in AR(1) parameters:")
        print(f"  ρ reduction: {100*(np.mean(rho_previous) - np.mean(rho_constrained))/np.mean(rho_previous):.1f}%")
        print(f"  σ_ε reduction: {100*(np.mean(sigma_epsilon_previous) - np.mean(sigma_epsilon_constrained))/np.mean(sigma_epsilon_previous):.1f}%")

    # Check if constraints are working
    print(f"\nConstraint Checks:")
    print(f"  ρ range: [{np.min(rho_constrained):.3f}, {np.max(rho_constrained):.3f}] (should be within [-0.5, 0.5])")

    if np.max(rho_constrained) > 0.5 or np.min(rho_constrained) < -0.5:
        print("  ⚠️ WARNING: ρ outside expected range!")
    else:
        print("  ✓ ρ within constrained range")

    return {
        'rho': rho_constrained,
        'sigma_epsilon': sigma_epsilon_constrained,
        'gamma_s': gamma_s_constrained,
        'gamma_a': gamma_a_constrained
    }


def analyze_variance_decomposition(fit, df_weight):
    """Analyze variance decomposition for the constrained model."""
    print("\n" + "="*60)
    print("Variance Decomposition Analysis")
    print("="*60)

    # Extract epsilon values (AR innovations)
    epsilon_samples = fit.stan_variable('epsilon')
    epsilon_mean = np.mean(epsilon_samples, axis=0)

    # Extract mu_no_ar (structural model without AR)
    mu_no_ar_samples = fit.stan_variable('mu_no_ar')
    mu_no_ar_mean = np.mean(mu_no_ar_samples, axis=0)

    # Calculate variance proportions
    total_variance = np.var(df_weight['weight_std'].values)
    structural_variance = np.var(mu_no_ar_mean)
    ar_variance = np.var(epsilon_mean)

    print(f"\nVariance Decomposition:")
    print(f"  Total variance in weight data: {total_variance:.4f}")
    print(f"  Variance from structural model (fitness + spline): {structural_variance:.4f} ({100*structural_variance/total_variance:.1f}%)")
    print(f"  Variance from AR(1) component: {ar_variance:.4f} ({100*ar_variance/total_variance:.1f}%)")
    print(f"  Unexplained variance (noise + interaction): {total_variance - structural_variance - ar_variance:.4f} ({100*(total_variance - structural_variance - ar_variance)/total_variance:.1f}%)")

    # Check if AR(1) variance is reasonable
    ar_proportion = ar_variance / total_variance
    if ar_proportion > 0.3:
        print(f"\n⚠️ WARNING: AR(1) component explains {100*ar_proportion:.1f}% of variance - still too high!")
        print("  Consider: even stronger constraints or remove AR(1) entirely")
    elif ar_proportion > 0.15:
        print(f"\n⚠️ NOTE: AR(1) component explains {100*ar_proportion:.1f}% of variance - moderate")
        print("  This might be acceptable for residual autocorrelation")
    elif ar_proportion > 0.05:
        print(f"\n✓ GOOD: AR(1) component explains {100*ar_proportion:.1f}% of variance - reasonable")
        print("  Constraints are working effectively")
    else:
        print(f"\n✓ EXCELLENT: AR(1) component explains {100*ar_proportion:.1f}% of variance - minimal")
        print("  AR(1) is properly constrained to capture only residual correlation")

    return {
        'epsilon_mean': epsilon_mean,
        'mu_no_ar_mean': mu_no_ar_mean,
        'variance_proportions': {
            'total': total_variance,
            'structural': structural_variance,
            'ar': ar_variance,
            'unexplained': total_variance - structural_variance - ar_variance
        }
    }


def create_comparison_visualizations(fit_constrained, variance_results, df_weight, output_dir):
    """Create visualizations comparing constrained vs previous models."""
    print("\nCreating comparison visualizations...")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # 1. AR(1) parameter distributions with constraints shown
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ρ distribution with constraint lines
    rho_samples = fit_constrained.stan_variable('rho')
    axes[0].hist(rho_samples, bins=30, density=True, alpha=0.7, edgecolor='black')
    axes[0].axvline(np.mean(rho_samples), color='red', linestyle='--', label=f'Mean: {np.mean(rho_samples):.3f}')
    axes[0].axvline(-0.5, color='orange', linestyle=':', label='Lower bound: -0.5')
    axes[0].axvline(0.5, color='orange', linestyle=':', label='Upper bound: 0.5')
    axes[0].axvline(0, color='black', linestyle='-', alpha=0.3)
    axes[0].set_xlabel('ρ (autocorrelation)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Constrained AR(1) Autocorrelation Coefficient\nRange: [-0.5, 0.5]')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # σ_ε distribution
    sigma_samples = fit_constrained.stan_variable('sigma_epsilon')
    axes[1].hist(sigma_samples, bins=30, density=True, alpha=0.7, edgecolor='black')
    axes[1].axvline(np.mean(sigma_samples), color='red', linestyle='--', label=f'Mean: {np.mean(sigma_samples):.3f}')
    axes[1].axvline(0.1, color='orange', linestyle=':', label='Prior mean: 0.1')
    axes[1].set_xlabel('σ_ε (innovation std)')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Constrained AR(1) Innovation Standard Deviation\nPrior: exponential(10)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'constrained_ar_parameters.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Variance decomposition comparison
    fig, ax = plt.subplots(figsize=(8, 8))

    variance_data = variance_results['variance_proportions']
    labels = ['Structural Model', 'AR(1) Component', 'Unexplained']
    sizes = [
        variance_data['structural'] / variance_data['total'] * 100,
        variance_data['ar'] / variance_data['total'] * 100,
        variance_data['unexplained'] / variance_data['total'] * 100
    ]

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct='%1.1f%%',
        startangle=90, colors=['#4CAF50', '#2196F3', '#FF9800']
    )

    ax.set_title('Variance Decomposition in Heavily Constrained AR(1) Model')
    plt.tight_layout()
    plt.savefig(output_dir / 'constrained_variance_decomposition.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. AR innovations time series (should be small)
    fig, ax = plt.subplots(figsize=(14, 6))

    # Sort by timestamp (need df_weight)
    df_sorted = df_weight.sort_values('timestamp')
    times = df_sorted['timestamp'].values

    ax.plot(times, variance_results['epsilon_mean'], 'b-', alpha=0.7, label='AR(1) Innovations')
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel('Time')
    ax.set_ylabel('AR(1) Innovation Value')
    ax.set_title('Constrained AR(1) Innovations Over Time\n(Should be small and uncorrelated)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Format x-axis
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'constrained_ar_innovations.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Visualizations saved to {output_dir}/")

    return output_dir


def create_constrained_summary(fit, variance_results, output_dir):
    """Create summary report for constrained model."""
    report_path = Path(output_dir) / 'constrained_analysis_summary.md'

    with open(report_path, 'w') as f:
        f.write("# Heavily Constrained AR(1) Model Analysis Summary\n\n")
        f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Heavy Constraints Applied\n\n")
        f.write("1. **ρ constrained to [-0.5, 0.5]** in parameter declaration\n")
        f.write("2. **Strong prior on ρ**: `normal(0, 0.1)` (centered at 0, small variance)\n")
        f.write("3. **Very strong prior on σ_ε**: `exponential(10)` (mean=0.1, very small)\n")
        f.write("4. **No local shrinkage parameters** - single σ_ε for all observations\n\n")

        f.write("## AR(1) Parameter Results\n\n")
        rho_samples = fit.stan_variable('rho')
        sigma_samples = fit.stan_variable('sigma_epsilon')

        f.write(f"- ρ (autocorrelation): {np.mean(rho_samples):.3f} ")
        f.write(f"[{np.percentile(rho_samples, 2.5):.3f}, {np.percentile(rho_samples, 97.5):.3f}]\n")
        f.write(f"- σ_ε (innovation std): {np.mean(sigma_samples):.3f} ")
        f.write(f"[{np.percentile(sigma_samples, 2.5):.3f}, {np.percentile(sigma_samples, 97.5):.3f}]\n\n")

        f.write("## Fitness Effects\n\n")
        gamma_s_samples = fit.stan_variable('gamma_s')
        gamma_a_samples = fit.stan_variable('gamma_a')

        f.write(f"- γ_s (strength effect): {np.mean(gamma_s_samples):.3f} ")
        f.write(f"[{np.percentile(gamma_s_samples, 2.5):.3f}, {np.percentile(gamma_s_samples, 97.5):.3f}]\n")
        f.write(f"- γ_a (aerobic effect): {np.mean(gamma_a_samples):.3f} ")
        f.write(f"[{np.percentile(gamma_a_samples, 2.5):.3f}, {np.percentile(gamma_a_samples, 97.5):.3f}]\n\n")

        f.write("## Variance Decomposition\n\n")
        variance_data = variance_results['variance_proportions']
        f.write(f"- Total variance in weight data: {variance_data['total']:.4f}\n")
        f.write(f"- Structural model (fitness + spline): {variance_data['structural']:.4f} ")
        f.write(f"({100*variance_data['structural']/variance_data['total']:.1f}%)\n")
        f.write(f"- AR(1) component: {variance_data['ar']:.4f} ")
        f.write(f"({100*variance_data['ar']/variance_data['total']:.1f}%)\n")
        f.write(f"- Unexplained (noise + interaction): {variance_data['unexplained']:.4f} ")
        f.write(f"({100*variance_data['unexplained']/variance_data['total']:.1f}%)\n\n")

        ar_proportion = variance_data['ar'] / variance_data['total']
        if ar_proportion > 0.3:
            f.write("## ⚠️ CRITICAL WARNING\n\n")
            f.write(f"AR(1) component still explains {100*ar_proportion:.1f}% of variance!\n")
            f.write("Even heavy constraints failed. Consider:\n")
            f.write("1. Removing AR(1) component entirely\n")
            f.write("2. Using even stronger priors\n")
            f.write("3. Investigating if fitness model is fundamentally wrong\n")
        elif ar_proportion > 0.15:
            f.write("## ⚠️ MODERATE WARNING\n\n")
            f.write(f"AR(1) component explains {100*ar_proportion:.1f}% of variance.\n")
            f.write("Constraints helped but AR(1) still substantial.\n")
        elif ar_proportion > 0.05:
            f.write("## ✓ SUCCESS\n\n")
            f.write(f"AR(1) component explains {100*ar_proportion:.1f}% of variance.\n")
            f.write("Heavy constraints successfully reduced AR(1) dominance.\n")
        else:
            f.write("## ✓ EXCELLENT SUCCESS\n\n")
            f.write(f"AR(1) component explains {100*ar_proportion:.1f}% of variance.\n")
            f.write("AR(1) is properly constrained to capture only residual correlation.\n")

        f.write("\n## Key Insight\n\n")
        f.write("The original problem was **per-observation shrinkage parameters** (`lambda[i]`)\n")
        f.write("which made AR(1) too flexible. Removing those and adding constraints\n")
        f.write("should fix the variance absorption issue.\n")

        f.write("\n## Generated Files\n\n")
        f.write("1. `constrained_ar_parameters.png` - AR(1) parameters with constraint lines\n")
        f.write("2. `constrained_variance_decomposition.png` - Variance proportions\n")
        f.write("3. `constrained_ar_innovations.png` - AR(1) innovations over time\n")
        f.write("4. `constrained_analysis_summary.md` - This summary file\n")

    print(f"✓ Summary report saved to {report_path}")


def main():
    """Main test function."""
    print("Testing heavily constrained AR(1) model")
    print("=" * 60)

    # Prepare test data
    stan_data, df_weight, df_daily = prepare_test_data()

    print(f"\nData dimensions:")
    print(f"  Number of days (D): {stan_data['D']}")
    print(f"  Number of weight observations: {stan_data['N_weight']}")

    # Run the constrained model
    fit = run_constrained_model(stan_data)

    if fit is None:
        print("Failed to run model. Exiting.")
        return

    # Compare with previous results
    comparison_results = compare_with_previous(fit)

    # Analyze variance decomposition
    variance_results = analyze_variance_decomposition(fit, df_weight)

    # Create visualizations
    output_dir = "docs/constrained_ar_analysis"
    create_comparison_visualizations(fit, variance_results, df_weight, output_dir)

    # Create summary report
    create_constrained_summary(fit, variance_results, output_dir)

    print("\n" + "="*60)
    print("Heavily Constrained AR(1) Model Test Complete")
    print("="*60)

    ar_proportion = variance_results['variance_proportions']['ar'] / variance_results['variance_proportions']['total']
    print(f"\nAR(1) variance proportion: {100*ar_proportion:.1f}%")

    if ar_proportion < 0.15:
        print("✓ SUCCESS: AR(1) component is now properly constrained")
    else:
        print("⚠️ WARNING: AR(1) still too dominant, consider stronger measures")

    print(f"\nResults saved to: {output_dir}/")


if __name__ == "__main__":
    main()