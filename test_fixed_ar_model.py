#!/usr/bin/env python3
"""Test the fixed AR(1) model with simple AR component (no local shrinkage)."""

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


def run_fixed_model(stan_data):
    """Run the fixed AR(1) model."""
    print("\nRunning fixed AR(1) model...")

    model_path = "stan/weight_state_space_training_decay_aerobic_symmetric_student_ar_spline_fixed.stan"

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


def analyze_ar_component(fit, df_weight):
    """Analyze the AR(1) component results."""
    print("\n" + "="*60)
    print("AR(1) Component Analysis")
    print("="*60)

    # Extract AR(1) parameters
    rho_samples = fit.stan_variable('rho')
    sigma_epsilon_samples = fit.stan_variable('sigma_epsilon')

    print(f"\nAR(1) Parameter Summary:")
    print(f"  ρ (autocorrelation): {np.mean(rho_samples):.3f} [{np.percentile(rho_samples, 2.5):.3f}, {np.percentile(rho_samples, 97.5):.3f}]")
    print(f"  σ_ε (innovation std): {np.mean(sigma_epsilon_samples):.3f} [{np.percentile(sigma_epsilon_samples, 2.5):.3f}, {np.percentile(sigma_epsilon_samples, 97.5):.3f}]")

    # Extract fitness effects
    gamma_s_samples = fit.stan_variable('gamma_s')
    gamma_a_samples = fit.stan_variable('gamma_a')

    print(f"\nFitness Effects Summary:")
    print(f"  γ_s (strength effect): {np.mean(gamma_s_samples):.3f} [{np.percentile(gamma_s_samples, 2.5):.3f}, {np.percentile(gamma_s_samples, 97.5):.3f}]")
    print(f"  γ_a (aerobic effect): {np.mean(gamma_a_samples):.3f} [{np.percentile(gamma_a_samples, 2.5):.3f}, {np.percentile(gamma_a_samples, 97.5):.3f}]")

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

    # Check if AR(1) variance is reasonable (< 30% of total)
    ar_proportion = ar_variance / total_variance
    if ar_proportion > 0.3:
        print(f"\n⚠️ WARNING: AR(1) component explains {100*ar_proportion:.1f}% of variance - still too high!")
        print("  Consider: stronger priors on σ_ε or ρ")
    elif ar_proportion > 0.15:
        print(f"\n⚠️ NOTE: AR(1) component explains {100*ar_proportion:.1f}% of variance - moderate")
        print("  This might be reasonable for residual autocorrelation")
    else:
        print(f"\n✓ GOOD: AR(1) component explains {100*ar_proportion:.1f}% of variance - reasonable")

    return {
        'rho': rho_samples,
        'sigma_epsilon': sigma_epsilon_samples,
        'gamma_s': gamma_s_samples,
        'gamma_a': gamma_a_samples,
        'epsilon_mean': epsilon_mean,
        'mu_no_ar_mean': mu_no_ar_mean,
        'variance_proportions': {
            'total': total_variance,
            'structural': structural_variance,
            'ar': ar_variance,
            'unexplained': total_variance - structural_variance - ar_variance
        }
    }


def create_visualizations(fit, analysis_results, df_weight):
    """Create visualizations of AR(1) component analysis."""
    print("\nCreating visualizations...")

    # Create output directory
    output_dir = Path("docs/fixed_ar_analysis")
    output_dir.mkdir(exist_ok=True)

    # 1. AR(1) parameter distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ρ distribution
    axes[0].hist(analysis_results['rho'], bins=30, density=True, alpha=0.7, edgecolor='black')
    axes[0].axvline(np.mean(analysis_results['rho']), color='red', linestyle='--', label=f'Mean: {np.mean(analysis_results["rho"]):.3f}')
    axes[0].set_xlabel('ρ (autocorrelation)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('AR(1) Autocorrelation Coefficient')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # σ_ε distribution
    axes[1].hist(analysis_results['sigma_epsilon'], bins=30, density=True, alpha=0.7, edgecolor='black')
    axes[1].axvline(np.mean(analysis_results['sigma_epsilon']), color='red', linestyle='--', label=f'Mean: {np.mean(analysis_results["sigma_epsilon"]):.3f}')
    axes[1].set_xlabel('σ_ε (innovation std)')
    axes[1].set_ylabel('Density')
    axes[1].set_title('AR(1) Innovation Standard Deviation')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'ar_parameters.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Fitness effects
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(analysis_results['gamma_s'], bins=30, density=True, alpha=0.7, edgecolor='black')
    axes[0].axvline(np.mean(analysis_results['gamma_s']), color='red', linestyle='--', label=f'Mean: {np.mean(analysis_results["gamma_s"]):.3f}')
    axes[0].axvline(0, color='black', linestyle='-', alpha=0.5)
    axes[0].set_xlabel('γ_s (strength effect)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Strength Fitness Effect on Weight')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(analysis_results['gamma_a'], bins=30, density=True, alpha=0.7, edgecolor='black')
    axes[1].axvline(np.mean(analysis_results['gamma_a']), color='red', linestyle='--', label=f'Mean: {np.mean(analysis_results["gamma_a"]):.3f}')
    axes[1].axvline(0, color='black', linestyle='-', alpha=0.5)
    axes[1].set_xlabel('γ_a (aerobic effect)')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Aerobic Fitness Effect on Weight')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'fitness_effects.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Time series of AR innovations
    fig, ax = plt.subplots(figsize=(14, 6))

    # Sort by timestamp
    df_sorted = df_weight.sort_values('timestamp')
    times = df_sorted['timestamp'].values

    ax.plot(times, analysis_results['epsilon_mean'], 'b-', alpha=0.7, label='AR(1) Innovations')
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel('Time')
    ax.set_ylabel('AR(1) Innovation Value')
    ax.set_title('AR(1) Innovations Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Format x-axis
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'ar_innovations_time_series.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Variance decomposition pie chart
    fig, ax = plt.subplots(figsize=(8, 8))

    variance_data = analysis_results['variance_proportions']
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

    ax.set_title('Variance Decomposition in Fixed AR(1) Model')
    plt.tight_layout()
    plt.savefig(output_dir / 'variance_decomposition.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Visualizations saved to {output_dir}/")

    return output_dir


def create_summary_report(analysis_results, output_dir):
    """Create a summary report of the analysis."""
    report_path = output_dir / 'analysis_summary.md'

    with open(report_path, 'w') as f:
        f.write("# Fixed AR(1) Model Analysis Summary\n\n")
        f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Key Changes from Original Model\n\n")
        f.write("1. **Removed local shrinkage parameters** (`lambda[i]`) - AR(1) now has single `sigma_epsilon`\n")
        f.write("2. **Stronger priors on AR(1) parameters**:\n")
        f.write("   - `rho` transformed beta prior (centered near 0)\n")
        f.write("   - `sigma_epsilon ~ exponential(5)` (mean=0.2)\n")
        f.write("3. **Simplified AR(1) process**: Standard AR(1) without horseshoe complications\n\n")

        f.write("## AR(1) Parameter Results\n\n")
        f.write(f"- ρ (autocorrelation): {np.mean(analysis_results['rho']):.3f} ")
        f.write(f"[{np.percentile(analysis_results['rho'], 2.5):.3f}, {np.percentile(analysis_results['rho'], 97.5):.3f}]\n")
        f.write(f"- σ_ε (innovation std): {np.mean(analysis_results['sigma_epsilon']):.3f} ")
        f.write(f"[{np.percentile(analysis_results['sigma_epsilon'], 2.5):.3f}, {np.percentile(analysis_results['sigma_epsilon'], 97.5):.3f}]\n\n")

        f.write("## Fitness Effects\n\n")
        f.write(f"- γ_s (strength effect): {np.mean(analysis_results['gamma_s']):.3f} ")
        f.write(f"[{np.percentile(analysis_results['gamma_s'], 2.5):.3f}, {np.percentile(analysis_results['gamma_s'], 97.5):.3f}]\n")
        f.write(f"- γ_a (aerobic effect): {np.mean(analysis_results['gamma_a']):.3f} ")
        f.write(f"[{np.percentile(analysis_results['gamma_a'], 2.5):.3f}, {np.percentile(analysis_results['gamma_a'], 97.5):.3f}]\n\n")

        f.write("## Variance Decomposition\n\n")
        variance_data = analysis_results['variance_proportions']
        f.write(f"- Total variance in weight data: {variance_data['total']:.4f}\n")
        f.write(f"- Structural model (fitness + spline): {variance_data['structural']:.4f} ")
        f.write(f"({100*variance_data['structural']/variance_data['total']:.1f}%)\n")
        f.write(f"- AR(1) component: {variance_data['ar']:.4f} ")
        f.write(f"({100*variance_data['ar']/variance_data['total']:.1f}%)\n")
        f.write(f"- Unexplained (noise + interaction): {variance_data['unexplained']:.4f} ")
        f.write(f"({100*variance_data['unexplained']/variance_data['total']:.1f}%)\n\n")

        ar_proportion = variance_data['ar'] / variance_data['total']
        if ar_proportion > 0.3:
            f.write("## ⚠️ WARNING\n\n")
            f.write(f"AR(1) component still explains {100*ar_proportion:.1f}% of variance - too high!\n")
            f.write("Consider even stronger priors on σ_ε or ρ.\n")
        elif ar_proportion > 0.15:
            f.write("## NOTE\n\n")
            f.write(f"AR(1) component explains {100*ar_proportion:.1f}% of variance - moderate.\n")
            f.write("This might be reasonable for residual autocorrelation.\n")
        else:
            f.write("## ✓ SUCCESS\n\n")
            f.write(f"AR(1) component explains {100*ar_proportion:.1f}% of variance - reasonable.\n")
            f.write("The fixed model successfully reduced AR(1) dominance.\n")

        f.write("\n## Generated Files\n\n")
        f.write("1. `ar_parameters.png` - Distributions of AR(1) parameters\n")
        f.write("2. `fitness_effects.png` - Distributions of fitness effects\n")
        f.write("3. `ar_innovations_time_series.png` - AR(1) innovations over time\n")
        f.write("4. `variance_decomposition.png` - Pie chart of variance proportions\n")
        f.write("5. `analysis_summary.md` - This summary file\n")

    print(f"✓ Summary report saved to {report_path}")


def main():
    """Main test function."""
    print("Testing fixed AR(1) model with simple AR component")
    print("=" * 60)

    # Prepare test data
    stan_data, df_weight, df_daily = prepare_test_data()

    print(f"\nData dimensions:")
    print(f"  Number of days (D): {stan_data['D']}")
    print(f"  Number of weight observations: {stan_data['N_weight']}")

    # Run the fixed model
    fit = run_fixed_model(stan_data)

    if fit is None:
        print("Failed to run model. Exiting.")
        return

    # Analyze AR component
    analysis_results = analyze_ar_component(fit, df_weight)

    # Create visualizations
    output_dir = create_visualizations(fit, analysis_results, df_weight)

    # Create summary report
    create_summary_report(analysis_results, output_dir)

    print("\n" + "="*60)
    print("Fixed AR(1) Model Test Complete")
    print("="*60)
    print(f"\nKey improvement: AR(1) now has single parameters (ρ, σ_ε)")
    print("instead of per-observation shrinkage parameters.")
    print(f"\nResults saved to: {output_dir}/")


if __name__ == "__main__":
    main()