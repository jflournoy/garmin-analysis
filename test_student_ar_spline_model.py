#!/usr/bin/env python3
"""Test symmetric model with Student-t likelihood, AR(1) correlation, and daily spline."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cmdstanpy
import scipy.stats as stats

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.data.weight import load_weight_data
from src.data.intensity import load_intensity_by_activity

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def prepare_data_with_hour():
    """Prepare data with hour_of_day for spline model."""
    print("Preparing data with hour_of_day for spline model...")

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
    df_daily = pd.merge(df_daily, df_act, on='date', how='left')

    # Fill missing values with 0
    df_daily['strength_training'] = df_daily['strength_training'].fillna(0.0)
    df_daily['walking'] = df_daily['walking'].fillna(0.0)
    df_daily['cycling'] = df_daily['cycling'].fillna(0.0)

    # Combine walking and cycling into aerobic intensity
    df_daily['aerobic_intensity'] = df_daily['walking'] + df_daily['cycling']

    # Standardize intensity (shift so min=0)
    for intensity_type in ['strength_training', 'aerobic_intensity']:
        min_val = df_daily[intensity_type].min()
        std = df_daily[intensity_type].std()
        if std > 0:
            df_daily[f'{intensity_type}_std'] = (df_daily[intensity_type] - min_val) / std
        else:
            df_daily[f'{intensity_type}_std'] = df_daily[intensity_type] - min_val

    # Standardize weight
    weight_mean = df_weight['weight_lbs'].mean()
    weight_std = df_weight['weight_lbs'].std()
    df_weight['weight_std'] = (df_weight['weight_lbs'] - weight_mean) / weight_std

    # Map weight observations to days
    df_weight['date'] = df_weight['timestamp'].dt.date
    date_to_idx = {date: i+1 for i, date in enumerate(date_range.date)}
    df_weight['day_idx'] = df_weight['date'].map(date_to_idx)
    df_weight = df_weight[df_weight['day_idx'].notna()]

    # Sort by timestamp for AR(1) process
    df_weight = df_weight.sort_values('timestamp').reset_index(drop=True)

    # Extract hour of day from timestamp
    df_weight['hour_of_day'] = df_weight['timestamp'].dt.hour + df_weight['timestamp'].dt.minute / 60.0

    print(f"\nHour of day statistics:")
    print(f"  Min: {df_weight['hour_of_day'].min():.2f}")
    print(f"  Max: {df_weight['hour_of_day'].max():.2f}")
    print(f"  Mean: {df_weight['hour_of_day'].mean():.2f}")
    print(f"  Std: {df_weight['hour_of_day'].std():.2f}")

    # Prepare Stan data
    K = 2  # Number of Fourier harmonics (24h and 12h cycles)
    stan_data = {
        'D': D,
        'strength_intensity': df_daily['strength_training_std'].values.astype(float),
        'aerobic_intensity': df_daily['aerobic_intensity_std'].values.astype(float),
        'N_weight': len(df_weight),
        'y_weight': df_weight['weight_std'].values.astype(float),
        'day_idx': df_weight['day_idx'].values.astype(int),
        'hour_of_day': df_weight['hour_of_day'].values.astype(float),
        'K': K,
    }

    print(f"\nData loaded: {len(df_daily)} days, {len(df_weight)} weight observations")
    print(f"Weight: mean={weight_mean:.2f} lbs, std={weight_std:.2f} lbs")
    print(f"Fourier harmonics: K={K} (24h and 12h cycles)")

    return stan_data, df_weight, df_daily, weight_mean, weight_std, K


def fit_model(model_file, stan_data, model_name):
    """Fit a model and return results."""
    print(f"\nFitting {model_name}...")
    print("-"*40)

    model = cmdstanpy.CmdStanModel(stan_file=model_file)

    fit = model.sample(
        data=stan_data,
        chains=2,
        iter_warmup=300,
        iter_sampling=300,
        adapt_delta=0.99,
        max_treedepth=15,
        show_progress=True,
        seed=12345,
    )

    # Check diagnostics
    try:
        diagnose = fit.diagnose()
        if "Divergent" in diagnose:
            print(f"WARNING: {model_name} has divergent transitions")
        if "maximum treedepth" in diagnose:
            print(f"WARNING: {model_name} hit maximum treedepth")
    except Exception as e:
        print(f"Could not run diagnose: {e}")

    return fit


def extract_results(fit, model_name, weight_std, K=0):
    """Extract key results from model fit."""
    draws_df = fit.draws_pd()

    # Key parameters to check
    key_params = ['gamma_s', 'gamma_a', 'weight_intercept', 'nu', 'rho', 'sigma_epsilon', 'sigma_fourier']

    results = {}
    for param in key_params:
        if param in draws_df.columns:
            mean_val = draws_df[param].mean()
            std_val = draws_df[param].std()
            ci_lower = np.percentile(draws_df[param], 2.5)
            ci_upper = np.percentile(draws_df[param], 97.5)
            results[param] = {
                'mean': mean_val,
                'std': std_val,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
            }

    # Extract Fourier coefficients if available
    if K > 0:
        for k in range(1, K+1):
            for coef_type in ['a_sin', 'a_cos']:
                param_name = f'{coef_type}[{k}]'
                if param_name in draws_df.columns:
                    mean_val = draws_df[param_name].mean()
                    results[f'{coef_type}_{k}'] = {'mean': mean_val}

    # Calculate weight effects in original units
    if 'gamma_s' in results and 'gamma_a' in results:
        gamma_s_lbs = results['gamma_s']['mean'] * weight_std
        gamma_a_lbs = results['gamma_a']['mean'] * weight_std

        results['gamma_s_lbs'] = gamma_s_lbs
        results['gamma_a_lbs'] = gamma_a_lbs

    # Calculate log likelihood for model comparison
    log_lik_cols = [col for col in draws_df.columns if col.startswith('log_lik_weight[')]
    if log_lik_cols:
        log_lik_vals = draws_df[log_lik_cols].values
        # Sum across observations for each sample
        log_lik_per_sample = np.sum(log_lik_vals, axis=1)
        results['log_lik_mean'] = np.mean(log_lik_per_sample)
        results['log_lik_std'] = np.std(log_lik_per_sample)

    results['model_name'] = model_name
    results['draws_df'] = draws_df

    return results


def compare_models(student_ar_results, student_ar_spline_results, K):
    """Compare Student-t AR vs Student-t AR Spline models."""
    print("\n" + "="*80)
    print("MODEL COMPARISON: Student-t AR vs Student-t AR Spline")
    print("="*80)

    print("\nParameter Comparison:")
    print("-"*80)
    print(f"{'Parameter':20} {'Student-t AR':>12} {'Student-t AR Spline':>12} {'Difference':>12}")
    print("-"*80)

    for param in ['gamma_s', 'gamma_a', 'weight_intercept', 'nu', 'rho', 'sigma_epsilon']:
        if param in student_ar_results and param in student_ar_spline_results:
            ar_mean = student_ar_results[param]['mean']
            spline_mean = student_ar_spline_results[param]['mean']
            diff = spline_mean - ar_mean
            print(f"{param:20} {ar_mean:12.3f} {spline_mean:12.3f} {diff:12.3f}")

    print("\nWeight Effects in Original Units (lbs per fitness unit):")
    print("-"*80)
    print(f"{'Model':25} {'Strength (lbs)':>15} {'Aerobic (lbs)':>15}")
    print("-"*80)

    for model_name, results in [('Student-t AR', student_ar_results), ('Student-t AR Spline', student_ar_spline_results)]:
        if 'gamma_s_lbs' in results and 'gamma_a_lbs' in results:
            print(f"{model_name:25} {results['gamma_s_lbs']:15.3f} {results['gamma_a_lbs']:15.3f}")

    # Model fit comparison
    print("\nModel Fit Comparison:")
    print("-"*80)

    if 'log_lik_mean' in student_ar_results and 'log_lik_mean' in student_ar_spline_results:
        ar_ll = student_ar_results['log_lik_mean']
        spline_ll = student_ar_spline_results['log_lik_mean']

        # Calculate WAIC approximation
        ar_waic = -2 * (ar_ll - student_ar_results['log_lik_std']**2)
        spline_waic = -2 * (spline_ll - student_ar_spline_results['log_lik_std']**2)

        print(f"Log-likelihood:")
        print(f"  Student-t AR: {ar_ll:.1f}")
        print(f"  Student-t AR Spline: {spline_ll:.1f}")
        print(f"  Difference: {spline_ll - ar_ll:.1f}")

        print(f"\nApproximate WAIC (lower is better):")
        print(f"  Student-t AR: {ar_waic:.1f}")
        print(f"  Student-t AR Spline: {spline_waic:.1f}")
        print(f"  Difference: {spline_waic - ar_waic:.1f}")

        if spline_waic < ar_waic:
            print(f"\n→ Student-t AR Spline model has better fit (ΔWAIC = {spline_waic - ar_waic:.1f})")
        else:
            print(f"\n→ Student-t AR model has better fit (ΔWAIC = {ar_waic - spline_waic:.1f})")

    # Student-t specific parameters
    if 'nu' in student_ar_spline_results:
        nu_mean = student_ar_spline_results['nu']['mean']
        print(f"\nStudent-t degrees of freedom (ν): {nu_mean:.1f}")
        if nu_mean > 30:
            print(f"  → Effectively normal (ν > 30)")
        elif nu_mean > 10:
            print(f"  → Slightly heavy-tailed")
        elif nu_mean > 4:
            print(f"  → Moderately heavy-tailed")
        else:
            print(f"  → Very heavy-tailed")

    if 'rho' in student_ar_spline_results:
        rho_mean = student_ar_spline_results['rho']['mean']
        rho_ci_lower = student_ar_spline_results['rho']['ci_lower']
        rho_ci_upper = student_ar_spline_results['rho']['ci_upper']
        print(f"\nAR(1) correlation coefficient (ρ):")
        print(f"  Mean: {rho_mean:.3f}")
        print(f"  95% CI: [{rho_ci_lower:.3f}, {rho_ci_upper:.3f}]")
        if rho_ci_lower > 0:
            print(f"  → Significant positive autocorrelation")
        elif rho_ci_upper < 0:
            print(f"  → Significant negative autocorrelation")
        else:
            print(f"  → No strong evidence for autocorrelation")

    # Fourier coefficients
    if K > 0:
        print(f"\nFourier Coefficients (K={K}):")
        print(f"  sigma_fourier: {student_ar_spline_results.get('sigma_fourier', {}).get('mean', np.nan):.3f}")
        for k in range(1, K+1):
            for coef_type in ['a_sin', 'a_cos']:
                param_name = f'{coef_type}_{k}'
                if param_name in student_ar_spline_results:
                    val = student_ar_spline_results[param_name]['mean']
                    print(f"  {coef_type}[{k}]: {val:.3f}")


def analyze_spline_residuals(student_ar_spline_results, df_weight):
    """Analyze residuals from Student-t AR Spline model."""
    print("\n" + "="*80)
    print("STUDENT-T AR SPLINE MODEL RESIDUAL ANALYSIS")
    print("="*80)

    draws_df = student_ar_spline_results['draws_df']

    # Extract residual columns
    resid_cols = [col for col in draws_df.columns if col.startswith('residual[')]
    if not resid_cols:
        print("No residual columns found in output")
        return None

    # Sort columns by index
    resid_cols_sorted = sorted(resid_cols, key=lambda x: int(x.split('[')[1].split(']')[0]))

    # Calculate mean residual for each observation
    mean_residuals = np.array([draws_df[col].mean() for col in resid_cols_sorted])

    print(f"\nResidual statistics:")
    print(f"  Number of residuals: {len(mean_residuals)}")
    print(f"  Mean: {np.mean(mean_residuals):.3f} (should be ~0)")
    print(f"  Std: {np.std(mean_residuals):.3f}")
    print(f"  Min/Max: [{np.min(mean_residuals):.3f}, {np.max(mean_residuals):.3f}]")

    # Normality test on residuals
    if len(mean_residuals) < 5000:
        shapiro_stat, shapiro_p = stats.shapiro(mean_residuals)
        print(f"\nNormality test on mean residuals:")
        print(f"  Shapiro-Wilk: W = {shapiro_stat:.3f}, p = {shapiro_p:.3e}")
        if shapiro_p < 0.05:
            print(f"  → REJECT normality at α=0.05")
        else:
            print(f"  → FAIL TO REJECT normality at α=0.05")

    # Autocorrelation in residuals
    if len(mean_residuals) > 1:
        lag1_corr = np.corrcoef(mean_residuals[:-1], mean_residuals[1:])[0, 1]
        print(f"\nResidual autocorrelation (after modeling AR(1)):")
        print(f"  Lag-1 autocorrelation: {lag1_corr:.3f}")

        # Compare with estimated rho
        if 'rho' in student_ar_spline_results:
            rho_mean = student_ar_spline_results['rho']['mean']
            print(f"  Model estimated ρ: {rho_mean:.3f}")
            print(f"  Difference: {lag1_corr - rho_mean:.3f}")

    return mean_residuals


def create_spline_visualization(student_ar_spline_results, df_weight, K, mean_residuals):
    """Create visualization of spline model results."""
    print("\nCreating spline model visualization...")

    output_dir = Path("output/student_ar_spline_comparison")
    output_dir.mkdir(exist_ok=True, parents=True)

    draws_df = student_ar_spline_results['draws_df']

    # Extract spline component
    spline_cols = [col for col in draws_df.columns if col.startswith('f_daily_stored[')]
    if spline_cols:
        spline_cols_sorted = sorted(spline_cols, key=lambda x: int(x.split('[')[1].split(']')[0]))
        mean_spline = np.array([draws_df[col].mean() for col in spline_cols_sorted])

        # Plot spline vs hour of day
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Student-t AR Spline Model Analysis', fontsize=16)

        # 1. Spline component vs hour of day
        ax = axes[0, 0]
        ax.scatter(df_weight['hour_of_day'], mean_spline, alpha=0.6, s=20)
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Spline Component (standardized)')
        ax.set_title('Daily Spline Component')
        ax.grid(True, alpha=0.3)

        # Sort by hour for smoother plot
        hour_sorted_idx = np.argsort(df_weight['hour_of_day'].values)
        ax.plot(df_weight['hour_of_day'].values[hour_sorted_idx],
                mean_spline[hour_sorted_idx], 'r-', alpha=0.7, linewidth=2)

        # 2. Residual distribution
        ax = axes[0, 1]
        if mean_residuals is not None:
            ax.hist(mean_residuals, bins=20, density=True, alpha=0.7,
                   color='green', edgecolor='black')

            # Overlay normal distribution
            xmin, xmax = ax.get_xlim()
            x = np.linspace(xmin, xmax, 100)
            normal_pdf = stats.norm.pdf(x, np.mean(mean_residuals), np.std(mean_residuals))
            ax.plot(x, normal_pdf, 'k-', linewidth=2, label='Normal fit')

            ax.axvline(0, color='red', linestyle='--', alpha=0.7)
            ax.set_xlabel('Residuals (standardized)')
            ax.set_ylabel('Density')
            ax.set_title('Residual Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 3. Residuals vs hour of day
        ax = axes[0, 2]
        if mean_residuals is not None:
            ax.scatter(df_weight['hour_of_day'], mean_residuals, alpha=0.6, s=20)
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            ax.set_xlabel('Hour of Day')
            ax.set_ylabel('Residuals (standardized)')
            ax.set_title('Residuals vs Hour of Day')
            ax.grid(True, alpha=0.3)

        # 4. Student-t degrees of freedom
        ax = axes[1, 0]
        if 'nu' in student_ar_spline_results:
            nu_samples = draws_df['nu'].values
            ax.hist(nu_samples, bins=30, density=True, alpha=0.7, color='purple')
            ax.axvline(student_ar_spline_results['nu']['mean'], color='red', linestyle='--',
                      label=f'Mean: {student_ar_spline_results["nu"]["mean"]:.1f}')
            ax.axvline(30, color='black', linestyle=':', alpha=0.5, label='ν=30 (≈normal)')
            ax.set_xlabel('Degrees of freedom (ν)')
            ax.set_ylabel('Density')
            ax.set_title('Student-t Degrees of Freedom')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 5. AR(1) correlation coefficient
        ax = axes[1, 1]
        if 'rho' in student_ar_spline_results:
            rho_samples = draws_df['rho'].values
            ax.hist(rho_samples, bins=30, density=True, alpha=0.7, color='orange')
            ax.axvline(student_ar_spline_results['rho']['mean'], color='red', linestyle='--',
                      label=f'Mean: {student_ar_spline_results["rho"]["mean"]:.3f}')
            ax.axvline(0, color='black', linestyle='-', alpha=0.5)
            ax.set_xlabel('AR(1) coefficient (ρ)')
            ax.set_ylabel('Density')
            ax.set_title('Autocorrelation Coefficient')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 6. Summary statistics
        ax = axes[1, 2]
        ax.axis('off')

        summary_text = "STUDENT-T AR SPLINE MODEL\n"
        summary_text += "="*40 + "\n\n"

        # Add parameter estimates
        summary_text += "Key Parameters:\n"
        summary_text += "-"*20 + "\n"

        for param in ['gamma_s', 'gamma_a', 'weight_intercept', 'nu', 'rho']:
            if param in student_ar_spline_results:
                mean_val = student_ar_spline_results[param]['mean']
                summary_text += f"{param}: {mean_val:.3f}\n"

        # Add Fourier coefficients
        if K > 0:
            summary_text += f"\nFourier Coefficients (K={K}):\n"
            summary_text += "-"*20 + "\n"
            for k in range(1, K+1):
                for coef_type in ['a_sin', 'a_cos']:
                    param_name = f'{coef_type}_{k}'
                    if param_name in student_ar_spline_results:
                        val = student_ar_spline_results[param_name]['mean']
                        summary_text += f"{coef_type}[{k}]: {val:.3f}\n"

        ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
               verticalalignment='center', fontsize=9, fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(output_dir / 'spline_model_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Visualization saved to {output_dir}/spline_model_analysis.png")


def main():
    """Main comparison function."""
    print("="*80)
    print("MODEL COMPARISON: Student-t AR vs Student-t AR Spline")
    print("="*80)

    # Prepare data with hour_of_day
    stan_data, df_weight, df_daily, weight_mean, weight_std, K = prepare_data_with_hour()

    # First, we need Student-t AR results for comparison
    # Let's run it with the same data (without hour_of_day for the non-spline version)
    print("\n" + "="*80)
    print("First, running Student-t AR model for comparison...")
    print("="*80)

    # Create data without hour_of_day for non-spline model
    stan_data_no_spline = stan_data.copy()
    # Remove hour_of_day and K for non-spline model
    stan_data_no_spline.pop('hour_of_day', None)
    stan_data_no_spline.pop('K', None)

    # Fit Student-t AR model
    student_ar_fit = fit_model(
        "stan/weight_state_space_training_decay_aerobic_symmetric_student_ar.stan",
        stan_data_no_spline,
        "Student-t AR(1) Model"
    )
    student_ar_results = extract_results(student_ar_fit, "Student-t AR", weight_std)

    # Fit Student-t AR Spline model
    print("\n" + "="*80)
    print("Now running Student-t AR Spline model...")
    print("="*80)

    student_ar_spline_fit = fit_model(
        "stan/weight_state_space_training_decay_aerobic_symmetric_student_ar_spline.stan",
        stan_data,
        "Student-t AR(1) Spline Model"
    )
    student_ar_spline_results = extract_results(student_ar_spline_fit, "Student-t AR Spline", weight_std, K)

    # Compare models
    compare_models(student_ar_results, student_ar_spline_results, K)

    # Analyze Student-t AR Spline residuals
    mean_residuals = analyze_spline_residuals(student_ar_spline_results, df_weight)

    # Create visualization
    create_spline_visualization(student_ar_spline_results, df_weight, K, mean_residuals)

    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)

    return student_ar_results, student_ar_spline_results


if __name__ == "__main__":
    student_ar_results, student_ar_spline_results = main()