#!/usr/bin/env python3
"""Test symmetric model with Student-t likelihood and AR(1) correlation."""

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


def prepare_data():
    """Prepare data for model comparison."""
    print("Preparing data for model comparison...")

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

    # Prepare Stan data
    stan_data = {
        'D': D,
        'strength_intensity': df_daily['strength_training_std'].values.astype(float),
        'aerobic_intensity': df_daily['aerobic_intensity_std'].values.astype(float),
        'N_weight': len(df_weight),
        'y_weight': df_weight['weight_std'].values.astype(float),
        'day_idx': df_weight['day_idx'].values.astype(int),
    }

    print(f"Data loaded: {len(df_daily)} days, {len(df_weight)} weight observations")
    print(f"Weight: mean={weight_mean:.2f} lbs, std={weight_std:.2f} lbs")

    return stan_data, df_weight, df_daily, weight_mean, weight_std


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


def extract_results(fit, model_name, weight_std):
    """Extract key results from model fit."""
    draws_df = fit.draws_pd()

    # Key parameters to check
    key_params = ['gamma_s', 'gamma_a', 'weight_intercept', 'nu', 'rho', 'sigma_epsilon']

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


def compare_models(original_results, student_ar_results):
    """Compare original vs Student-t AR model."""
    print("\n" + "="*80)
    print("MODEL COMPARISON: Original vs Student-t AR(1)")
    print("="*80)

    print("\nParameter Comparison:")
    print("-"*80)
    print(f"{'Parameter':15} {'Original':>12} {'Student-t AR':>12} {'Difference':>12}")
    print("-"*80)

    for param in ['gamma_s', 'gamma_a', 'weight_intercept']:
        if param in original_results and param in student_ar_results:
            orig_mean = original_results[param]['mean']
            stu_mean = student_ar_results[param]['mean']
            diff = stu_mean - orig_mean
            print(f"{param:15} {orig_mean:12.3f} {stu_mean:12.3f} {diff:12.3f}")

    print("\nWeight Effects in Original Units (lbs per fitness unit):")
    print("-"*80)
    print(f"{'Model':20} {'Strength (lbs)':>15} {'Aerobic (lbs)':>15}")
    print("-"*80)

    for model_name, results in [('Original', original_results), ('Student-t AR', student_ar_results)]:
        if 'gamma_s_lbs' in results and 'gamma_a_lbs' in results:
            print(f"{model_name:20} {results['gamma_s_lbs']:15.3f} {results['gamma_a_lbs']:15.3f}")

    # Model fit comparison
    print("\nModel Fit Comparison:")
    print("-"*80)

    if 'log_lik_mean' in original_results and 'log_lik_mean' in student_ar_results:
        orig_ll = original_results['log_lik_mean']
        stu_ll = student_ar_results['log_lik_mean']

        # Calculate WAIC approximation
        orig_waic = -2 * (orig_ll - original_results['log_lik_std']**2)
        stu_waic = -2 * (stu_ll - student_ar_results['log_lik_std']**2)

        print(f"Log-likelihood:")
        print(f"  Original: {orig_ll:.1f}")
        print(f"  Student-t AR: {stu_ll:.1f}")
        print(f"  Difference: {stu_ll - orig_ll:.1f}")

        print(f"\nApproximate WAIC (lower is better):")
        print(f"  Original: {orig_waic:.1f}")
        print(f"  Student-t AR: {stu_waic:.1f}")
        print(f"  Difference: {stu_waic - orig_waic:.1f}")

        if stu_waic < orig_waic:
            print(f"\n→ Student-t AR model has better fit (ΔWAIC = {stu_waic - orig_waic:.1f})")
        else:
            print(f"\n→ Original model has better fit (ΔWAIC = {orig_waic - stu_waic:.1f})")

    # Student-t specific parameters
    if 'nu' in student_ar_results:
        nu_mean = student_ar_results['nu']['mean']
        print(f"\nStudent-t degrees of freedom (ν): {nu_mean:.1f}")
        if nu_mean > 30:
            print(f"  → Effectively normal (ν > 30)")
        elif nu_mean > 10:
            print(f"  → Slightly heavy-tailed")
        elif nu_mean > 4:
            print(f"  → Moderately heavy-tailed")
        else:
            print(f"  → Very heavy-tailed")

    if 'rho' in student_ar_results:
        rho_mean = student_ar_results['rho']['mean']
        rho_ci_lower = student_ar_results['rho']['ci_lower']
        rho_ci_upper = student_ar_results['rho']['ci_upper']
        print(f"\nAR(1) correlation coefficient (ρ):")
        print(f"  Mean: {rho_mean:.3f}")
        print(f"  95% CI: [{rho_ci_lower:.3f}, {rho_ci_upper:.3f}]")
        if rho_ci_lower > 0:
            print(f"  → Significant positive autocorrelation")
        elif rho_ci_upper < 0:
            print(f"  → Significant negative autocorrelation")
        else:
            print(f"  → No strong evidence for autocorrelation")


def analyze_student_ar_residuals(student_ar_results, df_weight):
    """Analyze residuals from Student-t AR model."""
    print("\n" + "="*80)
    print("STUDENT-T AR MODEL RESIDUAL ANALYSIS")
    print("="*80)

    draws_df = student_ar_results['draws_df']

    # Extract residual columns
    resid_cols = [col for col in draws_df.columns if col.startswith('residual[')]
    if not resid_cols:
        print("No residual columns found in output")
        return

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
        if 'rho' in student_ar_results:
            rho_mean = student_ar_results['rho']['mean']
            print(f"  Model estimated ρ: {rho_mean:.3f}")
            print(f"  Difference: {lag1_corr - rho_mean:.3f}")

    return mean_residuals


def create_comparison_visualization(original_results, student_ar_results, mean_residuals):
    """Create visualization comparing models."""
    print("\nCreating comparison visualization...")

    output_dir = Path("output/student_ar_comparison")
    output_dir.mkdir(exist_ok=True, parents=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Model Comparison: Original vs Student-t AR(1)', fontsize=16)

    # 1. Gamma_s comparison
    ax = axes[0, 0]
    models = ['Original', 'Student-t AR']
    gamma_s_means = []
    gamma_s_cis = []

    for model_name, results in [('Original', original_results), ('Student-t AR', student_ar_results)]:
        if 'gamma_s' in results:
            gamma_s_means.append(results['gamma_s']['mean'])
            gamma_s_cis.append([
                results['gamma_s']['ci_lower'],
                results['gamma_s']['ci_upper']
            ])

    if gamma_s_means:
        y_pos = np.arange(len(models))
        ax.errorbar(gamma_s_means, y_pos,
                   xerr=[[m - ci[0] for m, ci in zip(gamma_s_means, gamma_s_cis)],
                         [ci[1] - m for m, ci in zip(gamma_s_means, gamma_s_cis)]],
                   fmt='o', capsize=5, color='blue')
        ax.axvline(0, color='black', linestyle='-', alpha=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(models)
        ax.set_xlabel('gamma_s (strength effect)')
        ax.set_title('Strength Effect Comparison')
        ax.grid(True, alpha=0.3)

    # 2. Gamma_a comparison
    ax = axes[0, 1]
    gamma_a_means = []
    gamma_a_cis = []

    for model_name, results in [('Original', original_results), ('Student-t AR', student_ar_results)]:
        if 'gamma_a' in results:
            gamma_a_means.append(results['gamma_a']['mean'])
            gamma_a_cis.append([
                results['gamma_a']['ci_lower'],
                results['gamma_a']['ci_upper']
            ])

    if gamma_a_means:
        y_pos = np.arange(len(models))
        ax.errorbar(gamma_a_means, y_pos,
                   xerr=[[m - ci[0] for m, ci in zip(gamma_a_means, gamma_a_cis)],
                         [ci[1] - m for m, ci in zip(gamma_a_means, gamma_a_cis)]],
                   fmt='o', capsize=5, color='green')
        ax.axvline(0, color='black', linestyle='-', alpha=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(models)
        ax.set_xlabel('gamma_a (aerobic effect)')
        ax.set_title('Aerobic Effect Comparison')
        ax.grid(True, alpha=0.3)

    # 3. Student-t degrees of freedom
    ax = axes[0, 2]
    if 'nu' in student_ar_results:
        nu_samples = student_ar_results['draws_df']['nu'].values
        ax.hist(nu_samples, bins=30, density=True, alpha=0.7, color='purple')
        ax.axvline(student_ar_results['nu']['mean'], color='red', linestyle='--',
                  label=f'Mean: {student_ar_results["nu"]["mean"]:.1f}')
        ax.axvline(30, color='black', linestyle=':', alpha=0.5, label='ν=30 (≈normal)')
        ax.set_xlabel('Degrees of freedom (ν)')
        ax.set_ylabel('Density')
        ax.set_title('Student-t Degrees of Freedom')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 4. AR(1) correlation coefficient
    ax = axes[1, 0]
    if 'rho' in student_ar_results:
        rho_samples = student_ar_results['draws_df']['rho'].values
        ax.hist(rho_samples, bins=30, density=True, alpha=0.7, color='orange')
        ax.axvline(student_ar_results['rho']['mean'], color='red', linestyle='--',
                  label=f'Mean: {student_ar_results["rho"]["mean"]:.3f}')
        ax.axvline(0, color='black', linestyle='-', alpha=0.5)
        ax.set_xlabel('AR(1) coefficient (ρ)')
        ax.set_ylabel('Density')
        ax.set_title('Autocorrelation Coefficient')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 5. Residual distribution from Student-t AR model
    ax = axes[1, 1]
    if mean_residuals is not None:
        ax.hist(mean_residuals, bins=20, density=True, alpha=0.7, color='green', edgecolor='black')

        # Overlay normal distribution
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        normal_pdf = stats.norm.pdf(x, np.mean(mean_residuals), np.std(mean_residuals))
        ax.plot(x, normal_pdf, 'k-', linewidth=2, label='Normal fit')

        ax.axvline(0, color='red', linestyle='--', alpha=0.7)
        ax.set_xlabel('Residuals (standardized)')
        ax.set_ylabel('Density')
        ax.set_title('Student-t AR Model Residuals')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 6. Model fit comparison
    ax = axes[1, 2]
    ax.axis('off')

    summary_text = "MODEL COMPARISON SUMMARY\n"
    summary_text += "="*40 + "\n\n"

    # Add parameter comparison
    summary_text += "Parameter Estimates:\n"
    summary_text += "-"*20 + "\n"

    for param in ['gamma_s', 'gamma_a']:
        if param in original_results and param in student_ar_results:
            orig = original_results[param]['mean']
            stu = student_ar_results[param]['mean']
            diff = stu - orig
            pct_diff = (diff / orig * 100) if orig != 0 else 0

            summary_text += f"{param}:\n"
            summary_text += f"  Original: {orig:.3f}\n"
            summary_text += f"  Student-t AR: {stu:.3f}\n"
            summary_text += f"  Δ: {diff:.3f} ({pct_diff:.1f}%)\n\n"

    # Add model fit
    if 'log_lik_mean' in original_results and 'log_lik_mean' in student_ar_results:
        orig_ll = original_results['log_lik_mean']
        stu_ll = student_ar_results['log_lik_mean']
        ll_diff = stu_ll - orig_ll

        summary_text += "Model Fit:\n"
        summary_text += "-"*20 + "\n"
        summary_text += f"Log-likelihood:\n"
        summary_text += f"  Original: {orig_ll:.1f}\n"
        summary_text += f"  Student-t AR: {stu_ll:.1f}\n"
        summary_text += f"  Δ: {ll_diff:.1f}\n"

        if ll_diff > 0:
            summary_text += f"  → Student-t AR fits better\n"
        else:
            summary_text += f"  → Original fits better\n"

    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
           verticalalignment='center', fontsize=9, fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Visualization saved to {output_dir}/model_comparison.png")


def main():
    """Main comparison function."""
    print("="*80)
    print("MODEL COMPARISON: Original vs Student-t AR(1)")
    print("="*80)

    # Prepare data
    stan_data, df_weight, df_daily, weight_mean, weight_std = prepare_data()

    # Fit original model
    original_fit = fit_model(
        "stan/weight_state_space_training_decay_aerobic_symmetric.stan",
        stan_data,
        "Original Symmetric Model"
    )
    original_results = extract_results(original_fit, "Original", weight_std)

    # Fit Student-t AR model
    student_ar_fit = fit_model(
        "stan/weight_state_space_training_decay_aerobic_symmetric_student_ar.stan",
        stan_data,
        "Student-t AR(1) Model"
    )
    student_ar_results = extract_results(student_ar_fit, "Student-t AR", weight_std)

    # Compare models
    compare_models(original_results, student_ar_results)

    # Analyze Student-t AR residuals
    mean_residuals = analyze_student_ar_residuals(student_ar_results, df_weight)

    # Create visualization
    create_comparison_visualization(original_results, student_ar_results, mean_residuals)

    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)

    return original_results, student_ar_results


if __name__ == "__main__":
    original_results, student_ar_results = main()