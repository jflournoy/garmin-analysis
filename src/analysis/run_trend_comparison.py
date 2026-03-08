#!/usr/bin/env python3
"""Run trend model and compare posteriors with original model.

Depends on Issue #11 (trend model creation). Runner script with data loading,
MCMC, side-by-side comparison table, density plots, LOO-CV via arviz,
and interpretation.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import cmdstanpy
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.weight import load_weight_data
from src.data.intensity import load_intensity_by_activity

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def prepare_data_with_4h_intervals():
    """Prepare data with 4-hour prediction intervals (reused from generate_component_predictions.py)."""
    print("Preparing data with 4-hour prediction intervals...")

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

    # Prepare Stan data with 4-hour intervals
    H = 7
    pred_hours = np.array([0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0])
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

    return stan_data, df_weight, df_daily, weight_mean, weight_std


def _extract_and_save_trend_metadata(fit):
    """Extract posterior metadata from trend model fit and save for report generation."""
    try:
        summaries = fit.summary()

        # Parameters to extract (including delta for trend model)
        scalar_params = [
            'delta', 'rho', 'sigma_epsilon', 'gamma_s', 'gamma_a', 'weight_intercept', 'nu',
            'sigma_w', 'sigma_fourier', 'beta_s', 'beta_a',
            'alpha_d_s', 'alpha_m_s', 'alpha_d_a', 'alpha_m_a'
        ]

        posterior_metadata = {}

        # Extract each parameter
        for param in scalar_params:
            if param in summaries.index:
                row = summaries.loc[param]
                posterior_metadata[param] = {
                    'mean': float(row['Mean']),
                    'sd': float(row['StdDev']),
                    'lower_ci': float(row['2.5%']) if '2.5%' in row.index else float('nan'),
                    'upper_ci': float(row['97.5%']) if '97.5%' in row.index else float('nan'),
                    'rhat': float(row['R_hat']) if not np.isnan(row['R_hat']) else None,
                    'ess_bulk': float(row['Bulk_ESS']) if not np.isnan(row['Bulk_ESS']) else None,
                    'ess_tail': float(row['Tail_ESS']) if not np.isnan(row['Tail_ESS']) else None,
                }

        # MCMC diagnostics
        diagnostics = {
            'num_chains': fit.num_chains,
            'num_draws_sampling': fit.num_draws_sampling,
            'num_draws_warmup': fit.num_draws_warmup,
        }

        metadata = {
            'posterior': posterior_metadata,
            'diagnostics': diagnostics,
        }

        # Save to JSON in docs directory
        output_dir = Path(__file__).parent.parent.parent / 'docs' / 'component_predictions'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / 'posterior_metadata_trend.json'

        with open(output_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n✓ Extracted trend model posterior metadata")
        print(f"  Output: {output_file}")

    except Exception as e:
        print(f"\n⚠ Warning: Could not extract trend metadata: {e}")


def run_trend_model(stan_data):
    """Run the trend model."""
    print("\n" + "="*60)
    print("Running Trend Model (with linear time trend)")
    print("="*60)

    model_path = "stan/weight_state_space_training_decay_aerobic_symmetric_student_ar_spline_constrained_trend.stan"

    try:
        model = cmdstanpy.CmdStanModel(stan_file=model_path)
        print(f"✓ Trend model compiled successfully")
    except Exception as e:
        print(f"✗ Trend model compilation failed: {e}")
        return None

    print("Running MCMC sampling...")
    try:
        # Set output directory for chain files
        output_dir = Path(__file__).parent.parent.parent / 'output' / 'trend_model_current'
        output_dir.mkdir(parents=True, exist_ok=True)

        fit = model.sample(
            data=stan_data,
            chains=4,
            parallel_chains=4,
            iter_warmup=500,
            iter_sampling=500,
            seed=12345,
            adapt_delta=0.95,
            max_treedepth=12,
            output_dir=str(output_dir),
            show_console=False,
            show_progress=True
        )

        # Check for convergence warnings
        if fit.diagnose() is not None:
            print("⚠ Warning: Convergence issues detected in trend model sampling")

        # Extract and save metadata for report generation
        _extract_and_save_trend_metadata(fit)

        return fit
    except Exception as e:
        print(f"✗ Trend model sampling failed: {e}")
        return None


def load_original_model_fit(stan_data):
    """Load or re-run the original (no-trend) model."""
    print("\n" + "="*60)
    print("Loading Original Model (without linear time trend)")
    print("="*60)

    model_path = "stan/weight_state_space_training_decay_aerobic_symmetric_student_ar_spline_constrained.stan"

    try:
        model = cmdstanpy.CmdStanModel(stan_file=model_path)
        print(f"✓ Original model compiled successfully")
    except Exception as e:
        print(f"✗ Original model compilation failed: {e}")
        return None

    print("Running MCMC sampling...")
    try:
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

        # Check for convergence warnings
        if fit.diagnose() is not None:
            print("⚠ Warning: Convergence issues detected in original model sampling")

        return fit
    except Exception as e:
        print(f"✗ Original model sampling failed: {e}")
        return None


def extract_posteriors(fit, model_name=""):
    """Extract posterior summaries from fit object."""
    summaries = fit.summary()

    posteriors = {}
    params = ['rho', 'sigma_epsilon', 'gamma_s', 'gamma_a', 'weight_intercept', 'nu',
              'sigma_w', 'sigma_fourier', 'beta_s', 'beta_a',
              'alpha_d_s', 'alpha_m_s', 'alpha_d_a', 'alpha_m_a']

    for param in params:
        if param in summaries.index:
            row = summaries.loc[param]
            try:
                posteriors[param] = {
                    'mean': float(row['Mean']),
                    'sd': float(row['StdDev']),
                    'lower_ci': float(row['2.5%']) if '2.5%' in row.index else float('nan'),
                    'upper_ci': float(row['97.5%']) if '97.5%' in row.index else float('nan'),
                }
            except (KeyError, ValueError) as e:
                print(f"  Warning: Could not extract posterior for {param}: {e}")
                continue

    # If trend parameter exists, add it
    if 'delta' in summaries.index:
        row = summaries.loc['delta']
        try:
            posteriors['delta'] = {
                'mean': float(row['Mean']),
                'sd': float(row['StdDev']),
                'lower_ci': float(row['2.5%']) if '2.5%' in row.index else float('nan'),
                'upper_ci': float(row['97.5%']) if '97.5%' in row.index else float('nan'),
            }
        except (KeyError, ValueError) as e:
            print(f"  Warning: Could not extract posterior for delta: {e}")

    return posteriors


def print_comparison_table(posteriors_no_trend, posteriors_trend):
    """Print side-by-side posterior comparison table."""
    print("\n" + "="*90)
    print("POSTERIOR COMPARISON TABLE")
    print("="*90)

    params = ['delta', 'gamma_s', 'gamma_a', 'rho', 'sigma_epsilon', 'weight_intercept',
              'nu', 'sigma_w', 'sigma_fourier', 'beta_s', 'beta_a',
              'alpha_d_s', 'alpha_m_s', 'alpha_d_a', 'alpha_m_a']

    print(f"\n{'Parameter':<20} {'No-Trend Model':<30} {'Trend Model':<30}")
    print("-" * 90)

    for param in params:
        no_trend_str = "N/A"
        trend_str = "N/A"

        if param in posteriors_no_trend:
            p = posteriors_no_trend[param]
            no_trend_str = f"{p['mean']:7.4f} [{p['lower_ci']:7.4f}, {p['upper_ci']:7.4f}]"

        if param in posteriors_trend:
            p = posteriors_trend[param]
            trend_str = f"{p['mean']:7.4f} [{p['lower_ci']:7.4f}, {p['upper_ci']:7.4f}]"

        print(f"{param:<20} {no_trend_str:<30} {trend_str:<30}")


def plot_gamma_s_comparison(fit_no_trend, fit_trend):
    """Plot overlay posterior density for gamma_s under both models."""
    print("\nGenerating posterior density plot for gamma_s...")

    gamma_s_no_trend = fit_no_trend.stan_variable('gamma_s').flatten()
    gamma_s_trend = fit_trend.stan_variable('gamma_s').flatten()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot densities
    ax.hist(gamma_s_no_trend, bins=30, alpha=0.5, density=True, label='No-Trend Model', color='blue')
    ax.hist(gamma_s_trend, bins=30, alpha=0.5, density=True, label='Trend Model', color='red')

    ax.set_xlabel('γ_s (Strength Effect on Weight)')
    ax.set_ylabel('Posterior Density')
    ax.set_title('Posterior Distribution of Strength Effect (γ_s)\nComparing Models With and Without Linear Trend')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Save plot
    output_dir = Path(__file__).parent.parent.parent / 'docs' / 'trend_comparison'
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_file = output_dir / 'gamma_s_comparison.png'

    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved density plot to {plot_file}")
    plt.close()


def compute_loo_comparison(fit_no_trend, fit_trend):
    """Compute LOO-CV comparison using arviz."""
    print("\n" + "="*60)
    print("LOO-CV Model Comparison")
    print("="*60)

    try:
        # Extract log likelihood from both models
        log_lik_no_trend = fit_no_trend.stan_variable('log_lik_weight')
        log_lik_trend = fit_trend.stan_variable('log_lik_weight')

        # Compute LOO
        loo_no_trend = az.loo(log_lik_no_trend)
        loo_trend = az.loo(log_lik_trend)

        print(f"\nNo-Trend Model:")
        print(f"  ELPD LOO: {loo_no_trend.elpd_loo:.2f}")
        print(f"  SE: {loo_no_trend.se_elpd_loo:.2f}")

        print(f"\nTrend Model:")
        print(f"  ELPD LOO: {loo_trend.elpd_loo:.2f}")
        print(f"  SE: {loo_trend.se_elpd_loo:.2f}")

        # Comparison
        elpd_diff = loo_trend.elpd_loo - loo_no_trend.elpd_loo
        se_diff = np.sqrt(loo_trend.se_elpd_loo**2 + loo_no_trend.se_elpd_loo**2)

        print(f"\nComparison (Trend - No-Trend):")
        print(f"  ELPD difference: {elpd_diff:.2f}")
        print(f"  SE difference: {se_diff:.2f}")

        if elpd_diff > 0:
            print(f"  ✓ Trend model has BETTER fit (higher ELPD)")
        elif elpd_diff < -se_diff:
            print(f"  ✗ No-trend model has CLEARLY BETTER fit")
        else:
            print(f"  ≈ Models have similar fit (difference within SE)")

    except Exception as e:
        print(f"Warning: LOO-CV computation failed: {e}")


def save_comparison_json(posteriors_no_trend, posteriors_trend, stan_data):
    """Save comparison to JSON file."""
    comparison = {
        'data_summary': {
            'D': stan_data['D'],
            'N_weight': stan_data['N_weight'],
        },
        'posteriors_no_trend': posteriors_no_trend,
        'posteriors_trend': posteriors_trend,
    }

    output_dir = Path(__file__).parent.parent.parent / 'docs' / 'trend_comparison'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'comparison_results.json'

    with open(output_file, 'w') as f:
        json.dump(comparison, f, indent=2)

    print(f"✓ Saved comparison to {output_file}")


def main():
    """Main entry point."""
    print("\n" + "="*60)
    print("TREND MODEL INVESTIGATION")
    print("="*60)

    # Prepare data
    stan_data, df_weight, df_daily, weight_mean, weight_std = prepare_data_with_4h_intervals()
    print(f"\n✓ Data prepared: D={stan_data['D']} days, N_weight={stan_data['N_weight']} observations")

    # Run both models
    fit_no_trend = load_original_model_fit(stan_data)
    if fit_no_trend is None:
        print("Failed to fit no-trend model")
        sys.exit(1)

    fit_trend = run_trend_model(stan_data)
    if fit_trend is None:
        print("\n⚠ Warning: Trend model fitting failed or did not converge properly")
        print("Continuing with no-trend model results only...")
        posteriors_trend = None
    else:
        # Extract posteriors from trend model if successful
        posteriors_trend = extract_posteriors(fit_trend, "trend")

    # Extract posteriors from no-trend model (always needed)
    posteriors_no_trend = extract_posteriors(fit_no_trend, "no-trend")

    # Print comparison table if trend model succeeded
    if posteriors_trend is not None:
        print_comparison_table(posteriors_no_trend, posteriors_trend)

        # Interpretation guidance
        print("\n" + "="*60)
        print("INTERPRETATION GUIDANCE")
        print("="*60)

        if 'delta' in posteriors_trend:
            delta_mean = posteriors_trend['delta']['mean']
            delta_lower = posteriors_trend['delta']['lower_ci']
            delta_upper = posteriors_trend['delta']['upper_ci']

            print(f"\nLinear Trend Parameter (δ):")
            print(f"  Mean: {delta_mean:.4f} standardized units over full time range")
            print(f"  95% CI: [{delta_lower:.4f}, {delta_upper:.4f}]")

            # Convert to lbs (roughly: 1 std unit ≈ 2.77 lbs over full range)
            lbs_over_full_range = delta_mean * 2.77
            lbs_per_year = lbs_over_full_range / 2.5  # ~2.5 years of data

            print(f"  Implied: {lbs_over_full_range:.2f} lbs over full time range (~2.5 years)")
            print(f"           or {lbs_per_year:.2f} lbs/year")

            if abs(delta_lower) < 0.01 and abs(delta_upper) < 0.01:
                print(f"  → NO meaningful secular trend detected")
            elif delta_lower > 0:
                print(f"  → SIGNIFICANT upward trend in weight over time")
            elif delta_upper < 0:
                print(f"  → SIGNIFICANT downward trend in weight over time")
            else:
                print(f"  → UNCERTAIN trend (CI includes zero)")

        # Compare gamma_s
        gamma_s_no_trend = posteriors_no_trend['gamma_s']
        gamma_s_trend = posteriors_trend['gamma_s']

        print(f"\nStrength Effect (γ_s) Comparison:")
        print(f"  No-trend model: {gamma_s_no_trend['mean']:.4f} [{gamma_s_no_trend['lower_ci']:.4f}, {gamma_s_no_trend['upper_ci']:.4f}]")
        print(f"  Trend model:    {gamma_s_trend['mean']:.4f} [{gamma_s_trend['lower_ci']:.4f}, {gamma_s_trend['upper_ci']:.4f}]")

        change_in_mean = gamma_s_trend['mean'] - gamma_s_no_trend['mean']
        pct_change = 100 * change_in_mean / gamma_s_no_trend['mean'] if gamma_s_no_trend['mean'] != 0 else 0

        print(f"  Change: {change_in_mean:+.4f} ({pct_change:+.1f}%)")

        if abs(change_in_mean) < 0.05:
            print(f"  → Strength effect is ROBUST to trend adjustment (effect not confounded)")
        elif change_in_mean < -0.05:
            print(f"  → Strength effect SHRINKS significantly in trend model")
            print(f"     This suggests trend was partially confounded with strength effect")
        else:
            print(f"  → Strength effect INCREASES in trend model (unexpected)")

        # Generate plots and comparisons
        plot_gamma_s_comparison(fit_no_trend, fit_trend)
        compute_loo_comparison(fit_no_trend, fit_trend)
        save_comparison_json(posteriors_no_trend, posteriors_trend, stan_data)
    else:
        print("\n" + "="*60)
        print("NO-TREND MODEL RESULTS ONLY")
        print("="*60)
        print("\nUnable to fit trend model. Showing base model results:")
        print(f"\nStrength Effect (γ_s):")
        gamma_s_no_trend = posteriors_no_trend['gamma_s']
        print(f"  {gamma_s_no_trend['mean']:.4f} [{gamma_s_no_trend['lower_ci']:.4f}, {gamma_s_no_trend['upper_ci']:.4f}]")

    print("\n" + "="*60)
    print("✓ Trend model comparison complete")
    print("="*60)


if __name__ == '__main__':
    main()
