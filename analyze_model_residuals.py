#!/usr/bin/env python3
"""Analyze model residuals from symmetric normal model."""

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


def extract_model_residuals():
    """Extract residuals from model fit."""
    print("Loading model results to extract residuals...")

    # First, let's check if we have existing model results
    output_dir = Path("output/symmetric_outcome_comparison")
    if not output_dir.exists():
        print("No existing model results found. Need to run model first.")
        return None

    # Load the model fit from the comparison script
    # We'll need to re-run a simpler version to get proper residuals
    print("Re-running model for residual extraction...")

    # Load data
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

    # Prepare Stan data
    stan_data = {
        'D': D,
        'strength_intensity': df_daily['strength_training_std'].values.astype(float),
        'aerobic_intensity': df_daily['aerobic_intensity_std'].values.astype(float),
        'N_weight': len(df_weight),
        'y_weight': df_weight['weight_std'].values.astype(float),
        'day_idx': df_weight['day_idx'].values.astype(int),
    }

    # Fit model with fewer iterations for speed
    print("\nFitting model for residual analysis...")
    model = cmdstanpy.CmdStanModel(
        stan_file="stan/weight_state_space_training_decay_aerobic_symmetric.stan"
    )

    fit = model.sample(
        data=stan_data,
        chains=1,
        iter_warmup=200,
        iter_sampling=200,
        adapt_delta=0.95,
        show_progress=True,
        seed=12345,
    )

    # Extract draws
    draws_df = fit.draws_pd()

    # Get posterior means of key parameters
    gamma_s_mean = draws_df['gamma_s'].mean()
    gamma_a_mean = draws_df['gamma_a'].mean()
    intercept_mean = draws_df['weight_intercept'].mean()

    print(f"\nModel parameters (posterior means):")
    print(f"  gamma_s (strength): {gamma_s_mean:.3f}")
    print(f"  gamma_a (aerobic): {gamma_a_mean:.3f}")
    print(f"  intercept: {intercept_mean:.3f}")

    # Extract fitness states - need to be careful with indexing
    # The model stores fitness states as generated quantities
    fitness_pattern_s = 'strength_fitness_stored['
    fitness_pattern_a = 'aerobic_fitness_stored['

    # Find columns matching the patterns
    s_cols = [col for col in draws_df.columns if fitness_pattern_s in col]
    a_cols = [col for col in draws_df.columns if fitness_pattern_a in col]

    if not s_cols or not a_cols:
        print("WARNING: Could not find fitness state columns")
        print(f"Available columns starting with patterns:")
        print(f"  {fitness_pattern_s}: {[c for c in draws_df.columns if c.startswith('strength')][:5]}...")
        print(f"  {fitness_pattern_a}: {[c for c in draws_df.columns if c.startswith('aerobic')][:5]}...")
        return None

    # Sort columns by index
    s_cols_sorted = sorted(s_cols, key=lambda x: int(x.split('[')[1].split(']')[0]))
    a_cols_sorted = sorted(a_cols, key=lambda x: int(x.split('[')[1].split(']')[0]))

    # Calculate mean fitness across samples for each day
    strength_fitness_means = np.array([draws_df[col].mean() for col in s_cols_sorted])
    aerobic_fitness_means = np.array([draws_df[col].mean() for col in a_cols_sorted])

    print(f"\nExtracted fitness states:")
    print(f"  Strength fitness: {len(strength_fitness_means)} days")
    print(f"  Aerobic fitness: {len(aerobic_fitness_means)} days")

    # Calculate predictions and residuals
    predictions = []
    residuals_std = []

    for i, row in df_weight.iterrows():
        day_idx = row['day_idx'] - 1  # Convert to 0-index for Python

        if day_idx < len(strength_fitness_means):
            # Calculate predicted value
            pred = (intercept_mean +
                   gamma_s_mean * strength_fitness_means[day_idx] +
                   gamma_a_mean * aerobic_fitness_means[day_idx])

            actual = row['weight_std']
            residual = actual - pred

            predictions.append(pred)
            residuals_std.append(residual)
        else:
            print(f"WARNING: Day index {day_idx} out of bounds for fitness arrays")
            predictions.append(np.nan)
            residuals_std.append(np.nan)

    df_weight['predicted_std'] = predictions
    df_weight['residual_std'] = residuals_std

    # Convert to original units
    df_weight['predicted_lbs'] = df_weight['predicted_std'] * weight_std + weight_mean
    df_weight['residual_lbs'] = df_weight['residual_std'] * weight_std

    # Remove any NaN values
    valid_mask = ~np.isnan(residuals_std)
    residuals_std = np.array(residuals_std)[valid_mask]
    residuals_lbs = np.array(df_weight['residual_lbs'])[valid_mask]

    print(f"\nResidual statistics:")
    print(f"  Number of valid residuals: {len(residuals_std)}")
    print(f"  Mean (std): {np.mean(residuals_std):.3f} (should be ~0)")
    print(f"  Std (std): {np.std(residuals_std):.3f}")
    print(f"  Min/Max (std): [{np.min(residuals_std):.3f}, {np.max(residuals_std):.3f}]")
    print(f"  Mean (lbs): {np.mean(residuals_lbs):.3f} lbs")
    print(f"  Std (lbs): {np.std(residuals_lbs):.3f} lbs")

    return df_weight, residuals_std, residuals_lbs, weight_mean, weight_std


def analyze_residuals(df_weight, residuals_std, residuals_lbs):
    """Analyze residual distribution and patterns."""
    print("\n" + "="*80)
    print("RESIDUAL ANALYSIS")
    print("="*80)

    # Basic statistics
    print(f"\nResidual distribution:")
    print(f"  Skewness: {stats.skew(residuals_std):.3f}")
    print(f"  Kurtosis: {stats.kurtosis(residuals_std):.3f}")

    # Normality tests
    print(f"\nNormality tests for residuals:")

    # Shapiro-Wilk
    if len(residuals_std) < 5000:
        shapiro_stat, shapiro_p = stats.shapiro(residuals_std)
        print(f"  Shapiro-Wilk: W = {shapiro_stat:.3f}, p = {shapiro_p:.3e}")
        if shapiro_p < 0.05:
            print(f"    → REJECT normality at α=0.05")
        else:
            print(f"    → FAIL TO REJECT normality at α=0.05")

    # Autocorrelation
    print(f"\nAutocorrelation analysis:")
    if len(residuals_std) > 1:
        # Simple lag-1 autocorrelation
        lag1_corr = np.corrcoef(residuals_std[:-1], residuals_std[1:])[0, 1]
        print(f"  Lag-1 autocorrelation: {lag1_corr:.3f}")

        # Durbin-Watson statistic approximation
        dw = np.sum(np.diff(residuals_std)**2) / np.sum(residuals_std**2)
        print(f"  Durbin-Watson statistic: {dw:.3f}")
        print(f"    Interpretation: {interpret_durbin_watson(dw)}")

    # Heteroscedasticity check
    print(f"\nHeteroscedasticity check:")
    # Correlation between absolute residuals and fitted values
    if 'predicted_std' in df_weight.columns:
        abs_residuals = np.abs(residuals_std)
        predicted = df_weight['predicted_std'].dropna().values

        if len(abs_residuals) == len(predicted):
            corr = np.corrcoef(abs_residuals, predicted)[0, 1]
            print(f"  Correlation(|residuals|, predicted): {corr:.3f}")
            if abs(corr) > 0.3:
                print(f"    → Possible heteroscedasticity")
            else:
                print(f"    → No strong evidence of heteroscedasticity")

    return residuals_std, residuals_lbs


def interpret_durbin_watson(dw):
    """Interpret Durbin-Watson statistic."""
    if dw < 1.5:
        return "Positive autocorrelation"
    elif dw > 2.5:
        return "Negative autocorrelation"
    else:
        return "No significant autocorrelation"


def create_residual_visualizations(df_weight, residuals_std, residuals_lbs):
    """Create visualizations of residuals."""
    print("\nCreating residual visualizations...")

    # Create output directory
    output_dir = Path("output/residual_analysis")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Model Residual Analysis - Symmetric Normal Model', fontsize=16)

    # 1. Residual distribution histogram
    ax = axes[0, 0]
    n_bins = min(20, int(np.sqrt(len(residuals_std))))
    ax.hist(residuals_std, bins=n_bins, density=True, alpha=0.7,
           color='green', edgecolor='black')

    # Overlay normal distribution
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    normal_pdf = stats.norm.pdf(x, np.mean(residuals_std), np.std(residuals_std))
    ax.plot(x, normal_pdf, 'k-', linewidth=2, label='Normal fit')

    ax.axvline(0, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('Residuals (standardized)')
    ax.set_ylabel('Density')
    ax.set_title('Residual Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Q-Q plot for residuals
    ax = axes[0, 1]
    stats.probplot(residuals_std, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot: Residuals vs Normal')
    ax.grid(True, alpha=0.3)

    # 3. Residuals vs fitted values
    ax = axes[0, 2]
    if 'predicted_std' in df_weight.columns:
        predicted = df_weight['predicted_std'].dropna().values
        if len(predicted) == len(residuals_std):
            ax.scatter(predicted, residuals_std, alpha=0.6, s=20)
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)

            # Add LOWESS smoother if available
            try:
                from statsmodels.nonparametric.smoothers_lowess import lowess
                smoothed = lowess(residuals_std, predicted, frac=0.3)
                ax.plot(smoothed[:, 0], smoothed[:, 1], 'r-', linewidth=2, alpha=0.8)
            except ImportError:
                pass

            ax.set_xlabel('Predicted Values (standardized)')
            ax.set_ylabel('Residuals (standardized)')
            ax.set_title('Residuals vs Fitted Values')
    else:
        ax.text(0.5, 0.5, 'No predicted values available',
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Residuals vs Fitted Values')
    ax.grid(True, alpha=0.3)

    # 4. Residuals vs time
    ax = axes[1, 0]
    df_sorted = df_weight.sort_values('timestamp')
    valid_mask = ~np.isnan(df_sorted['residual_std'])
    ax.plot(df_sorted['timestamp'][valid_mask], df_sorted['residual_std'][valid_mask],
           'o-', markersize=4, linewidth=1, alpha=0.7)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('Date')
    ax.set_ylabel('Residuals (standardized)')
    ax.set_title('Residuals vs Time')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 5. Residuals vs observed values
    ax = axes[1, 1]
    observed = df_weight['weight_std'].values[~np.isnan(residuals_std)]
    ax.scatter(observed, residuals_std, alpha=0.6, s=20)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('Observed Values (standardized)')
    ax.set_ylabel('Residuals (standardized)')
    ax.set_title('Residuals vs Observed Values')
    ax.grid(True, alpha=0.3)

    # 6. Summary statistics
    ax = axes[1, 2]
    ax.axis('off')

    summary_text = (
        f"RESIDUAL ANALYSIS SUMMARY\n"
        f"{'='*40}\n"
        f"Number of residuals: {len(residuals_std)}\n"
        f"Mean: {np.mean(residuals_std):.3f}\n"
        f"Std: {np.std(residuals_std):.3f}\n"
        f"Min/Max: [{np.min(residuals_std):.3f}, {np.max(residuals_std):.3f}]\n"
        f"Skewness: {stats.skew(residuals_std):.3f}\n"
        f"Kurtosis: {stats.kurtosis(residuals_std):.3f}\n\n"
    )

    # Add normality test results
    if len(residuals_std) < 5000:
        shapiro_stat, shapiro_p = stats.shapiro(residuals_std)
        summary_text += f"Shapiro-Wilk: W = {shapiro_stat:.3f}\n"
        summary_text += f"p-value: {shapiro_p:.3e}\n"

    # Add autocorrelation
    if len(residuals_std) > 1:
        lag1_corr = np.corrcoef(residuals_std[:-1], residuals_std[1:])[0, 1]
        summary_text += f"\nLag-1 autocorrelation: {lag1_corr:.3f}\n"

    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
           verticalalignment='center', fontsize=10, fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / 'residual_analysis_comprehensive.png',
               dpi=150, bbox_inches='tight')
    plt.close()

    # Create additional detailed plots
    create_detailed_residual_plots(df_weight, residuals_std, residuals_lbs, output_dir)

    print(f"\nVisualizations saved to {output_dir}/")


def create_detailed_residual_plots(df_weight, residuals_std, residuals_lbs, output_dir):
    """Create additional detailed residual plots."""
    # 1. Residual distribution with KDE
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram with KDE
    ax1.hist(residuals_std, bins=20, density=True, alpha=0.5,
            color='green', edgecolor='black', label='Histogram')

    # KDE
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(residuals_std)
    x_range = np.linspace(np.min(residuals_std), np.max(residuals_std), 200)
    ax1.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')

    # Normal fit
    normal_fit = stats.norm.pdf(x_range, np.mean(residuals_std), np.std(residuals_std))
    ax1.plot(x_range, normal_fit, 'k--', linewidth=2, label='Normal fit')

    ax1.axvline(0, color='red', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Residuals (standardized)')
    ax1.set_ylabel('Density')
    ax1.set_title('Residual Distribution: Histogram, KDE, and Normal Fit')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Residuals in original units over time
    ax2.plot(df_weight['timestamp'], df_weight['residual_lbs'],
            'o-', markersize=4, linewidth=1, alpha=0.7)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax2.fill_between(df_weight['timestamp'], -1, 1, alpha=0.2, color='gray',
                    label='±1 lb band')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Residuals (lbs)')
    ax2.set_title('Residuals Over Time (Original Units)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_dir / 'residuals_detailed.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Residual autocorrelation plot
    fig, ax = plt.subplots(figsize=(10, 6))

    max_lag = min(20, len(residuals_std) - 1)
    lags = range(1, max_lag + 1)
    autocorrs = []

    for lag in lags:
        if lag < len(residuals_std):
            corr = np.corrcoef(residuals_std[:-lag], residuals_std[lag:])[0, 1]
            autocorrs.append(corr)

    ax.bar(lags, autocorrs, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='±0.2 threshold')
    ax.axhline(y=-0.2, color='red', linestyle='--', alpha=0.5)

    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('Residual Autocorrelation Function')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'residual_autocorrelation.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Save residuals data
    residuals_df = pd.DataFrame({
        'timestamp': df_weight['timestamp'],
        'weight_lbs': df_weight['weight_lbs'],
        'predicted_lbs': df_weight['predicted_lbs'],
        'residual_lbs': df_weight['residual_lbs'],
        'residual_std': df_weight['residual_std']
    })
    residuals_df.to_csv(output_dir / 'residuals_data.csv', index=False)


def main():
    """Main function."""
    print("="*80)
    print("MODEL RESIDUAL ANALYSIS")
    print("Symmetric Normal Model")
    print("="*80)

    # Extract residuals
    result = extract_model_residuals()
    if result is None:
        print("Failed to extract residuals.")
        return

    df_weight, residuals_std, residuals_lbs, weight_mean, weight_std = result

    # Analyze residuals
    analyze_residuals(df_weight, residuals_std, residuals_lbs)

    # Create visualizations
    create_residual_visualizations(df_weight, residuals_std, residuals_lbs)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()