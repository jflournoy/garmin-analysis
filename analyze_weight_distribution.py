#!/usr/bin/env python3
"""Analyze weight distribution and residuals from symmetric normal model."""

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


def load_and_prepare_data():
    """Load and prepare data for analysis."""
    print("Loading data for weight distribution analysis...")

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

    print(f"Data loaded: {len(df_daily)} days, {len(df_weight)} weight observations")
    print(f"Weight: mean={weight_mean:.2f} lbs, std={weight_std:.2f} lbs")

    return stan_data, df_weight, df_daily, weight_mean, weight_std


def fit_model_and_get_residuals():
    """Fit symmetric normal model and calculate residuals."""
    print("\nFitting symmetric normal model...")

    stan_data, df_weight, df_daily, weight_mean, weight_std = load_and_prepare_data()

    # Fit model
    model = cmdstanpy.CmdStanModel(
        stan_file="stan/weight_state_space_training_decay_aerobic_symmetric.stan"
    )

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

    # Extract posterior samples
    draws_df = fit.draws_pd()

    # Get parameter means
    gamma_s_mean = draws_df['gamma_s'].mean()
    gamma_a_mean = draws_df['gamma_a'].mean()
    intercept_mean = draws_df['weight_intercept'].mean()

    print(f"\nModel parameters:")
    print(f"  gamma_s (strength): {gamma_s_mean:.3f}")
    print(f"  gamma_a (aerobic): {gamma_a_mean:.3f}")
    print(f"  intercept: {intercept_mean:.3f}")

    # Extract fitness states (approximate - using mean across samples)
    fitness_cols_s = [col for col in draws_df.columns if col.startswith('strength_fitness_stored[')]
    fitness_cols_a = [col for col in draws_df.columns if col.startswith('aerobic_fitness_stored[')]

    if fitness_cols_s and fitness_cols_a:
        # Calculate mean fitness across samples for each day
        strength_fitness_means = np.zeros(len(fitness_cols_s))
        aerobic_fitness_means = np.zeros(len(fitness_cols_a))

        for i in range(len(fitness_cols_s)):
            strength_fitness_means[i] = draws_df[f'strength_fitness_stored[{i+1}]'].mean()
            aerobic_fitness_means[i] = draws_df[f'aerobic_fitness_stored[{i+1}]'].mean()
    else:
        print("WARNING: Could not extract fitness states")
        strength_fitness_means = np.zeros(stan_data['D'])
        aerobic_fitness_means = np.zeros(stan_data['D'])

    # Calculate predicted values and residuals
    predictions = []
    residuals = []

    for i, row in df_weight.iterrows():
        day_idx = row['day_idx'] - 1  # Convert to 0-index

        if day_idx < len(strength_fitness_means):
            pred = (intercept_mean +
                   gamma_s_mean * strength_fitness_means[day_idx] +
                   gamma_a_mean * aerobic_fitness_means[day_idx])
            actual = row['weight_std']
            residual = actual - pred

            predictions.append(pred)
            residuals.append(residual)
        else:
            predictions.append(np.nan)
            residuals.append(np.nan)

    df_weight['predicted_std'] = predictions
    df_weight['residual_std'] = residuals

    # Convert back to original units
    df_weight['predicted_lbs'] = df_weight['predicted_std'] * weight_std + weight_mean
    df_weight['residual_lbs'] = df_weight['residual_std'] * weight_std

    print(f"\nResidual statistics (standardized):")
    print(f"  Mean: {np.mean(residuals):.3f}")
    print(f"  Std: {np.std(residuals):.3f}")
    print(f"  Min: {np.min(residuals):.3f}")
    print(f"  Max: {np.max(residuals):.3f}")

    return df_weight, residuals, weight_mean, weight_std, draws_df


def analyze_distribution(df_weight, residuals, weight_mean, weight_std):
    """Analyze weight distribution and residuals."""
    print("\n" + "="*80)
    print("WEIGHT DISTRIBUTION ANALYSIS")
    print("="*80)

    # Weight data in original units
    weight_data = df_weight['weight_lbs'].values
    weight_std_data = df_weight['weight_std'].values

    print(f"\nWeight data (original units, lbs):")
    print(f"  N = {len(weight_data)}")
    print(f"  Mean = {np.mean(weight_data):.2f}")
    print(f"  Std = {np.std(weight_data):.2f}")
    print(f"  Min = {np.min(weight_data):.2f}")
    print(f"  Max = {np.max(weight_data):.2f}")
    print(f"  Range = {np.max(weight_data) - np.min(weight_data):.2f}")

    # Normality tests
    print(f"\nNormality tests for weight data:")

    # Shapiro-Wilk test (for n < 5000)
    if len(weight_data) < 5000:
        shapiro_stat, shapiro_p = stats.shapiro(weight_data)
        print(f"  Shapiro-Wilk: W = {shapiro_stat:.3f}, p = {shapiro_p:.3e}")

    # Kolmogorov-Smirnov test against normal distribution
    ks_stat, ks_p = stats.kstest(weight_std_data, 'norm', args=(0, 1))
    print(f"  Kolmogorov-Smirnov: D = {ks_stat:.3f}, p = {ks_p:.3e}")

    # Anderson-Darling test
    anderson_result = stats.anderson(weight_data, dist='norm')
    print(f"  Anderson-Darling: A² = {anderson_result.statistic:.3f}")
    print(f"  Critical values: {anderson_result.critical_values}")
    print(f"  Significance levels: {anderson_result.significance_level}")

    # Skewness and kurtosis
    skewness = stats.skew(weight_data)
    kurtosis = stats.kurtosis(weight_data)
    print(f"\nDistribution shape:")
    print(f"  Skewness = {skewness:.3f} (0 = symmetric)")
    print(f"  Kurtosis = {kurtosis:.3f} (3 = normal)")

    # Residual analysis
    print(f"\nResidual analysis:")
    print(f"  Mean residual = {np.mean(residuals):.3f} (should be ~0)")
    print(f"  Std residual = {np.std(residuals):.3f}")
    print(f"  Skewness = {stats.skew(residuals):.3f}")
    print(f"  Kurtosis = {stats.kurtosis(residuals):.3f}")

    # Durbin-Watson test for autocorrelation
    if len(residuals) > 1:
        # Simple autocorrelation calculation
        resid_array = np.array(residuals)
        autocorr = np.corrcoef(resid_array[:-1], resid_array[1:])[0, 1]
        print(f"  Lag-1 autocorrelation = {autocorr:.3f}")

    return weight_data, weight_std_data


def create_visualizations(df_weight, residuals, weight_data, weight_std_data):
    """Create comprehensive visualizations."""
    print("\nCreating visualizations...")

    # Create output directory
    output_dir = Path("output/weight_distribution_analysis")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))

    # 1. Weight distribution histogram with normal overlay
    ax1 = plt.subplot(3, 3, 1)
    ax1.hist(weight_data, bins=20, density=True, alpha=0.7, color='blue', edgecolor='black')

    # Overlay normal distribution
    xmin, xmax = ax1.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, np.mean(weight_data), np.std(weight_data))
    ax1.plot(x, p, 'k', linewidth=2, label='Normal fit')

    ax1.set_xlabel('Weight (lbs)')
    ax1.set_ylabel('Density')
    ax1.set_title('Weight Distribution with Normal Overlay')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Q-Q plot for weight data
    ax2 = plt.subplot(3, 3, 2)
    stats.probplot(weight_data, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot: Weight vs Normal Distribution')
    ax2.grid(True, alpha=0.3)

    # 3. Time series of weight measurements
    ax3 = plt.subplot(3, 3, 3)
    df_weight_sorted = df_weight.sort_values('timestamp')
    ax3.plot(df_weight_sorted['timestamp'], df_weight_sorted['weight_lbs'],
            'o-', markersize=4, linewidth=1, alpha=0.7)
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Weight (lbs)')
    ax3.set_title('Weight Time Series')
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 4. Residual distribution
    ax4 = plt.subplot(3, 3, 4)
    ax4.hist(residuals, bins=20, density=True, alpha=0.7, color='green', edgecolor='black')

    # Overlay normal distribution on residuals
    xmin, xmax = ax4.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, np.mean(residuals), np.std(residuals))
    ax4.plot(x, p, 'k', linewidth=2, label='Normal fit')

    ax4.axvline(0, color='red', linestyle='--', alpha=0.7)
    ax4.set_xlabel('Residuals (standardized)')
    ax4.set_ylabel('Density')
    ax4.set_title('Residual Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Q-Q plot for residuals
    ax5 = plt.subplot(3, 3, 5)
    stats.probplot(residuals, dist="norm", plot=ax5)
    ax5.set_title('Q-Q Plot: Residuals vs Normal')
    ax5.grid(True, alpha=0.3)

    # 6. Residuals vs fitted values
    ax6 = plt.subplot(3, 3, 6)
    ax6.scatter(df_weight['predicted_std'], residuals, alpha=0.6, s=20)
    ax6.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax6.set_xlabel('Predicted Values (standardized)')
    ax6.set_ylabel('Residuals (standardized)')
    ax6.set_title('Residuals vs Fitted Values')
    ax6.grid(True, alpha=0.3)

    # 7. Residuals vs time
    ax7 = plt.subplot(3, 3, 7)
    df_weight_sorted = df_weight.sort_values('timestamp')
    ax7.plot(df_weight_sorted['timestamp'], df_weight_sorted['residual_std'],
            'o-', markersize=4, linewidth=1, alpha=0.7)
    ax7.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax7.set_xlabel('Date')
    ax7.set_ylabel('Residuals (standardized)')
    ax7.set_title('Residuals vs Time')
    ax7.grid(True, alpha=0.3)
    plt.setp(ax7.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 8. ACF plot for residuals
    ax8 = plt.subplot(3, 3, 8)
    from statsmodels.graphics.tsaplots import plot_acf
    try:
        plot_acf(residuals, ax=ax8, lags=20, alpha=0.05)
        ax8.set_title('Residual Autocorrelation')
        ax8.set_ylim([-0.3, 1.1])
    except ImportError:
        ax8.text(0.5, 0.5, 'statsmodels not available\nfor ACF plot',
                ha='center', va='center', transform=ax8.transAxes)
        ax8.set_title('ACF Plot (statsmodels required)')

    # 9. Summary statistics text
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')

    summary_text = (
        f"WEIGHT DATA SUMMARY\n"
        f"-------------------\n"
        f"N = {len(weight_data)}\n"
        f"Mean = {np.mean(weight_data):.2f} lbs\n"
        f"Std = {np.std(weight_data):.2f} lbs\n"
        f"Range = [{np.min(weight_data):.1f}, {np.max(weight_data):.1f}]\n"
        f"Skewness = {stats.skew(weight_data):.3f}\n"
        f"Kurtosis = {stats.kurtosis(weight_data):.3f}\n\n"
        f"RESIDUALS SUMMARY\n"
        f"-----------------\n"
        f"Mean = {np.mean(residuals):.3f}\n"
        f"Std = {np.std(residuals):.3f}\n"
        f"Skewness = {stats.skew(residuals):.3f}\n"
        f"Kurtosis = {stats.kurtosis(residuals):.3f}"
    )

    ax9.text(0.1, 0.5, summary_text, transform=ax9.transAxes,
            verticalalignment='center', fontsize=10, fontfamily='monospace')

    plt.suptitle('Weight Distribution and Residual Analysis - Symmetric Normal Model', fontsize=16)
    plt.tight_layout()

    # Save figure
    plt.savefig(output_dir / 'weight_distribution_residuals_analysis.png',
               dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Visualizations saved to {output_dir}/weight_distribution_residuals_analysis.png")

    # Create additional detailed plots
    create_detailed_plots(df_weight, residuals, weight_data, output_dir)


def create_detailed_plots(df_weight, residuals, weight_data, output_dir):
    """Create additional detailed plots."""
    # 1. Detailed histogram with kernel density estimate
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Weight distribution with KDE
    ax1.hist(weight_data, bins=25, density=True, alpha=0.7, color='blue', edgecolor='black')

    # Add KDE
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(weight_data)
    x_range = np.linspace(np.min(weight_data), np.max(weight_data), 100)
    ax1.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')

    # Add normal fit
    normal_fit = stats.norm.pdf(x_range, np.mean(weight_data), np.std(weight_data))
    ax1.plot(x_range, normal_fit, 'k--', linewidth=2, label='Normal fit')

    ax1.set_xlabel('Weight (lbs)')
    ax1.set_ylabel('Density')
    ax1.set_title('Weight Distribution: Histogram, KDE, and Normal Fit')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Residuals with KDE
    ax2.hist(residuals, bins=25, density=True, alpha=0.7, color='green', edgecolor='black')

    # Add KDE for residuals
    kde_resid = gaussian_kde(residuals)
    x_range_resid = np.linspace(np.min(residuals), np.max(residuals), 100)
    ax2.plot(x_range_resid, kde_resid(x_range_resid), 'r-', linewidth=2, label='KDE')

    # Add normal fit for residuals
    normal_fit_resid = stats.norm.pdf(x_range_resid, np.mean(residuals), np.std(residuals))
    ax2.plot(x_range_resid, normal_fit_resid, 'k--', linewidth=2, label='Normal fit')

    ax2.axvline(0, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Residuals (standardized)')
    ax2.set_ylabel('Density')
    ax2.set_title('Residual Distribution: Histogram, KDE, and Normal Fit')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'detailed_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Box plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.boxplot(weight_data, vert=True, patch_artist=True,
               boxprops=dict(facecolor='lightblue'),
               medianprops=dict(color='red'))
    ax1.set_ylabel('Weight (lbs)')
    ax1.set_title('Weight Data Box Plot')
    ax1.grid(True, alpha=0.3)

    ax2.boxplot(residuals, vert=True, patch_artist=True,
               boxprops=dict(facecolor='lightgreen'),
               medianprops=dict(color='red'))
    ax2.set_ylabel('Residuals (standardized)')
    ax2.set_title('Residuals Box Plot')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'box_plots.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Additional plots saved to {output_dir}/")


def main():
    """Main analysis function."""
    print("="*80)
    print("WEIGHT DISTRIBUTION AND RESIDUAL ANALYSIS")
    print("Symmetric Normal Model")
    print("="*80)

    # Fit model and get residuals
    df_weight, residuals, weight_mean, weight_std, draws_df = fit_model_and_get_residuals()

    # Analyze distributions
    weight_data, weight_std_data = analyze_distribution(df_weight, residuals, weight_mean, weight_std)

    # Create visualizations
    create_visualizations(df_weight, residuals, weight_data, weight_std_data)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

    # Save data for further analysis
    output_dir = Path("output/weight_distribution_analysis")
    df_weight.to_csv(output_dir / 'weight_data_with_residuals.csv', index=False)

    print(f"\nData saved to {output_dir}/weight_data_with_residuals.csv")
    print(f"Check {output_dir}/ for all visualizations")


if __name__ == "__main__":
    main()