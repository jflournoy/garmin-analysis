#!/usr/bin/env python3
"""Simple analysis of weight distribution and residuals."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.data.weight import load_weight_data

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def analyze_weight_distribution():
    """Analyze weight distribution from raw data."""
    print("Analyzing weight distribution...")
    print("="*80)

    # Load weight data
    data_dir = "data"
    df_weight = load_weight_data(data_dir)

    weight_data = df_weight['weight_lbs'].values
    n = len(weight_data)

    print(f"Number of weight observations: {n}")
    print(f"Time period: {df_weight['timestamp'].min().date()} to {df_weight['timestamp'].max().date()}")
    print(f"Days with measurements: {(df_weight['timestamp'].max() - df_weight['timestamp'].min()).days} days")

    # Basic statistics
    print(f"\nBasic statistics (lbs):")
    print(f"  Mean: {np.mean(weight_data):.2f}")
    print(f"  Std: {np.std(weight_data):.2f}")
    print(f"  Min: {np.min(weight_data):.2f}")
    print(f"  Max: {np.max(weight_data):.2f}")
    print(f"  Range: {np.max(weight_data) - np.min(weight_data):.2f}")
    print(f"  Median: {np.median(weight_data):.2f}")
    print(f"  IQR: {np.percentile(weight_data, 75) - np.percentile(weight_data, 25):.2f}")

    # Percentiles
    print(f"\nPercentiles (lbs):")
    for p in [5, 10, 25, 50, 75, 90, 95]:
        print(f"  {p}th: {np.percentile(weight_data, p):.2f}")

    # Normality tests
    print(f"\nNormality tests:")

    # Shapiro-Wilk test
    if n < 5000:
        shapiro_stat, shapiro_p = stats.shapiro(weight_data)
        print(f"  Shapiro-Wilk: W = {shapiro_stat:.3f}, p = {shapiro_p:.3e}")
        if shapiro_p < 0.05:
            print(f"    → REJECT normality at α=0.05")
        else:
            print(f"    → FAIL TO REJECT normality at α=0.05")

    # Skewness and kurtosis
    skewness = stats.skew(weight_data)
    kurtosis = stats.kurtosis(weight_data)

    print(f"\nDistribution shape:")
    print(f"  Skewness: {skewness:.3f}")
    print(f"    Interpretation: {interpret_skewness(skewness)}")
    print(f"  Kurtosis: {kurtosis:.3f}")
    print(f"    Interpretation: {interpret_kurtosis(kurtosis)}")

    # Check for multimodality using Hartigan's dip test if available
    try:
        from scipy.stats import diptest
        dip_stat, dip_p = diptest.diptest(weight_data)
        print(f"\nMultimodality test (Hartigan's dip):")
        print(f"  D = {dip_stat:.3f}, p = {dip_p:.3e}")
        if dip_p < 0.05:
            print(f"    → Evidence for multimodality")
        else:
            print(f"    → No strong evidence for multimodality")
    except ImportError:
        print(f"\nNote: Install 'diptest' package for multimodality testing")

    return df_weight, weight_data


def interpret_skewness(skewness):
    """Interpret skewness value."""
    if abs(skewness) < 0.5:
        return "Approximately symmetric"
    elif 0.5 <= abs(skewness) < 1:
        return f"Moderately {('left' if skewness < 0 else 'right')}-skewed"
    else:
        return f"Highly {('left' if skewness < 0 else 'right')}-skewed"


def interpret_kurtosis(kurtosis):
    """Interpret kurtosis value."""
    # Note: scipy returns excess kurtosis (kurtosis - 3)
    excess_kurtosis = kurtosis  # scipy already gives excess kurtosis

    if abs(excess_kurtosis) < 0.5:
        return "Approximately normal kurtosis"
    elif excess_kurtosis > 0.5:
        return "Leptokurtic (heavy-tailed)"
    else:
        return "Platykurtic (light-tailed)"


def create_weight_visualizations(df_weight, weight_data):
    """Create visualizations of weight distribution."""
    print("\nCreating visualizations...")

    # Create output directory
    output_dir = Path("output/weight_distribution")
    output_dir.mkdir(exist_ok=True, parents=True)

    # 1. Time series plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Weight Distribution Analysis', fontsize=16)

    # Time series
    ax = axes[0, 0]
    df_sorted = df_weight.sort_values('timestamp')
    ax.plot(df_sorted['timestamp'], df_sorted['weight_lbs'],
           'o-', markersize=4, linewidth=1, alpha=0.7)
    ax.axhline(y=np.mean(weight_data), color='red', linestyle='--',
              alpha=0.7, label=f'Mean: {np.mean(weight_data):.1f} lbs')
    ax.set_xlabel('Date')
    ax.set_ylabel('Weight (lbs)')
    ax.set_title('Weight Time Series')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Histogram with normal overlay
    ax = axes[0, 1]
    n_bins = min(30, int(np.sqrt(len(weight_data))))
    ax.hist(weight_data, bins=n_bins, density=True, alpha=0.7,
           color='blue', edgecolor='black')

    # Overlay normal distribution
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    normal_pdf = stats.norm.pdf(x, np.mean(weight_data), np.std(weight_data))
    ax.plot(x, normal_pdf, 'k-', linewidth=2, label='Normal fit')

    ax.set_xlabel('Weight (lbs)')
    ax.set_ylabel('Density')
    ax.set_title('Histogram with Normal Overlay')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Q-Q plot
    ax = axes[0, 2]
    stats.probplot(weight_data, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot vs Normal Distribution')
    ax.grid(True, alpha=0.3)

    # Box plot
    ax = axes[1, 0]
    ax.boxplot(weight_data, vert=True, patch_artist=True,
              boxprops=dict(facecolor='lightblue'),
              medianprops=dict(color='red', linewidth=2),
              whiskerprops=dict(color='black'),
              capprops=dict(color='black'),
              flierprops=dict(marker='o', color='red', alpha=0.5))
    ax.set_ylabel('Weight (lbs)')
    ax.set_title('Box Plot')
    ax.grid(True, alpha=0.3)

    # Violin plot
    ax = axes[1, 1]
    ax.violinplot(weight_data, vert=True, showmeans=True, showmedians=True)
    ax.set_ylabel('Weight (lbs)')
    ax.set_title('Violin Plot')
    ax.grid(True, alpha=0.3)

    # ECDF plot
    ax = axes[1, 2]
    sorted_data = np.sort(weight_data)
    ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ax.plot(sorted_data, ecdf, 'b-', linewidth=2)

    # Overlay normal CDF
    normal_cdf = stats.norm.cdf(sorted_data, np.mean(weight_data), np.std(weight_data))
    ax.plot(sorted_data, normal_cdf, 'r--', linewidth=2, alpha=0.7, label='Normal CDF')

    ax.set_xlabel('Weight (lbs)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Empirical CDF vs Normal CDF')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'weight_distribution_comprehensive.png',
               dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Detailed distribution plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Kernel density estimate
    ax1.hist(weight_data, bins=n_bins, density=True, alpha=0.5,
            color='blue', edgecolor='black', label='Histogram')

    # KDE
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(weight_data)
    x_range = np.linspace(np.min(weight_data), np.max(weight_data), 200)
    ax1.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')

    # Normal fit
    normal_fit = stats.norm.pdf(x_range, np.mean(weight_data), np.std(weight_data))
    ax1.plot(x_range, normal_fit, 'k--', linewidth=2, label='Normal fit')

    ax1.set_xlabel('Weight (lbs)')
    ax1.set_ylabel('Density')
    ax1.set_title('Weight Distribution: Histogram, KDE, and Normal Fit')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Cumulative distribution comparison
    ax2.plot(sorted_data, ecdf, 'b-', linewidth=2, label='Empirical CDF')
    ax2.plot(sorted_data, normal_cdf, 'r--', linewidth=2, label='Normal CDF')

    # Add KS statistic annotation
    ks_stat = np.max(np.abs(ecdf - normal_cdf))
    ax2.annotate(f'KS statistic = {ks_stat:.3f}',
                xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax2.set_xlabel('Weight (lbs)')
    ax2.set_ylabel('Cumulative Probability')
    ax2.set_title('CDF Comparison: Empirical vs Normal')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'weight_distribution_detailed.png',
               dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Summary statistics plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    # Calculate additional statistics
    cv = (np.std(weight_data) / np.mean(weight_data)) * 100  # Coefficient of variation
    mad = np.median(np.abs(weight_data - np.median(weight_data)))  # Median absolute deviation

    summary_text = (
        f"WEIGHT DATA SUMMARY\n"
        f"{'='*40}\n"
        f"Sample size: {len(weight_data)} measurements\n"
        f"Time period: {df_weight['timestamp'].min().date()} to {df_weight['timestamp'].max().date()}\n"
        f"Days: {(df_weight['timestamp'].max() - df_weight['timestamp'].min()).days}\n\n"
        f"DESCRIPTIVE STATISTICS\n"
        f"{'-'*40}\n"
        f"Mean: {np.mean(weight_data):.2f} lbs\n"
        f"Std: {np.std(weight_data):.2f} lbs\n"
        f"Coefficient of variation: {cv:.1f}%\n"
        f"Median: {np.median(weight_data):.2f} lbs\n"
        f"MAD: {mad:.2f} lbs\n"
        f"Range: [{np.min(weight_data):.1f}, {np.max(weight_data):.1f}] lbs\n"
        f"IQR: [{np.percentile(weight_data, 25):.1f}, {np.percentile(weight_data, 75):.1f}] lbs\n\n"
        f"DISTRIBUTION SHAPE\n"
        f"{'-'*40}\n"
        f"Skewness: {stats.skew(weight_data):.3f}\n"
        f"  → {interpret_skewness(stats.skew(weight_data))}\n"
        f"Kurtosis: {stats.kurtosis(weight_data):.3f}\n"
        f"  → {interpret_kurtosis(stats.kurtosis(weight_data))}\n\n"
        f"NORMALITY TESTS\n"
        f"{'-'*40}\n"
    )

    # Add normality test results
    if len(weight_data) < 5000:
        shapiro_stat, shapiro_p = stats.shapiro(weight_data)
        summary_text += f"Shapiro-Wilk: W = {shapiro_stat:.3f}, p = {shapiro_p:.3e}\n"

    ks_stat, ks_p = stats.kstest(
        (weight_data - np.mean(weight_data)) / np.std(weight_data),
        'norm'
    )
    summary_text += f"Kolmogorov-Smirnov: D = {ks_stat:.3f}, p = {ks_p:.3e}\n"

    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
           verticalalignment='center', fontsize=10, fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.savefig(output_dir / 'weight_summary_statistics.png',
               dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nVisualizations saved to {output_dir}/")

    # Save data
    df_weight.to_csv(output_dir / 'weight_data.csv', index=False)
    print(f"Data saved to {output_dir}/weight_data.csv")


def main():
    """Main function."""
    print("="*80)
    print("WEIGHT DISTRIBUTION ANALYSIS")
    print("="*80)

    # Analyze weight distribution
    df_weight, weight_data = analyze_weight_distribution()

    # Create visualizations
    create_weight_visualizations(df_weight, weight_data)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()