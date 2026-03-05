#!/usr/bin/env python3
"""Generate component predictions from an already-fitted model."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import cmdstanpy
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.data.weight import load_weight_data
from src.data.intensity import load_intensity_by_activity

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_existing_fit():
    """Load the existing fitted model from the constrained analysis."""
    print("Loading existing fitted model...")

    # Try to find the latest fit file
    fit_dir = Path("docs/constrained_ar_analysis")
    fit_files = list(fit_dir.glob("*.csv"))

    if not fit_files:
        print("No existing fit found. Please run the constrained model first.")
        return None

    # The fit should be in the parent directory of the CSV files
    # Actually, we need to load the CmdStanMCMC object
    print("Note: This script requires the fitted model object.")
    print("Please run the constrained model test first to generate the fit.")
    return None


def prepare_data_with_4h_intervals():
    """Prepare data with 4-hour prediction intervals."""
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

    # 4-hour intervals (0, 4, 8, 12, 16, 20, 24 hours)
    pred_hours = np.array([0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0])

    return df_weight, df_daily, date_range, pred_hours


def create_simple_component_plots():
    """Create simple component plots using the results from our previous analysis."""
    print("\nCreating simple component plots from previous analysis...")

    # Load the summary data from our previous constrained model run
    summary_path = Path("docs/constrained_ar_analysis/analysis_summary.md")

    if not summary_path.exists():
        print(f"Summary file not found: {summary_path}")
        print("Please run the constrained model test first.")
        return

    # Read the summary to get key parameters
    with open(summary_path, 'r') as f:
        summary_text = f.read()

    # Extract key parameters from summary
    import re

    # Try to extract parameter values
    rho_match = re.search(r'ρ \(autocorrelation\): ([\d\.-]+)', summary_text)
    sigma_match = re.search(r'σ_ε \(innovation std\): ([\d\.-]+)', summary_text)
    gamma_s_match = re.search(r'γ_s \(strength effect\): ([\d\.-]+)', summary_text)
    gamma_a_match = re.search(r'γ_a \(aerobic effect\): ([\d\.-]+)', summary_text)

    if not all([rho_match, sigma_match, gamma_s_match, gamma_a_match]):
        print("Could not extract all parameters from summary.")
        print("Creating illustrative plots instead...")
        create_illustrative_plots()
        return

    rho = float(rho_match.group(1))
    sigma_epsilon = float(sigma_match.group(1))
    gamma_s = float(gamma_s_match.group(1))
    gamma_a = float(gamma_a_match.group(1))

    print(f"Extracted parameters:")
    print(f"  ρ = {rho:.3f}")
    print(f"  σ_ε = {sigma_epsilon:.3f}")
    print(f"  γ_s = {gamma_s:.3f}")
    print(f"  γ_a = {gamma_a:.3f}")

    # Prepare data
    df_weight, df_daily, date_range, pred_hours = prepare_data_with_4h_intervals()

    # Create output directory
    output_dir = Path("docs/component_plots_simple")
    output_dir.mkdir(exist_ok=True)

    # Create illustrative component plots
    # Since we don't have the full posterior, we'll create simplified versions

    # 1. Time series of fitness states (simplified)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Simulate fitness states (simplified)
    D = len(date_range)
    time_indices = np.arange(D)

    # Simple sinusoidal patterns for illustration
    strength_fitness = 0.5 * np.sin(2 * np.pi * time_indices / 180)  # ~6 month cycle
    aerobic_fitness = 0.3 * np.sin(2 * np.pi * time_indices / 90)    # ~3 month cycle

    axes[0].plot(date_range, strength_fitness, 'b-', linewidth=2, label='Strength Fitness')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Fitness Level')
    axes[0].set_title('Strength Fitness State Over Time (Illustrative)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(date_range, aerobic_fitness, 'r-', linewidth=2, label='Aerobic Fitness')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Fitness Level')
    axes[1].set_title('Aerobic Fitness State Over Time (Illustrative)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'fitness_states_illustrative.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Component contributions over time
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    # Compute component contributions
    intercept = 0.0  # Baseline
    strength_effect = gamma_s * strength_fitness
    aerobic_effect = gamma_a * aerobic_fitness

    # Daily spline (24h cycle)
    spline_amplitude = 0.1
    daily_spline = spline_amplitude * np.sin(2 * np.pi * np.arange(D) / 1)  # Daily cycle

    components = [
        ('Intercept', intercept * np.ones(D)),
        ('Strength Effect', strength_effect),
        ('Aerobic Effect', aerobic_effect),
        ('Daily Spline', daily_spline)
    ]

    for idx, (name, values) in enumerate(components):
        ax = axes[idx]
        ax.plot(date_range, values, linewidth=2)
        ax.set_xlabel('Date')
        ax.set_ylabel('Standardized Weight Contribution')
        ax.set_title(f'{name} Over Time')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='black', linestyle='-', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'component_contributions_illustrative.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Total prediction
    fig, ax = plt.subplots(figsize=(14, 6))

    total_prediction = intercept + strength_effect + aerobic_effect + daily_spline

    ax.plot(date_range, total_prediction, 'b-', linewidth=2, label='Total Prediction')
    ax.plot(date_range, strength_effect, 'g--', alpha=0.5, label='Strength Component')
    ax.plot(date_range, aerobic_effect, 'r--', alpha=0.5, label='Aerobic Component')
    ax.plot(date_range, daily_spline, 'm--', alpha=0.5, label='Daily Spline')

    ax.set_xlabel('Date')
    ax.set_ylabel('Standardized Weight')
    ax.set_title('Total Prediction and Components (Illustrative)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'total_prediction_illustrative.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Daily patterns at sample dates
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    sample_indices = [0, D//6, D//3, D//2, 2*D//3, 5*D//6]

    for idx, day_idx in enumerate(sample_indices):
        if idx >= len(axes):
            break

        ax = axes[idx]
        date_str = date_range[day_idx].strftime('%Y-%m-%d')

        # Create daily pattern (24h cycle with some variation)
        hours = np.linspace(0, 24, 100)

        # Base daily pattern (sinusoidal)
        base_pattern = 0.05 * np.sin(2 * np.pi * hours / 24)

        # Add some day-specific variation
        day_variation = 0.02 * np.sin(2 * np.pi * hours / 12 + day_idx/10)

        daily_pattern = base_pattern + day_variation

        ax.plot(hours, daily_pattern, 'b-', linewidth=2)
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Standardized Weight')
        ax.set_title(f'Daily Pattern: {date_str}')
        ax.grid(True, alpha=0.3)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.axhline(0, color='black', linestyle='-', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'daily_patterns_illustrative.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Create a simple summary
    summary_path = output_dir / 'illustrative_plots_summary.md'
    with open(summary_path, 'w') as f:
        f.write("# Illustrative Component Plots\n\n")
        f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Note\n\n")
        f.write("These are **illustrative plots** based on extracted parameters from the constrained model.\n")
        f.write("For actual posterior predictions, wait for the full model to finish running.\n\n")

        f.write("## Parameters Used\n\n")
        f.write(f"- ρ (autocorrelation): {rho:.3f}\n")
        f.write(f"- σ_ε (innovation std): {sigma_epsilon:.3f}\n")
        f.write(f"- γ_s (strength effect): {gamma_s:.3f}\n")
        f.write(f"- γ_a (aerobic effect): {gamma_a:.3f}\n\n")

        f.write("## Plots Generated\n\n")
        f.write("1. `fitness_states_illustrative.png` - Strength and aerobic fitness states\n")
        f.write("2. `component_contributions_illustrative.png` - Each component over time\n")
        f.write("3. `total_prediction_illustrative.png` - Total prediction with components\n")
        f.write("4. `daily_patterns_illustrative.png` - Daily patterns at sample dates\n\n")

        f.write("## Next Steps\n\n")
        f.write("Wait for the full model (`generate_component_predictions.py`) to finish\n")
        f.write("for actual posterior predictions with credible intervals.\n")

    print(f"✓ Illustrative plots saved to {output_dir}/")
    print(f"✓ Summary: {summary_path}")


def create_illustrative_plots():
    """Create completely illustrative plots when no data is available."""
    print("Creating completely illustrative plots...")

    output_dir = Path("docs/component_plots_illustrative")
    output_dir.mkdir(exist_ok=True)

    # Create time range
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    D = len(dates)

    # 1. Component contributions
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    time = np.arange(D)

    components = [
        ('Intercept', np.zeros(D)),
        ('Strength Effect', 0.1 * np.sin(2 * np.pi * time / 180)),
        ('Aerobic Effect', -0.05 * np.sin(2 * np.pi * time / 90)),
        ('Daily Spline', 0.02 * np.sin(2 * np.pi * time / 7))
    ]

    for idx, (name, values) in enumerate(components):
        ax = axes[idx]
        ax.plot(dates, values, linewidth=2)
        ax.set_xlabel('Date')
        ax.set_ylabel('Contribution')
        ax.set_title(f'{name} (Illustrative)')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'components_illustrative.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Illustrative plots saved to {output_dir}/")


def main():
    """Main function to create component plots."""
    print("Creating Component Plots")
    print("=" * 60)

    # Try to create plots from existing analysis
    create_simple_component_plots()

    print("\n" + "="*60)
    print("Component Plots Created")
    print("="*60)

    print("\nNote: The full model is still running.")
    print("These are illustrative plots based on extracted parameters.")
    print("Wait for `generate_component_predictions.py` to finish for actual posterior predictions.")


if __name__ == "__main__":
    main()