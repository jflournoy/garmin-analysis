#!/usr/bin/env python3
"""Regenerate plots from existing predictions with proper data-driven scaling.

This script reads existing component predictions from CSV files and
regenerates the visualizations with proper y-axis scaling (data-driven,
not including 0). No model fitting is required.

Usage:
  uv run python regenerate_plots_with_proper_scaling.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.weight import load_weight_data
from src.data.workout import load_workout_data

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_component_predictions(predictions_dir="docs/component_predictions"):
    """Load component predictions from CSV files (always in lbs)."""
    print(f"Loading component predictions from {predictions_dir}/...")

    predictions_dir = Path(predictions_dir)

    combined_path = predictions_dir / "all_component_predictions.csv"
    if not combined_path.exists():
        raise FileNotFoundError(f"Could not find component predictions in {predictions_dir}/")

    df_all = pd.read_csv(combined_path)
    df_all['date'] = pd.to_datetime(df_all['date'])

    print(f"  ✓ Loaded {len(df_all)} predictions")
    print(f"  Date range: {df_all['date'].min().date()} to {df_all['date'].max().date()}")
    print(f"  Hours: {sorted(df_all['hour'].unique())}")

    return df_all


def extract_components_from_dataframe(df_all):
    """Extract component data from the combined dataframe."""
    print("\nExtracting component data...")

    # Get unique dates and hours
    dates = np.sort(df_all['date'].unique())
    hours = np.sort(df_all['hour'].unique())

    D = len(dates)
    H = len(hours)

    # Create date and hour index mappings
    date_to_idx = {date: i for i, date in enumerate(dates)}
    hour_to_idx = {hour: i for i, hour in enumerate(hours)}

    # Initialize component arrays
    component_names = ['intercept', 'strength', 'aerobic', 'spline', 'total', 'full_model']
    components = {}

    for component_name in component_names:
        mean = np.zeros((D, H))
        lower = np.zeros((D, H))
        upper = np.zeros((D, H))

        # Fill arrays
        for _, row in df_all.iterrows():
            d_idx = date_to_idx[row['date']]
            h_idx = hour_to_idx[row['hour']]

            mean[d_idx, h_idx] = row[f'{component_name}_mean']
            lower[d_idx, h_idx] = row[f'{component_name}_lower']
            upper[d_idx, h_idx] = row[f'{component_name}_upper']

        components[component_name] = (mean, lower, upper)
        print(f"  ✓ {component_name}: {D} days × {H} hours")

    return components, dates, hours


def load_activity_impulses(data_dir="data"):
    """Load activity impulse data (duration_min * avg_hr) for strength and aerobic events."""
    aerobic_types = ['cycling', 'treadmill_running', 'walking', 'indoor_cycling',
                     'elliptical', 'indoor_cardio', 'e_bike_fitness', 'floor_climbing']

    results = {}
    for label, types in [('strength', ['strength_training']), ('aerobic', aerobic_types)]:
        try:
            df = load_workout_data(data_dir=data_dir, activity_type=types,
                                   include_exercise_details=False)
            if df.empty:
                results[label] = pd.DataFrame(columns=['date', 'impulse'])
                continue
            # duration is in ms; convert to minutes
            df = df.dropna(subset=['date', 'avg_hr', 'duration']).copy()
            df['duration_min'] = df['duration'] / 60000.0
            df['impulse'] = df['duration_min'] * df['avg_hr']
            results[label] = df[['date', 'impulse']].copy()
            print(f"  ✓ {label}: {len(df)} events, "
                  f"impulse range {df['impulse'].min():.0f}–{df['impulse'].max():.0f}")
        except Exception as e:
            print(f"  ⚠ Could not load {label} activities: {e}")
            results[label] = pd.DataFrame(columns=['date', 'impulse'])

    return results


def create_component_visualizations(components, df_weight, dates, hours, output_dir,
                                    activity_impulses=None):
    """Create visualizations with proper data-driven scaling."""
    print(f"\nCreating visualizations with proper scaling...")

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Convert to numpy arrays for convenience
    dates_np = np.array(dates)
    hours_np = np.array(hours)

    # Find noon index
    noon_idx = np.where(hours_np == 12.0)[0]
    if len(noon_idx) == 0:
        noon_idx = np.argmin(np.abs(hours_np - 12.0))
    else:
        noon_idx = noon_idx[0]

    # Prepare afternoon weight data (for total panel)
    if df_weight is not None:
        afternoon_start = 13.0
        afternoon_end = 20.0
        df_weight_pm = df_weight[(df_weight['hour_of_day'] >= afternoon_start) &
                                 (df_weight['hour_of_day'] <= afternoon_end)].copy()
    else:
        df_weight_pm = pd.DataFrame()

    # 1. Fitness State Time Series with Predictions — four panels, different data per panel
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    # Panel 0: Strength fitness component + strength training impulse events
    ax = axes[0]
    mean, lower, upper = components['strength']
    ax.fill_between(dates_np, lower[:, noon_idx], upper[:, noon_idx],
                    alpha=0.3, color='skyblue', label='95% CI')
    ax.plot(dates_np, mean[:, noon_idx], 'b-', linewidth=1.5, label='Strength fitness state')
    ax.set_xlabel('Date')
    ax.set_ylabel('Contribution to weight (lbs)')
    ax.set_title('Strength Fitness Component')
    ax.grid(True, alpha=0.3)

    if activity_impulses and not activity_impulses['strength'].empty:
        df_str = activity_impulses['strength']
        # Secondary axis for impulse
        ax2 = ax.twinx()
        ax2.bar(df_str['date'], df_str['impulse'], width=1, alpha=0.35, color='orange',
                label='Workout impulse\n(min×bpm)')
        ax2.set_ylabel('Impulse (min × bpm)', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
    else:
        ax.legend()

    # Set y limits for fitness state axis
    all_y = np.concatenate([mean[:, noon_idx], lower[:, noon_idx], upper[:, noon_idx]])
    y_pad = (all_y.max() - all_y.min()) * 0.05
    ax.set_ylim(all_y.min() - y_pad, all_y.max() + y_pad)
    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # Panel 1: Aerobic fitness component + aerobic impulse events
    ax = axes[1]
    mean, lower, upper = components['aerobic']
    ax.fill_between(dates_np, lower[:, noon_idx], upper[:, noon_idx],
                    alpha=0.3, color='lightgreen', label='95% CI')
    ax.plot(dates_np, mean[:, noon_idx], 'g-', linewidth=1.5, label='Aerobic fitness state')
    ax.set_xlabel('Date')
    ax.set_ylabel('Contribution to weight (lbs)')
    ax.set_title('Aerobic Fitness Component')
    ax.grid(True, alpha=0.3)

    if activity_impulses and not activity_impulses['aerobic'].empty:
        df_aer = activity_impulses['aerobic']
        ax2 = ax.twinx()
        ax2.bar(df_aer['date'], df_aer['impulse'], width=1, alpha=0.35, color='purple',
                label='Workout impulse\n(min×bpm)')
        ax2.set_ylabel('Impulse (min × bpm)', color='purple')
        ax2.tick_params(axis='y', labelcolor='purple')
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
    else:
        ax.legend()

    all_y = np.concatenate([mean[:, noon_idx], lower[:, noon_idx], upper[:, noon_idx]])
    y_pad = (all_y.max() - all_y.min()) * 0.05
    ax.set_ylim(all_y.min() - y_pad, all_y.max() + y_pad)
    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # Panel 2: Daily spline component — averaged intraday pattern (hour of day)
    ax = axes[2]
    mean_spline, lower_spline, upper_spline = components['spline']
    # Average over all days to get the typical daily pattern
    mean_daily = mean_spline.mean(axis=0)
    lower_daily = lower_spline.mean(axis=0)
    upper_daily = upper_spline.mean(axis=0)
    ax.fill_between(hours_np, lower_daily, upper_daily,
                    alpha=0.3, color='orange', label='95% CI (mean)')
    ax.plot(hours_np, mean_daily, 'o-', linewidth=2, markersize=4, color='darkorange',
            label='Mean daily pattern')
    ax.set_xlabel('Hour of day')
    ax.set_ylabel('Contribution to weight (lbs)')
    ax.set_title('Daily Spline Component (Intraday Pattern)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    all_y = np.concatenate([mean_daily, lower_daily, upper_daily])
    y_pad = (all_y.max() - all_y.min()) * 0.05
    ax.set_ylim(all_y.min() - y_pad, all_y.max() + y_pad)

    # Panel 3: Total prediction + actual weight measurements
    ax = axes[3]
    mean_total, lower_total, upper_total = components['total']
    ax.fill_between(dates_np, lower_total[:, noon_idx], upper_total[:, noon_idx],
                    alpha=0.3, color='skyblue', label='95% CI')
    ax.plot(dates_np, mean_total[:, noon_idx], 'b-', linewidth=1.5, label='Total prediction (1 PM)')
    if len(df_weight_pm) > 0:
        ax.scatter(df_weight_pm['timestamp'], df_weight_pm['weight_lbs'],
                   alpha=0.6, s=30, color='red', edgecolor='black', linewidth=0.5,
                   label='Actual weight (1–8 PM)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Weight (lbs)')
    ax.set_title('Total Prediction (All Components)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    all_y = np.concatenate([mean_total[:, noon_idx], lower_total[:, noon_idx],
                            upper_total[:, noon_idx]])
    if len(df_weight_pm) > 0:
        all_y = np.concatenate([all_y, df_weight_pm['weight_lbs'].values])
    y_pad = (all_y.max() - all_y.min()) * 0.05
    ax.set_ylim(all_y.min() - y_pad, all_y.max() + y_pad)
    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    output_filename = 'component_time_series_noon_lbs.png'
    plt.savefig(output_dir / output_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/{output_filename}")

    # 2. Heatmap of total predictions over time
    fig, ax = plt.subplots(figsize=(14, 8))

    mean_total, _, _ = components['total']

    # Create heatmap - use ordinal dates for plotting
    # Convert dates to ordinal numbers (days since 0001-01-01)
    dates_ordinal = np.array([pd.Timestamp(d).toordinal() for d in dates_np])

    im = ax.imshow(mean_total.T, aspect='auto', cmap='RdYlBu_r',
                   extent=[dates_ordinal[0], dates_ordinal[-1],
                           hours_np[0], hours_np[-1]],
                   origin='lower')

    # Add actual weight data points on heatmap
    for _, row in df_weight.iterrows():
        date_ordinal = pd.Timestamp(row['date']).toordinal()
        ax.scatter(date_ordinal, row['hour_of_day'],
                  s=20, color='red', edgecolor='white', linewidth=0.5, alpha=0.7)

    ax.set_xlabel('Date')
    ax.set_ylabel('Hour of Day')
    ax.set_title('Total Predictions Heatmap with Actual Weight Measurements')

    # Format x-axis with dates (include year)
    date_ticks = np.linspace(dates_ordinal[0], dates_ordinal[-1], 8)
    date_labels = [pd.Timestamp.fromordinal(int(d)).strftime('%m/%d\n%Y') for d in date_ticks]
    ax.set_xticks(date_ticks)
    ax.set_xticklabels(date_labels)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Weight (lbs)')

    ax.xaxis.set_major_locator(plt.MaxNLocator(8))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    output_filename = 'total_predictions_heatmap_lbs.png'
    plt.savefig(output_dir / output_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/{output_filename}")

    # 3. Daily patterns analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Spline-only daily pattern
    ax = axes[0]
    mean_spline, _, _ = components['spline']
    mean_spline_daily = mean_spline.mean(axis=0)

    ax.plot(hours_np, mean_spline_daily, 'o-', linewidth=3, markersize=6,
            color='orange', label='Spline Component')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Weight (lbs)')
    ax.set_title('Mean Daily Spline Pattern')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Set y-axis limits
    y_min = np.min(mean_spline_daily)
    y_max = np.max(mean_spline_daily)
    y_range = y_max - y_min
    y_padding = y_range * 0.05
    ax.set_ylim(y_min - y_padding, y_max + y_padding)

    # Right: Daily pattern variability
    ax = axes[1]
    n_days_to_plot = min(50, len(dates))
    day_indices = np.linspace(0, len(dates)-1, n_days_to_plot, dtype=int)

    for day_idx in day_indices:
        ax.plot(hours_np, mean_total[day_idx, :], 'gray', alpha=0.1, linewidth=0.5)

    # Use total mean pattern (not spline-only) for right panel
    mean_total_daily = mean_total.mean(axis=0)
    ax.plot(hours_np, mean_total_daily, 'b-', linewidth=3, label='Mean Total Pattern')

    # Add all actual weight data
    if df_weight is not None and len(df_weight) > 0:
        ax.scatter(df_weight['hour_of_day'], df_weight['weight_lbs'],
                  alpha=0.3, s=20, color='red', label='Actual Weight')

    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Weight (lbs)')
    ax.set_title(f'Daily Pattern Variability ({n_days_to_plot} sample days)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Set y-axis limits
    all_y_data = []
    for day_idx in day_indices:
        all_y_data.append(mean_total[day_idx, :])
    all_y_data.append(mean_total_daily)
    if df_weight is not None and len(df_weight) > 0:
        all_y_data.append(df_weight['weight_lbs'].values)

    all_y_data_flat = np.concatenate([d.flatten() for d in all_y_data])

    y_min = np.min(all_y_data_flat)
    y_max = np.max(all_y_data_flat)
    y_range = y_max - y_min
    y_padding = y_range * 0.05
    ax.set_ylim(y_min - y_padding, y_max + y_padding)

    plt.tight_layout()
    output_filename = 'daily_patterns_analysis_lbs.png'
    plt.savefig(output_dir / output_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/{output_filename}")

    # 5. Prediction vs actual scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Get predictions at actual measurement times
    if df_weight is not None and len(df_weight) > 0:
        predicted_weights = []
        actual_weights = []

        for _, row in df_weight.iterrows():
            # Find closest date and hour
            date_diffs = np.abs(dates_np - pd.Timestamp(row['date']).to_datetime64())
            date_idx = np.argmin(date_diffs)

            hour_diffs = np.abs(hours_np - row['hour_of_day'])
            hour_idx = np.argmin(hour_diffs)

            predicted_weight = mean_total[date_idx, hour_idx]
            actual_weight = row['weight_lbs']

            predicted_weights.append(predicted_weight)
            actual_weights.append(actual_weight)

        ax.scatter(actual_weights, predicted_weights, alpha=0.6, s=20)

        # Add diagonal line
        min_val = min(min(actual_weights), min(predicted_weights))
        max_val = max(max(actual_weights), max(predicted_weights))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect Prediction')

        ax.set_xlabel('Actual Weight (lbs)')
        ax.set_ylabel('Predicted Weight (lbs)')
        ax.set_title('Prediction vs Actual Weight')

        # Add correlation coefficient
        correlation = np.corrcoef(actual_weights, predicted_weights)[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                transform=ax.transAxes, fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No weight data available',
                ha='center', va='center', fontsize=14,
                transform=ax.transAxes)
        ax.set_title('Prediction vs Actual Weight (No Data)')

    plt.tight_layout()
    output_filename = 'prediction_vs_actual_scatter_lbs.png'
    plt.savefig(output_dir / output_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/{output_filename}")

    # 5. AR(1) component visualization
    # The AR(1) process models temporal autocorrelation in the residuals at measurement times.
    # We visualize this by computing residuals (actual - structural prediction) and showing
    # their temporal structure.
    if df_weight is not None and len(df_weight) > 0:
        # Compute residuals: actual weight - total structural prediction (no AR)
        df_resid = df_weight.sort_values('timestamp').copy()
        resid_vals = []
        resid_dates = []

        for _, row in df_resid.iterrows():
            date_diffs = np.abs(dates_np - pd.Timestamp(row['date']).to_datetime64())
            d_idx = np.argmin(date_diffs)
            h_idx = np.argmin(np.abs(hours_np - row['hour_of_day']))
            structural_pred = mean_total[d_idx, h_idx]
            resid = row['weight_lbs'] - structural_pred
            resid_vals.append(resid)
            resid_dates.append(row['timestamp'])

        resid_vals = np.array(resid_vals)
        resid_dates = np.array(resid_dates)

        # Load rho / sigma_epsilon posteriors if available
        meta_path = Path(output_dir) / 'posterior_metadata.json'
        rho_mean = rho_lower = rho_upper = None
        sig_eps_mean = None
        if meta_path.exists():
            import json
            with open(meta_path) as f:
                meta = json.load(f)
            post = meta.get('posterior', {})
            if 'rho' in post:
                rho_mean = post['rho']['mean']
                rho_lower = post['rho'].get('lower_ci')
                rho_upper = post['rho'].get('upper_ci')
            if 'sigma_epsilon' in post:
                sig_eps_mean = post['sigma_epsilon']['mean']

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Panel A: residuals over time
        ax = axes[0]
        ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
        ax.plot(resid_dates, resid_vals, 'o-', color='steelblue', linewidth=0.8,
                markersize=4, alpha=0.7, label='Residual (actual − structural)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Residual (lbs)')
        ax.set_title('Residuals from Structural Prediction\n(AR(1) signal)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        r_pad = np.abs(resid_vals).max() * 1.1
        ax.set_ylim(-r_pad, r_pad)
        ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        # Panel B: lag-1 scatter (consecutive residuals)
        ax = axes[1]
        if len(resid_vals) > 1:
            ax.scatter(resid_vals[:-1], resid_vals[1:], alpha=0.6, s=30, color='steelblue',
                       edgecolor='navy', linewidth=0.4)
            # Fit a line for visual reference
            lag1_corr = np.corrcoef(resid_vals[:-1], resid_vals[1:])[0, 1]
            xr = np.array([resid_vals.min(), resid_vals.max()])
            # Simple OLS slope through origin
            slope_ols = np.sum(resid_vals[:-1] * resid_vals[1:]) / np.sum(resid_vals[:-1] ** 2)
            ax.plot(xr, slope_ols * xr, 'r-', linewidth=1.5,
                    label=f'OLS slope = {slope_ols:.2f}')
            ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
            ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
            # Annotate with posterior rho
            note = f'Empirical lag-1 r = {lag1_corr:.2f}'
            if rho_mean is not None:
                note += f'\nPosterior ρ = {rho_mean:.2f} [{rho_lower:.2f}, {rho_upper:.2f}]'
            ax.text(0.05, 0.95, note, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        ax.set_xlabel('Residual at time t')
        ax.set_ylabel('Residual at time t+1')
        ax.set_title('Lag-1 Autocorrelation of Residuals\n(AR(1) ρ structure)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Panel C: ACF of residuals
        ax = axes[2]
        max_lag = min(20, len(resid_vals) - 1)
        lags = np.arange(0, max_lag + 1)
        acf_vals = [1.0]  # lag 0
        for lag in range(1, max_lag + 1):
            if len(resid_vals) > lag:
                r = np.corrcoef(resid_vals[:-lag], resid_vals[lag:])[0, 1]
                acf_vals.append(r)
            else:
                acf_vals.append(np.nan)
        acf_vals = np.array(acf_vals)

        ax.bar(lags, acf_vals, color='steelblue', alpha=0.7, width=0.5)
        # 95% confidence bound for white noise: ±1.96/sqrt(n)
        ci_bound = 1.96 / np.sqrt(len(resid_vals))
        ax.axhline(ci_bound, color='red', linestyle='--', linewidth=1,
                   label=f'95% CI (white noise): ±{ci_bound:.2f}')
        ax.axhline(-ci_bound, color='red', linestyle='--', linewidth=1)
        ax.axhline(0, color='black', linewidth=0.8)
        if rho_mean is not None:
            # Overlay theoretical AR(1) ACF: rho^lag
            theoretical_acf = np.array([rho_mean ** lag for lag in lags])
            ax.plot(lags, theoretical_acf, 'g--', linewidth=1.5,
                    label=f'AR(1) theory (ρ={rho_mean:.2f})')
        ax.set_xlabel('Lag (measurement intervals)')
        ax.set_ylabel('Autocorrelation')
        ax.set_title('ACF of Residuals\nvs theoretical AR(1)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.05, 1.05)

        plt.tight_layout()
        output_filename = 'ar1_component_lbs.png'
        plt.savefig(output_dir / output_filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_dir}/{output_filename}")


def create_fitness_impulse_viz(components, dates, hours, output_dir, activity_impulses=None):
    """Four-panel: raw impulse bars vs. model fitness state contribution (strength & aerobic).

    Top row = strength, bottom row = aerobic.
    Left col: raw impulse bars + model fitness contribution (γ * state) over time.
    Right col: scaled impulse (decay-convolved with posterior parameters) vs. model fitness state.
    The right panel reveals how the leaky-integrator decay kernel transforms bursty impulse
    signals into smooth fitness states, and whether the impulse alone predicts the state.
    """
    import json

    output_dir = Path(output_dir)
    dates_np = np.array(dates)

    hours_np = np.array(hours)
    noon_idx = np.where(hours_np == 12.0)[0]
    noon_idx = noon_idx[0] if len(noon_idx) else np.argmin(np.abs(hours_np - 12.0))

    # Load posterior parameters for decay reconstruction
    meta_path = output_dir / 'posterior_metadata.json'
    params = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        post = meta.get('posterior', {})
        for key in ('alpha_d_s', 'alpha_m_s', 'beta_s', 'gamma_s',
                    'alpha_d_a', 'alpha_m_a', 'beta_a', 'gamma_a'):
            if key in post:
                params[key] = post[key]['mean']

    # Build a daily impulse array aligned to model dates
    def build_daily_impulse(impulse_df, date_array):
        """Return impulse array aligned to date_array (0 on days with no workout)."""
        imp = np.zeros(len(date_array))
        if impulse_df is None or impulse_df.empty:
            return imp
        dates_pd = pd.to_datetime(date_array)
        for _, row in impulse_df.iterrows():
            ev_date = pd.to_datetime(row['date'])
            diffs = np.abs((dates_pd - ev_date).total_seconds())
            idx = np.argmin(diffs)
            imp[idx] += row['impulse']
        return imp

    # Reconstruct fitness state via forward simulation with posterior-mean parameters
    def simulate_fitness_state(impulse_arr, alpha_d, alpha_m, beta, gamma,
                               trained_days=None):
        """Simulate γ*fitness[t] using posterior-mean decay parameters.

        trained_days: boolean array, True on days where impulse > 0.
        """
        n = len(impulse_arr)
        if trained_days is None:
            trained_days = impulse_arr > 0
        state = np.zeros(n)
        for t in range(1, n):
            alpha_eff = alpha_d + (1 - alpha_d) * alpha_m * float(trained_days[t - 1])
            state[t] = alpha_eff * state[t - 1] + beta * impulse_arr[t - 1] * float(trained_days[t - 1])
        return gamma * state

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))

    configs = [
        ('strength', 0, 'steelblue', 'orange',
         'alpha_d_s', 'alpha_m_s', 'beta_s', 'gamma_s'),
        ('aerobic',  1, 'seagreen',  'mediumpurple',
         'alpha_d_a', 'alpha_m_a', 'beta_a', 'gamma_a'),
    ]

    for label, row_i, state_color, imp_color, ad_k, am_k, b_k, g_k in configs:
        # Extract model component (γ*fitness, lbs)
        mean_comp, lower_comp, upper_comp = components[label]
        comp_noon = mean_comp[:, noon_idx]
        lower_noon = lower_comp[:, noon_idx]
        upper_noon = upper_comp[:, noon_idx]

        # Daily impulse array
        imp_df = activity_impulses.get(label) if activity_impulses else None
        imp_arr = build_daily_impulse(imp_df, dates_np)
        trained = imp_arr > 0

        # Posterior-mean simulated fitness contribution
        alpha_d = params.get(ad_k, 0.9)
        alpha_m = params.get(am_k, 0.5)
        beta    = params.get(b_k, 0.5)
        gamma   = params.get(g_k, 0.1)
        simulated = simulate_fitness_state(imp_arr, alpha_d, alpha_m, beta, gamma, trained)

        # ------ LEFT: impulse bars + model fitness contribution ------
        ax = axes[row_i, 0]
        ax2 = ax.twinx()

        # Impulse bars on secondary axis
        ax2.bar(dates_np, imp_arr, width=1, alpha=0.35, color=imp_color,
                label='Workout impulse (min×bpm)')
        ax2.set_ylabel('Impulse (min × bpm)', color=imp_color, fontsize=9)
        ax2.tick_params(axis='y', labelcolor=imp_color)

        # Model fitness state on primary axis
        ax.fill_between(dates_np, lower_noon, upper_noon, alpha=0.25, color=state_color)
        ax.plot(dates_np, comp_noon, color=state_color, linewidth=1.5,
                label=f'Model γ·fitness (lbs)')
        ax.set_ylabel('Weight contribution (lbs)', color=state_color, fontsize=9)
        ax.tick_params(axis='y', labelcolor=state_color)
        ax.set_title(f'{label.capitalize()} — Impulse vs. Model State', fontsize=10)
        ax.set_xlabel('Date')
        ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        lines1, lbls1 = ax.get_legend_handles_labels()
        lines2, lbls2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, lbls1 + lbls2, loc='upper left', fontsize=8)

        # ------ RIGHT: simulated (decay-convolved) vs. model fitness ------
        ax = axes[row_i, 1]
        ax.fill_between(dates_np, lower_noon, upper_noon, alpha=0.2, color=state_color,
                        label='Model 95% CI')
        ax.plot(dates_np, comp_noon, color=state_color, linewidth=2,
                label=f'Model γ·fitness')
        ax.plot(dates_np, simulated, color=imp_color, linewidth=1.5, linestyle='--',
                label=f'Simulated (β·decay·impulse, posterior params)')
        ax.set_title(f'{label.capitalize()} — Model vs. Decay-Convolved Impulse', fontsize=10)
        ax.set_ylabel('Weight contribution (lbs)', fontsize=9)
        ax.set_xlabel('Date')
        ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

        note = (f'α_d={alpha_d:.3f}, α_m={alpha_m:.3f}\n'
                f'β={beta:.3f}, γ={gamma:.3f}')
        ax.text(0.98, 0.05, note, transform=ax.transAxes, fontsize=8,
                ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        # Set shared y limits for left panel too
        all_y = np.concatenate([comp_noon, lower_noon, upper_noon, simulated])
        y_pad = (all_y.max() - all_y.min()) * 0.08
        axes[row_i, 1].set_ylim(all_y.min() - y_pad, all_y.max() + y_pad)
        all_y2 = np.concatenate([comp_noon, lower_noon, upper_noon])
        y_pad2 = (all_y2.max() - all_y2.min()) * 0.08
        axes[row_i, 0].set_ylim(all_y2.min() - y_pad2, all_y2.max() + y_pad2)

    fig.suptitle('Exercise Impulse vs. Model Fitness State Contributions',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    output_filename = 'fitness_impulse_vs_state_lbs.png'
    plt.savefig(output_dir / output_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/{output_filename}")


def main():
    """Main function to regenerate plots with proper scaling."""
    print("="*60)
    print("Regenerating Plots with Proper Data-Driven Scaling")
    print("="*60)
    print("\nThis script reads existing predictions and regenerates plots")
    print("with proper y-axis scaling (data-driven, not including 0).")
    print("No model fitting is required.\n")

    # Load weight data
    print("Loading weight data...")
    data_dir = "data"
    try:
        df_weight = load_weight_data(data_dir)
        # Prepare weight data
        df_weight['date'] = df_weight['timestamp'].dt.date
        df_weight['hour_of_day'] = df_weight['timestamp'].dt.hour + df_weight['timestamp'].dt.minute / 60.0
        print(f"  ✓ Loaded {len(df_weight)} weight measurements")
        print(f"  Weight range: {df_weight['weight_lbs'].min():.1f} to {df_weight['weight_lbs'].max():.1f} lbs")
    except FileNotFoundError as e:
        print(f"  ⚠ Warning: Weight data not found ({e})")
        print(f"  Proceeding without actual weight measurements (will not overlay on plots)")
        df_weight = None

    # Load component predictions
    try:
        df_all = load_component_predictions()
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease run one of the component prediction scripts first:")
        print("  uv run python generate_component_predictions.py")
        print("  uv run python generate_component_predictions_with_raw_weights.py")
        return

    # Extract components from dataframe
    components, dates, hours = extract_components_from_dataframe(df_all)

    # Load activity impulse data for overlay on fitness component panels
    print("\nLoading activity impulse data...")
    try:
        activity_impulses = load_activity_impulses(data_dir)
    except Exception as e:
        print(f"  ⚠ Warning: Could not load activity impulses ({e})")
        activity_impulses = None

    # Create output directory
    output_dir = "docs/component_predictions"

    # Create visualizations with proper scaling
    create_component_visualizations(components, df_weight, dates, hours, output_dir,
                                    activity_impulses=activity_impulses)

    # Create fitness impulse vs. state visualization
    create_fitness_impulse_viz(components, dates, hours, output_dir,
                               activity_impulses=activity_impulses)

    print("\n" + "="*60)
    print("Plot Regeneration Complete")
    print("="*60)
    print(f"\nSummary:")
    print(f"  - Source: Existing predictions from docs/component_predictions/")
    print(f"  - Scale: lbs")
    print(f"  - Output directory: {output_dir}/")
    print(f"  - Files: 5 plots with proper data-driven scaling")
    print(f"\nKey improvements:")
    print(f"  1. Strength/aerobic panels show activity impulse events (dur×HR)")
    print(f"  2. Spline panel shows intraday hour-of-day pattern (not time series)")
    print(f"  3. Component Contributions at Sample Dates removed")
    print(f"  4. AR(1) component shown via residual time series, lag-1 scatter, and ACF")
    print(f"  5. Y-axis scaled to data range with 5% padding")


if __name__ == "__main__":
    main()