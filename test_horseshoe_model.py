#!/usr/bin/env python3
"""Test the horseshoe prior model with predictions for all days."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import cmdstanpy

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.data.weight import load_weight_data
from src.data.intensity import load_intensity_by_activity


def prepare_test_data():
    """Prepare minimal test data for model compilation."""
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


def test_model_compilation():
    """Test that the horseshoe model compiles."""
    print("\nTesting model compilation...")

    model_path = "stan/weight_state_space_training_decay_aerobic_symmetric_student_ar_spline_horseshoe.stan"

    try:
        model = cmdstanpy.CmdStanModel(stan_file=model_path)
        print(f"✓ Model compiled successfully: {model_path}")
        return model
    except Exception as e:
        print(f"✗ Model compilation failed: {e}")
        return None


def main():
    """Main test function."""
    print("Testing horseshoe prior model with predictions for all days")
    print("=" * 60)

    # Prepare test data
    stan_data, df_weight, df_daily = prepare_test_data()

    print(f"\nData dimensions:")
    print(f"  Number of days (D): {stan_data['D']}")
    print(f"  Number of weight observations: {stan_data['N_weight']}")
    print(f"  Fourier harmonics (K): {stan_data['K']}")

    # Test model compilation
    model = test_model_compilation()

    if model is not None:
        print("\nModel structure highlights:")
        print("  1. Horseshoe prior on AR(1) innovation scale (sigma_epsilon)")
        print("  2. More informative prior on rho: normal(0, 0.3)")
        print("  3. Generates predictions for all days (y_pred_all_days)")
        print("  4. AR component omitted for daily predictions (measurement-time specific)")

        print("\nKey changes from original model:")
        print("  - sigma_epsilon ~ horseshoe (tau ~ cauchy(0, 0.1), lambda ~ cauchy(0, 1))")
        print("  - rho ~ normal(0, 0.3) [was normal(0, 0.5)]")
        print("  - Added y_pred_all_days in generated quantities")
        print("  - Daily predictions use average hour for spline component")

        print("\n✓ Model ready for testing with actual data sampling.")


if __name__ == "__main__":
    main()