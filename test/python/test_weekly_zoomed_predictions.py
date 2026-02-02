#!/usr/bin/env python3
"""Test weekly zoomed prediction plot function."""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.fit_weight import (
    fit_weight_model_spline_optimized,
    extract_predictions,
)
from src.models.plot_cyclic import plot_weekly_zoomed_predictions


def test_weekly_zoomed_plot():
    """Test the weekly zoomed prediction plot function."""
    print("Testing weekly zoomed prediction plot function...")

    # Fit model with prediction grid (hourly predictions every 5 days for speed)
    result = fit_weight_model_spline_optimized(
        data_dir="data",
        chains=1,
        iter_warmup=50,
        iter_sampling=50,
        fourier_harmonics=2,
        use_sparse=True,
        n_inducing_points=10,
        include_prediction_grid=True,
        prediction_hour=8.0,
        prediction_hour_step=3,  # 3-hour steps for fewer prediction points
        prediction_step_days=5,  # Every 5 days for speed
        cache=False,
        force_refit=True,
    )

    fit, idata, df, stan_data = result
    print(f"Fit successful. N_pred = {stan_data.get('N_pred', 0)}")

    # Extract predictions
    predictions = extract_predictions(idata, stan_data)
    if not predictions:
        print("No predictions extracted")
        sys.exit(1)

    print(f"Predictions extracted. Shape f_pred_mean: {predictions['f_pred_mean'].shape}")

    # Test 1: Default (middle of prediction range)
    print("\nTest 1: Default week (middle of prediction range)")
    try:
        plot_weekly_zoomed_predictions(
            predictions=predictions,
            df=df,
            stan_data=stan_data,
            model_name="Spline Optimized",
            output_path="output/weekly_zoomed_default.png",
            show_ci=True,
            show_observations=True,
        )
        print("  ✓ Default week plot created")
    except Exception as e:
        print(f"  ✗ Default week plot failed: {e}")
        raise

    # Test 2: Specific target date (use a date in the middle of data range)
    # Find a date with predictions
    start_timestamp = df["timestamp"].min()
    t_pred_days = predictions["t_pred"]
    dates_pred = [start_timestamp + pd.Timedelta(days=float(d)) for d in t_pred_days]
    mid_idx = len(dates_pred) // 2
    target_date = dates_pred[mid_idx]

    print(f"\nTest 2: Specific target date ({target_date.date()})")
    try:
        plot_weekly_zoomed_predictions(
            predictions=predictions,
            df=df,
            stan_data=stan_data,
            model_name="Spline Optimized",
            output_path="output/weekly_zoomed_target.png",
            target_date=target_date,
            show_ci=True,
            show_observations=True,
        )
        print("  ✓ Target date week plot created")
    except Exception as e:
        print(f"  ✗ Target date week plot failed: {e}")
        raise

    # Test 3: Explicit week start date
    week_start = target_date - pd.Timedelta(days=target_date.dayofweek)
    print(f"\nTest 3: Explicit week start ({week_start.date()})")
    try:
        plot_weekly_zoomed_predictions(
            predictions=predictions,
            df=df,
            stan_data=stan_data,
            model_name="Spline Optimized",
            output_path="output/weekly_zoomed_explicit.png",
            week_start_date=week_start,
            show_ci=True,
            show_observations=True,
        )
        print("  ✓ Explicit week start plot created")
    except Exception as e:
        print(f"  ✗ Explicit week start plot failed: {e}")
        raise

    # Test 4: Without CI and observations
    print("\nTest 4: Without CI and observations")
    try:
        plot_weekly_zoomed_predictions(
            predictions=predictions,
            df=df,
            stan_data=stan_data,
            model_name="Spline Optimized",
            output_path="output/weekly_zoomed_simple.png",
            target_date=target_date,
            show_ci=False,
            show_observations=False,
        )
        print("  ✓ Simple plot created")
    except Exception as e:
        print(f"  ✗ Simple plot failed: {e}")
        raise

    print("\nAll weekly zoomed plot tests passed!")
    print("Plots saved to output/weekly_zoomed_*.png")

    return True


if __name__ == "__main__":
    try:
        test_weekly_zoomed_plot()
        sys.exit(0)
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)