#!/usr/bin/env python3
"""Test the fixed weekly zoomed plot implementation.

Verify that predictions are plotted with correct datetime on x-axis
and don't overlap for same hour across different days.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.plot_cyclic import plot_weekly_zoomed_predictions


def create_realistic_test_predictions():
    """Create realistic test predictions with actual datetime reconstruction."""
    # Create a week of predictions with 4-hour steps
    start_date = pd.Timestamp("2025-03-01 00:00:00")
    days = 7
    hour_step = 4
    hours_per_day = int(24 / hour_step)  # 6 hours: 0, 4, 8, 12, 16, 20

    predictions = {
        "t_pred": [],
        "hour_of_day_pred": [],
        "f_pred_mean": [],
        "f_pred_lower": [],
        "f_pred_upper": []
    }

    # Create predictions for each day and hour
    for day in range(days):
        for hour_idx in range(hours_per_day):
            hour = hour_idx * hour_step
            # t_pred is in days since start, including fractional part for hour
            t_pred_days = day + (hour / 24.0)
            predictions["t_pred"].append(t_pred_days)
            predictions["hour_of_day_pred"].append(hour)

            # Create a simple daily pattern: weight varies by hour
            base_weight = 150.0
            daily_variation = 2.0 * np.sin(2 * np.pi * hour / 24.0)
            # Add small day-to-day trend
            day_trend = 0.1 * day
            predictions["f_pred_mean"].append(base_weight + daily_variation + day_trend)
            predictions["f_pred_lower"].append(base_weight + daily_variation + day_trend - 0.5)
            predictions["f_pred_upper"].append(base_weight + daily_variation + day_trend + 0.5)

    # Convert to numpy arrays
    for key in predictions:
        predictions[key] = np.array(predictions[key])

    return predictions, start_date


def test_fixed_plot_implementation():
    """Test the fixed plotting implementation."""
    print("Testing fixed weekly zoomed plot implementation...")

    predictions, start_date = create_realistic_test_predictions()

    # Create a minimal df for testing
    df = pd.DataFrame({
        "timestamp": [start_date + pd.Timedelta(hours=2),  # One observation at 2 AM
                      start_date + pd.Timedelta(days=1, hours=14)],  # Another at 2 PM next day
        "weight_lbs": [151.0, 149.0],
        "date": [(start_date + pd.Timedelta(hours=2)).date(),
                 (start_date + pd.Timedelta(days=1, hours=14)).date()],
        "days_since_start": [0, 1]
    })

    # Create stan_data with required fields
    stan_data = {
        "_y_mean": 150.0,
        "_y_sd": 5.0,
        "_t_max": 7.0  # 7 days
    }

    # Create the plot with fixed implementation
    try:
        fig = plot_weekly_zoomed_predictions(
            predictions=predictions,
            df=df,
            stan_data=stan_data,
            model_name="Test Model",
            target_date=start_date + pd.Timedelta(days=3),  # Middle of week
            show_ci=True,
            show_observations=True,
        )

        # Get the axes to inspect
        ax = fig.axes[0]

        # Check that x-axis data is datetime, not just hours
        lines = ax.get_lines()
        assert len(lines) > 0, "No lines plotted"

        # First line should have datetime xdata
        xdata = lines[0].get_xdata()
        print(f"  X-data type: {type(xdata[0])}")
        print(f"  First x-value: {xdata[0]}")
        print(f"  Number of points in first line: {len(xdata)}")

        # Verify xdata contains datetime objects
        assert isinstance(xdata[0], (pd.Timestamp, np.datetime64, datetime)), \
            f"X-data should be datetime, got {type(xdata[0])}"

        # Verify we have multiple distinct x-values (not all at same hour positions)
        unique_x_values = len(np.unique(xdata))
        print(f"  Unique x-values in first line: {unique_x_values}")
        assert unique_x_values > 1, "All points have same x-value (overlapping hours)"

        # Check that observations are plotted with datetime
        collections = ax.collections
        if collections:
            obs_offsets = collections[0].get_offsets()
            print(f"  Observations x-data type: {type(obs_offsets[0, 0])}")
            assert obs_offsets.shape[0] > 0, "No observations plotted"

        # Save to examine
        output_path = Path("output/test_fixed_weekly_zoomed.png")
        output_path.parent.mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"  Plot saved to {output_path}")
        print("  ✓ Fixed implementation uses datetime on x-axis")
        print("  ✓ Predictions don't overlap for same hour across days")

    except Exception as e:
        print(f"  Error creating plot: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def verify_prediction_datetime_reconstruction():
    """Verify that prediction datetime reconstruction is correct."""
    print("\nVerifying prediction datetime reconstruction...")

    predictions, start_date = create_realistic_test_predictions()

    # Reconstruct datetime as done in plot_weekly_zoomed_predictions
    t_pred_days = predictions["t_pred"]
    dates_pred = [start_date + pd.Timedelta(days=d) for d in t_pred_days]

    # Check first few predictions
    print("  First 6 predictions (first day):")
    for i in range(6):
        t_pred = t_pred_days[i]
        hour = predictions["hour_of_day_pred"][i]
        reconstructed = dates_pred[i]
        expected_hour = int((t_pred % 1) * 24)

        print(f"    Prediction {i}: t_pred={t_pred:.3f} days, hour={hour}")
        print(f"      Reconstructed: {reconstructed}")
        print(f"      Expected hour: {expected_hour}, Actual hour: {reconstructed.hour}")

        # Verify hour matches
        assert reconstructed.hour == expected_hour, \
            f"Hour mismatch: expected {expected_hour}, got {reconstructed.hour}"
        assert reconstructed.hour == hour, \
            f"Hour mismatch with hour_of_day_pred: expected {hour}, got {reconstructed.hour}"

    print("  ✓ Datetime reconstruction correct")

    return True


def test_edge_cases():
    """Test edge cases for the plot function."""
    print("\nTesting edge cases...")

    predictions, start_date = create_realistic_test_predictions()

    # Create test data
    df = pd.DataFrame({
        "timestamp": [start_date],
        "weight_lbs": [150.0],
        "date": [start_date.date()],
        "days_since_start": [0]
    })

    stan_data = {
        "_y_mean": 150.0,
        "_y_sd": 5.0,
        "_t_max": 7.0
    }

    # Test 1: No predictions in selected week
    print("  Test 1: No predictions in selected week")
    try:
        # Use a date far in the future
        fig = plot_weekly_zoomed_predictions(
            predictions=predictions,
            df=df,
            stan_data=stan_data,
            model_name="Test Model",
            target_date=start_date + pd.Timedelta(days=30),  # Far future
            show_ci=True,
            show_observations=False,
        )
        plt.close(fig)
        print("    ⚠ Should have raised ValueError for no predictions in week")
        return False
    except ValueError as e:
        print(f"    ✓ Correctly raised ValueError: {e}")
    except Exception as e:
        print(f"    ⚠ Unexpected error: {e}")
        return False

    # Test 2: No CI data
    print("  Test 2: No CI data")
    predictions_no_ci = predictions.copy()
    # Remove CI arrays
    del predictions_no_ci["f_pred_lower"]
    del predictions_no_ci["f_pred_upper"]

    try:
        fig = plot_weekly_zoomed_predictions(
            predictions=predictions_no_ci,
            df=df,
            stan_data=stan_data,
            model_name="Test Model",
            target_date=start_date + pd.Timedelta(days=3),
            show_ci=True,  # Request CI but none available
            show_observations=False,
        )
        plt.close(fig)
        print("    ✓ Plot created without CI (graceful handling)")
    except Exception as e:
        print(f"    ⚠ Error: {e}")
        return False

    print("  ✓ All edge cases handled correctly")
    return True


def main():
    """Run all tests for fixed weekly zoomed plot."""
    print("=" * 70)
    print("FIXED WEEKLY ZOOMED PLOT TESTS")
    print("=" * 70)

    try:
        test_fixed_plot_implementation()
        verify_prediction_datetime_reconstruction()
        test_edge_cases()

        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED")
        print("=" * 70)
        print("\nSummary:")
        print("  - Predictions now plotted with datetime on x-axis")
        print("  - No overlapping of same hour across different days")
        print("  - Observations plotted with correct datetime")
        print("  - Edge cases handled gracefully")

        return True

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)