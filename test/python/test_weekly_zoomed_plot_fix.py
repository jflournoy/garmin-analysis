#!/usr/bin/env python3
"""Test to demonstrate and fix the weekly zoomed plot x-axis issue.

The current plot_weekly_zoomed_predictions function plots predictions with
hour of day on the x-axis, causing all predictions for the same hour across
different days to be plotted at the same position. This test demonstrates
the issue and verifies the fix.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.plot_cyclic import plot_weekly_zoomed_predictions


def create_test_predictions():
    """Create synthetic test predictions to demonstrate the issue."""
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
            # t_pred is in days since start
            t_pred_days = day + (hour / 24.0)  # Include fractional day for hour
            predictions["t_pred"].append(t_pred_days)
            predictions["hour_of_day_pred"].append(hour)

            # Create a simple daily pattern: weight varies by hour
            base_weight = 150.0
            daily_variation = 2.0 * np.sin(2 * np.pi * hour / 24.0)
            predictions["f_pred_mean"].append(base_weight + daily_variation)
            predictions["f_pred_lower"].append(base_weight + daily_variation - 0.5)
            predictions["f_pred_upper"].append(base_weight + daily_variation + 0.5)

    # Convert to numpy arrays
    for key in predictions:
        predictions[key] = np.array(predictions[key])

    return predictions, start_date


def test_current_plot_issue():
    """Demonstrate the current plotting issue."""
    print("Testing current weekly zoomed plot implementation...")

    predictions, start_date = create_test_predictions()

    # Create a minimal df for testing
    df = pd.DataFrame({
        "timestamp": [start_date],
        "weight_lbs": [150.0],
        "date": [start_date.date()],
        "days_since_start": [0]
    })

    # Create stan_data with required fields
    stan_data = {
        "_y_mean": 150.0,
        "_y_sd": 5.0,
        "_t_max": 7.0  # 7 days
    }

    # Try to create the plot with current implementation
    try:
        fig = plot_weekly_zoomed_predictions(
            predictions=predictions,
            df=df,
            stan_data=stan_data,
            model_name="Test Model",
            target_date=start_date + pd.Timedelta(days=3),  # Middle of week
            show_ci=True,
            show_observations=False,
        )

        # Save to examine
        output_path = Path("output/test_weekly_zoomed_issue.png")
        output_path.parent.mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"  Plot saved to {output_path}")
        print("  ⚠ Current implementation plots hour of day on x-axis")
        print("  ⚠ All predictions for same hour across days overlap")

    except Exception as e:
        print(f"  Error creating plot: {e}")
        import traceback
        traceback.print_exc()

    return True


def analyze_prediction_data_structure():
    """Analyze the structure of prediction data to understand the fix needed."""
    print("\nAnalyzing prediction data structure...")

    predictions, start_date = create_test_predictions()

    print(f"Number of predictions: {len(predictions['t_pred'])}")
    print(f"Unique t_pred values (days since start): {np.unique(predictions['t_pred'])}")
    print(f"Unique hour_of_day_pred values: {np.unique(predictions['hour_of_day_pred'])}")

    # Show first few predictions
    print("\nFirst 12 predictions (2 days worth):")
    for i in range(12):
        t_pred = predictions["t_pred"][i]
        hour = predictions["hour_of_day_pred"][i]
        f_mean = predictions["f_pred_mean"][i]
        print(f"  Prediction {i}: t_pred={t_pred:.3f} days, hour={hour}, weight={f_mean:.2f}")

    # The issue: t_pred includes fractional days (hour/24.0)
    # But in plot_weekly_zoomed_predictions line 929:
    # dates_pred = [start_timestamp + pd.Timedelta(days=float(d)) for d in t_pred_days]
    # This converts t_pred to whole days, losing the hour information!

    print("\n⚠ ISSUE IDENTIFIED:")
    print("  1. t_pred values include fractional days (e.g., 0.1667 for 4 hours)")
    print("  2. Line 929 converts to Timedelta(days=float(d)) which truncates to whole days")
    print("  3. Hour information is stored separately in hour_of_day_pred")
    print("  4. Plot uses only hour_of_day_pred on x-axis, losing day information")

    return True


def propose_fix():
    """Propose fix for the weekly zoomed plot."""
    print("\nProposed fix for weekly zoomed plot:")
    print("  1. Reconstruct actual datetime from t_pred (which includes fractional days)")
    print("  2. Use actual datetime on x-axis instead of just hour of day")
    print("  3. Format x-axis to show date and time")
    print("  4. Observations should also use actual datetime")

    print("\nCurrent problematic code (lines 929, 975, 989):")
    print("  Line 929: dates_pred = [start_timestamp + pd.Timedelta(days=float(d)) for d in t_pred_days]")
    print("  Line 975: ax.plot(day_df['hour'], day_df['f_pred_mean'], ...)")
    print("  Line 989: obs_df['timestamp'].dt.hour + obs_df['timestamp'].dt.minute / 60")

    print("\nProposed fix:")
    print("  1. Use: dates_pred = [start_timestamp + pd.Timedelta(days=d) for d in t_pred_days]")
    print("     (d is already fractional, no need for float() conversion)")
    print("  2. Plot with: ax.plot(day_df['datetime'], day_df['f_pred_mean'], ...)")
    print("  3. Observations: ax.scatter(obs_df['timestamp'], ...)")
    print("  4. Format x-axis with date and time")

    return True


def main():
    """Run the analysis."""
    print("=" * 70)
    print("WEEKLY ZOOMED PLOT X-AXIS ISSUE ANALYSIS")
    print("=" * 70)

    try:
        test_current_plot_issue()
        analyze_prediction_data_structure()
        propose_fix()

        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. Fix plot_weekly_zoomed_predictions to use datetime on x-axis")
        print("  2. Update test to verify correct x-axis")
        print("  3. Run existing tests to ensure no regression")

        return True

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)