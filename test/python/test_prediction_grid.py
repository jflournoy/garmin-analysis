#!/usr/bin/env python3
"""Test prediction grid generation and validation.

This test verifies that prediction grids are generated correctly with proper
dimensions and timestamps. If the time step is M hours and the number of days is N days,
we should see int(24/M) x N predicted points with the correct day-time stamps.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.weight import prepare_stan_data


def test_prediction_grid_dimensions():
    """Test that prediction grid has correct dimensions."""
    print("Testing prediction grid dimensions...")

    # Create synthetic test data
    dates = pd.date_range(start="2025-01-01", end="2025-01-10", freq="D")
    df = pd.DataFrame({
        "timestamp": dates,
        "weight_lbs": np.random.normal(150, 5, len(dates)),
        "date": dates.date,
        "days_since_start": np.arange(len(dates))
    })

    # Test 1: Single hour per day (M = None, N = 10 days)
    print("\nTest 1: Single hour per day (M = None, N = 10 days)")
    stan_data = prepare_stan_data(
        df,
        include_prediction_grid=True,
        prediction_hour=8.0,
        prediction_step_days=1,
    )

    assert "N_pred" in stan_data, "N_pred should be in stan_data"
    assert "t_pred" in stan_data, "t_pred should be in stan_data"
    assert "hour_of_day_pred" in stan_data, "hour_of_day_pred should be in stan_data"

    N_pred = stan_data["N_pred"]
    t_pred = np.array(stan_data["t_pred"])
    hour_of_day_pred = np.array(stan_data["hour_of_day_pred"])

    print(f"  N_pred: {N_pred}")
    print(f"  t_pred shape: {t_pred.shape}")
    print(f"  hour_of_day_pred shape: {hour_of_day_pred.shape}")

    # With single hour per day and daily step, should have 10 predictions
    assert N_pred == 10, f"Expected 10 predictions, got {N_pred}"
    assert len(t_pred) == 10, f"Expected 10 time points, got {len(t_pred)}"
    assert len(hour_of_day_pred) == 10, f"Expected 10 hour values, got {len(hour_of_day_pred)}"

    # All hours should be 8.0
    assert np.allclose(hour_of_day_pred, 8.0), f"All hours should be 8.0, got {hour_of_day_pred}"

    print("  ✓ Test 1 passed")

    # Test 2: Multiple hours per day (M = 4 hours, N = 10 days)
    print("\nTest 2: Multiple hours per day (M = 4 hours, N = 10 days)")
    stan_data = prepare_stan_data(
        df,
        include_prediction_grid=True,
        prediction_hour=8.0,  # Ignored when prediction_hour_step is provided
        prediction_hour_step=4.0,
        prediction_step_days=1,
    )

    N_pred = stan_data["N_pred"]
    t_pred = np.array(stan_data["t_pred"])
    hour_of_day_pred = np.array(stan_data["hour_of_day_pred"])

    print(f"  N_pred: {N_pred}")
    print(f"  t_pred shape: {t_pred.shape}")
    print(f"  hour_of_day_pred shape: {hour_of_day_pred.shape}")

    # With 4-hour steps: 0, 4, 8, 12, 16, 20 = 6 hours per day
    # 10 days * 6 hours = 60 predictions
    expected_hours_per_day = int(24 / 4)  # Should be 6
    expected_predictions = 10 * expected_hours_per_day  # Should be 60

    assert N_pred == expected_predictions, f"Expected {expected_predictions} predictions, got {N_pred}"
    assert len(t_pred) == expected_predictions, f"Expected {expected_predictions} time points, got {len(t_pred)}"
    assert len(hour_of_day_pred) == expected_predictions, f"Expected {expected_predictions} hour values, got {len(hour_of_day_pred)}"

    # Check hour values
    unique_hours = np.unique(hour_of_day_pred)
    print(f"  Unique hours: {unique_hours}")
    expected_hours = [0, 4, 8, 12, 16, 20]
    assert np.allclose(unique_hours, expected_hours), f"Expected hours {expected_hours}, got {unique_hours}"

    # Check that hours repeat correctly across days
    hours_per_day = len(unique_hours)
    for day_idx in range(10):
        day_start = day_idx * hours_per_day
        day_end = (day_idx + 1) * hours_per_day
        day_hours = hour_of_day_pred[day_start:day_end]
        assert np.allclose(day_hours, expected_hours), f"Day {day_idx} hours incorrect: {day_hours}"

    print("  ✓ Test 2 passed")

    # Test 3: 2-day step (M = 4 hours, N = 5 prediction days)
    print("\nTest 3: 2-day step (M = 4 hours, N = 5 prediction days)")
    stan_data = prepare_stan_data(
        df,
        include_prediction_grid=True,
        prediction_hour_step=4.0,
        prediction_step_days=2,
    )

    N_pred = stan_data["N_pred"]
    t_pred = np.array(stan_data["t_pred"])
    hour_of_day_pred = np.array(stan_data["hour_of_day_pred"])

    print(f"  N_pred: {N_pred}")
    print(f"  t_pred shape: {t_pred.shape}")
    print(f"  hour_of_day_pred shape: {hour_of_day_pred.shape}")

    # With 2-day steps: days 0, 2, 4, 6, 8 = 5 days
    # 5 days * 6 hours = 30 predictions
    expected_days = 5  # 0, 2, 4, 6, 8
    expected_predictions = expected_days * 6  # 6 hours per day

    assert N_pred == expected_predictions, f"Expected {expected_predictions} predictions, got {N_pred}"

    print("  ✓ Test 3 passed")

    # Test 4: Verify t_pred scaling
    print("\nTest 4: Verify t_pred scaling")
    # t_pred should be scaled to [0, 1] based on t_max
    df["days_since_start"].max()
    t_pred_scaled = np.array(stan_data["t_pred"])

    # Check that all values are between 0 and 1
    assert np.all(t_pred_scaled >= 0), f"t_pred values should be >= 0, got min {t_pred_scaled.min()}"
    assert np.all(t_pred_scaled <= 1), f"t_pred values should be <= 1, got max {t_pred_scaled.max()}"

    print(f"  t_pred range: [{t_pred_scaled.min():.4f}, {t_pred_scaled.max():.4f}]")
    print("  ✓ Test 4 passed")

    # Test 5: Edge case - hour step that doesn't divide 24 evenly
    print("\nTest 5: Edge case - hour step that doesn't divide 24 evenly (M = 7 hours)")
    stan_data = prepare_stan_data(
        df,
        include_prediction_grid=True,
        prediction_hour_step=7.0,
        prediction_step_days=1,
    )

    N_pred = stan_data["N_pred"]
    hour_of_day_pred = np.array(stan_data["hour_of_day_pred"])
    unique_hours = np.unique(hour_of_day_pred)

    print(f"  N_pred: {N_pred}")
    print(f"  Unique hours: {unique_hours}")

    # With 7-hour steps: 0, 7, 14, 21 = 4 hours per day
    expected_hours = [0, 7, 14, 21]
    assert np.allclose(unique_hours, expected_hours), f"Expected hours {expected_hours}, got {unique_hours}"
    assert N_pred == 10 * 4, f"Expected 40 predictions, got {N_pred}"

    print("  ✓ Test 5 passed")

    print("\n✅ All prediction grid dimension tests passed!")
    return True


def test_prediction_grid_timestamps():
    """Test that prediction grid timestamps are correct."""
    print("\nTesting prediction grid timestamps...")

    # Create synthetic test data with specific start date
    start_date = pd.Timestamp("2025-01-01 00:00:00")
    dates = pd.date_range(start=start_date, periods=5, freq="D")
    df = pd.DataFrame({
        "timestamp": dates,
        "weight_lbs": np.random.normal(150, 5, len(dates)),
        "date": dates.date,
        "days_since_start": np.arange(len(dates))
    })

    # Test with 3-hour steps, daily predictions
    stan_data = prepare_stan_data(
        df,
        include_prediction_grid=True,
        prediction_hour_step=3.0,
        prediction_step_days=1,
    )

    t_pred = np.array(stan_data["t_pred"])
    hour_of_day_pred = np.array(stan_data["hour_of_day_pred"])

    # Reconstruct timestamps from t_pred and hour_of_day_pred
    t_max = df["days_since_start"].max()
    t_pred_days = t_pred * t_max  # Convert back to days

    # Check that t_pred_days are integers (since we use daily steps)
    # They should be 0, 1, 2, 3, 4
    expected_days = np.arange(5)

    # Get unique day indices from t_pred_days (accounting for floating point)
    unique_day_indices = np.unique(np.round(t_pred_days))
    assert np.allclose(unique_day_indices, expected_days), f"Expected days {expected_days}, got {unique_day_indices}"

    # Check hour distribution
    unique_hours = np.unique(hour_of_day_pred)
    expected_hours = [0, 3, 6, 9, 12, 15, 18, 21]  # 3-hour steps
    assert np.allclose(unique_hours, expected_hours), f"Expected hours {expected_hours}, got {unique_hours}"

    # Verify ordering: days then hours within each day
    hours_per_day = len(expected_hours)
    for day_idx in range(5):
        day_start = day_idx * hours_per_day
        day_end = (day_idx + 1) * hours_per_day
        day_hours = hour_of_day_pred[day_start:day_end]
        assert np.allclose(day_hours, expected_hours), f"Day {day_idx} hours incorrect"

        # Check that t_pred values increase monotonically
        day_t_pred = t_pred[day_start:day_end]
        assert np.all(np.diff(day_t_pred) >= 0), f"Day {day_idx} t_pred not monotonic"

    print("  ✓ Prediction grid timestamps test passed")

    # Test reconstruction of actual datetime
    print("\nTesting datetime reconstruction...")
    # The first prediction should be at start_date + 0 days + 0 hours = 2025-01-01 00:00:00
    # The second prediction should be at start_date + 0 days + 3 hours = 2025-01-01 03:00:00
    # etc.

    # For day 0, hour 0
    first_t_pred_days = t_pred_days[0]
    first_hour = hour_of_day_pred[0]

    expected_first_timestamp = start_date + pd.Timedelta(days=first_t_pred_days, hours=first_hour)
    assert expected_first_timestamp == start_date, f"First timestamp should be {start_date}, got {expected_first_timestamp}"

    # For day 0, hour 3 (second prediction)
    second_t_pred_days = t_pred_days[1]
    second_hour = hour_of_day_pred[1]

    expected_second_timestamp = start_date + pd.Timedelta(days=second_t_pred_days, hours=second_hour)
    expected_time = start_date + pd.Timedelta(hours=3)
    assert expected_second_timestamp == expected_time, f"Second timestamp should be {expected_time}, got {expected_second_timestamp}"

    print("  ✓ Datetime reconstruction test passed")

    print("\n✅ All prediction grid timestamp tests passed!")
    return True


def test_prediction_function_returns_per_row():
    """Test that prediction function returns point predictions for each row."""
    print("\nTesting prediction function returns per-row predictions...")

    # This test will need to be implemented after we have the prediction extraction function
    # For now, we'll create a placeholder test

    print("  ⚠ Prediction function test placeholder - to be implemented")
    print("  This test will verify that extract_predictions() returns predictions for each row")

    return True


def main():
    """Run all prediction grid tests."""
    print("=" * 70)
    print("PREDICTION GRID GENERATION TESTS")
    print("=" * 70)

    try:
        test_prediction_grid_dimensions()
        test_prediction_grid_timestamps()
        test_prediction_function_returns_per_row()

        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED")
        print("=" * 70)
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