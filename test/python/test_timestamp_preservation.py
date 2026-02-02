"""Test timestamp preservation in weight data loading."""
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.weight import load_weight_data, prepare_stan_data


def test_load_weight_data_returns_timestamps():
    """Test that load_weight_data preserves timestamp information."""
    # Load data
    df = load_weight_data()

    # Check required columns
    assert "date" in df.columns
    assert "weight_lbs" in df.columns
    assert "days_since_start" in df.columns

    # NEW: Check for timestamp column
    assert "timestamp" in df.columns, "DataFrame should have 'timestamp' column"

    # Check timestamp is datetime type
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"]), \
        "timestamp should be datetime type"

    # Check we have reasonable number of rows
    assert len(df) > 100, f"Expected >100 rows, got {len(df)}"

    # Check timestamps have time component (not all midnight)
    hour_counts = df["timestamp"].dt.hour.value_counts()
    unique_hours = len(hour_counts)
    assert unique_hours > 1, f"Timestamps should have varying hours, got {unique_hours} unique hours"

    print(f"✓ Data loaded: {len(df)} rows with timestamps")
    print(f"✓ Hour distribution: {sorted(hour_counts.index.tolist())}")


def test_prepare_stan_data_includes_hour_info():
    """Test that prepare_stan_data includes hour-of-day and period information."""
    # Load data
    df = load_weight_data()

    # Prepare Stan data
    stan_data = prepare_stan_data(df)

    # Check existing fields
    assert "N" in stan_data
    assert "t" in stan_data
    assert "y" in stan_data

    # NEW: Check for hour-related fields
    assert "t_hours" in stan_data, "Should have t_hours (continuous hours)"
    assert "hour_of_day" in stan_data, "Should have hour_of_day (0-24)"
    assert "period_daily" in stan_data, "Should have period_daily parameter"

    # Check dimensions
    N = stan_data["N"]
    assert len(stan_data["t_hours"]) == N, f"t_hours should have length {N}"
    assert len(stan_data["hour_of_day"]) == N, f"hour_of_day should have length {N}"

    # Check hour_of_day values are in [0, 24)
    hour_of_day = stan_data["hour_of_day"]
    assert all(0 <= h < 24 for h in hour_of_day), \
        f"hour_of_day should be in [0, 24), got min={min(hour_of_day)}, max={max(hour_of_day)}"

    # Check period_daily is reasonable (24 / total_hours)
    t_hours = stan_data["t_hours"]
    total_hours = max(t_hours) - min(t_hours)
    expected_period = 24.0 / total_hours
    actual_period = stan_data["period_daily"]

    # Allow small floating point differences
    assert abs(actual_period - expected_period) < 1e-10, \
        f"period_daily should be 24/total_hours ≈ {expected_period}, got {actual_period}"

    print(f"✓ Stan data includes hour info: N={N}, period_daily={actual_period:.6f}")
    print(f"✓ Hour range: {min(hour_of_day):.1f}-{max(hour_of_day):.1f}")


def test_backward_compatibility():
    """Test that existing models can still use the data (t field preserved)."""
    df = load_weight_data()
    stan_data = prepare_stan_data(df)

    # Original fields should still exist
    assert "t" in stan_data
    assert "y" in stan_data

    # t should still be scaled to [0, 1]
    t = stan_data["t"]
    assert min(t) >= 0 and max(t) <= 1, f"t should be in [0, 1], got [{min(t)}, {max(t)}]"

    # t_hours should be continuous hours (not scaled to [0, 1])
    t_hours = stan_data["t_hours"]
    assert max(t_hours) > 1, f"t_hours should be in hours, max={max(t_hours)}"

    print("✓ Backward compatibility maintained: t in [0,1], t_hours in hours")


def test_hour_distribution_matches_raw_data():
    """Test that hour distribution matches what we saw in raw JSON analysis."""
    df = load_weight_data()

    # Get hour distribution
    hours = df["timestamp"].dt.hour
    hour_counts = hours.value_counts().sort_index()

    # Expected: Most measurements in 14:00-17:00 GMT (morning US time)
    # Missing: 06:00-12:00 GMT (sleeping hours)

    peak_hours = [14, 15, 16, 17]
    missing_hours = list(range(6, 13))  # 06:00-12:00

    # Check peak hours have measurements
    for h in peak_hours:
        assert h in hour_counts.index, f"Peak hour {h}:00 should have measurements"

    # Check some missing hours (allow some measurements in these hours)
    # But they should have fewer measurements than peak hours
    if missing_hours[0] in hour_counts.index:
        peak_count = sum(hour_counts.get(h, 0) for h in peak_hours)
        missing_count = sum(hour_counts.get(h, 0) for h in missing_hours[:3])
        assert peak_count > missing_count, \
            "Peak hours should have more measurements than missing hours"

    print(f"✓ Hour distribution: {sorted(hour_counts.index.tolist())}")
    print(f"✓ Peak hours {peak_hours} have measurements")


if __name__ == "__main__":
    # Run tests
    print("Running timestamp preservation tests...")
    print("=" * 60)

    test_load_weight_data_returns_timestamps()
    print()

    test_prepare_stan_data_includes_hour_info()
    print()

    test_backward_compatibility()
    print()

    test_hour_distribution_matches_raw_data()
    print()

    print("=" * 60)
    print("All tests passed! ✓")