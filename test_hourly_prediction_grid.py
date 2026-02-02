#!/usr/bin/env python3
"""Test hourly prediction grid functionality."""
import sys
sys.path.insert(0, '.')

import numpy as np
from src.models.fit_weight import (
    fit_weight_model_spline_optimized,
    fit_weight_model_cyclic_optimized,
    fit_weight_model_optimized,
    fit_weight_model_flexible_optimized,
    extract_predictions
)
from src.data.weight import load_weight_data, prepare_stan_data


def test_hourly_grid_data_preparation():
    """Test that prepare_stan_data generates correct hourly prediction grid."""
    print("Testing hourly prediction grid data preparation...")

    df = load_weight_data("data")

    # Test with 6-hour step (4 predictions per day)
    stan_data = prepare_stan_data(
        df,
        fourier_harmonics=2,
        use_sparse=False,
        include_prediction_grid=True,
        prediction_hour_step=6.0,
        prediction_step_days=1
    )

    n_pred = stan_data.get('N_pred', 0)
    t_pred = np.array(stan_data.get('t_pred', []))
    hour_of_day_pred = np.array(stan_data.get('hour_of_day_pred', []))

    print(f"  N_pred with 6-hour step: {n_pred}")
    print(f"  t_pred shape: {len(t_pred)}")
    print(f"  hour_of_day_pred shape: {len(hour_of_day_pred)}")

    # Calculate expected predictions
    t = df["days_since_start"].values
    t_min = t.min()
    t_max = t.max()
    t_pred_days = np.arange(t_min, t_max + 1, 1)  # daily step
    t_pred_days = t_pred_days[t_pred_days <= t_max]

    # With 6-hour step: 0, 6, 12, 18 hours (4 per day)
    prediction_hours = np.arange(0, 24, 6)
    prediction_hours = prediction_hours[prediction_hours < 24]

    expected_n_pred = len(t_pred_days) * len(prediction_hours)

    print(f"  Expected N_pred: {expected_n_pred}")
    print(f"  Actual N_pred: {n_pred}")

    assert n_pred == expected_n_pred, f"Expected {expected_n_pred} predictions, got {n_pred}"
    assert len(t_pred) == n_pred, "t_pred length mismatch"
    assert len(hour_of_day_pred) == n_pred, "hour_of_day_pred length mismatch"

    # Check hour values are correct
    unique_hours = np.unique(hour_of_day_pred)
    print(f"  Unique hours in grid: {sorted(unique_hours)}")
    assert np.allclose(sorted(unique_hours), prediction_hours), f"Hours don't match expected {prediction_hours}"

    print("  ✓ Hourly grid data preparation test passed")
    return True


def test_model_with_hourly_grid(model_name, fit_func, **model_kwargs):
    """Test a specific model with hourly prediction grid."""
    print(f"\nTesting {model_name} with hourly prediction grid...")

    try:
        # Fit model with minimal iterations
        result = fit_func(
            data_dir="data",
            chains=1,
            iter_warmup=10,
            iter_sampling=10,
            include_prediction_grid=True,
            prediction_hour_step=6.0,  # 4 predictions per day
            prediction_step_days=1,
            cache=False,
            force_refit=True,
            **model_kwargs
        )

        fit, idata, df, stan_data = result
        n_pred = stan_data.get('N_pred', 0)

        print("  Model fitted successfully")
        print(f"  N_pred: {n_pred}")

        # Extract predictions
        predictions = extract_predictions(idata, stan_data)

        if predictions:
            print(f"  Predictions extracted: {len(predictions['t_pred'])} points")
            print(f"  Prediction range: {predictions['f_pred_mean'].min():.2f} - {predictions['f_pred_mean'].max():.2f} lbs")

            # Verify prediction count matches
            assert len(predictions['t_pred']) == n_pred, "Prediction count mismatch"
            assert len(predictions['hour_of_day_pred']) == n_pred, "Hour array length mismatch"

            # Check that hour_of_day_pred is in predictions
            assert 'hour_of_day_pred' in predictions, "hour_of_day_pred missing from predictions"

            # Check hour values are correct (0, 6, 12, 18)
            unique_hours = np.unique(predictions['hour_of_day_pred'])
            expected_hours = np.array([0.0, 6.0, 12.0, 18.0])
            assert np.allclose(sorted(unique_hours), expected_hours), f"Hours don't match expected {expected_hours}"

            print(f"  ✓ {model_name} hourly prediction test passed")
            return True
        else:
            print(f"  ⚠ No predictions extracted for {model_name}")
            return False

    except Exception as e:
        print(f"  ✗ Error testing {model_name}: {e}")
        raise


def main():
    """Run all hourly prediction grid tests."""
    print("=" * 70)
    print("HOURLY PREDICTION GRID TESTS")
    print("=" * 70)

    all_passed = True

    # Test 1: Data preparation
    try:
        test_hourly_grid_data_preparation()
    except Exception as e:
        print(f"✗ Data preparation test failed: {e}")
        all_passed = False

    # Test 2: Spline optimized model
    try:
        test_model_with_hourly_grid(
            "spline_optimized",
            fit_weight_model_spline_optimized,
            fourier_harmonics=2,
            use_sparse=False,
            n_inducing_points=50
        )
    except Exception as e:
        print(f"✗ Spline optimized model test failed: {e}")
        all_passed = False

    # Test 3: Cyclic optimized model
    try:
        test_model_with_hourly_grid(
            "cyclic_optimized",
            fit_weight_model_cyclic_optimized,
            use_sparse=False,
            n_inducing_points=50
        )
    except Exception as e:
        print(f"✗ Cyclic optimized model test failed: {e}")
        all_passed = False

    # Test 4: Original optimized model
    try:
        test_model_with_hourly_grid(
            "original_optimized",
            fit_weight_model_optimized,
            use_sparse=False,
            n_inducing_points=50
        )
    except Exception as e:
        print(f"✗ Original optimized model test failed: {e}")
        all_passed = False

    # Test 5: Flexible optimized model
    try:
        test_model_with_hourly_grid(
            "flexible_optimized",
            fit_weight_model_flexible_optimized,
            alpha_prior_sd=1.0,
            rho_prior_shape=5.0,
            rho_prior_scale=1.0,
            sigma_prior_sd=0.5,
            use_sparse=False,
            n_inducing_points=50
        )
    except Exception as e:
        print(f"✗ Flexible optimized model test failed: {e}")
        all_passed = False

    # Test 6: Different hour steps
    print("\nTesting different hour step values...")
    try:
        df = load_weight_data("data")

        for hour_step in [2.0, 3.0, 4.0, 8.0]:
            stan_data = prepare_stan_data(
                df,
                fourier_harmonics=2,
                use_sparse=False,
                include_prediction_grid=True,
                prediction_hour_step=hour_step,
                prediction_step_days=1
            )

            n_pred = stan_data.get('N_pred', 0)
            hour_of_day_pred = np.array(stan_data.get('hour_of_day_pred', []))

            # Calculate expected hours
            expected_hours = np.arange(0, 24, hour_step)
            expected_hours = expected_hours[expected_hours < 24]

            # Calculate expected count
            t = df["days_since_start"].values
            t_min = t.min()
            t_max = t.max()
            t_pred_days = np.arange(t_min, t_max + 1, 1)
            t_pred_days = t_pred_days[t_pred_days <= t_max]
            expected_n_pred = len(t_pred_days) * len(expected_hours)

            # Check
            unique_hours = np.unique(hour_of_day_pred)
            assert np.allclose(sorted(unique_hours), expected_hours), f"Hours don't match for step {hour_step}"
            assert n_pred == expected_n_pred, f"Prediction count mismatch for step {hour_step}"

            print(f"  ✓ Hour step {hour_step}: {len(expected_hours)} hours/day, {n_pred} total predictions")

    except Exception as e:
        print(f"✗ Hour step test failed: {e}")
        all_passed = False

    # Summary
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL HOURLY PREDICTION GRID TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)