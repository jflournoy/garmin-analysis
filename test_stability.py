#!/usr/bin/env python
"""Test numerical stability fixes."""
import sys
sys.path.insert(0, "src")

from src.models.fit_weight import fit_weight_model_spline_optimized

def main():
    print("Testing numerical stability with bounds...")
    try:
        # Run with minimal iterations
        fit, idata, df, stan_data = fit_weight_model_spline_optimized(
            data_dir="data",
            chains=1,
            iter_warmup=10,
            iter_sampling=10,
            use_sparse=False,
            include_prediction_grid=True,
            prediction_hour=8.0,
            prediction_step_days=1,
            cache=False,
            force_refit=True,
        )
        print("✓ Model fitting completed without infinite values error")

        # Check for warnings (cmdstanpy stores warnings in stderr)
        # Check divergent transitions
        if hasattr(fit, 'divergences'):
            divergences = fit.divergences
            if divergences.sum() > 0:
                print(f"  Divergent transitions: {divergences.sum()}")
            else:
                print("  ✓ No divergent transitions")
        else:
            print("  (divergences attribute not available)")

        return 0

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())