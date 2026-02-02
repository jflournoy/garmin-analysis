#!/usr/bin/env python3
"""Quick test of WAIC/LOO comparison function."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.fit_weight import (
    fit_weight_model,
    fit_weight_model_flexible,
    fit_weight_model_cyclic,
    fit_weight_model_spline,
    compare_models_waic_loo,
)

def test_waic_loo_single_model():
    """Test WAIC/LOO with just original model (minimal iterations)."""
    print("Testing WAIC/LOO with original model (chains=1, iter=10)...")
    try:
        fit, idata, df, stan_data = fit_weight_model(
            chains=1,
            iter_warmup=10,
            iter_sampling=10,
            cache=False,
            force_refit=True,
        )
        print(f"  Model fitted successfully: {fit.runset._args}")

        # Test WAIC/LOO comparison with single model
        print("\nComputing WAIC/LOO...")
        df_compare = compare_models_waic_loo(
            idata_original=idata,
            print_summary=True,
        )
        print("\nComparison DataFrame:")
        print(df_compare)

        # Check that WAIC and LOO values are present
        assert df_compare.index[0] == "original"
        assert df_compare["waic"].notna().all()
        assert df_compare["loo"].notna().all()
        print("✓ WAIC/LOO computed successfully")

        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_waic_loo_multiple_models():
    """Test WAIC/LOO with multiple models (minimal iterations)."""
    print("\nTesting WAIC/LOO with multiple models...")
    try:
        # Fit original model (already fitted above, but refit for consistency)
        print("  Fitting original model...")
        fit_orig, idata_orig, df_orig, stan_data_orig = fit_weight_model(
            chains=1, iter_warmup=10, iter_sampling=10, cache=False, force_refit=True
        )

        # Fit flexible model
        print("  Fitting flexible model...")
        fit_flex, idata_flex, df_flex, stan_data_flex = fit_weight_model_flexible(
            chains=1, iter_warmup=10, iter_sampling=10, cache=False, force_refit=True
        )

        # Fit cyclic model (requires hour data)
        print("  Fitting cyclic model...")
        fit_cyclic, idata_cyclic, df_cyclic, stan_data_cyclic = fit_weight_model_cyclic(
            chains=1, iter_warmup=10, iter_sampling=10, cache=False, force_refit=True
        )

        # Fit spline model (requires hour data and K)
        print("  Fitting spline model...")
        fit_spline, idata_spline, df_spline, stan_data_spline = fit_weight_model_spline(
            chains=1, iter_warmup=10, iter_sampling=10, cache=False, force_refit=True,
            fourier_harmonics=2,
        )

        # Compare all four models
        print("\nComparing all four models...")
        df_compare = compare_models_waic_loo(
            idata_original=idata_orig,
            idata_flexible=idata_flex,
            idata_cyclic=idata_cyclic,
            idata_spline=idata_spline,
            print_summary=True,
        )

        print("\nFull comparison table:")
        print(df_compare)

        # Check all models present
        expected_models = {"original", "flexible", "cyclic", "spline"}
        assert set(df_compare.index) == expected_models
        print("✓ All models compared successfully")

        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 70)
    print("WAIC/LOO COMPARISON TEST (single and multiple models)")
    print("=" * 70)

    success1 = test_waic_loo_single_model()
    success2 = test_waic_loo_multiple_models()

    if success1 and success2:
        print("\n" + "=" * 70)
        print("ALL TESTS PASSED ✓")
        sys.exit(0)
    else:
        print("\n" + "=" * 70)
        print("TESTS FAILED ✗")
        sys.exit(1)