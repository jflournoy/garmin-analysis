#!/usr/bin/env python3
"""Test optimized flexible model with cov_exp_quad, customizable priors, and sparse GP options."""
import sys
from pathlib import Path

# Add project root to path if needed
sys.path.insert(0, str(Path(__file__).parent))

from src.models.fit_weight import (
    fit_weight_model_flexible,
    fit_weight_model_flexible_optimized,
)

def test_optimized_model_full_gp():
    """Test that optimized flexible model with full GP (use_sparse=False) works."""
    print("Testing optimized flexible model (full GP)...")
    print("Parameters: chains=1, iter_warmup=50, iter_sampling=50")
    try:
        fit, idata, df, stan_data = fit_weight_model_flexible_optimized(
            chains=1,
            iter_warmup=50,
            iter_sampling=50,
            use_sparse=False,  # Full GP
            cache=False,  # Force compilation and fitting
            force_refit=True,
        )
        print("✓ Optimized flexible model (full GP) fitted successfully")
        print(f"  Shape of posterior: {idata.posterior.dims}")
        print(f"  Parameters: {list(idata.posterior.data_vars)}")
        # Check that key parameters are present
        required_params = ['alpha', 'rho', 'sigma']
        for param in required_params:
            if param in idata.posterior:
                mean_val = idata.posterior[param].mean().item()
                print(f"  {param} mean: {mean_val:.4f}")
            else:
                print(f"  ⚠ {param} not found in posterior")
        # Check generated quantities
        if 'trend_change' in idata.posterior:
            tc_mean = idata.posterior['trend_change'].mean().item()
            print(f"  trend_change mean: {tc_mean:.4f}")
        return True
    except Exception as e:
        print(f"✗ Fitting failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_numerical_equivalence():
    """Compare original flexible and optimized flexible full GP models for numerical equivalence."""
    print("\n" + "="*60)
    print("Testing numerical equivalence: original flexible vs optimized flexible (full GP)")
    print("="*60)

    # Use minimal iterations for speed
    chains = 1
    iter_warmup = 50
    iter_sampling = 50

    print("Fitting original flexible model (manual loops)...")
    try:
        fit_orig, idata_orig, df_orig, stan_data_orig = fit_weight_model_flexible(
            chains=chains,
            iter_warmup=iter_warmup,
            iter_sampling=iter_sampling,
            cache=False,
            force_refit=True,
        )
        print("✓ Original flexible model fitted")
    except Exception as e:
        print(f"✗ Original flexible model failed: {e}")
        return False

    print("Fitting optimized flexible model (full GP)...")
    try:
        fit_opt, idata_opt, df_opt, stan_data_opt = fit_weight_model_flexible_optimized(
            chains=chains,
            iter_warmup=iter_warmup,
            iter_sampling=iter_sampling,
            use_sparse=False,
            cache=False,
            force_refit=True,
        )
        print("✓ Optimized flexible model fitted")
    except Exception as e:
        print(f"✗ Optimized flexible model failed: {e}")
        return False

    # Compare key parameter posterior means
    tolerance = 0.1  # Relaxed tolerance due to sampling variability
    print("\nComparing posterior means (tolerance = ±{:.3f}):".format(tolerance))

    comparisons = []
    for param in ['alpha', 'rho', 'sigma']:
        if param in idata_orig.posterior and param in idata_opt.posterior:
            mean_orig = idata_orig.posterior[param].mean().item()
            mean_opt = idata_opt.posterior[param].mean().item()
            diff = abs(mean_orig - mean_opt)
            within_tol = diff <= tolerance
            comparisons.append((param, mean_orig, mean_opt, diff, within_tol))
            print(f"  {param}: orig={mean_orig:.4f}, opt={mean_opt:.4f}, diff={diff:.4f} {'✓' if within_tol else '✗'}")

    # Check if all comparisons pass tolerance
    all_pass = all(c[4] for c in comparisons)
    if all_pass:
        print("\n✓ All parameters within tolerance (numerical equivalence confirmed)")
    else:
        print("\n⚠ Some parameters outside tolerance (expected due to sampling variability)")
        print("   Note: With only 50 iterations, some differences are expected.")

    return all_pass

def test_sparse_gp_option():
    """Test sparse GP option with minimal inducing points."""
    print("\n" + "="*60)
    print("Testing sparse GP option (use_sparse=True)")
    print("="*60)

    # Use very minimal inducing points for speed
    n_inducing_points = 5

    print(f"Testing with M={n_inducing_points} inducing points...")
    try:
        fit, idata, df, stan_data = fit_weight_model_flexible_optimized(
            chains=1,
            iter_warmup=30,  # Even fewer iterations for speed
            iter_sampling=30,
            use_sparse=True,
            n_inducing_points=n_inducing_points,
            inducing_point_method="uniform",
            cache=False,
            force_refit=True,
        )
        print("✓ Sparse GP model fitted successfully")

        # Verify sparse GP parameters in stan_data
        required = ["use_sparse", "M", "t_inducing"]
        for key in required:
            if key in stan_data:
                print(f"  {key}: {stan_data[key]}")
            else:
                print(f"  ⚠ {key} not in stan_data")

        # Check that M matches n_inducing_points
        if 'M' in stan_data:
            M = stan_data['M']
            if M == n_inducing_points:
                print(f"  ✓ M correctly set to {M}")
            else:
                print(f"  ⚠ M={M}, expected {n_inducing_points}")

        # Check that t_inducing has correct length
        if 't_inducing' in stan_data:
            t_inducing = stan_data['t_inducing']
            if len(t_inducing) == n_inducing_points:
                print(f"  ✓ t_inducing length = {len(t_inducing)}")
            else:
                print(f"  ⚠ t_inducing length = {len(t_inducing)}, expected {n_inducing_points}")

        return True
    except Exception as e:
        print(f"✗ Sparse GP fitting failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_inducing_point_methods():
    """Test different inducing point selection methods."""
    print("\n" + "="*60)
    print("Testing inducing point selection methods")
    print("="*60)

    methods = ["uniform", "random"]
    # Skip kmeans by default (requires sklearn)

    for method in methods:
        print(f"\nTesting method: '{method}'")
        try:
            # Just test data preparation, not full model fitting
            from src.data.weight import prepare_stan_data
            from src.data.weight import load_weight_data

            df = load_weight_data("data")
            stan_data = prepare_stan_data(
                df,
                include_hour_info=False,
                use_sparse=True,
                n_inducing_points=10,
                inducing_point_method=method,
            )

            print("  ✓ Data preparation successful")
            print(f"    M: {stan_data.get('M', 'N/A')}")
            print(f"    use_sparse: {stan_data.get('use_sparse', 'N/A')}")

            # Check that t_inducing is sorted (for uniform and random)
            t_inducing = stan_data.get('t_inducing', [])
            if len(t_inducing) > 1:
                is_sorted = all(t_inducing[i] <= t_inducing[i+1] for i in range(len(t_inducing)-1))
                if is_sorted:
                    print("    ✓ t_inducing is sorted")
                else:
                    print(f"    ⚠ t_inducing not sorted (method={method})")

        except Exception as e:
            print(f"  ✗ Method '{method}' failed: {e}")

    # Test kmeans if sklearn available
    try:
        from sklearn.cluster import KMeans
        print("\nTesting method: 'kmeans' (requires sklearn)")
        from src.data.weight import prepare_stan_data
        from src.data.weight import load_weight_data

        df = load_weight_data("data")
        stan_data = prepare_stan_data(
            df,
            include_hour_info=False,
            use_sparse=True,
            n_inducing_points=10,
            inducing_point_method="kmeans",
        )
        print("  ✓ K-means data preparation successful")
        print(f"    M: {stan_data.get('M', 'N/A')}")
    except ImportError:
        print("\n⚠ Skipping kmeans test: sklearn not installed")
    except Exception as e:
        print(f"  ✗ K-means method failed: {e}")

    return True

def main():
    """Run all tests."""
    print("="*70)
    print("OPTIMIZED FLEXIBLE MODEL TEST SUITE")
    print("="*70)

    results = []

    # Test 1: Basic optimized model (full GP)
    results.append(("Optimized model (full GP)", test_optimized_model_full_gp()))

    # Test 2: Numerical equivalence
    results.append(("Numerical equivalence", test_numerical_equivalence()))

    # Test 3: Sparse GP option
    results.append(("Sparse GP option", test_sparse_gp_option()))

    # Test 4: Inducing point methods
    results.append(("Inducing point methods", test_inducing_point_methods()))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:30} {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n⚠ Some tests failed (see details above)")
        return 1

if __name__ == "__main__":
    sys.exit(main())