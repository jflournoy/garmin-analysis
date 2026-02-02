"""Test cross-lagged GP model with different lag values for weight vs workouts."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data import load_weight_data
from src.data.workout import load_workout_data, prepare_workout_aggregates
from src.models.fit_weight import fit_crosslagged_model


def test_crosslagged_lags():
    """Test cross-lagged model with different lag values (0, 1, 2, 3, 7 days)."""
    output_dir = Path("output/crosslagged_lags")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Testing Cross-Lagged GP Model with Different Lags ===")
    print("Model: weight(t) = β·workouts(t-τ) + weight_intrinsic(t)")
    print("Testing τ = 0, 1, 2, 3, 7 days")
    print()

    # Load data
    print("1. Loading data...")
    df_weight_raw = load_weight_data()
    df_workouts_raw = load_workout_data(activity_type="strength_training")

    print(f"   Weight measurements: {len(df_weight_raw)}")
    print(f"   Strength training workouts: {len(df_workouts_raw)}")

    # Aggregate workouts to daily count
    df_workouts_agg = prepare_workout_aggregates(
        df_workouts_raw,
        aggregation="daily",
        metric="count"
    )
    print(f"   Workout days (with workouts): {len(df_workouts_agg)}")
    print(f"   Date range workouts: {df_workouts_agg['date'].min().date()} to {df_workouts_agg['date'].max().date()}")

    # Test different lags
    lags = [0, 1, 2, 3, 7]
    results = []

    for lag_days in lags:
        print(f"\n{'='*60}")
        print(f"Testing lag τ = {lag_days} days")
        print(f"{'='*60}")

        try:
            # Fit model with minimal iterations for testing
            fit, idata, stan_data = fit_crosslagged_model(
                df_weight=df_weight_raw,
                df_workout=df_workouts_agg,
                workout_value_col="workout_count",
                lag_days=lag_days,
                chains=2,
                iter_warmup=100,
                iter_sampling=100,
                use_sparse=True,
                n_inducing_points=30,
                cache=False,
                force_refit=True,
                include_prediction_grid=False,
            )

            # Extract posterior summary for beta
            posterior = idata.posterior
            if 'beta' in posterior:
                beta_samples = posterior['beta'].values.flatten()
                beta_mean = np.mean(beta_samples)
                beta_std = np.std(beta_samples)
                beta_ci_low = np.percentile(beta_samples, 2.5)
                beta_ci_high = np.percentile(beta_samples, 97.5)

                # Convert to original units if available
                if 'beta_original_units' in posterior:
                    beta_orig_samples = posterior['beta_original_units'].values.flatten()
                    beta_orig_mean = np.mean(beta_orig_samples)
                    beta_orig_std = np.std(beta_orig_samples)
                    beta_orig_ci_low = np.percentile(beta_orig_samples, 2.5)
                    beta_orig_ci_high = np.percentile(beta_orig_samples, 97.5)
                else:
                    beta_orig_mean = beta_orig_std = beta_orig_ci_low = beta_orig_ci_high = np.nan

                # Compute WAIC and LOO
                try:
                    waic = az.waic(idata, var_name="log_lik_weight")
                    loo = az.loo(idata, var_name="log_lik_weight")
                    waic_value = waic.waic
                    loo_value = loo.loo
                except Exception as e:
                    print(f"   Warning: Could not compute WAIC/LOO: {e}")
                    waic_value = loo_value = np.nan

                # Store results
                result = {
                    'lag_days': lag_days,
                    'beta_mean': beta_mean,
                    'beta_std': beta_std,
                    'beta_ci_low': beta_ci_low,
                    'beta_ci_high': beta_ci_high,
                    'beta_orig_mean': beta_orig_mean,
                    'beta_orig_std': beta_orig_std,
                    'beta_orig_ci_low': beta_orig_ci_low,
                    'beta_orig_ci_high': beta_orig_ci_high,
                    'waic': waic_value,
                    'loo': loo_value,
                    'n_eff_beta': float(az.ess(beta_samples)) if hasattr(az, 'ess') else np.nan,
                    'rhat_beta': float(az.rhat(posterior['beta']).values) if 'beta' in posterior else np.nan,
                }
                results.append(result)

                print(f"   β (standardized): {beta_mean:.3f} [{beta_ci_low:.3f}, {beta_ci_high:.3f}]")
                if not np.isnan(beta_orig_mean):
                    print(f"   β (original units): {beta_orig_mean:.3f} [{beta_orig_ci_low:.3f}, {beta_orig_ci_high:.3f}]")
                print(f"   WAIC: {waic_value:.1f}" if not np.isnan(waic_value) else "   WAIC: N/A")
                print(f"   LOO: {loo_value:.1f}" if not np.isnan(loo_value) else "   LOO: N/A")

                # Save model results
                lag_dir = output_dir / f"lag_{lag_days}days"
                lag_dir.mkdir(exist_ok=True)

                # Save summary
                summary = az.summary(idata, round_to=3)
                summary.to_csv(lag_dir / "posterior_summary.csv")

                # Save beta posterior samples
                beta_df = pd.DataFrame({'beta': beta_samples})
                beta_df.to_csv(lag_dir / "beta_posterior.csv", index=False)

                # Plot beta posterior
                plt.figure(figsize=(8, 5))
                plt.hist(beta_samples, bins=30, alpha=0.7, density=True, edgecolor='black')
                plt.axvline(0, color='red', linestyle='--', label='Zero effect')
                plt.xlabel('β (causal effect: workouts → weight)')
                plt.ylabel('Density')
                plt.title(f'Posterior of β (lag τ={lag_days} days)')
                plt.legend()
                plt.tight_layout()
                plt.savefig(lag_dir / "beta_posterior.png", dpi=150)
                plt.close()

                print(f"   Saved results to {lag_dir}")

            else:
                print("   Warning: 'beta' not found in posterior")
                results.append({'lag_days': lag_days, 'error': 'beta not found'})

        except Exception as e:
            print(f"   Error fitting model with lag {lag_days}: {e}")
            import traceback
            traceback.print_exc()
            results.append({'lag_days': lag_days, 'error': str(e)})

    # Create summary table
    if results:
        print(f"\n{'='*60}")
        print("SUMMARY OF ALL LAGS")
        print(f"{'='*60}")

        df_results = pd.DataFrame(results)
        # Reorder columns
        cols = ['lag_days', 'beta_mean', 'beta_std', 'beta_ci_low', 'beta_ci_high',
                'beta_orig_mean', 'beta_orig_std', 'beta_orig_ci_low', 'beta_orig_ci_high',
                'waic', 'loo', 'n_eff_beta', 'rhat_beta']
        df_results = df_results[[c for c in cols if c in df_results.columns]]

        print("\nComparison table:")
        print(df_results.to_string(index=False))

        # Save comparison
        df_results.to_csv(output_dir / "lag_comparison.csv", index=False)

        # Plot comparison
        plt.figure(figsize=(10, 6))
        lags_vals = [r['lag_days'] for r in results if 'beta_mean' in r]
        beta_means = [r['beta_mean'] for r in results if 'beta_mean' in r]
        beta_cis_low = [r['beta_ci_low'] for r in results if 'beta_mean' in r]
        beta_cis_high = [r['beta_ci_high'] for r in results if 'beta_mean' in r]

        if lags_vals:
            plt.errorbar(lags_vals, beta_means,
                        yerr=[np.array(beta_means) - np.array(beta_cis_low),
                              np.array(beta_cis_high) - np.array(beta_means)],
                        fmt='o-', capsize=5, capthick=2)
            plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            plt.xlabel('Lag τ (days)')
            plt.ylabel('β (causal effect)')
            plt.title('Causal effect of workouts on weight across different lags')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "beta_vs_lag.png", dpi=150)
            plt.close()

        # Find best lag by WAIC (if available)
        if 'waic' in df_results.columns and not df_results['waic'].isna().all():
            best_idx = df_results['waic'].idxmin()
            best_lag = df_results.loc[best_idx, 'lag_days']
            best_waic = df_results.loc[best_idx, 'waic']
            print(f"\nBest model by WAIC: lag τ = {best_lag} days (WAIC = {best_waic:.1f})")

        # Interpretation
        print("\n=== INTERPRETATION ===")
        print("β > 0: Workouts cause weight gain (muscle mass)")
        print("β < 0: Workouts cause weight loss (fat loss)")
        print("β ≈ 0: No causal effect detected")
        print("\nNote: Models assume no unmeasured confounding.")
        print("Standardized β: weight change in SD per 1 SD increase in workouts")
        print("Original units: weight change in lbs per unit increase in workout metric")

    print(f"\nAnalysis complete. Results saved to {output_dir}")


def quick_test():
    """Quick test with minimal iterations to verify the model works."""
    print("=== Quick Test of Cross-Lagged Model ===")

    # Load small subset of data
    df_weight = load_weight_data()
    df_workouts = load_workout_data(activity_type="strength_training")
    df_workouts_agg = prepare_workout_aggregates(df_workouts, aggregation="daily", metric="count")

    print(f"Weight: {len(df_weight)} measurements")
    print(f"Workouts: {len(df_workouts_agg)} days with workouts")

    # Test with lag=2 days and minimal iterations
    print("\nFitting model with τ=2 days (minimal iterations for testing)...")
    try:
        fit, idata, stan_data = fit_crosslagged_model(
            df_weight=df_weight,
            df_workout=df_workouts_agg,
            workout_value_col="workout_count",
            lag_days=2.0,
            chains=1,
            iter_warmup=5,
            iter_sampling=5,
            use_sparse=True,
            n_inducing_points=10,
            cache=False,
            force_refit=True,
        )
        print("Model fitted successfully!")
        print(f"Stan data keys: {list(stan_data.keys())[:10]}...")
        if 'beta' in idata.posterior:
            beta_mean = idata.posterior['beta'].mean().values.item()
            print(f"β mean: {beta_mean:.3f}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run quick test first
    if quick_test():
        print("\nQuick test passed! Running full lag comparison...")
        # Uncomment for full analysis
        test_crosslagged_lags()
    else:
        print("\nQuick test failed. Check model compilation and data preparation.")