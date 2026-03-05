#!/usr/bin/env python3
"""Test the updated state-space spline model with adapt_delta=0.99 and generate visualizations."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.fit_weight import fit_state_space_model_impulse_spline
from src.models.plot_cyclic import plot_state_space_spline_decomposition
from src.models.state_space_utils import (
    compute_fitness_expectation,
    interpolate_fitness_to_timestamps,
    predict_weight_components,
    create_prediction_dataframe
)


def test_updated_model():
    """Test the updated state-space spline model with adapt_delta=0.99."""
    print("Testing updated state-space spline model (adapt_delta=0.99)...")
    print("=" * 60)

    # Test with moderate iterations for reasonable runtime
    # Using defaults: adapt_delta=0.99, max_treedepth=12
    # But reduced iterations for faster testing
    print("1. Fitting updated model (warmup=500, sampling=500, adapt_delta=0.99)...")
    try:
        fit, idata, df_weight, df_intensity, stan_data = fit_state_space_model_impulse_spline(
            chains=2,
            iter_warmup=500,
            iter_sampling=500,
            fourier_harmonics=2,
            cache=False,
            force_refit=True,
            include_prediction_grid=True,
            prediction_step_days=7,
            adapt_delta=0.99,  # Explicitly set (already default)
            max_treedepth=12,  # Explicitly set (already default)
        )
        print("✓ Model fitted successfully with adapt_delta=0.99")
        print(f"  Posterior shape: {idata.posterior.dims}")
        print(f"  Parameters: {list(idata.posterior.data_vars)}")

        # Check for divergent transitions
        if hasattr(fit, 'diagnose'):
            print("\n  Checking for divergent transitions...")
            # Get diagnostics
            diag = fit.diagnose()
            if 'divergent transitions' in diag:
                import re
                div_match = re.search(r'(\d+) divergent transitions', diag)
                if div_match:
                    n_div = int(div_match.group(1))
                    total_iter = fit.num_draws * fit.num_chains
                    div_pct = (n_div / total_iter) * 100
                    print(f"  Divergent transitions: {n_div}/{total_iter} ({div_pct:.1f}%)")
                else:
                    print("  No divergent transitions found (or could not parse)")
            else:
                print("  Could not extract divergent transitions from diagnostics")

    except Exception as e:
        print(f"✗ Model fitting failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Check parameter estimates
    print("\n2. Checking parameter estimates...")
    for param in ['alpha', 'psi', 'beta', 'gamma']:
        if param in idata.posterior:
            samples = idata.posterior[param].values
            mean = samples.mean()
            std = samples.std()
            print(f"  {param}: {mean:.3f} ± {std:.3f}")

    # Check daily variance metrics
    print("\n3. Checking daily variance metrics...")
    if 'prop_variance_daily' in idata.posterior:
        prop = idata.posterior['prop_variance_daily'].values.mean()
        print(f"  Proportion of variance from daily component: {prop:.3%}")
    if 'daily_amplitude' in idata.posterior:
        amp = idata.posterior['daily_amplitude'].values.mean()
        print(f"  Daily amplitude (standardized): {amp:.3f}")

    # Create visualization
    print("\n4. Creating comprehensive visualization...")
    try:
        output_dir = Path("output/updated_spline_test")
        output_dir.mkdir(exist_ok=True)

        fig = plot_state_space_spline_decomposition(
            idata=idata,
            df_weight=df_weight,
            df_intensity=df_intensity,
            stan_data=stan_data,
            model_name="Updated State-Space Spline Model (adapt_delta=0.99)",
            output_path=str(output_dir / "spline_decomposition.png"),
            show_ci=True
        )
        print("✓ Visualization created successfully")
        print(f"  Saved to: {output_dir}/spline_decomposition.png")

        # Save additional diagnostics
        print("\n5. Saving additional diagnostics...")

        # Save parameter summary
        param_summary = []
        for param in ['alpha', 'psi', 'beta', 'gamma', 'sigma_w', 'alpha_gp', 'rho_gp']:
            if param in idata.posterior:
                samples = idata.posterior[param].values.flatten()
                param_summary.append({
                    'parameter': param,
                    'mean': np.mean(samples),
                    'std': np.std(samples),
                    '2.5%': np.percentile(samples, 2.5),
                    '50%': np.percentile(samples, 50),
                    '97.5%': np.percentile(samples, 97.5),
                })

        param_df = pd.DataFrame(param_summary)
        param_df.to_csv(output_dir / "parameter_summary.csv", index=False)
        print(f"  Parameter summary saved to: {output_dir}/parameter_summary.csv")

        # Save fitness expectations
        print("\n6. Computing and saving fitness expectations...")
        try:
            # Get posterior means for parameters
            alpha_mean = idata.posterior['alpha'].values.mean()
            beta_mean = idata.posterior['beta'].values.mean()
            psi_mean = idata.posterior['psi'].values.mean()
            intensity = stan_data['intensity']

            # Compute deterministic fitness expectation
            fitness, impulse = compute_fitness_expectation(
                alpha=alpha_mean,
                beta=beta_mean,
                psi=psi_mean,
                intensity=intensity
            )

            # Create date range
            D = len(intensity)
            start_date_str = stan_data.get('_start_date', None)
            if start_date_str:
                start_date = pd.Timestamp(start_date_str)
            else:
                start_date = pd.Timestamp('2020-01-01')

            date_range = pd.date_range(start=start_date, periods=D, freq='D')

            # Save to CSV
            fitness_df = pd.DataFrame({
                'date': date_range,
                'fitness': fitness,
                'impulse': impulse,
                'intensity': intensity
            })
            fitness_df.to_csv(output_dir / "fitness_expectations.csv", index=False)
            print(f"  Fitness expectations saved to: {output_dir}/fitness_expectations.csv")

        except Exception as e:
            print(f"  Note: Fitness expectation computation failed: {e}")

        # Close figure to free memory
        plt.close(fig)

    except Exception as e:
        print(f"✗ Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n" + "=" * 60)
    print("Updated model test completed successfully!")
    print("\nSummary:")
    print(f"- Model: State-space spline with adapt_delta=0.99")
    print(f"- Iterations: 500 warmup, 500 sampling per chain (2 chains)")
    print(f"- Parameters updated: α, ψ, β, γ, α_gp, ρ_gp priors")
    print(f"- Initial values: Set to prior means")
    print(f"- Visualizations: Generated in output/updated_spline_test/")
    print("\nNext steps for production:")
    print("1. Run with full iterations (warmup=1000, sampling=2000)")
    print("2. Use 4 chains for better convergence diagnostics")
    print("3. Monitor divergent transitions and R-hat values")

    return 0


if __name__ == "__main__":
    sys.exit(test_updated_model())