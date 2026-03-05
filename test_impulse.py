#!/usr/bin/env python3
"""Quick test of impulse-response state-space model."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.fit_weight import fit_state_space_model_impulse

def main():
    """Run quick test with minimal iterations."""
    print("Testing impulse-response state-space model...")

    try:
        fit, idata, df_weight, df_intensity, stan_data = fit_state_space_model_impulse(
            data_dir="data",
            output_dir="output/test_impulse",
            chains=2,
            iter_warmup=100,
            iter_sampling=100,
            n_inducing_points=20,
            cache=False,
        )

        print("\n=== Model fitting successful ===")
        print(f"Number of samples: {len(idata.posterior.draw) * len(idata.posterior.chain)}")
        print(f"Parameters: {list(idata.posterior.data_vars)}")

        # Print parameter summaries
        import arviz as az
        summary = az.summary(idata, var_names=["alpha", "psi", "beta", "gamma", "sigma_w", "alpha_gp", "rho_gp"])
        print("\nParameter estimates:")
        print(summary)

        # Check for divergent transitions
        if hasattr(fit, 'diagnose'):
            print("\nDiagnostics:")
            print(fit.diagnose())

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()