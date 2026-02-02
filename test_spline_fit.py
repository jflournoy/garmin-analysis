#!/usr/bin/env python3
"""Test spline model fitting with minimal iterations."""
import sys
from pathlib import Path

# Add project root to path if needed
sys.path.insert(0, str(Path(__file__).parent))

from src.models.fit_weight import fit_weight_model_spline

def main():
    print("Testing spline model fitting...")
    print("Parameters: chains=1, iter_warmup=50, iter_sampling=50, fourier_harmonics=2")
    try:
        fit, idata, df, stan_data = fit_weight_model_spline(
            chains=1,
            iter_warmup=50,
            iter_sampling=50,
            fourier_harmonics=2,
            cache=False,  # Force compilation and fitting
            force_refit=True,
        )
        print("✓ Spline model fitted successfully")
        print(f"  Shape of posterior: {idata.posterior.dims}")
        print(f"  Parameters: {list(idata.posterior.data_vars)}")
        # Check that Fourier coefficients are present
        if 'a_sin' in idata.posterior:
            print(f"  a_sin shape: {idata.posterior['a_sin'].shape}")
        if 'a_cos' in idata.posterior:
            print(f"  a_cos shape: {idata.posterior['a_cos'].shape}")
        # Check sigma
        if 'sigma' in idata.posterior:
            sigma_mean = idata.posterior['sigma'].mean().item()
            print(f"  sigma mean: {sigma_mean:.4f}")
        return 0
    except Exception as e:
        print(f"✗ Fitting failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())