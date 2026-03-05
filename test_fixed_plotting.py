#!/usr/bin/env python3
"""Test the fixed plot_spline_daily_pattern function."""

import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.fit_weight import fit_state_space_model_impulse_spline
from src.models.plot_cyclic import plot_spline_daily_pattern

def main():
    print("Testing fixed plot_spline_daily_pattern function...")
    print("=" * 60)

    # Run quick fit to get data
    print("Running quick model fit...")
    fit, idata, df_weight, df_intensity, stan_data = fit_state_space_model_impulse_spline(
        chains=2,
        iter_warmup=100,
        iter_sampling=200,
        fourier_harmonics=2,
        cache=False,
        force_refit=True,
    )

    print("✓ Model fitted successfully")

    # Test the plotting function
    print("\nTesting plot_spline_daily_pattern function...")
    try:
        fig = plot_spline_daily_pattern(
            idata=idata,
            stan_data=stan_data,
            output_path="output/test_spline_daily_pattern.png",
            n_hours_grid=100,
        )
        print("✓ plot_spline_daily_pattern succeeded!")
        print("  Saved to: output/test_spline_daily_pattern.png")
        plt.close(fig)
    except Exception as e:
        print(f"✗ plot_spline_daily_pattern failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n" + "=" * 60)
    print("Plotting function test completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())