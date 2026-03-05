#!/usr/bin/env python3
"""Run enhanced state-space spline model with full settings and generate visualizations."""

import sys
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.fit_weight import fit_state_space_model_impulse_spline
from src.models.plot_cyclic import (
    plot_state_space_spline_decomposition,
    plot_state_space_expectations,
    plot_spline_daily_pattern,
    plot_state_space_component_details,
)

def main():
    print("Running enhanced state-space spline model with full settings...")
    print("=" * 60)
    print("Settings:")
    print("  chains: 4")
    print("  warmup: 1000")
    print("  sampling: 2000")
    print("  adapt_delta: 0.99")
    print("  max_treedepth: 12")
    print("  fourier_harmonics: 2")
    print("=" * 60)

    # Fit model with full settings (defaults: chains=4, warmup=1000, sampling=2000, adapt_delta=0.99)
    fit, idata, df_weight, df_intensity, stan_data = fit_state_space_model_impulse_spline(
        cache=False,  # Don't use cache to ensure fresh fit
        force_refit=True,
        include_prediction_grid=True,
        prediction_step_days=7,
        fourier_harmonics=2,
    )

    print("\n✓ Model fitted successfully")
    print(f"  Posterior dimensions: {idata.posterior.dims}")
    print(f"  Number of parameters: {len(idata.posterior.data_vars)}")

    # Create output directory for visualizations
    output_dir = Path("output/enhanced_full")
    output_dir.mkdir(exist_ok=True)

    print("\nGenerating visualizations...")

    # 1. Comprehensive decomposition plot
    print("  1. Creating comprehensive decomposition plot...")
    fig1 = plot_state_space_spline_decomposition(
        idata=idata,
        df_weight=df_weight,
        df_intensity=df_intensity,
        stan_data=stan_data,
        model_name="Enhanced State-Space Spline Model (Full)",
        output_path=output_dir / "spline_decomposition.png",
        show_ci=True,
    )
    plt.close(fig1)

    # 2. State-space expectations plot
    print("  2. Creating state-space expectations plot...")
    fig2 = plot_state_space_expectations(
        idata=idata,
        df_weight=df_weight,
        df_intensity=df_intensity,
        stan_data=stan_data,
        model_name="Enhanced State-Space Spline Model (Full)",
        output_path=output_dir / "state_space_expectations.png",
        show_ci=True,
    )
    plt.close(fig2)

    # 3. Spline daily pattern plot
    print("  3. Creating spline daily pattern plot...")
    fig3 = plot_spline_daily_pattern(
        idata=idata,
        stan_data=stan_data,
        output_path=output_dir / "spline_daily_pattern.png",
        n_hours_grid=100,
    )
    plt.close(fig3)

    # 4. Component details plot
    print("  4. Creating component details plot...")
    fig4 = plot_state_space_component_details(
        idata=idata,
        df_weight=df_weight,
        df_intensity=df_intensity,
        stan_data=stan_data,
        model_name="Enhanced State-Space Spline Model (Full)",
        output_path=output_dir / "component_details.png",
        show_ci=True,
    )
    plt.close(fig4)

    # Copy visualizations to report directory
    report_viz_dir = Path("docs/weight_fitness_report/visualizations")
    report_viz_dir.mkdir(exist_ok=True)

    viz_files = [
        ("spline_decomposition.png", "spline_decomposition.png"),
        ("component_details.png", "component_details.png"),
        ("spline_daily_pattern.png", "spline_daily_pattern.png"),
        ("state_space_expectations.png", "state_space_expectations.png"),
    ]

    for src_name, dst_name in viz_files:
        src = output_dir / src_name
        dst = report_viz_dir / dst_name
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  Copied {src_name} to report visualizations")
        else:
            print(f"  WARNING: {src} not found")

    # Print parameter summaries
    print("\nParameter estimates (mean ± SD):")
    for param in ['alpha', 'psi', 'beta', 'gamma', 'sigma_w', 'alpha_gp', 'rho_gp']:
        if param in idata.posterior:
            samples = idata.posterior[param].values.flatten()
            mean = samples.mean()
            std = samples.std()
            print(f"  {param}: {mean:.3f} ± {std:.3f}")

    if 'prop_variance_daily' in idata.posterior:
        prop = idata.posterior['prop_variance_daily'].values.mean()
        print(f"  Proportion of variance from daily component: {prop:.3%}")

    if 'daily_amplitude' in idata.posterior:
        amp = idata.posterior['daily_amplitude'].values.mean()
        print(f"  Daily amplitude (standardized): {amp:.3f}")

    print(f"\n✓ All visualizations saved to {output_dir}/")
    print("=" * 60)
    print("Enhanced model analysis completed successfully!")

if __name__ == "__main__":
    main()