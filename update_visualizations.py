#!/usr/bin/env python3
"""Update visualizations for weight fitness report with enhanced intensity display."""

import sys
from pathlib import Path
import shutil

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.analysis.analyze_state_space_impulse import ImpulseStateSpaceAnalyzer
from src.data.weight import load_weight_data
from src.data.intensity import load_intensity_data
from src.models.fit_weight import fit_state_space_model_impulse


def main():
    # Use the same output directory as the cached fit
    output_dir = Path("output/state_space_impulse_updated")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create analyzer instance with same parameters as cached fit
    analyzer = ImpulseStateSpaceAnalyzer(
        output_dir=str(output_dir),
        data_dir="data",
        activity_types=['strength_training', 'walking', 'cycling'],
        chains=4,
        iter_warmup=500,  # Matches cached fit
        iter_sampling=500,  # Matches cached fit
        adapt_delta=0.95,
        max_treedepth=12,
        use_sparse=True,
        n_inducing_points=50,
    )

    print("Loading data...")
    analyzer.load_data()

    print("Fitting model (should load from cache)...")
    # Override cache parameter by directly calling fit function with cache=True
    # The analyzer's fit_model method uses cache=False, so we call directly
    fit, idata, df_weight, df_intensity, stan_data = fit_state_space_model_impulse(
        data_dir=analyzer.data_dir,
        df_weight=analyzer.df_weight,
        df_intensity=analyzer.df_intensity,
        output_dir=analyzer.output_dir,
        chains=analyzer.chains,
        iter_warmup=analyzer.iter_warmup,
        iter_sampling=analyzer.iter_sampling,
        adapt_delta=analyzer.adapt_delta,
        max_treedepth=analyzer.max_treedepth,
        use_sparse=analyzer.use_sparse,
        n_inducing_points=analyzer.n_inducing_points,
        cache=True,  # Enable cache to load existing fit
    )

    analyzer.fit = fit
    analyzer.idata = idata
    analyzer.df_weight = df_weight
    analyzer.df_intensity = df_intensity
    analyzer.stan_data = stan_data

    print("Creating visualizations with enhanced intensity display...")
    analyzer.create_visualizations()

    # Copy updated visualizations to docs folder
    docs_vis_dir = Path("docs/weight_fitness_report/visualizations")
    docs_vis_dir.mkdir(parents=True, exist_ok=True)

    vis_files = [
        "data_overview.png",
        "posterior_distributions.png",
        "parameter_relationships.png",
        "state_space_expectations.png",
        "trace_plots.png",
    ]

    for vis_file in vis_files:
        src = output_dir / "visualizations" / vis_file
        dst = docs_vis_dir / vis_file
        if src.exists():
            shutil.copy2(src, dst)
            print(f"Copied {vis_file} to docs")
        else:
            print(f"Warning: {src} does not exist")

    print("Visualizations updated successfully.")


if __name__ == "__main__":
    main()