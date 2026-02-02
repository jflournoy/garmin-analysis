#!/usr/bin/env python3
"""Benchmark performance improvements from optimized models with cov_exp_quad."""
import sys
import time
import argparse
from pathlib import Path
import pandas as pd

# Add project root to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.fit_weight import (
    fit_weight_model,
    fit_weight_model_optimized,
    fit_weight_model_cyclic,
    fit_weight_model_cyclic_optimized,
    fit_weight_model_flexible,
    fit_weight_model_flexible_optimized,
    fit_weight_model_spline,
    fit_weight_model_spline_optimized,
)

def benchmark_model(model_name, fit_function, **kwargs):
    """Benchmark a single model fitting."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_name}")
    print(f"{'='*60}")

    start_time = time.time()
    try:
        fit, idata, df, stan_data = fit_function(**kwargs)
        elapsed = time.time() - start_time

        # Extract key metrics
        sigma_mean = idata.posterior["sigma"].mean().item()
        sigma_sd = idata.posterior["sigma"].std().item()

        # Compute WAIC if log_likelihood group exists
        waic = None
        if "log_likelihood" in idata:
            import arviz as az
            waic_result = az.waic(idata)
            # Convert to -2*elpd scale (lower is better)
            waic = -2 * waic_result.elpd_waic

        print("  Status: ✓ SUCCESS")
        print(f"  Time: {elapsed:.1f} seconds")
        print(f"  σ: {sigma_mean:.4f} ± {sigma_sd:.4f}")
        if waic is not None:
            print(f"  WAIC: {waic:.1f}")

        return {
            "model": model_name,
            "success": True,
            "time_seconds": elapsed,
            "sigma_mean": sigma_mean,
            "sigma_sd": sigma_sd,
            "waic": waic,
        }
    except Exception as e:
        elapsed = time.time() - start_time
        print("  Status: ✗ FAILED")
        print(f"  Time: {elapsed:.1f} seconds")
        print(f"  Error: {e}")
        return {
            "model": model_name,
            "success": False,
            "time_seconds": elapsed,
            "sigma_mean": None,
            "sigma_sd": None,
            "waic": None,
        }

def main():
    """Run benchmarks for all models."""
    parser = argparse.ArgumentParser(
        description="Benchmark performance improvements from optimized models"
    )
    parser.add_argument(
        "--chains",
        type=int,
        default=2,
        help="Number of MCMC chains (default: 2)",
    )
    parser.add_argument(
        "--iter-warmup",
        type=int,
        default=200,
        help="Warmup iterations per chain (default: 200)",
    )
    parser.add_argument(
        "--iter-sampling",
        type=int,
        default=200,
        help="Sampling iterations per chain (default: 200)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/benchmark",
        help="Output directory for results (default: output/benchmark)",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Use cached model fits if available (default: False)",
    )
    parser.add_argument(
        "--force-refit",
        action="store_true",
        help="Force refit even if cached model exists (default: False)",
    )
    args = parser.parse_args()

    print("="*70)
    print("PERFORMANCE BENCHMARK: OPTIMIZED MODELS vs ORIGINAL")
    print("="*70)
    print(f"Configuration: chains={args.chains}, "
          f"warmup={args.iter_warmup}, sampling={args.iter_sampling}")
    print(f"Cache: {args.cache}, Force refit: {args.force_refit}")
    print(f"Output directory: {args.output_dir}")

    # Common parameters for all models
    common_params = {
        "chains": args.chains,
        "iter_warmup": args.iter_warmup,
        "iter_sampling": args.iter_sampling,
        "cache": args.cache,
        "force_refit": args.force_refit,
    }

    # Model configurations
    benchmarks = [
        ("Original (manual loops)", fit_weight_model, {}),
        ("Original (optimized)", fit_weight_model_optimized, {"use_sparse": False}),
        ("Original (optimized, sparse)", fit_weight_model_optimized, {"use_sparse": True, "n_inducing_points": 50}),
        ("Cyclic (manual loops)", fit_weight_model_cyclic, {}),
        ("Cyclic (optimized)", fit_weight_model_cyclic_optimized, {"use_sparse": False}),
        ("Cyclic (optimized, sparse)", fit_weight_model_cyclic_optimized, {"use_sparse": True, "n_inducing_points": 50}),
        ("Flexible (manual loops)", fit_weight_model_flexible, {}),
        ("Flexible (optimized)", fit_weight_model_flexible_optimized, {"use_sparse": False}),
        ("Flexible (optimized, sparse)", fit_weight_model_flexible_optimized, {"use_sparse": True, "n_inducing_points": 50}),
        ("Spline (manual loops)", fit_weight_model_spline, {}),
        ("Spline (optimized)", fit_weight_model_spline_optimized, {"use_sparse": False}),
        ("Spline (optimized, sparse)", fit_weight_model_spline_optimized, {"use_sparse": True, "n_inducing_points": 50}),
    ]

    results = []
    for model_name, fit_func, extra_params in benchmarks:
        params = common_params.copy()
        params.update(extra_params)
        result = benchmark_model(model_name, fit_func, **params)
        results.append(result)

    # Create summary DataFrame
    df_results = pd.DataFrame(results)
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)

    # Display results table
    print("\nPerformance comparison:")
    for _, row in df_results.iterrows():
        if row["success"]:
            time_str = f"{row['time_seconds']:.1f}s"
            sigma_str = f"{row['sigma_mean']:.4f}"
            waic_str = f"{row['waic']:.1f}" if row['waic'] is not None else "N/A"
            print(f"  {row['model']:30} {time_str:>8}  σ={sigma_str}  WAIC={waic_str}")
        else:
            print(f"  {row['model']:30} FAILED")

    # Calculate speedup ratios
    print("\nSpeedup ratios (original time / optimized time):")
    for orig_name, opt_name in [
        ("Original (manual loops)", "Original (optimized)"),
        ("Cyclic (manual loops)", "Cyclic (optimized)"),
        ("Flexible (manual loops)", "Flexible (optimized)"),
        ("Spline (manual loops)", "Spline (optimized)"),
    ]:
        orig_row = df_results[df_results["model"] == orig_name].iloc[0]
        opt_row = df_results[df_results["model"] == opt_name].iloc[0]

        if orig_row["success"] and opt_row["success"]:
            speedup = orig_row["time_seconds"] / opt_row["time_seconds"]
            print(f"  {orig_name:30} → {speedup:.2f}x speedup")
        else:
            print(f"  {orig_name:30} → Cannot compute speedup (model failed)")

    # Sparse vs full GP speedup
    print("\nSparse GP speedup ratios (full GP time / sparse GP time):")
    for full_name, sparse_name in [
        ("Original (optimized)", "Original (optimized, sparse)"),
        ("Cyclic (optimized)", "Cyclic (optimized, sparse)"),
        ("Flexible (optimized)", "Flexible (optimized, sparse)"),
        ("Spline (optimized)", "Spline (optimized, sparse)"),
    ]:
        full_row = df_results[df_results["model"] == full_name].iloc[0]
        sparse_row = df_results[df_results["model"] == sparse_name].iloc[0]

        if full_row["success"] and sparse_row["success"]:
            speedup = full_row["time_seconds"] / sparse_row["time_seconds"]
            print(f"  {full_name:30} → {speedup:.2f}x speedup")
        else:
            print(f"  {full_name:30} → Cannot compute speedup (model failed)")

    # Save results to CSV
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "optimized_benchmark.csv"
    df_results.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Generate markdown report
    md_path = output_dir / "optimized_benchmark.md"
    with open(md_path, "w") as f:
        f.write("# Performance Benchmark: Optimized Models\n\n")
        f.write(f"*Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")

        f.write("## Results Summary\n\n")
        f.write("| Model | Time (s) | σ | WAIC | Success |\n")
        f.write("|-------|----------|---|------|---------|\n")
        for _, row in df_results.iterrows():
            time_str = f"{row['time_seconds']:.1f}" if row['success'] else "—"
            sigma_str = f"{row['sigma_mean']:.4f}" if row['sigma_mean'] is not None else "—"
            waic_str = f"{row['waic']:.1f}" if row['waic'] is not None else "—"
            success_str = "✓" if row['success'] else "✗"
            f.write(f"| {row['model']} | {time_str} | {sigma_str} | {waic_str} | {success_str} |\n")

        f.write("\n## Speedup Ratios\n\n")
        f.write("### Manual vs Optimized (Full GP)\n")
        f.write("| Original Model | Optimized Model | Speedup |\n")
        f.write("|----------------|-----------------|---------|\n")
        for orig_name, opt_name in [
            ("Original (manual loops)", "Original (optimized)"),
            ("Cyclic (manual loops)", "Cyclic (optimized)"),
            ("Flexible (manual loops)", "Flexible (optimized)"),
            ("Spline (manual loops)", "Spline (optimized)"),
        ]:
            orig_row = df_results[df_results["model"] == orig_name].iloc[0]
            opt_row = df_results[df_results["model"] == opt_name].iloc[0]
            if orig_row["success"] and opt_row["success"]:
                speedup = orig_row["time_seconds"] / opt_row["time_seconds"]
                f.write(f"| {orig_name} | {opt_name} | {speedup:.2f}x |\n")
            else:
                f.write(f"| {orig_name} | {opt_name} | — |\n")

        f.write("\n### Sparse GP vs Full GP\n")
        f.write("| Full GP Model | Sparse GP Model | Speedup |\n")
        f.write("|---------------|-----------------|---------|\n")
        for full_name, sparse_name in [
            ("Original (optimized)", "Original (optimized, sparse)"),
            ("Cyclic (optimized)", "Cyclic (optimized, sparse)"),
            ("Flexible (optimized)", "Flexible (optimized, sparse)"),
            ("Spline (optimized)", "Spline (optimized, sparse)"),
        ]:
            full_row = df_results[df_results["model"] == full_name].iloc[0]
            sparse_row = df_results[df_results["model"] == sparse_name].iloc[0]
            if full_row["success"] and sparse_row["success"]:
                speedup = full_row["time_seconds"] / sparse_row["time_seconds"]
                f.write(f"| {full_name} | {sparse_name} | {speedup:.2f}x |\n")
            else:
                f.write(f"| {full_name} | {sparse_name} | — |\n")

        f.write("\n## Interpretation\n\n")
        f.write("- **Speedup > 1.0**: Optimized model is faster\n")
        f.write("- **Speedup < 1.0**: Optimized model is slower (unexpected)\n")
        f.write("- **σ comparison**: Lower σ indicates better fit (less measurement error)\n")
        f.write("- **WAIC comparison**: Lower WAIC indicates better predictive performance\n")
        f.write("\n## Notes\n\n")
        f.write(f"- Benchmark uses chains={args.chains}, "
                f"iter_warmup={args.iter_warmup}, iter_sampling={args.iter_sampling}\n")
        f.write("- Includes both full GP and sparse GP (50 inducing points) comparisons\n")
        f.write("- Sparse GP uses projected process (DIC) approximation with uniform inducing points\n")
        f.write("- Times include model compilation (first run) and sampling\n")
        f.write("- Results may vary due to random sampling variability\n")

    print(f"Markdown report saved to: {md_path}")

    return 0 if df_results["success"].all() else 1

if __name__ == "__main__":
    sys.exit(main())