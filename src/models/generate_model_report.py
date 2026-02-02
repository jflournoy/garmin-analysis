#!/usr/bin/env python3
"""Generate comprehensive model comparison report with WAIC/LOO metrics.

This script fits all four weight GP models (original, flexible, cyclic, spline),
computes WAIC and LOO-CV information criteria, generates comparison plots,
and creates a markdown report with tables and embedded figures.

Usage:
    python src/models/generate_model_report.py [--chains N] [--iter-sampling N] [--output-dir DIR]

Example:
    python src/models/generate_model_report.py --chains 4 --iter-sampling 500 --output-dir output/model_report

The report will be saved as `model_comparison_report.md` in the output directory.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.models.fit_weight import (
    fit_weight_model,
    fit_weight_model_flexible,
    fit_weight_model_cyclic,
    fit_weight_model_spline,
    fit_weight_model_spline_optimized,
    fit_weight_model_optimized,
    fit_weight_model_cyclic_optimized,
    fit_weight_model_flexible_optimized,
    compare_models_sigma,
    compare_models_all,
    compare_models_waic_loo,
)
from src.models.plot_cyclic import (
    plot_cyclic_components,
    plot_daily_pattern,
    plot_spline_daily_pattern,
    plot_models_comparison_all,
    plot_model_full_expectation,
)


def format_metric(value, format_str=".2f"):
    """Format metric value, handling None."""
    if value is None:
        return "—"
    return format(value, format_str)


def generate_markdown_report(
    output_dir: Path,
    df_waic_loo: pd.DataFrame,
    sigma_comparison: dict,
    data_info: dict,
    timestamp: str,
) -> Path:
    """Generate markdown report with model comparison results.

    Args:
        output_dir: Directory to save report and plots
        df_waic_loo: DataFrame with WAIC/LOO comparison results
        sigma_comparison: Dictionary with sigma comparison metrics
        data_info: Dictionary with data information (N, date range, etc.)
        timestamp: Report generation timestamp

    Returns:
        Path to generated markdown file
    """
    report_path = output_dir / "model_comparison_report.md"

    with open(report_path, 'w') as f:
        # Title and metadata
        f.write("# Model Comparison Report - Weight GP Models\n\n")
        f.write(f"*Generated: {timestamp}*\n")
        f.write(f"*Output directory: `{output_dir}`*\n\n")

        # Data summary
        f.write("## Data Summary\n\n")
        f.write(f"- **Observations**: {data_info['N']}\n")
        f.write(f"- **Date range**: {data_info['date_min']} to {data_info['date_max']}\n")
        f.write(f"- **Days with multiple measurements**: {data_info['days_multiple']}/{data_info['total_days']} ")
        f.write(f"({data_info['days_multiple_pct']:.1%})\n")
        if data_info['days_multiple_pct'] < 0.1:
            f.write("\n> ⚠ **Data sparsity note**: With sparse intraday data, daily components may capture residual variation ")
            f.write("rather than true daily cyclic patterns.\n\n")

        # Model descriptions
        f.write("## Model Descriptions\n\n")
        f.write("1. **Original GP**: Baseline Gaussian process model\n")
        f.write("2. **Flexible GP**: Customizable priors for α, ρ, σ\n")
        f.write("3. **Cyclic GP**: Trend + daily periodic kernel (24-hour period)\n")
        f.write("4. **Spline GP**: Trend + Fourier harmonics for daily cycles\n")
        f.write("5. **Original Optimized**: Original GP with efficient `gp_exp_quad_cov`\n")
        f.write("6. **Flexible Optimized**: Flexible GP with efficient covariance\n")
        f.write("7. **Cyclic Optimized**: Cyclic GP with efficient periodic covariance\n")
        f.write("8. **Spline Optimized**: Spline GP with efficient covariance\n\n")

        # Information Criteria Comparison
        f.write("## Information Criteria Comparison\n\n")
        f.write("WAIC (Widely Applicable Information Criterion) and LOO-CV (Leave-One-Out Cross-Validation) ")
        f.write("are Bayesian model comparison metrics. Lower values indicate better predictive performance.\n\n")

        # WAIC/LOO table
        f.write("### WAIC and LOO-CV Results\n\n")
        f.write("| Model | WAIC | WAIC SE | p_waic | LOO | LOO SE | p_loo | WAIC Weight | LOO Weight |\n")
        f.write("|-------|------|---------|--------|-----|--------|-------|-------------|------------|\n")

        for model_name in df_waic_loo.index:
            row = df_waic_loo.loc[model_name]
            f.write(f"| {model_name} | ")
            f.write(f"{format_metric(row.get('waic'), '.1f')} | ")
            f.write(f"{format_metric(row.get('waic_se'), '.1f')} | ")
            f.write(f"{format_metric(row.get('p_waic'), '.1f')} | ")
            f.write(f"{format_metric(row.get('loo'), '.1f')} | ")
            f.write(f"{format_metric(row.get('loo_se'), '.1f')} | ")
            f.write(f"{format_metric(row.get('p_loo'), '.1f')} | ")
            f.write(f"{format_metric(row.get('waic_weight'), '.3f')} | ")
            f.write(f"{format_metric(row.get('loo_weight'), '.3f')} |\n")

        # Best model identification
        best_waic = df_waic_loo['waic_weight'].idxmax() if df_waic_loo['waic_weight'].notna().any() else None
        best_loo = df_waic_loo['loo_weight'].idxmax() if df_waic_loo['loo_weight'].notna().any() else None

        f.write("\n**Interpretation**:\n")
        if best_waic:
            f.write(f"- **Best model by WAIC**: {best_waic} (weight={df_waic_loo.loc[best_waic, 'waic_weight']:.3f})\n")
        if best_loo:
            f.write(f"- **Best model by LOO-CV**: {best_loo} (weight={df_waic_loo.loc[best_loo, 'loo_weight']:.3f})\n")
        f.write("- **Weights**: Probability that model is best given data (sum to 1.0)\n")
        f.write("- **p_waic/p_loo**: Effective number of parameters (higher = more complex)\n\n")

        # Sigma Comparison
        f.write("## Measurement Error (σ) Comparison\n\n")
        f.write("Sigma represents measurement error + residual variation. Lower values indicate better model fit.\n\n")

        f.write("### Sigma Values (standardized scale)\n\n")
        f.write("| Model | σ |\n")
        f.write("|-------|---|\n")
        sigma_keys = [
            "sigma_original", "sigma_flexible", "sigma_cyclic", "sigma_spline",
            "sigma_original_optimized", "sigma_flexible_optimized",
            "sigma_cyclic_optimized", "sigma_spline_optimized"
        ]
        for key in sigma_keys:
            if key in sigma_comparison and sigma_comparison[key] is not None:
                model_name = key.replace("sigma_", "")
                f.write(f"| {model_name} | {sigma_comparison[key]:.4f} |\n")

        # Reductions
        f.write("\n### Sigma Reductions\n\n")
        reductions = [
            ("original → cyclic", "sigma_reduction_original_cyclic", "sigma_reduction_pct_original_cyclic"),
            ("cyclic → spline", "sigma_reduction_cyclic_spline", "sigma_reduction_pct_cyclic_spline"),
            ("original → spline", "sigma_reduction_original_spline", "sigma_reduction_pct_original_spline"),
            ("original → original_optimized", "sigma_reduction_original_original_optimized", "sigma_reduction_pct_original_original_optimized"),
            ("cyclic → cyclic_optimized", "sigma_reduction_cyclic_cyclic_optimized", "sigma_reduction_pct_cyclic_cyclic_optimized"),
            ("spline → spline_optimized", "sigma_reduction_spline_spline_optimized", "sigma_reduction_pct_spline_spline_optimized"),
        ]

        for label, abs_key, pct_key in reductions:
            if abs_key in sigma_comparison and sigma_comparison[abs_key] is not None:
                abs_val = sigma_comparison[abs_key]
                pct_val = sigma_comparison.get(pct_key, 0)
                f.write(f"- **{label}**: {abs_val:.4f} ({pct_val:.1f}% reduction)\n")

        # Daily component metrics
        f.write("\n### Daily Component Analysis\n\n")
        daily_metrics = [
            ("cyclic", "daily_amplitude_cyclic", "prop_variance_daily_cyclic"),
            ("spline", "daily_amplitude_spline", "prop_variance_daily_spline"),
        ]

        for model_name, amp_key, var_key in daily_metrics:
            if amp_key in sigma_comparison and sigma_comparison[amp_key] is not None:
                amp = sigma_comparison[amp_key]
                var = sigma_comparison.get(var_key, 0)
                f.write(f"**{model_name.capitalize()} model**:\n")
                f.write(f"- Daily amplitude: {amp:.4f}\n")
                f.write(f"- Proportion of variance: {var:.3f}\n\n")

        # Figures
        f.write("## Visualizations\n\n")

        # List available figures
        figure_files = [
            ("cyclic_components.png", "Cyclic model components (trend + daily)"),
            ("daily_pattern.png", "Daily pattern from cyclic model"),
            ("spline_daily_pattern.png", "Spline daily pattern with detrended data"),
            ("model_comparison_all.png", "Sigma comparison across all models"),
        ]

        for fig_file, description in figure_files:
            fig_path = output_dir / fig_file
            if fig_path.exists():
                # Use relative path for markdown
                f.write(f"### {description}\n\n")
                f.write(f"![{description}]({fig_file})\n\n")

        # Full expectation plots (dynamically discovered)
        full_exp_files = list(output_dir.glob("full_expectation_*.png"))
        if full_exp_files:
            f.write("### Full Model Expectations\n\n")
            f.write("Individual plots showing the complete predicted weight trajectory ")
            f.write("(all model components combined) for each model:\n\n")

            # Sort by model name for consistent ordering
            full_exp_files.sort()
            for plot_path in full_exp_files:
                model_name = plot_path.stem.replace("full_expectation_", "")
                f.write(f"#### {model_name} Model\n\n")
                f.write(f"![Full expectation for {model_name} model]({plot_path.name})\n\n")

        # Recommendations
        f.write("## Recommendations\n\n")

        # Based on information criteria
        if best_waic and best_loo:
            if best_waic == best_loo:
                f.write(f"1. **Primary recommendation**: Use **{best_waic}** model ")
                f.write("(best by both WAIC and LOO-CV)\n")
            else:
                f.write(f"1. **WAIC suggests**: {best_waic} model\n")
                f.write(f"2. **LOO-CV suggests**: {best_loo} model\n")
                f.write(f"3. **Consider**: {best_loo} if predictive accuracy is priority ")
                f.write("(LOO-CV approximates leave-one-out predictive performance)\n")

        # Based on sigma reduction
        if "sigma_reduction_original_spline" in sigma_comparison:
            reduction = sigma_comparison["sigma_reduction_original_spline"]
            if reduction > 0.1:  # Arbitrary threshold
                f.write("2. **Sigma reduction**: Spline model reduces measurement error ")
                f.write(f"by {reduction:.4f} units\n")

        # Daily component note
        if "prop_variance_daily_spline" in sigma_comparison:
            prop_var = sigma_comparison["prop_variance_daily_spline"]
            if prop_var > 0.05:
                f.write(f"3. **Daily variation**: {prop_var:.1%} of variance from daily component ")
                f.write("(worth modeling)\n")
            else:
                f.write(f"3. **Daily variation**: Minimal ({prop_var:.1%}), ")
                f.write("consider simpler model without daily component\n")

        # Next steps
        f.write("\n## Next Steps\n\n")
        f.write("1. **Increase iterations**: For more reliable WAIC/LOO estimates, run with ")
        f.write("`--chains 4 --iter-sampling 1000 --iter-warmup 500`\n")
        f.write("2. **Check diagnostics**: Ensure R-hat < 1.01 and ESS > 400 for key parameters\n")
        f.write("3. **Validate with new data**: If possible, hold out recent data for predictive validation\n")
        f.write("4. **Explore sensitivity**: Try different Fourier harmonics with spline model\n")

        # Command to reproduce
        f.write("\n## Reproduction\n\n")
        f.write("To reproduce this analysis:\n")
        f.write("```bash\n")
        f.write("uv run python -m src.models.generate_model_report.py \\\n")
        f.write(f"  --output-dir {output_dir} \\\n")
        f.write("  --chains 4 --iter-warmup 500 --iter-sampling 500\n")
        f.write("```\n")

    print(f"✓ Report saved to {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive model comparison report with WAIC/LOO metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Path to data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/model_report",
        help="Directory for output report and plots",
    )
    parser.add_argument(
        "--chains",
        type=int,
        default=4,
        help="Number of MCMC chains",
    )
    parser.add_argument(
        "--iter-warmup",
        type=int,
        default=500,
        help="Warmup iterations per chain",
    )
    parser.add_argument(
        "--iter-sampling",
        type=int,
        default=500,
        help="Sampling iterations per chain",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching (force refit)",
    )
    parser.add_argument(
        "--force-refit",
        action="store_true",
        help="Force refit even if cached results exist",
    )
    parser.add_argument(
        "--fourier-harmonics",
        type=int,
        default=2,
        help="Number of Fourier harmonics for spline model",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 70)
    print("COMPREHENSIVE MODEL COMPARISON REPORT")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Chains: {args.chains}, Warmup: {args.iter_warmup}, Sampling: {args.iter_sampling}")
    print(f"Caching: {not args.no_cache}, Force refit: {args.force_refit}")
    print()

    # Load data first to get info
    from src.data.weight import load_weight_data, prepare_stan_data
    print("Loading weight data...")
    df = load_weight_data(args.data_dir)
    stan_data = prepare_stan_data(df, fourier_harmonics=args.fourier_harmonics)

    # Data information for report
    days_with_multiple = (df.groupby('date').size() > 1).sum()
    total_days = df['date'].nunique()

    data_info = {
        'N': stan_data['N'],
        'date_min': df['date'].min().date(),
        'date_max': df['date'].max().date(),
        'days_multiple': days_with_multiple,
        'total_days': total_days,
        'days_multiple_pct': days_with_multiple / total_days if total_days > 0 else 0,
    }

    print(f"  {data_info['N']} observations from {data_info['date_min']} to {data_info['date_max']}")
    print(f"  Days with multiple measurements: {days_with_multiple}/{total_days} ({data_info['days_multiple_pct']:.1%})")
    if data_info['days_multiple_pct'] < 0.1:
        print("  ⚠ Sparse intraday data: limited ability to estimate true daily cyclic patterns")
    print()

    # Initialize model results dictionary
    model_results = {}

    # Fit all models (original + optimized variants)
    models_to_fit = [
        ("original", fit_weight_model, {}),
        ("flexible", fit_weight_model_flexible, {}),
        ("cyclic", fit_weight_model_cyclic, {}),
        ("spline", fit_weight_model_spline, {"fourier_harmonics": args.fourier_harmonics}),
        ("original_optimized", fit_weight_model_optimized, {}),
        ("flexible_optimized", fit_weight_model_flexible_optimized, {}),
        ("cyclic_optimized", fit_weight_model_cyclic_optimized, {}),
        ("spline_optimized", fit_weight_model_spline_optimized, {"fourier_harmonics": args.fourier_harmonics}),
    ]

    for model_name, fit_func, kwargs in models_to_fit:
        print(f"Fitting {model_name} model...")
        try:
            fit, idata, df_model, stan_data_model = fit_func(
                data_dir=args.data_dir,
                chains=args.chains,
                iter_warmup=args.iter_warmup,
                iter_sampling=args.iter_sampling,
                cache=not args.no_cache,
                force_refit=args.force_refit,
                **kwargs,
            )
            model_results[model_name] = {
                'fit': fit,
                'idata': idata,
                'df': df_model,
                'stan_data': stan_data_model,
            }
            print(f"  ✓ {model_name.capitalize()} model fitted successfully")
        except Exception as e:
            print(f"  ✗ Error fitting {model_name} model: {e}")
            # Continue with other models
            continue

    if len(model_results) < 2:
        print("Error: Need at least 2 models for comparison")
        sys.exit(1)

    print()
    print("-" * 60)
    print("MODEL COMPARISONS")
    print("-" * 60)

    # WAIC/LOO comparison
    print("\nComputing WAIC and LOO-CV comparison...")
    try:
        df_waic_loo = compare_models_waic_loo(
            idata_original=model_results.get('original', {}).get('idata'),
            idata_flexible=model_results.get('flexible', {}).get('idata'),
            idata_cyclic=model_results.get('cyclic', {}).get('idata'),
            idata_spline=model_results.get('spline', {}).get('idata'),
            idata_spline_optimized=model_results.get('spline_optimized', {}).get('idata'),
            idata_original_optimized=model_results.get('original_optimized', {}).get('idata'),
            idata_flexible_optimized=model_results.get('flexible_optimized', {}).get('idata'),
            idata_cyclic_optimized=model_results.get('cyclic_optimized', {}).get('idata'),
            model_names=None,  # Use default names based on provided models
            print_summary=True,
        )
    except Exception as e:
        print(f"✗ WAIC/LOO comparison failed: {e}")
        df_waic_loo = pd.DataFrame()

    # Sigma comparison (original, cyclic, spline)
    print("\nComputing sigma comparison...")
    sigma_comparison = {}
    if 'original' in model_results and 'cyclic' in model_results:
        if 'spline' in model_results:
            sigma_comparison = compare_models_all(
                idata_original=model_results['original']['idata'],
                idata_cyclic=model_results['cyclic']['idata'],
                idata_spline=model_results['spline']['idata'],
                stan_data=model_results['cyclic']['stan_data'],
                print_summary=True,
            )
        else:
            sigma_comparison = compare_models_sigma(
                model_results['original']['idata'],
                model_results['cyclic']['idata'],
                model_results['cyclic']['stan_data'],
                print_summary=True,
            )

    # Add sigma values for all models to comparison
    for model_name, results in model_results.items():
        if 'idata' in results and 'sigma' in results['idata'].posterior:
            sigma_key = f"sigma_{model_name}"
            sigma_val = results['idata'].posterior["sigma"].values.mean()
            sigma_comparison[sigma_key] = sigma_val

    # Compute sigma reductions between original and optimized versions
    for base_model in ["original", "flexible", "cyclic", "spline"]:
        base_key = f"sigma_{base_model}"
        opt_key = f"sigma_{base_model}_optimized"

        if base_key in sigma_comparison and opt_key in sigma_comparison:
            sigma_base = sigma_comparison[base_key]
            sigma_opt = sigma_comparison[opt_key]

            # Compute absolute reduction
            reduction_key = f"sigma_reduction_{base_model}_{base_model}_optimized"
            sigma_comparison[reduction_key] = sigma_base - sigma_opt

            # Compute percentage reduction
            if sigma_base > 0:
                pct_key = f"sigma_reduction_pct_{base_model}_{base_model}_optimized"
                sigma_comparison[pct_key] = (sigma_base - sigma_opt) / sigma_base * 100

    # Generate plots
    print()
    print("Generating visualizations...")

    # 1. Cyclic components plot
    if 'cyclic' in model_results:
        components_path = output_dir / "cyclic_components.png"
        try:
            plot_cyclic_components(
                model_results['cyclic']['idata'],
                model_results['cyclic']['df'],
                model_results['cyclic']['stan_data'],
                output_path=str(components_path),
                show_trend_component=True,
                show_daily_component=True,
            )
            print(f"  ✓ Cyclic components plot saved to {components_path}")
        except Exception as e:
            print(f"  ⚠ Could not generate cyclic components plot: {e}")

    # 2. Daily pattern plot (cyclic)
    if 'cyclic' in model_results:
        daily_pattern_path = output_dir / "daily_pattern.png"
        try:
            plot_daily_pattern(
                model_results['cyclic']['idata'],
                model_results['cyclic']['df'],
                model_results['cyclic']['stan_data'],
                output_path=str(daily_pattern_path),
            )
            print(f"  ✓ Daily pattern plot saved to {daily_pattern_path}")
        except Exception as e:
            print(f"  ⚠ Could not generate daily pattern plot: {e}")

    # 3. Spline daily pattern plot
    if 'spline' in model_results:
        spline_pattern_path = output_dir / "spline_daily_pattern.png"
        try:
            plot_spline_daily_pattern(
                model_results['spline']['idata'],
                model_results['spline']['stan_data'],
                output_path=str(spline_pattern_path),
            )
            print(f"  ✓ Spline daily pattern plot saved to {spline_pattern_path}")
        except Exception as e:
            print(f"  ⚠ Could not generate spline daily pattern plot: {e}")

    # 4. Model comparison plot (sigma comparison)
    if 'original' in model_results and 'cyclic' in model_results:
        comparison_path = output_dir / "model_comparison_all.png"
        try:
            if 'spline' in model_results:
                plot_models_comparison_all(
                    idata_original=model_results['original']['idata'],
                    idata_cyclic=model_results['cyclic']['idata'],
                    idata_spline=model_results['spline']['idata'],
                    stan_data=model_results['cyclic']['stan_data'],
                    output_path=str(comparison_path),
                )
            else:
                # Use simpler comparison if spline not available
                from src.models.plot_cyclic import plot_model_comparison
                plot_model_comparison(
                    model_results['original']['idata'],
                    model_results['cyclic']['idata'],
                    model_results['cyclic']['stan_data'],
                    output_path=str(comparison_path),
                )
            print(f"  ✓ Model comparison plot saved to {comparison_path}")
        except Exception as e:
            print(f"  ⚠ Could not generate model comparison plot: {e}")

    # 5. Full expectation plots for each model
    print("\nGenerating full expectation plots for each model...")
    full_expectation_plots = []
    for model_name, results in model_results.items():
        if 'idata' in results and 'df' in results and 'stan_data' in results:
            plot_path = output_dir / f"full_expectation_{model_name}.png"
            try:
                plot_model_full_expectation(
                    idata=results['idata'],
                    df=results['df'],
                    stan_data=results['stan_data'],
                    model_name=model_name,
                    output_path=str(plot_path),
                    show_observations=True,
                    show_ci=True,
                )
                full_expectation_plots.append((model_name, plot_path))
                print(f"  ✓ Full expectation plot for {model_name} saved to {plot_path}")
            except Exception as e:
                print(f"  ⚠ Could not generate full expectation plot for {model_name}: {e}")

    # Generate markdown report
    print()
    print("Generating markdown report...")
    try:
        report_path = generate_markdown_report(
            output_dir=output_dir,
            df_waic_loo=df_waic_loo,
            sigma_comparison=sigma_comparison,
            data_info=data_info,
            timestamp=timestamp,
        )
        print(f"✓ Comprehensive report generated: {report_path}")
    except Exception as e:
        print(f"✗ Report generation failed: {e}")
        import traceback
        traceback.print_exc()

    # Final summary
    print()
    print("=" * 70)
    print("REPORT GENERATION COMPLETE")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Markdown report: {output_dir}/model_comparison_report.md")

    # List generated files
    print("\nGenerated files:")
    for file in output_dir.glob("*"):
        if file.is_file():
            print(f"  • {file.name}")

    print("\nTo view the report:")
    print(f"  cat {output_dir}/model_comparison_report.md")
    print("or open in a markdown viewer/editor.")
    print()


if __name__ == "__main__":
    main()