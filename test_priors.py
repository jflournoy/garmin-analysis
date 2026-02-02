"""Test different prior configurations for the weight GP model."""
import warnings
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models.fit_weight import fit_weight_model_flexible


def test_prior_config(name, alpha_sd, rho_shape, rho_scale, sigma_sd, chains=2, iter_warmup=100, iter_sampling=100):
    """Test a specific prior configuration."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    print(f"  alpha ~ normal(0, {alpha_sd})")
    print(f"  rho ~ inv_gamma({rho_shape}, {rho_scale})")
    print(f"  sigma ~ normal(0, {sigma_sd})")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        try:
            fit, idata, df, stan_data = fit_weight_model_flexible(
                stan_file="stan/weight_gp_flexible.stan",
                chains=chains,
                iter_warmup=iter_warmup,
                iter_sampling=iter_sampling,
                alpha_prior_sd=alpha_sd,
                rho_prior_shape=rho_shape,
                rho_prior_scale=rho_scale,
                sigma_prior_sd=sigma_sd,
            )

            # Check warnings
            if w:
                print(f"\nWarnings ({len(w)}):")
                for warning in w:
                    print(f"  - {warning.category.__name__}: {warning.message}")

            # Compute key statistics
            posterior = idata.posterior
            y_sd = stan_data["_y_sd"]
            y_mean = stan_data["_y_mean"]

            # Hyperparameters
            alpha_mean = posterior["alpha"].mean().item()
            rho_mean = posterior["rho"].mean().item()
            sigma_mean = posterior["sigma"].mean().item()

            # Trend change (original scale)
            trend_change = posterior["trend_change"].mean().item() * y_sd

            # Uncertainty statistics (from f samples)
            f_samples = posterior["f"].values  # shape: (chain, draw, obs)
            f_std = f_samples.std(axis=(0, 1)) * y_sd  # standard deviation in lbs
            avg_uncertainty = f_std.mean()
            max_uncertainty = f_std.max()
            min_uncertainty = f_std.min()

            print("\nResults:")
            print(f"  alpha (posterior mean): {alpha_mean:.3f}")
            print(f"  rho (posterior mean): {rho_mean:.3f}")
            print(f"  sigma (posterior mean): {sigma_mean:.3f}")
            print(f"  Trend change: {trend_change:.2f} lbs")
            print(f"  Avg uncertainty (std dev): {avg_uncertainty:.2f} lbs")
            print(f"  Uncertainty range: {min_uncertainty:.2f} to {max_uncertainty:.2f} lbs")

            # Check for divergent transitions
            try:
                divergent = fit.diagnose().get('divergent_iterations', 0)
                print(f"  Divergent transitions: {divergent}")
            except AttributeError:
                print("  Divergent transitions: diagnose returned string")

            # Compute posterior mean of f for plotting
            f_mean = f_samples.mean(axis=(0, 1)) * y_sd + y_mean

            return {
                "name": name,
                "fit": fit,
                "idata": idata,
                "df": df,
                "stan_data": stan_data,
                "f_mean": f_mean,
                "f_std": f_std,
                "alpha_mean": alpha_mean,
                "rho_mean": rho_mean,
                "sigma_mean": sigma_mean,
                "trend_change": trend_change,
                "avg_uncertainty": avg_uncertainty,
                "warnings": w,
            }

        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()
            return None


def plot_comparison(results, output_path="output/prior_comparison.png"):
    """Plot comparison of different prior configurations."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Colors for different configurations
    colors = plt.cm.Set1(np.linspace(0, 1, len(results)))

    # Plot 1: Fitted curves
    ax = axes[0, 0]
    for i, res in enumerate(results):
        if res is None:
            continue
        ax.plot(res["df"]["date"], res["f_mean"], label=res["name"], color=colors[i], linewidth=2)
    ax.scatter(results[0]["df"]["date"], results[0]["df"]["weight_lbs"],
               alpha=0.5, s=20, color="black", label="Observations", zorder=10)
    ax.set_xlabel("Date")
    ax.set_ylabel("Weight (lbs)")
    ax.set_title("Fitted GP Means - Prior Comparison")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Plot 2: Uncertainty (standard deviation) over time
    ax = axes[0, 1]
    for i, res in enumerate(results):
        if res is None:
            continue
        ax.plot(res["df"]["date"], res["f_std"], label=res["name"], color=colors[i], linewidth=1.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Uncertainty (lbs)")
    ax.set_title("Uncertainty Over Time (Std Dev)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Plot 3: Hyperparameter posteriors (violin plots)
    ax = axes[1, 0]
    param_data = []
    param_names = []
    for i, res in enumerate(results):
        if res is None:
            continue
        # Extract posterior samples for this configuration
        posterior = res["idata"].posterior
        for param in ["alpha", "rho", "sigma"]:
            samples = posterior[param].values.flatten()
            param_data.append(samples)
            param_names.append(f"{res['name']}\n{param}")

    if param_data:
        ax.violinplot(param_data, showmeans=True, showmedians=True)
        ax.set_xticks(range(1, len(param_names) + 1))
        ax.set_xticklabels(param_names, rotation=45, ha="right")
        ax.set_ylabel("Parameter Value")
        ax.set_title("Posterior Distributions of Hyperparameters")
        ax.grid(True, alpha=0.3)

    # Plot 4: Summary statistics bar chart
    ax = axes[1, 1]
    if results and results[0] is not None:
        names = [r["name"] for r in results if r is not None]
        trend_changes = [r["trend_change"] for r in results if r is not None]
        avg_uncertainties = [r["avg_uncertainty"] for r in results if r is not None]

        x = np.arange(len(names))
        width = 0.35

        bars1 = ax.bar(x - width/2, trend_changes, width, label='Trend Change (lbs)', color='skyblue')
        bars2 = ax.bar(x + width/2, avg_uncertainties, width, label='Avg Uncertainty (lbs)', color='lightcoral')

        ax.set_xlabel("Prior Configuration")
        ax.set_ylabel("Value (lbs)")
        ax.set_title("Trend Change vs Average Uncertainty")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nComparison plot saved to {output_path}")
    return fig


def main():
    """Run prior sensitivity analysis."""
    print("Running prior sensitivity analysis for weight GP model")
    print("Testing different prior configurations for wigglyness...")

    # Define prior configurations to test
    configs = [
        {
            "name": "Default (Smooth)",
            "alpha_sd": 1.0,
            "rho_shape": 5.0,
            "rho_scale": 1.0,
            "sigma_sd": 0.5,
        },
        {
            "name": "More Wiggly",
            "alpha_sd": 2.0,
            "rho_shape": 2.0,
            "rho_scale": 0.5,
            "sigma_sd": 1.0,
        },
        {
            "name": "Very Wiggly",
            "alpha_sd": 3.0,
            "rho_shape": 1.5,
            "rho_scale": 0.2,
            "sigma_sd": 1.5,
        },
        {
            "name": "Exponential Prior",
            "alpha_sd": 1.0,
            "rho_shape": 1.0,  # inv_gamma(1,1) is heavy-tailed
            "rho_scale": 1.0,
            "sigma_sd": 0.5,
        },
    ]

    results = []
    for config in configs:
        result = test_prior_config(
            name=config["name"],
            alpha_sd=config["alpha_sd"],
            rho_shape=config["rho_shape"],
            rho_scale=config["rho_scale"],
            sigma_sd=config["sigma_sd"],
            chains=2,
            iter_warmup=100,
            iter_sampling=100,
        )
        results.append(result)

    # Create comparison plot
    valid_results = [r for r in results if r is not None]
    if valid_results:
        plot_comparison(valid_results)

    print("\n" + "="*60)
    print("PRIOR SENSITIVITY ANALYSIS COMPLETE")
    print("="*60)

    # Summary table
    print("\nSummary Table:")
    print("-" * 80)
    print(f"{'Configuration':<20} {'Trend Change':<15} {'Avg Uncertainty':<15} {'alpha':<10} {'rho':<10} {'sigma':<10}")
    print("-" * 80)
    for res in valid_results:
        print(f"{res['name']:<20} {res['trend_change']:<15.2f} {res['avg_uncertainty']:<15.2f} "
              f"{res['alpha_mean']:<10.3f} {res['rho_mean']:<10.3f} {res['sigma_mean']:<10.3f}")

    # Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS:")
    print("="*60)
    print("1. Choose prior configuration that balances fit and uncertainty")
    print("2. Consider using 'More Wiggly' if data shows short-term fluctuations")
    print("3. Monitor divergent transitions and adjust priors if needed")
    print("4. Consider increasing sampling iterations for final analysis")


if __name__ == "__main__":
    main()