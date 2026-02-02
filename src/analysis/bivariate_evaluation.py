"""Evaluate bivariate GP model results."""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import arviz as az

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.fit_weight import fit_bivariate_model


def main():
    output_dir = Path("output/bivariate")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Fitting bivariate model (weight + resting heart rate)...")
    fit, idata, df, stan_data = fit_bivariate_model(
        chains=2,
        iter_warmup=100,
        iter_sampling=100,
        cache=False,
        force_refit=True,
        use_sparse=True,
        n_inducing_points=30,
    )

    print("\n=== Model Summary ===")
    print(f"Observations: {len(df)}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    # Extract posterior samples
    posterior = idata.posterior
    print("\nPosterior variables:", list(posterior.data_vars.keys()))

    # Correlation between latent processes
    if 'correlation' in posterior:
        corr_samples = posterior['correlation'].values.flatten()
        print("\nLatent correlation posterior:")
        print(f"  Mean: {np.mean(corr_samples):.3f}")
        print(f"  SD: {np.std(corr_samples):.3f}")
        print(f"  95% CI: [{np.percentile(corr_samples, 2.5):.3f}, {np.percentile(corr_samples, 97.5):.3f}]")
        # Compare with empirical correlation
        emp_corr = df[['weight_mean', 'resting_heart_rate']].corr().iloc[0, 1]
        print(f"  Empirical correlation: {emp_corr:.3f}")

        # Plot correlation posterior
        plt.figure(figsize=(8, 5))
        plt.hist(corr_samples, bins=30, alpha=0.7, density=True, edgecolor='black')
        plt.axvline(emp_corr, color='red', linestyle='--', label=f'Empirical = {emp_corr:.3f}')
        plt.xlabel('Latent correlation')
        plt.ylabel('Density')
        plt.title('Posterior distribution of latent correlation (weight vs resting heart rate)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'correlation_posterior.png', dpi=150)
        plt.close()
        print(f"Saved correlation posterior plot to {output_dir / 'correlation_posterior.png'}")

    # Check convergence (R-hat)
    rhat = az.rhat(idata)
    print("\nR-hat statistics (should be <1.01):")
    for var in rhat.data_vars:
        val = float(rhat[var].values)
        if val > 1.01:
            print(f"  {var}: {val:.3f} *** High!")
        else:
            print(f"  {var}: {val:.3f}")

    # Save summary table
    summary = az.summary(idata, round_to=3)
    summary.to_csv(output_dir / 'posterior_summary.csv')
    print(f"\nSaved posterior summary to {output_dir / 'posterior_summary.csv'}")

    # Plot trend components
    if 'f_trend' in posterior:
        # Posterior mean of trend for weight and other
        f_trend_weight = posterior['f_trend'].sel(output='weight').mean(dim=('chain', 'draw')).values
        f_trend_other = posterior['f_trend'].sel(output='other').mean(dim=('chain', 'draw')).values

        # Unscale using stored scaling parameters
        y_weight_mean = stan_data['_y_weight_mean']
        y_weight_sd = stan_data['_y_weight_sd']
        y_other_mean = stan_data['_y_other_mean']
        y_other_sd = stan_data['_y_other_sd']

        f_trend_weight_unscaled = f_trend_weight * y_weight_sd + y_weight_mean
        f_trend_other_unscaled = f_trend_other * y_other_sd + y_other_mean

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        axes[0].plot(df['date'], df['weight_mean'], 'k.', alpha=0.5, label='Weight (observed)')
        axes[0].plot(df['date'], f_trend_weight_unscaled, 'b-', label='GP trend (mean)')
        axes[0].set_ylabel('Weight (lbs)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(df['date'], df['resting_heart_rate'], 'k.', alpha=0.5, label='Resting HR (observed)')
        axes[1].plot(df['date'], f_trend_other_unscaled, 'r-', label='GP trend (mean)')
        axes[1].set_ylabel('Resting heart rate (bpm)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.suptitle('Bivariate GP Trend Components')
        plt.tight_layout()
        plt.savefig(output_dir / 'trend_components.png', dpi=150)
        plt.close()
        print(f"Saved trend components plot to {output_dir / 'trend_components.png'}")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()