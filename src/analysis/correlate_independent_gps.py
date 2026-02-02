"""Compute correlation between latent curves using independent GPs."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_weight_data, load_vo2max_data
from src.models.fit_weight import fit_gp_simple


def prepare_weight_for_simple_gp(df_weight: pd.DataFrame) -> pd.DataFrame:
    """Prepare weight data for simple GP (aggregate to daily if multiple measurements)."""
    # If multiple measurements per day, aggregate to daily mean
    df = df_weight.copy()
    df["date"] = df["timestamp"].dt.date
    df["date"] = pd.to_datetime(df["date"])

    # Group by date
    daily = df.groupby("date")["weight_lbs"].agg(["mean", "std", "count"]).reset_index()
    daily = daily.rename(columns={"mean": "weight", "date": "timestamp"})

    return daily


def align_to_common_time(df1, time_col1, df2, time_col2):
    """Align two time series to a common global time zero.

    Returns:
        df1_aligned, df2_aligned, global_t_min, global_t_max
        Each aligned DataFrame has new column 'days_since_global'.
    """
    # Convert to datetime if not already
    t1 = pd.to_datetime(df1[time_col1])
    t2 = pd.to_datetime(df2[time_col2])

    global_t_min = min(t1.min(), t2.min())
    global_t_max = max(t1.max(), t2.max())

    df1_aligned = df1.copy()
    df2_aligned = df2.copy()

    df1_aligned['days_since_global'] = (t1 - global_t_min).dt.days
    df2_aligned['days_since_global'] = (t2 - global_t_min).dt.days

    return df1_aligned, df2_aligned, global_t_min, global_t_max


def sample_latent_functions_on_grid(
    idata_weight,
    idata_other,
    stan_data_weight,
    stan_data_other,
    n_samples: int = 100,
) -> tuple:
    """Sample latent functions on common time grid using f_pred.

    Requires that both GPs were fitted with same prediction grid (t_pred).

    Returns:
        Tuple of (t_grid_days, f_weight_samples, f_other_samples)
        where each samples matrix is n_samples x len(t_grid_days)
        t_grid_days is absolute days (since global time zero).
    """
    # Check if f_pred exists in posterior_predictive (not posterior)
    if "f_pred" not in idata_weight.posterior_predictive or "f_pred" not in idata_other.posterior_predictive:
        raise ValueError("Both GPs must have prediction grid (f_pred) in posterior_predictive")

    # Extract f_pred samples
    f_weight_pred = idata_weight.posterior_predictive["f_pred"].values  # shape (chain, draw, N_pred)
    f_other_pred = idata_other.posterior_predictive["f_pred"].values

    # Ensure prediction grids are the same length (should be if same t_pred used)
    if f_weight_pred.shape[-1] != f_other_pred.shape[-1]:
        raise ValueError(f"Prediction grid sizes differ between GPs: {f_weight_pred.shape[-1]} vs {f_other_pred.shape[-1]}")

    # Reshape to (total_samples, N_pred)
    total_samples_weight = f_weight_pred.shape[0] * f_weight_pred.shape[1]
    total_samples_other = f_other_pred.shape[0] * f_other_pred.shape[1]

    f_weight_flat = f_weight_pred.reshape(total_samples_weight, -1)
    f_other_flat = f_other_pred.reshape(total_samples_other, -1)

    # If we have more samples than needed, randomly select n_samples
    if n_samples < total_samples_weight:
        idx = np.random.choice(total_samples_weight, n_samples, replace=False)
        f_weight_samples = f_weight_flat[idx]
        f_other_samples = f_other_flat[idx]
    else:
        f_weight_samples = f_weight_flat
        f_other_samples = f_other_flat
        n_samples = total_samples_weight

    # Get absolute prediction grid from stan_data (should be same for both)
    t_pred_days = np.array(stan_data_weight.get("_t_pred_days", []))
    if len(t_pred_days) == 0:
        t_pred_days = np.array(stan_data_other.get("_t_pred_days", []))
    if len(t_pred_days) == 0:
        raise ValueError("No _t_pred_days found in stan_data")

    # Ensure length matches
    if len(t_pred_days) != f_weight_samples.shape[1]:
        raise ValueError(f"t_pred_days length {len(t_pred_days)} doesn't match f_pred samples {f_weight_samples.shape[1]}")

    return t_pred_days, f_weight_samples, f_other_samples


def compute_correlation_from_samples(
    f_weight_samples: np.ndarray,
    f_other_samples: np.ndarray,
) -> dict:
    """Compute correlation statistics from latent function samples."""
    n_samples = f_weight_samples.shape[0]
    correlations = np.zeros(n_samples)

    for i in range(n_samples):
        # Pearson correlation between two vectors
        if np.std(f_weight_samples[i]) > 1e-10 and np.std(f_other_samples[i]) > 1e-10:
            corr = np.corrcoef(f_weight_samples[i], f_other_samples[i])[0, 1]
            correlations[i] = corr
        else:
            correlations[i] = np.nan

    # Remove NaN
    correlations = correlations[~np.isnan(correlations)]

    if len(correlations) == 0:
        return None

    return {
        "mean": np.mean(correlations),
        "std": np.std(correlations),
        "2.5%": np.percentile(correlations, 2.5),
        "50%": np.percentile(correlations, 50),
        "97.5%": np.percentile(correlations, 97.5),
        "samples": correlations,
    }


def analyze_weight_vo2max():
    """Analyze correlation between weight and VO2 max using independent GPs with common prediction grid."""
    output_dir = Path("output/independent_gp_correlation")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Weight vs VO2 Max Correlation Analysis ===")

    # Load data
    print("\n1. Loading data...")
    df_weight_raw = load_weight_data()
    df_vo2max = load_vo2max_data()

    print(f"   Weight: {len(df_weight_raw)} measurements")
    print(f"   VO2 max: {len(df_vo2max)} measurements")

    # Prepare weight data (aggregate to daily)
    df_weight_daily = prepare_weight_for_simple_gp(df_weight_raw)
    print(f"   Weight (daily): {len(df_weight_daily)} days")

    # Align both datasets to common global time
    print("\n2. Aligning to common global time...")
    df_weight_aligned, df_vo2_aligned, global_t_min, global_t_max = align_to_common_time(
        df_weight_daily, "timestamp", df_vo2max, "date"
    )
    print(f"   Global time range: {global_t_min.date()} to {global_t_max.date()} ({global_t_max - global_t_min} days)")

    # Create common prediction grid (daily)
    t_pred_days = np.arange(0, (global_t_max - global_t_min).days + 1, 1)
    print(f"   Common prediction grid: {len(t_pred_days)} daily points")

    # Fit simple GP to weight using global time alignment
    print("\n3. Fitting GP to weight (global time)...")
    fit_weight, idata_weight, df_weight_prep, stan_data_weight = fit_gp_simple(
        df_weight_aligned,
        time_col="timestamp_global",
        value_col="weight",
        chains=1,
        iter_warmup=5,
        iter_sampling=5,
        use_sparse=True,
        n_inducing_points=10,
        cache=False,
        force_refit=True,
        t_pred_days=t_pred_days,
    )

    # Fit simple GP to VO2 max using global time alignment
    print("\n4. Fitting GP to VO2 max (global time)...")
    fit_vo2, idata_vo2, df_vo2_prep, stan_data_vo2 = fit_gp_simple(
        df_vo2_aligned,
        time_col="timestamp_global",
        value_col="vo2_max",
        chains=1,
        iter_warmup=5,
        iter_sampling=5,
        use_sparse=True,
        n_inducing_points=10,
        cache=False,
        force_refit=True,
        t_pred_days=t_pred_days,
    )

    print("\n5. Computing empirical correlation...")
    # Merge on overlapping dates (using original timestamps)
    merged = pd.merge(df_weight_daily, df_vo2max, left_on="timestamp", right_on="date", how="inner")
    if len(merged) > 0:
        emp_corr = merged["weight"].corr(merged["vo2_max"])
        print(f"   Empirical correlation (aligned dates): {emp_corr:.3f} (n={len(merged)})")
    else:
        print("   No overlapping dates for empirical correlation")
        emp_corr = None

    # Plot both latent functions on global time axis
    print("\n6. Plotting latent functions...")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Convert days_since_global back to datetime for plotting
    weight_dates = global_t_min + pd.to_timedelta(df_weight_prep["days_since_global"], unit='D')
    vo2_dates = global_t_min + pd.to_timedelta(df_vo2_prep["days_since_global"], unit='D')

    # Weight
    ax = axes[0]
    ax.plot(weight_dates, df_weight_prep["weight"], 'k.', alpha=0.5, label='Weight (observed)')
    # Plot posterior mean of latent function (unscaled)
    f_weight_mean = idata_weight.posterior["f_trend"].mean(dim=("chain", "draw")).values
    f_weight_unscaled = f_weight_mean * stan_data_weight["_y_sd"] + stan_data_weight["_y_mean"]
    ax.plot(weight_dates, f_weight_unscaled, 'b-', label='GP trend (mean)')
    ax.set_ylabel('Weight (lbs)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # VO2 max
    ax = axes[1]
    ax.plot(vo2_dates, df_vo2_prep["vo2_max"], 'k.', alpha=0.5, label='VO2 max (observed)')
    f_vo2_mean = idata_vo2.posterior["f_trend"].mean(dim=("chain", "draw")).values
    f_vo2_unscaled = f_vo2_mean * stan_data_vo2["_y_sd"] + stan_data_vo2["_y_mean"]
    ax.plot(vo2_dates, f_vo2_unscaled, 'r-', label='GP trend (mean)')
    ax.set_ylabel('VO2 max (mL/kg/min)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Independent GP Fits: Weight and VO2 Max (Global Time Alignment)')
    plt.tight_layout()
    plt.savefig(output_dir / 'independent_gp_fits_global.png', dpi=150)
    plt.close()
    print(f"   Saved GP fits plot to {output_dir / 'independent_gp_fits_global.png'}")

    # Sample latent functions on common prediction grid
    print("\n7. Sampling latent functions on common grid...")
    t_grid_days, f_weight_samples, f_vo2_samples = sample_latent_functions_on_grid(
        idata_weight, idata_vo2, stan_data_weight, stan_data_vo2, n_samples=100
    )

    # Compute correlation posterior
    print("\n8. Computing latent correlation posterior...")
    corr_results = compute_correlation_from_samples(f_weight_samples, f_vo2_samples)
    if corr_results is not None:
        print(f"   Latent correlation posterior mean: {corr_results['mean']:.3f}")
        print(f"   95% CI: [{corr_results['2.5%']:.3f}, {corr_results['97.5%']:.3f}]")
        print(f"   Standard deviation: {corr_results['std']:.3f}")
    else:
        print("   Could not compute correlation (insufficient samples)")

    # Save results summary
    print("\n9. Saving results...")
    with open(output_dir / 'summary.txt', 'w') as f:
        f.write("Weight vs VO2 Max Correlation Analysis (Independent GPs with Common Grid)\n")
        f.write("==========================================================================\n\n")
        f.write(f"Weight measurements: {len(df_weight_raw)} (raw), {len(df_weight_daily)} (daily)\n")
        f.write(f"VO2 max measurements: {len(df_vo2max)}\n")
        f.write(f"Overlapping dates: {len(merged) if 'merged' in locals() else 0}\n")
        if emp_corr is not None:
            f.write(f"Empirical correlation (aligned dates): {emp_corr:.3f}\n")
        f.write(f"Common prediction grid: {len(t_pred_days)} daily points\n")
        f.write(f"Global time range: {global_t_min.date()} to {global_t_max.date()}\n")
        if corr_results is not None:
            f.write("\nLatent correlation (posterior):\n")
            f.write(f"  Mean: {corr_results['mean']:.3f}\n")
            f.write(f"  Std: {corr_results['std']:.3f}\n")
            f.write(f"  2.5%: {corr_results['2.5%']:.3f}\n")
            f.write(f"  50%: {corr_results['50%']:.3f}\n")
            f.write(f"  97.5%: {corr_results['97.5%']:.3f}\n")
        f.write("\nModel: Independent GPs with Student-t likelihood\n")
        f.write("  Weight GP: simple GP with squared exponential kernel\n")
        f.write("  VO2 max GP: simple GP with squared exponential kernel\n")
        f.write("  Common prediction grid: daily points across global time range\n")

    print(f"\nAnalysis complete. Results saved to {output_dir}")


if __name__ == "__main__":
    analyze_weight_vo2max()