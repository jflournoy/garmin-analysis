"""Visualization for cyclic GP model components."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .plot_zoom import zoom_to_date_range, zoom_to_preset


def plot_cyclic_components(
    idata,
    df,
    stan_data,
    output_path: str = None,
    show_daily_component: bool = True,
    show_trend_component: bool = True,
    show_weekly_component: bool = False,
):
    """Plot the cyclic GP model with separate trend, daily, and weekly components.

    Args:
        idata: ArviZ InferenceData from cyclic model
        df: Original DataFrame with timestamps
        stan_data: Stan data dictionary
        output_path: Path to save plot (optional)
        show_daily_component: Whether to show daily component plot
        show_trend_component: Whether to show trend component plot
        show_weekly_component: Whether to show weekly component plot (if available)
    """
    # Back-transform to original scale
    y_mean = stan_data["_y_mean"]
    y_sd = stan_data["_y_sd"]

    # Extract posterior samples
    f_trend_samples = idata.posterior["f_trend"].values
    f_daily_samples = idata.posterior["f_daily"].values
    f_total_samples = idata.posterior["f_total"].values

    # Compute means and credible intervals
    def compute_stats(samples):
        mean = samples.mean(axis=(0, 1)) * y_sd + y_mean
        lower = np.percentile(samples, 2.5, axis=(0, 1)) * y_sd + y_mean
        upper = np.percentile(samples, 97.5, axis=(0, 1)) * y_sd + y_mean
        return mean, lower, upper

    f_trend_mean, f_trend_lower, f_trend_upper = compute_stats(f_trend_samples)
    f_daily_mean, f_daily_lower, f_daily_upper = compute_stats(f_daily_samples)
    f_total_mean, f_total_lower, f_total_upper = compute_stats(f_total_samples)

    # Create figure
    n_plots = 1 + show_trend_component + show_daily_component
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots))

    if n_plots == 1:
        axes = [axes]

    plot_idx = 0

    # Plot 1: Total fit (trend + daily)
    ax = axes[plot_idx]
    ax.scatter(df["timestamp"], df["weight_lbs"], alpha=0.5, s=20, label="Observations")
    ax.plot(df["timestamp"], f_total_mean, "k-", linewidth=2, label="Total (trend + daily)")
    ax.fill_between(df["timestamp"], f_total_lower, f_total_upper, alpha=0.3, color="blue", label="95% CI")
    ax.set_xlabel("Date")
    ax.set_ylabel("Weight (lbs)")
    ax.set_title("Weight Over Time - Cyclic GP Model (Trend + Daily)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plot_idx += 1

    # Plot 2: Trend component
    if show_trend_component:
        ax = axes[plot_idx]
        ax.plot(df["timestamp"], f_trend_mean, "r-", linewidth=2, label="Trend component")
        ax.fill_between(df["timestamp"], f_trend_lower, f_trend_upper, alpha=0.3, color="red", label="95% CI")
        ax.set_xlabel("Date")
        ax.set_ylabel("Weight (lbs)")
        ax.set_title("Trend Component (Long-term changes)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Plot 3: Daily component
    if show_daily_component:
        ax = axes[plot_idx]

        # Plot daily component over time
        ax.plot(df["timestamp"], f_daily_mean, "g-", linewidth=2, label="Daily component")
        ax.fill_between(df["timestamp"], f_daily_lower, f_daily_upper, alpha=0.3, color="green", label="95% CI")

        # Add horizontal line at zero for reference
        ax.axhline(y=y_mean, color="gray", linestyle="--", alpha=0.5, label=f"Mean ({y_mean:.1f} lbs)")

        ax.set_xlabel("Date")
        ax.set_ylabel("Weight deviation (lbs)")
        ax.set_title("Daily Component (Intraday variation)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved cyclic components plot to {output_path}")

    return fig


def plot_daily_pattern(
    idata,
    df,
    stan_data,
    output_path: str = None,
):
    """Plot the estimated daily pattern (hour-of-day effect).

    Args:
        idata: ArviZ InferenceData from cyclic model
        df: Original DataFrame with timestamps
        stan_data: Stan data dictionary
        output_path: Path to save plot (optional)
    """
    # Extract hour of day from data
    if "hour_of_day" not in stan_data:
        raise ValueError("Stan data must include hour_of_day for daily pattern plot")

    hour_of_day = stan_data["hour_of_day"]
    f_daily_samples = idata.posterior["f_daily"].values

    # Back-transform to original scale
    y_sd = stan_data["_y_sd"]

    # Reshape samples for easier processing
    n_chains, n_draws, n_obs = f_daily_samples.shape
    f_daily_flat = f_daily_samples.reshape(-1, n_obs) * y_sd  # Shape: (n_samples, n_obs)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Daily pattern scatter with uncertainty
    ax = axes[0]
    for i in range(min(50, f_daily_flat.shape[0])):  # Plot first 50 samples
        ax.scatter(hour_of_day, f_daily_flat[i, :], alpha=0.05, s=10, color="blue")

    # Add mean line
    f_daily_mean = f_daily_flat.mean(axis=0)
    ax.plot(hour_of_day, f_daily_mean, "r-", linewidth=2, label="Mean daily pattern")

    ax.set_xlabel("Hour of day (GMT)")
    ax.set_ylabel("Weight deviation (lbs)")
    ax.set_title("Estimated Daily Weight Pattern")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Hourly distribution
    ax = axes[1]

    # Bin by hour and compute statistics
    hours = np.floor(hour_of_day).astype(int)
    unique_hours = np.unique(hours)

    hour_means = []
    hour_stds = []
    hour_counts = []

    for h in unique_hours:
        mask = hours == h
        if mask.sum() > 0:
            # Compute mean and std across samples for this hour
            hour_data = f_daily_flat[:, mask].mean(axis=1)  # Mean across observations in this hour
            hour_means.append(hour_data.mean())
            hour_stds.append(hour_data.std())
            hour_counts.append(mask.sum())

    # Create bar plot
    x_pos = np.arange(len(unique_hours))
    ax.bar(x_pos, hour_means, yerr=hour_stds, capsize=5, alpha=0.7, color="steelblue")
    ax.set_xlabel("Hour of day (GMT)")
    ax.set_ylabel("Mean weight deviation (lbs)")
    ax.set_title("Hourly Average Weight Pattern")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{int(h):02d}:00" if h.is_integer() else f"{h:02.1f}:00" for h in unique_hours])
    ax.grid(True, alpha=0.3, axis="y")

    # Add count labels
    for i, count in enumerate(hour_counts):
        ax.text(x_pos[i], hour_means[i] + (hour_stds[i] if i < len(hour_stds) else 0),
                f"n={count}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved daily pattern plot to {output_path}")

    return fig


def plot_model_comparison(
    idata_original,
    idata_cyclic,
    stan_data,
    output_path: str = None,
):
    """Plot comparison between original and cyclic models.

    Args:
        idata_original: InferenceData from original model
        idata_cyclic: InferenceData from cyclic model
        stan_data: Stan data dictionary
        output_path: Path to save plot (optional)
    """
    # Extract sigma estimates
    sigma_orig = idata_original.posterior["sigma"].values.flatten() * stan_data["_y_sd"]
    sigma_cyclic = idata_cyclic.posterior["sigma"].values.flatten() * stan_data["_y_sd"]

    # Extract daily amplitude
    daily_amplitude = idata_cyclic.posterior["daily_amplitude"].values.flatten() * stan_data["_y_sd"]

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: Sigma comparison
    ax = axes[0]
    ax.hist(sigma_orig, bins=30, alpha=0.5, label=f"Original: {sigma_orig.mean():.2f} ± {sigma_orig.std():.2f} lbs")
    ax.hist(sigma_cyclic, bins=30, alpha=0.5, label=f"Cyclic: {sigma_cyclic.mean():.2f} ± {sigma_cyclic.std():.2f} lbs")
    ax.set_xlabel("Sigma (measurement error, lbs)")
    ax.set_ylabel("Frequency")
    ax.set_title("Measurement Error Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Sigma reduction
    ax = axes[1]
    sigma_reduction = sigma_orig - sigma_cyclic
    ax.hist(sigma_reduction, bins=30, alpha=0.7, color="green")
    ax.axvline(x=0, color="red", linestyle="--", label="No reduction")
    ax.axvline(x=sigma_reduction.mean(), color="black", linestyle="-", label=f"Mean: {sigma_reduction.mean():.2f} lbs")
    ax.set_xlabel("Sigma reduction (lbs)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Sigma Reduction: {sigma_reduction.mean():.2f} lbs ({100*sigma_reduction.mean()/sigma_orig.mean():.1f}%)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Daily amplitude
    ax = axes[2]
    ax.hist(daily_amplitude, bins=30, alpha=0.7, color="purple")
    ax.axvline(x=daily_amplitude.mean(), color="black", linestyle="-", label=f"Mean: {daily_amplitude.mean():.2f} lbs")
    ax.set_xlabel("Daily amplitude (lbs)")
    ax.set_ylabel("Frequency")
    ax.set_title("Daily Variation Amplitude")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved model comparison plot to {output_path}")

    return fig


def plot_spline_daily_pattern(
    idata,
    stan_data,
    output_path: str = None,
    n_hours_grid: int = 100,
):
    """Plot the estimated daily pattern from spline model (Fourier harmonics).

    Args:
        idata: ArviZ InferenceData from spline model
        stan_data: Stan data dictionary with K (number of harmonics) and hour_of_day
        output_path: Path to save plot (optional)
        n_hours_grid: Number of points to evaluate across 0-24 hours
    """
    # Check required parameters
    if "K" not in stan_data:
        raise ValueError("Stan data must include K (number of Fourier harmonics) for spline pattern plot")
    if "hour_of_day" not in stan_data:
        raise ValueError("Stan data must include hour_of_day for spline pattern plot")

    K = stan_data["K"]
    hour_of_day_obs = np.array(stan_data["hour_of_day"])

    # Extract Fourier coefficients from posterior
    if "a_sin" not in idata.posterior or "a_cos" not in idata.posterior:
        raise ValueError("Spline model posterior must include a_sin and a_cos parameters")

    a_sin_samples = idata.posterior["a_sin"].values  # shape: (chain, draw, K)
    a_cos_samples = idata.posterior["a_cos"].values  # shape: (chain, draw, K)

    # Back-transform to original scale if needed
    # Note: Fourier coefficients are already in standardized scale
    # We need y_sd to convert daily component to original units
    y_sd = stan_data["_y_sd"]

    # Reshape for easier processing
    n_chains, n_draws, _ = a_sin_samples.shape
    n_samples = n_chains * n_draws
    a_sin_flat = a_sin_samples.reshape(n_samples, K) * y_sd  # Convert to original units
    a_cos_flat = a_cos_samples.reshape(n_samples, K) * y_sd  # Convert to original units

    # Create grid of hours for visualization
    hours_grid = np.linspace(0, 24, n_hours_grid)
    hours_scaled_grid = hours_grid / 24.0

    # Evaluate Fourier series for each posterior sample
    f_daily_grid = np.zeros((n_samples, n_hours_grid))
    for s in range(n_samples):
        for h_idx in range(n_hours_grid):
            hour_scaled = hours_scaled_grid[h_idx]
            val = 0.0
            for k in range(K):
                freq = 2.0 * np.pi * (k + 1)  # k is 0-indexed in Python, but harmonics start at 1
                val += a_sin_flat[s, k] * np.sin(freq * hour_scaled) + a_cos_flat[s, k] * np.cos(freq * hour_scaled)
            f_daily_grid[s, h_idx] = val

    # Compute statistics across posterior samples
    f_daily_mean = f_daily_grid.mean(axis=0)
    f_daily_grid.std(axis=0)
    f_daily_lower = np.percentile(f_daily_grid, 2.5, axis=0)
    f_daily_upper = np.percentile(f_daily_grid, 97.5, axis=0)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Continuous daily pattern with uncertainty
    ax = axes[0]
    # Plot individual samples (first 50)
    for s in range(min(50, n_samples)):
        ax.plot(hours_grid, f_daily_grid[s, :], alpha=0.05, color="blue", linewidth=0.5)

    # Plot mean and credible interval
    ax.plot(hours_grid, f_daily_mean, "r-", linewidth=2, label="Mean daily pattern")
    ax.fill_between(hours_grid, f_daily_lower, f_daily_upper, alpha=0.3, color="red", label="95% CI")

    ax.set_xlabel("Hour of day (GMT)")
    ax.set_ylabel("Weight deviation (lbs)")
    ax.set_title(f"Spline Daily Pattern (K={K} Fourier harmonics)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 24)

    # Plot 2: Detrended data residuals vs estimated daily pattern
    ax = axes[1]

    # Extract trend component and compute residuals
    if "f_trend" not in idata.posterior:
        raise ValueError("Spline model posterior must include f_trend for residual calculation")

    # Get scaled raw data and trend component
    y_scaled = np.array(stan_data["y"])  # Already centered and scaled, ensure numpy array
    f_trend_samples = idata.posterior["f_trend"].values  # shape: (chain, draw, obs)
    f_trend_mean = f_trend_samples.mean(axis=(0, 1))  # Mean across chains and draws

    # Compute residuals: raw data minus trend component (both in scaled space)
    residuals_scaled = y_scaled - f_trend_mean
    residuals_original = residuals_scaled * y_sd  # Convert to original units (lbs)

    # Plot detrended data residuals
    ax.scatter(
        hour_of_day_obs,
        residuals_original,
        alpha=0.6,
        s=30,
        color="blue",
        label="Detrended data (raw - trend)",
        zorder=10,
    )

    # Overlay the continuous daily pattern for comparison
    ax.plot(
        hours_grid,
        f_daily_mean,
        "r-",
        linewidth=2,
        alpha=0.8,
        label="Estimated daily pattern",
        zorder=5,
    )

    # Add horizontal line at zero for reference
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=1, zorder=1)

    ax.set_xlabel("Hour of day (GMT)")
    ax.set_ylabel("Weight deviation (lbs)")
    ax.set_title("Detrended Data vs. Estimated Daily Pattern")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 24)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved spline daily pattern plot to {output_path}")

    return fig


def plot_spline_weekly_pattern(
    idata,
    stan_data,
    output_path: str = None,
    n_days_grid: int = 100,
):
    """Plot the estimated weekly pattern from spline model with weekly Fourier harmonics.

    Args:
        idata: ArviZ InferenceData from weekly spline model
        stan_data: Stan data dictionary with L (weekly harmonics) and day_of_week
        output_path: Path to save plot (optional)
        n_days_grid: Number of points to evaluate across 0-7 days
    """
    # Check required parameters
    if "L" not in stan_data:
        raise ValueError("Stan data must include L (number of weekly Fourier harmonics) for weekly pattern plot")
    if "day_of_week" not in stan_data:
        raise ValueError("Stan data must include day_of_week for weekly pattern plot")

    L = stan_data["L"]
    day_of_week_obs = np.array(stan_data["day_of_week"])

    # Extract weekly Fourier coefficients from posterior
    if "b_sin" not in idata.posterior or "b_cos" not in idata.posterior:
        raise ValueError("Weekly spline model posterior must include b_sin and b_cos parameters")

    b_sin_samples = idata.posterior["b_sin"].values  # shape: (chain, draw, L)
    b_cos_samples = idata.posterior["b_cos"].values  # shape: (chain, draw, L)

    # Back-transform to original scale if needed
    # Note: Fourier coefficients are already in standardized scale
    # We need y_sd to convert weekly component to original units
    y_sd = stan_data["_y_sd"]

    # Reshape for easier processing
    n_chains, n_draws, _ = b_sin_samples.shape
    n_samples = n_chains * n_draws
    b_sin_flat = b_sin_samples.reshape(n_samples, L) * y_sd  # Convert to original units
    b_cos_flat = b_cos_samples.reshape(n_samples, L) * y_sd  # Convert to original units

    # Create grid of days for visualization (0-7 representing Monday-Sunday)
    days_grid = np.linspace(0, 7, n_days_grid)
    days_scaled_grid = days_grid / 7.0

    # Evaluate Fourier series for each posterior sample
    f_weekly_grid = np.zeros((n_samples, n_days_grid))
    for s in range(n_samples):
        for d_idx in range(n_days_grid):
            day_scaled = days_scaled_grid[d_idx]
            val = 0.0
            for l in range(L):
                freq = 2.0 * np.pi * (l + 1)  # l is 0-indexed in Python, but harmonics start at 1
                val += b_sin_flat[s, l] * np.sin(freq * day_scaled) + b_cos_flat[s, l] * np.cos(freq * day_scaled)
            f_weekly_grid[s, d_idx] = val

    # Compute statistics across posterior samples
    f_weekly_mean = f_weekly_grid.mean(axis=0)
    f_weekly_grid.std(axis=0)
    f_weekly_lower = np.percentile(f_weekly_grid, 2.5, axis=0)
    f_weekly_upper = np.percentile(f_weekly_grid, 97.5, axis=0)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Continuous weekly pattern with uncertainty
    ax = axes[0]
    # Plot individual samples (first 50)
    for s in range(min(50, n_samples)):
        ax.plot(days_grid, f_weekly_grid[s, :], alpha=0.05, color="blue", linewidth=0.5)

    # Plot mean and credible interval
    ax.plot(days_grid, f_weekly_mean, "r-", linewidth=2, label="Mean weekly pattern")
    ax.fill_between(days_grid, f_weekly_lower, f_weekly_upper, alpha=0.3, color="red", label="95% CI")

    ax.set_xlabel("Day of week (0=Monday, 7=Sunday)")
    ax.set_ylabel("Weight deviation (lbs)")
    ax.set_title(f"Spline Weekly Pattern (L={L} Fourier harmonics)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 7)

    # Plot 2: Detrended and dedaily data residuals vs estimated weekly pattern
    ax = axes[1]

    # Extract trend and daily components and compute residuals
    if "f_trend" not in idata.posterior or "f_daily" not in idata.posterior:
        raise ValueError("Weekly spline model posterior must include f_trend and f_daily for residual calculation")

    # Get scaled raw data and trend+daily components
    y_scaled = np.array(stan_data["y"])  # Already centered and scaled, ensure numpy array
    f_trend_samples = idata.posterior["f_trend"].values  # shape: (chain, draw, obs)
    f_daily_samples = idata.posterior["f_daily"].values  # shape: (chain, draw, obs)
    f_trend_mean = f_trend_samples.mean(axis=(0, 1))  # Mean across chains and draws
    f_daily_mean = f_daily_samples.mean(axis=(0, 1))  # Mean across chains and draws

    # Compute residuals: raw data minus trend and daily components (both in scaled space)
    residuals_scaled = y_scaled - f_trend_mean - f_daily_mean
    residuals_original = residuals_scaled * y_sd  # Convert to original units (lbs)

    # Plot detrended+dedaily data residuals
    ax.scatter(
        day_of_week_obs,
        residuals_original,
        alpha=0.6,
        s=30,
        color="blue",
        label="Detrended & dedaily data",
        zorder=10,
    )

    # Overlay the continuous weekly pattern for comparison
    ax.plot(
        days_grid,
        f_weekly_mean,
        "r-",
        linewidth=2,
        alpha=0.8,
        label="Estimated weekly pattern",
        zorder=5,
    )

    # Add horizontal line at zero for reference
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=1, zorder=1)

    ax.set_xlabel("Day of week (0=Monday, 7=Sunday)")
    ax.set_ylabel("Weight deviation (lbs)")
    ax.set_title("Detrended & Dedaily Data vs. Estimated Weekly Pattern")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 7)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved spline weekly pattern plot to {output_path}")

    return fig


def plot_models_comparison_all(
    idata_original,
    idata_cyclic,
    idata_spline=None,
    stan_data=None,
    output_path: str = None,
):
    """Plot comparison between original, cyclic, and spline models.

    Args:
        idata_original: InferenceData from original model
        idata_cyclic: InferenceData from cyclic model
        idata_spline: InferenceData from spline model (optional)
        stan_data: Stan data dictionary for back-transformation
        output_path: Path to save plot (optional)
    """
    if stan_data is None or "_y_sd" not in stan_data:
        raise ValueError("stan_data with _y_sd is required for unit conversion")

    y_sd = stan_data["_y_sd"]

    # Extract sigma estimates (convert to original units)
    sigma_orig = idata_original.posterior["sigma"].values.flatten() * y_sd
    sigma_cyclic = idata_cyclic.posterior["sigma"].values.flatten() * y_sd

    # Extract daily amplitude (convert to original units)
    daily_amplitude_cyclic = idata_cyclic.posterior["daily_amplitude"].values.flatten() * y_sd
    prop_variance_daily_cyclic = idata_cyclic.posterior["prop_variance_daily"].values.flatten()

    # Initialize spline metrics
    sigma_spline = daily_amplitude_spline = prop_variance_daily_spline = None
    if idata_spline is not None:
        sigma_spline = idata_spline.posterior["sigma"].values.flatten() * y_sd
        daily_amplitude_spline = idata_spline.posterior["daily_amplitude"].values.flatten() * y_sd
        prop_variance_daily_spline = idata_spline.posterior["prop_variance_daily"].values.flatten()

    # Determine number of subplots based on available models
    has_spline = idata_spline is not None
    n_plots = 4 if has_spline else 3
    fig_width = 15 if has_spline else 12
    fig, axes = plt.subplots(1, n_plots, figsize=(fig_width, 4))

    # Colors for each model
    colors = ["steelblue", "darkorange", "mediumseagreen"]

    # Plot 1: Sigma comparison
    ax = axes[0]
    ax.hist(sigma_orig, bins=30, alpha=0.6, color=colors[0],
            label=f"Original: {sigma_orig.mean():.2f} ± {sigma_orig.std():.2f} lbs")
    ax.hist(sigma_cyclic, bins=30, alpha=0.6, color=colors[1],
            label=f"Cyclic: {sigma_cyclic.mean():.2f} ± {sigma_cyclic.std():.2f} lbs")
    if has_spline:
        ax.hist(sigma_spline, bins=30, alpha=0.6, color=colors[2],
                label=f"Spline: {sigma_spline.mean():.2f} ± {sigma_spline.std():.2f} lbs")
    ax.set_xlabel("Measurement error (sigma, lbs)")
    ax.set_ylabel("Frequency")
    ax.set_title("Sigma Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Daily amplitude comparison
    ax = axes[1]
    ax.hist(daily_amplitude_cyclic, bins=30, alpha=0.7, color=colors[1],
            label=f"Cyclic: {daily_amplitude_cyclic.mean():.2f} ± {daily_amplitude_cyclic.std():.2f} lbs")
    if has_spline:
        ax.hist(daily_amplitude_spline, bins=30, alpha=0.7, color=colors[2],
                label=f"Spline: {daily_amplitude_spline.mean():.2f} ± {daily_amplitude_spline.std():.2f} lbs")
    ax.set_xlabel("Daily amplitude (lbs)")
    ax.set_ylabel("Frequency")
    ax.set_title("Daily Amplitude")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Proportion of variance from daily component
    ax = axes[2]
    ax.hist(prop_variance_daily_cyclic, bins=30, alpha=0.7, color=colors[1],
            label=f"Cyclic: {prop_variance_daily_cyclic.mean():.3f}")
    if has_spline:
        ax.hist(prop_variance_daily_spline, bins=30, alpha=0.7, color=colors[2],
                label=f"Spline: {prop_variance_daily_spline.mean():.3f}")
    ax.set_xlabel("Proportion of variance")
    ax.set_ylabel("Frequency")
    ax.set_title("Daily Variance Proportion")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Sigma reductions (only if spline available)
    if has_spline:
        ax = axes[3]
        # Calculate reductions
        sigma_reduction_oc = sigma_orig - sigma_cyclic
        sigma_reduction_cs = sigma_cyclic - sigma_spline
        sigma_reduction_os = sigma_orig - sigma_spline

        # Plot reductions
        ax.hist(sigma_reduction_oc, bins=30, alpha=0.5, color=colors[0],
                label=f"Original→Cyclic: {sigma_reduction_oc.mean():.2f} lbs")
        ax.hist(sigma_reduction_cs, bins=30, alpha=0.5, color=colors[1],
                label=f"Cyclic→Spline: {sigma_reduction_cs.mean():.2f} lbs")
        ax.hist(sigma_reduction_os, bins=30, alpha=0.5, color=colors[2],
                label=f"Original→Spline: {sigma_reduction_os.mean():.2f} lbs")

        ax.axvline(x=0, color="red", linestyle="--", alpha=0.7, linewidth=1, label="No reduction")
        ax.set_xlabel("Sigma reduction (lbs)")
        ax.set_ylabel("Frequency")
        ax.set_title("Sigma Reductions")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved all models comparison plot to {output_path}")

    return fig


def plot_model_full_expectation(
    idata,
    df,
    stan_data,
    model_name: str,
    output_path: str = None,
    show_observations: bool = True,
    show_ci: bool = True,
):
    """Plot the full expected prediction (combined predictors) for any model.

    This function works with any model type by detecting which prediction
    variables are available in the InferenceData:
    - Original models: "f" (single GP)
    - Cyclic/spline models: "f_total" (trend + daily)
    - Also checks for "f_trend" and "f_daily" for component plots

    Args:
        idata: ArviZ InferenceData from fitted model
        df: Original DataFrame with timestamps
        stan_data: Stan data dictionary for back-transformation
        model_name: Name of model for plot title
        output_path: Path to save plot (optional)
        show_observations: Whether to show data points
        show_ci: Whether to show 95% credible interval

    Returns:
        matplotlib Figure object
    """
    # Back-transform to original scale
    y_mean = stan_data["_y_mean"]
    y_sd = stan_data["_y_sd"]

    # Determine which prediction variable to use
    pred_var = None
    if "f_total" in idata.posterior:
        pred_var = "f_total"
    elif "f" in idata.posterior:
        pred_var = "f"
    else:
        # Try to find any prediction variable
        possible_vars = ["f_total", "f", "f_trend", "f_daily"]
        for var in possible_vars:
            if var in idata.posterior:
                pred_var = var
                break

    if pred_var is None:
        raise ValueError(f"No prediction variable found in model {model_name}. "
                         f"Available variables: {list(idata.posterior.keys())}")

    # Extract posterior samples for the prediction variable
    f_samples = idata.posterior[pred_var].values

    # Compute mean and credible intervals (back-transformed)
    def compute_stats(samples):
        mean = samples.mean(axis=(0, 1)) * y_sd + y_mean
        lower = np.percentile(samples, 2.5, axis=(0, 1)) * y_sd + y_mean
        upper = np.percentile(samples, 97.5, axis=(0, 1)) * y_sd + y_mean
        return mean, lower, upper

    f_mean, f_lower, f_upper = compute_stats(f_samples)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot observations if requested
    if show_observations:
        ax.scatter(df["timestamp"], df["weight_lbs"], alpha=0.5, s=20,
                  label="Observations", color="gray")

    # Plot mean prediction
    ax.plot(df["timestamp"], f_mean, "b-", linewidth=2,
            label=f"{model_name} prediction")

    # Plot credible interval if requested
    if show_ci:
        ax.fill_between(df["timestamp"], f_lower, f_upper, alpha=0.3,
                       color="blue", label="95% CI")

    # Add horizontal line at mean weight for reference
    ax.axhline(y=y_mean, color="k", linestyle="--", alpha=0.5,
               label=f"Mean weight ({y_mean:.1f} lbs)")

    ax.set_xlabel("Date")
    ax.set_ylabel("Weight (lbs)")
    ax.set_title(f"Full Expectation - {model_name} Model")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved full expectation plot for {model_name} to {output_path}")

    return fig


def plot_model_predictions(
    predictions,
    df,
    stan_data,
    model_name: str,
    output_path: str = None,
    show_observations: bool = True,
    show_ci: bool = True,
    zoom_to = None,
    zoom_reference_date = None,
):
    """Plot model predictions at unobserved time points.

    Args:
        predictions: Dictionary from extract_predictions() with keys
                     t_pred, f_pred_mean, f_pred_lower, f_pred_upper, etc.
        df: Original DataFrame with timestamps (for start date reference)
        stan_data: Stan data dictionary with _t_max for scaling
        model_name: Name of model for plot title
        output_path: Path to save plot (optional)
        show_observations: Whether to show data points
        show_ci: Whether to show 95% credible interval for predictions
        zoom_to: Optional zoom specification. Can be:
                - Tuple of (start_date, end_date) as pandas Timestamps
                - String preset: 'last_week', 'last_month', 'last_year', 'all'
        zoom_reference_date: Reference date for preset zooms (pandas Timestamp).
                            If None, uses latest date in prediction data.

    Returns:
        matplotlib Figure object
    """
    if not predictions:
        raise ValueError("Predictions dictionary is empty")

    # Convert prediction times to dates
    start_timestamp = df["timestamp"].min()
    # t_pred is in original days scale (see extract_predictions line 232)
    t_pred_days = predictions["t_pred"]
    hour_of_day_pred = predictions.get("hour_of_day_pred")
    if hour_of_day_pred is not None and len(hour_of_day_pred) == len(t_pred_days):
        dates_pred = [start_timestamp + pd.Timedelta(days=float(d)) + pd.Timedelta(hours=float(h))
                      for d, h in zip(t_pred_days, hour_of_day_pred)]
    else:
        dates_pred = [start_timestamp + pd.Timedelta(days=float(d)) for d in t_pred_days]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot observations if requested
    if show_observations:
        ax.scatter(df["timestamp"], df["weight_lbs"], alpha=0.5, s=20,
                  label="Observations", color="gray")

    # Plot prediction mean
    ax.plot(dates_pred, predictions["f_pred_mean"], "b-", linewidth=2,
            label=f"{model_name} prediction (unobserved days)")

    # Plot credible interval if requested
    if show_ci:
        ax.fill_between(dates_pred, predictions["f_pred_lower"],
                       predictions["f_pred_upper"], alpha=0.3,
                       color="blue", label="95% CI")

    # Add horizontal line at mean weight for reference
    y_mean = stan_data["_y_mean"]
    ax.axhline(y=y_mean, color="k", linestyle="--", alpha=0.5,
               label=f"Mean weight ({y_mean:.1f} lbs)")

    ax.set_xlabel("Date")
    ax.set_ylabel("Weight (lbs)")
    ax.set_title(f"Model Predictions for Unobserved Days - {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Apply zoom if requested
    if zoom_to is not None:
        if isinstance(zoom_to, tuple) and len(zoom_to) == 2:
            # Tuple of dates
            start_date, end_date = zoom_to
            zoom_to_date_range(ax, start_date, end_date)
        elif isinstance(zoom_to, str):
            # Preset string
            # Convert reference date to numeric if provided
            ref_num = None
            if zoom_reference_date is not None:
                from matplotlib.dates import date2num
                ref_num = date2num(zoom_reference_date)
            zoom_to_preset(ax, zoom_to, reference_date=ref_num)
        else:
            raise ValueError("zoom_to must be a tuple (start_date, end_date) or a preset string")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved prediction plot for {model_name} to {output_path}")

    return fig


def plot_hourly_predictions(
    predictions,
    df,
    stan_data,
    model_name: str,
    output_path: str = None,
    day_index: int = 0,
    show_ci: bool = True,
    show_all_days: bool = False,
):
    """Plot hourly predictions for a specific day or aggregated across days.

    Args:
        predictions: Dictionary from extract_predictions() with keys
                     t_pred, hour_of_day_pred, f_pred_mean, f_pred_lower, f_pred_upper, etc.
        df: Original DataFrame with timestamps (for start date reference)
        stan_data: Stan data dictionary with _t_max for scaling
        model_name: Name of model for plot title
        output_path: Path to save plot (optional)
        day_index: Which day to visualize (0-indexed within prediction grid)
        show_ci: Whether to show 95% credible interval
        show_all_days: If True, show aggregated hourly means across all days
                      If False, show predictions for specific day_index

    Returns:
        matplotlib Figure object
    """
    if not predictions:
        raise ValueError("Predictions dictionary is empty")

    # Extract prediction arrays
    t_pred = predictions["t_pred"]  # days since start
    hour_of_day_pred = predictions["hour_of_day_pred"]
    f_pred_mean = predictions["f_pred_mean"]
    f_pred_lower = predictions.get("f_pred_lower")
    f_pred_upper = predictions.get("f_pred_upper")

    # Create figure
    if show_all_days:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        axes = [ax]

    if show_all_days:
        # Plot 1: Aggregated hourly means across all days
        ax = axes[0]

        # Group predictions by hour of day
        unique_hours = np.unique(hour_of_day_pred)
        hour_means = []
        hour_stds = []
        hour_cis_lower = []
        hour_cis_upper = []

        for hour in unique_hours:
            mask = hour_of_day_pred == hour
            hour_means.append(f_pred_mean[mask].mean())
            hour_stds.append(f_pred_mean[mask].std())

            if f_pred_lower is not None and f_pred_upper is not None:
                hour_cis_lower.append(f_pred_lower[mask].mean())
                hour_cis_upper.append(f_pred_upper[mask].mean())

        # Plot hourly means with error bars
        x_pos = np.arange(len(unique_hours))
        ax.bar(x_pos, hour_means, yerr=hour_stds, capsize=5, alpha=0.7,
               color="steelblue", label="Mean ± SD across days")

        if show_ci and f_pred_lower is not None and f_pred_upper is not None:
            # Add CI as error bars or shaded region
            ci_lower = np.array(hour_means) - np.array(hour_cis_lower)
            ci_upper = np.array(hour_cis_upper) - np.array(hour_means)
            ax.errorbar(x_pos, hour_means, yerr=[ci_lower, ci_upper],
                       fmt='none', ecolor='red', capsize=3, alpha=0.7,
                       label="95% CI (mean)")

        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Predicted weight (lbs)")
        ax.set_title(f"Aggregated Hourly Predictions - {model_name}")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"{int(h):02d}:00" if h.is_integer() else f"{h:02.1f}:00" for h in unique_hours])
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # Plot 2: Hourly predictions for a specific day
        ax = axes[1]

        # Find predictions for the specified day_index
        # Assuming predictions are ordered by day then hour
        unique_days = np.unique(t_pred)
        if day_index >= len(unique_days):
            day_index = len(unique_days) - 1

        target_day = unique_days[day_index]
        day_mask = t_pred == target_day

        if day_mask.sum() == 0:
            raise ValueError(f"No predictions found for day index {day_index}")

        day_hours = hour_of_day_pred[day_mask]
        day_mean = f_pred_mean[day_mask]

        # Sort by hour
        sort_idx = np.argsort(day_hours)
        day_hours_sorted = day_hours[sort_idx]
        day_mean_sorted = day_mean[sort_idx]

        ax.plot(day_hours_sorted, day_mean_sorted, "bo-", linewidth=2,
                markersize=8, label=f"Day {day_index+1} predictions")

        if show_ci and f_pred_lower is not None and f_pred_upper is not None:
            day_lower = f_pred_lower[day_mask][sort_idx]
            day_upper = f_pred_upper[day_mask][sort_idx]
            ax.fill_between(day_hours_sorted, day_lower, day_upper,
                           alpha=0.3, color="blue", label="95% CI")

        # Add horizontal line at mean weight for reference
        y_mean = stan_data["_y_mean"]
        ax.axhline(y=y_mean, color="k", linestyle="--", alpha=0.5,
                   label=f"Mean weight ({y_mean:.1f} lbs)")

        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Predicted weight (lbs)")
        ax.set_title(f"Hourly Predictions for Day {day_index+1} - {model_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-1, 25)  # Slightly beyond 0-24 for visual comfort

    else:
        # Single plot: hourly predictions for specific day
        ax = axes[0]

        # Find predictions for the specified day_index
        unique_days = np.unique(t_pred)
        if day_index >= len(unique_days):
            day_index = len(unique_days) - 1

        target_day = unique_days[day_index]
        day_mask = t_pred == target_day

        if day_mask.sum() == 0:
            raise ValueError(f"No predictions found for day index {day_index}")

        day_hours = hour_of_day_pred[day_mask]
        day_mean = f_pred_mean[day_mask]

        # Sort by hour
        sort_idx = np.argsort(day_hours)
        day_hours_sorted = day_hours[sort_idx]
        day_mean_sorted = day_mean[sort_idx]

        ax.plot(day_hours_sorted, day_mean_sorted, "bo-", linewidth=2,
                markersize=8, label="Predictions at hour intervals")

        if show_ci and f_pred_lower is not None and f_pred_upper is not None:
            day_lower = f_pred_lower[day_mask][sort_idx]
            day_upper = f_pred_upper[day_mask][sort_idx]
            ax.fill_between(day_hours_sorted, day_lower, day_upper,
                           alpha=0.3, color="blue", label="95% CI")

        # Add horizontal line at mean weight for reference
        y_mean = stan_data["_y_mean"]
        ax.axhline(y=y_mean, color="k", linestyle="--", alpha=0.5,
                   label=f"Mean weight ({y_mean:.1f} lbs)")

        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Predicted weight (lbs)")
        ax.set_title(f"Hourly Predictions for Day {day_index+1} - {model_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-1, 25)  # Slightly beyond 0-24 for visual comfort

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved hourly prediction plot for {model_name} to {output_path}")

    return fig


def plot_weekly_zoomed_predictions(
    predictions,
    df,
    stan_data,
    model_name: str,
    output_path: str = None,
    target_date=None,
    week_start_date=None,
    show_ci: bool = True,
    show_observations: bool = True,
):
    """Plot hourly predictions zoomed into a specific week.

    Args:
        predictions: Dictionary from extract_predictions() with keys
                     t_pred, hour_of_day_pred, f_pred_mean, f_pred_lower, f_pred_upper, etc.
        df: Original DataFrame with timestamps (for start date reference)
        stan_data: Stan data dictionary with _t_max for scaling
        model_name: Name of model for plot title
        output_path: Path to save plot (optional)
        target_date: Target date (pandas Timestamp) to center week on.
                     If None, uses the middle of prediction range.
        week_start_date: Explicit start date of week (pandas Timestamp).
                         If provided, overrides target_date.
        show_ci: Whether to show 95% credible interval
        show_observations: Whether to show observed data points in the week

    Returns:
        matplotlib Figure object
    """
    if not predictions:
        raise ValueError("Predictions dictionary is empty")

    # Convert prediction times to dates
    start_timestamp = df["timestamp"].min()
    t_pred_days = predictions["t_pred"]
    hour_of_day_pred = predictions.get("hour_of_day_pred")
    if hour_of_day_pred is not None and len(hour_of_day_pred) == len(t_pred_days):
        dates_pred = [start_timestamp + pd.Timedelta(days=float(d)) + pd.Timedelta(hours=float(h))
                      for d, h in zip(t_pred_days, hour_of_day_pred)]
    else:
        dates_pred = [start_timestamp + pd.Timedelta(days=float(d)) for d in t_pred_days]

    # Create DataFrame for predictions
    pred_df = pd.DataFrame({
        "datetime": dates_pred,  # Full datetime including time component
        "hour": predictions["hour_of_day_pred"],  # Keep hour for reference if needed
        "f_pred_mean": predictions["f_pred_mean"],
    })
    if "f_pred_lower" in predictions and "f_pred_upper" in predictions:
        pred_df["f_pred_lower"] = predictions["f_pred_lower"]
        pred_df["f_pred_upper"] = predictions["f_pred_upper"]

    # Determine week start date
    if week_start_date is not None:
        start_date = week_start_date
    elif target_date is not None:
        # Start of week (Monday)
        start_date = target_date - pd.Timedelta(days=target_date.dayofweek)
    else:
        # Use middle of prediction range
        mid_idx = len(dates_pred) // 2
        mid_date = dates_pred[mid_idx]
        start_date = mid_date - pd.Timedelta(days=mid_date.dayofweek)

    end_date = start_date + pd.Timedelta(days=7)

    # Filter predictions within this week
    week_mask = (pred_df["datetime"] >= start_date) & (pred_df["datetime"] < end_date)
    if not week_mask.any():
        raise ValueError(f"No predictions found in week starting {start_date.date()}")

    week_df = pred_df[week_mask].copy()

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot each day separately with distinct colors
    # Group by date part (ignoring time) for coloring
    week_df["date_only"] = week_df["datetime"].dt.date
    unique_days = week_df["date_only"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_days)))

    for day_color, day_date in zip(colors, unique_days):
        day_mask = week_df["date_only"] == day_date
        day_df = week_df[day_mask].sort_values("datetime")

        # Plot mean prediction
        label = day_date.strftime("%a %Y-%m-%d")
        ax.plot(day_df["datetime"], day_df["f_pred_mean"], "o-",
                color=day_color, linewidth=2, markersize=6, label=label)

        # Plot credible interval
        if show_ci and "f_pred_lower" in day_df.columns and "f_pred_upper" in day_df.columns:
            ax.fill_between(day_df["datetime"], day_df["f_pred_lower"], day_df["f_pred_upper"],
                           alpha=0.3, color=day_color)

    # Add observations if requested
    if show_observations:
        # Filter observations within the week
        obs_mask = (df["timestamp"] >= start_date) & (df["timestamp"] < end_date)
        if obs_mask.any():
            obs_df = df[obs_mask]
            ax.scatter(obs_df["timestamp"], obs_df["weight_lbs"], color="black", s=50, zorder=5,
                      label="Observations", edgecolors="white", linewidth=1)

    # Add horizontal line at mean weight for reference
    y_mean = stan_data["_y_mean"]
    ax.axhline(y=y_mean, color="k", linestyle="--", alpha=0.5,
               label=f"Mean weight ({y_mean:.1f} lbs)")

    ax.set_xlabel("Date and Time")
    ax.set_ylabel("Weight (lbs)")
    ax.set_title(f"Hourly Predictions for Week Starting {start_date.date()} - {model_name}")
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    ax.grid(True, alpha=0.3)
    # Format x-axis to show date and time
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%a\n%H:%M'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    fig.autofmt_xdate()  # Rotate and align date labels

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved weekly zoomed prediction plot for {model_name} to {output_path}")

    return fig


if __name__ == "__main__":
    # Example usage
    print("Cyclic and spline model visualization module")
    print("Usage:")
    print("  from src.models.plot_cyclic import (")
    print("      plot_cyclic_components,")
    print("      plot_daily_pattern,")
    print("      plot_model_comparison,")
    print("      plot_spline_daily_pattern,")
    print("      plot_models_comparison_all,")
    print("      plot_model_full_expectation,")
    print("      plot_model_predictions,")
    print("      plot_weekly_zoomed_predictions")
    print("  )")
    print()
    print("Functions:")
    print("  plot_cyclic_components() - Show trend vs. daily components")
    print("  plot_daily_pattern() - Visualize hour-of-day effects for cyclic model")
    print("  plot_spline_daily_pattern() - Visualize Fourier spline daily pattern")
    print("  plot_model_comparison() - Compare original vs. cyclic models")
    print("  plot_models_comparison_all() - Compare original, cyclic, and spline models")
    print("  plot_model_full_expectation() - Plot full prediction for any model")
    print("  plot_model_predictions() - Plot predictions at unobserved time points")
    print("  plot_weekly_zoomed_predictions() - Plot hourly predictions zoomed into a specific week")