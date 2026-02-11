"""Visualization for cyclic GP model components."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .plot_zoom import zoom_to_date_range, zoom_to_preset


def _compute_stats(samples, y_mean, y_sd):
    """Compute mean and credible intervals for posterior samples.

    Args:
        samples: Array of shape (chain, draw, n_obs) or (chain, draw)
        y_mean: Mean used for standardization (to add back)
        y_sd: Standard deviation used for standardization (to multiply)

    Returns:
        Tuple of (mean, lower, upper) arrays of shape (n_obs,) or scalar
    """
    mean = samples.mean(axis=(0, 1)) * y_sd + y_mean
    lower = np.percentile(samples, 2.5, axis=(0, 1)) * y_sd + y_mean
    upper = np.percentile(samples, 97.5, axis=(0, 1)) * y_sd + y_mean
    return mean, lower, upper


def _select_prediction_var(idata, model_name=None):
    """Select appropriate prediction variable from InferenceData.

    Checks for variables in order of preference:
    1. f_total (combined trend + daily)
    2. f (single GP)
    3. f_trend or f_daily (individual components)

    Args:
        idata: ArviZ InferenceData object
        model_name: Optional model name for error message

    Returns:
        String variable name

    Raises:
        ValueError if no prediction variable found
    """
    if "f_total" in idata.posterior:
        return "f_total"
    elif "f" in idata.posterior:
        return "f"
    else:
        # Try to find any prediction variable
        possible_vars = ["f_total", "f", "f_trend", "f_daily"]
        for var in possible_vars:
            if var in idata.posterior:
                return var
    name_part = f" in model {model_name}" if model_name else ""
    raise ValueError(
        f"No prediction variable found{name_part}. "
        f"Available variables: {list(idata.posterior.keys())}"
    )


def _compute_state_space_weight_samples(idata, stan_data):
    """Compute weight samples for state-space model from components.

    Args:
        idata: ArviZ InferenceData with state-space model parameters
        stan_data: Stan data dictionary with day_idx mapping

    Returns:
        Array of weight samples shape (chain, draw, N_weight)
    """
    gamma_samples = idata.posterior['gamma'].values  # shape (chain, draw)
    fitness_samples = idata.posterior['fitness_stored'].values  # (chain, draw, D)
    f_gp_samples = idata.posterior['f_gp_stored'].values  # (chain, draw, N_weight)

    # Get day indices for each weight observation (1-indexed in Stan)
    day_idx = stan_data['day_idx']  # shape (N_weight,)
    # Convert to 0-indexed for Python
    day_idx_zero = day_idx - 1

    # Compute weight expectation for each sample
    # gamma_samples[:, :, None] * fitness_samples[:, :, day_idx_zero] + f_gp_samples
    gamma_expanded = gamma_samples[:, :, np.newaxis]
    fitness_selected = fitness_samples[:, :, day_idx_zero]  # (chain, draw, N_weight)
    weight_samples = gamma_expanded * fitness_selected + f_gp_samples
    return weight_samples


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
    f_trend_mean, f_trend_lower, f_trend_upper = _compute_stats(f_trend_samples, y_mean, y_sd)
    f_daily_mean, f_daily_lower, f_daily_upper = _compute_stats(f_daily_samples, y_mean, y_sd)
    f_total_mean, f_total_lower, f_total_upper = _compute_stats(f_total_samples, y_mean, y_sd)

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

    # Get scaled weight data - handle different model types
    # Original spline model uses 'y', state-space model uses 'y_weight'
    if "y" in stan_data:
        y_scaled = np.array(stan_data["y"])  # Already centered and scaled
    elif "y_weight" in stan_data:
        y_scaled = np.array(stan_data["y_weight"])  # Already centered and scaled
    else:
        raise ValueError("Stan data must include either 'y' or 'y_weight' for weight data")

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
    # Handle different model types: original spline model has f_trend, state-space model has f_gp_stored
    if "f_trend" in idata.posterior:
        # Original spline model
        f_trend_samples = idata.posterior["f_trend"].values  # shape: (chain, draw, obs)
        f_trend_mean = f_trend_samples.mean(axis=(0, 1))  # Mean across chains and draws
        residuals_scaled = y_scaled - f_trend_mean
    elif "f_gp_stored" in idata.posterior and "fitness_stored" in idata.posterior and "gamma" in idata.posterior:
        # State-space model: need to compute fitness effect + GP trend
        # Get components
        fitness_samples = idata.posterior['fitness_stored'].values  # (chain, draw, D)
        f_gp_samples = idata.posterior['f_gp_stored'].values  # (chain, draw, N_weight)
        gamma_samples = idata.posterior['gamma'].values  # (chain, draw)

        # Map fitness to weight observation times via day_idx
        day_idx = stan_data['day_idx']  # length N_weight, values 1..D
        D = fitness_samples.shape[2]
        N_weight = len(day_idx)

        # Compute fitness effect at weight observation times
        fitness_effect_samples = np.zeros_like(f_gp_samples)
        for chain in range(fitness_samples.shape[0]):
            for draw in range(fitness_samples.shape[1]):
                fitness_effect_samples[chain, draw, :] = gamma_samples[chain, draw] * fitness_samples[chain, draw, day_idx - 1]

        # Total non-daily component = fitness_effect + f_gp
        non_daily_samples = fitness_effect_samples + f_gp_samples
        non_daily_mean = non_daily_samples.mean(axis=(0, 1))

        residuals_scaled = y_scaled - non_daily_mean
    else:
        raise ValueError(
            "Spline model posterior must include either f_trend (original model) or "
            "f_gp_stored + fitness_stored + gamma (state-space model) for residual calculation"
        )

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
    pred_var = _select_prediction_var(idata, model_name=model_name)

    # Extract posterior samples for the prediction variable
    f_samples = idata.posterior[pred_var].values

    # Compute mean and credible intervals (back-transformed)
    f_mean, f_lower, f_upper = _compute_stats(f_samples, y_mean, y_sd)

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

def plot_state_space_expectations(
    idata,
    df_weight,
    df_intensity,
    stan_data,
    model_name: str,
    output_path: str = None,
    show_ci: bool = True,
):
    """Plot state-space model expectations for fitness and weight with data overlays.

    Creates a 2-panel figure:
    1. Latent fitness state over time with workout intensity bars
    2. Weight predictions with weight observations

    Args:
        idata: ArviZ InferenceData from state-space model fit
        df_weight: DataFrame with weight observations (columns: timestamp, weight_lbs)
        df_intensity: DataFrame with daily intensity (columns: date, intensity)
        stan_data: Stan data dictionary with scaling parameters
        model_name: Name of model for plot title
        output_path: Path to save plot (optional)
        show_ci: Whether to show 95% credible intervals

    Returns:
        matplotlib Figure object
    """
    # Back-transform scaling parameters
    y_mean = stan_data.get("_y_mean", 0.0)
    y_sd = stan_data.get("_y_sd", 1.0)

    # Create figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # --- Panel 1: Fitness state ---
    ax_fitness = axes[0]

    # Check for fitness variable names (Stan stores as 'fitness_stored' or 'fitness')
    fitness_var = None
    for var in ['fitness_stored', 'fitness']:
        if var in idata.posterior:
            fitness_var = var
            break

    if fitness_var is None:
        raise ValueError(
            "No fitness variable found in model posterior. "
            f"Available variables: {list(idata.posterior.keys())}"
        )

    # Extract fitness samples (shape: chain, draw, D)
    fitness_samples = idata.posterior[fitness_var].values

    # Compute mean and credible intervals across chains and draws
    fitness_mean = fitness_samples.mean(axis=(0, 1))
    if show_ci:
        fitness_lower = np.percentile(fitness_samples, 2.5, axis=(0, 1))
        fitness_upper = np.percentile(fitness_samples, 97.5, axis=(0, 1))

    # Determine correct date range matching fitness dimension D
    D = fitness_samples.shape[2]

    # Validate D matches Stan data if available
    if 'D' in stan_data:
        if D != stan_data['D']:
            raise ValueError(
                f"Fitness dimension mismatch: fitness_samples.shape[2]={D}, "
                f"stan_data['D']={stan_data['D']}. Ensure consistent data preparation."
            )

    # Compute start date as min of normalized dates (without time components)
    # to match the date range used in prepare_state_space_data
    weight_dates = df_weight['timestamp'].dt.normalize()  # keep only date part
    intensity_dates = df_intensity['date'].dt.normalize()
    start_date = min(weight_dates.min(), intensity_dates.min())
    days = pd.date_range(start=start_date, periods=D, freq='D')

    # Get intensity values for the full date range (unstandardize if needed)
    if 'intensity' in stan_data:
        intensity_mean = stan_data.get('intensity_mean', 0.0)
        intensity_std = stan_data.get('intensity_std', 1.0)
        intensity_values = stan_data['intensity'] * intensity_std + intensity_mean
        # Ensure length matches D
        if len(intensity_values) != D:
            raise ValueError(
                f"Intensity array length mismatch: intensity length={len(intensity_values)}, D={D}. "
                "Check stan_data['intensity'] alignment."
            )
    else:
        # Fallback: map df_intensity to full date range, fill missing with 0
        intensity_df = df_intensity.set_index('date').reindex(days, fill_value=0.0)
        intensity_values = intensity_df['intensity'].values
        # Should already match D due to reindex, but verify
        if len(intensity_values) != D:
            raise ValueError(
                f"Intensity values length mismatch after reindex: {len(intensity_values)} != {D}"
            )

    # Validate fitness_mean length matches D (should already hold)
    if len(fitness_mean) != D:
        raise ValueError(
            f"Fitness mean length mismatch: {len(fitness_mean)} != {D}. "
            "Check fitness_samples shape."
        )

    # Plot fitness mean
    ax_fitness.plot(days, fitness_mean, 'b-', linewidth=2, label='Fitness state (mean)')

    # Plot credible interval
    if show_ci:
        ax_fitness.fill_between(
            days, fitness_lower, fitness_upper,
            alpha=0.3, color='blue', label='95% CI'
        )

    # Plot intensity bars on secondary y-axis
    ax_intensity = ax_fitness.twinx()
    ax_intensity.bar(
        days, intensity_values,
        width=1.0, alpha=0.6, color='orange', edgecolor='orange', linewidth=0.5,
        label='Workout intensity (bars)'
    )
    # Add intensity line trend
    ax_intensity.plot(
        days, intensity_values, 'k-', linewidth=1, alpha=0.7,
        label='Workout intensity (line)'
    )

    # Labels and titles for fitness panel
    ax_fitness.set_xlabel('Date')
    ax_fitness.set_ylabel('Fitness (standardized)', color='blue')
    ax_fitness.tick_params(axis='y', labelcolor='blue')
    ax_fitness.set_title(f'Latent Fitness State - {model_name}')
    ax_fitness.legend(loc='upper left')
    ax_fitness.grid(True, alpha=0.3)

    ax_intensity.set_ylabel('Workout intensity', color='orange')
    ax_intensity.tick_params(axis='y', labelcolor='orange')
    ax_intensity.legend(loc='upper right')

    # --- Panel 2: Weight expectations ---
    ax_weight = axes[1]

    # Check if this is a state-space model
    if 'fitness_stored' in idata.posterior and 'f_gp_stored' in idata.posterior:
        # State-space model: compute weight expectation from components
        weight_samples = _compute_state_space_weight_samples(idata, stan_data)

    else:
        # Standard GP model: use existing prediction variable
        pred_var = _select_prediction_var(idata, model_name=model_name)
        weight_samples = idata.posterior[pred_var].values

    # Compute mean and credible intervals (back-transformed)
    f_mean, f_lower, f_upper = _compute_stats(weight_samples, y_mean, y_sd)

    # Plot weight observations
    ax_weight.scatter(
        df_weight['timestamp'], df_weight['weight_lbs'],
        alpha=0.5, s=20, label='Observations', color='gray'
    )

    # Plot mean prediction
    ax_weight.plot(df_weight['timestamp'], f_mean, 'b-', linewidth=2,
                   label=f'{model_name} prediction')

    # Plot credible interval if requested
    if show_ci:
        ax_weight.fill_between(
            df_weight['timestamp'], f_lower, f_upper,
            alpha=0.3, color='blue', label='95% CI'
        )

    # Add horizontal line at mean weight for reference
    ax_weight.axhline(y=y_mean, color='k', linestyle='--', alpha=0.5,
                      label=f'Mean weight ({y_mean:.1f} lbs)')

    ax_weight.set_xlabel('Date')
    ax_weight.set_ylabel('Weight (lbs)')
    ax_weight.set_title(f'Weight Expectations - {model_name}')
    ax_weight.legend()
    ax_weight.grid(True, alpha=0.3)

    # Adjust layout
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'Saved state-space expectations plot for {model_name} to {output_path}')

    return fig


def plot_state_space_expectations_with_activity_breakdown(
    idata,
    df_weight,
    df_intensity,
    stan_data,
    model_name: str,
    df_intensity_by_activity: pd.DataFrame = None,
    output_path: str = None,
    show_ci: bool = True,
):
    """Plot state-space model expectations with activity type breakdown.

    Creates a 2-panel figure:
    1. Latent fitness state over time with stacked workout intensity bars by activity type
    2. Weight predictions with weight observations

    Args:
        idata: ArviZ InferenceData from state-space model fit
        df_weight: DataFrame with weight observations (columns: timestamp, weight_lbs)
        df_intensity: DataFrame with daily total intensity (columns: date, intensity)
        stan_data: Stan data dictionary with scaling parameters
        model_name: Name of model for plot title
        df_intensity_by_activity: Optional DataFrame with intensity by activity type
            (columns: 'date', plus columns for each activity type)
        output_path: Path to save plot (optional)
        show_ci: Whether to show 95% credible intervals

    Returns:
        matplotlib Figure object
    """
    # Back-transform scaling parameters
    y_mean = stan_data.get("_y_mean", 0.0)
    y_sd = stan_data.get("_y_sd", 1.0)

    # Create figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # --- Panel 1: Fitness state ---
    ax_fitness = axes[0]

    # Check for fitness variable names (Stan stores as 'fitness_stored' or 'fitness')
    fitness_var = None
    for var in ['fitness_stored', 'fitness']:
        if var in idata.posterior:
            fitness_var = var
            break

    if fitness_var is None:
        raise ValueError(
            "No fitness variable found in model posterior. "
            f"Available variables: {list(idata.posterior.keys())}"
        )

    # Extract fitness samples (shape: chain, draw, D)
    fitness_samples = idata.posterior[fitness_var].values

    # Compute mean and credible intervals across chains and draws
    fitness_mean = fitness_samples.mean(axis=(0, 1))
    if show_ci:
        fitness_lower = np.percentile(fitness_samples, 2.5, axis=(0, 1))
        fitness_upper = np.percentile(fitness_samples, 97.5, axis=(0, 1))

    # Determine correct date range matching fitness dimension D
    D = fitness_samples.shape[2]

    # Validate D matches Stan data if available
    if 'D' in stan_data:
        if D != stan_data['D']:
            raise ValueError(
                f"Fitness dimension mismatch: fitness_samples.shape[2]={D}, "
                f"stan_data['D']={stan_data['D']}. Ensure consistent data preparation."
            )

    # Compute start date as min of normalized dates (without time components)
    # to match the date range used in prepare_state_space_data
    weight_dates = df_weight['timestamp'].dt.normalize()  # keep only date part
    intensity_dates = df_intensity['date'].dt.normalize()
    start_date = min(weight_dates.min(), intensity_dates.min())
    days = pd.date_range(start=start_date, periods=D, freq='D')

    # Get intensity values for the full date range (unstandardize if needed)
    if 'intensity' in stan_data:
        intensity_mean = stan_data.get('intensity_mean', 0.0)
        intensity_std = stan_data.get('intensity_std', 1.0)
        intensity_values = stan_data['intensity'] * intensity_std + intensity_mean
        # Ensure length matches D
        if len(intensity_values) != D:
            raise ValueError(
                f"Intensity array length mismatch: intensity length={len(intensity_values)}, D={D}. "
                "Check stan_data['intensity'] alignment."
            )
    else:
        # Fallback: map df_intensity to full date range, fill missing with 0
        intensity_df = df_intensity.set_index('date').reindex(days, fill_value=0.0)
        intensity_values = intensity_df['intensity'].values
        # Should already match D due to reindex, but verify
        if len(intensity_values) != D:
            raise ValueError(
                f"Intensity values length mismatch after reindex: {len(intensity_values)} != {D}"
            )

    # Validate fitness_mean length matches D (should already hold)
    if len(fitness_mean) != D:
        raise ValueError(
            f"Fitness mean length mismatch: {len(fitness_mean)} != {D}. "
            "Check fitness_samples shape."
        )

    # Plot fitness mean
    ax_fitness.plot(days, fitness_mean, 'b-', linewidth=2, label='Fitness state (mean)')

    # Plot credible interval
    if show_ci:
        ax_fitness.fill_between(
            days, fitness_lower, fitness_upper,
            alpha=0.3, color='blue', label='95% CI'
        )

    # Plot intensity bars on secondary y-axis
    ax_intensity = ax_fitness.twinx()

    if df_intensity_by_activity is not None:
        # Map activity intensity to full date range
        activity_df = df_intensity_by_activity.set_index('date').reindex(days, fill_value=0.0)
        # Identify activity columns (excluding 'date' which is now index)
        activity_cols = [col for col in activity_df.columns if col != 'intensity']  # exclude total intensity column if present

        if len(activity_cols) > 0:
            # Plot stacked bars
            bottom = np.zeros(len(days))
            colors = plt.cm.Set3(np.linspace(0, 1, len(activity_cols)))
            for i, activity in enumerate(activity_cols):
                ax_intensity.bar(
                    days, activity_df[activity].values,
                    width=1.0, alpha=0.6, color=colors[i],
                    bottom=bottom, label=activity
                )
                bottom += activity_df[activity].values
        else:
            # Fallback to total intensity
            ax_intensity.bar(
                days, intensity_values,
                width=1.0, alpha=0.3, color='orange', label='Workout intensity'
            )
    else:
        # Plot total intensity bars
        ax_intensity.bar(
            days, intensity_values,
            width=1.0, alpha=0.3, color='orange', label='Workout intensity'
        )

    # Labels and titles for fitness panel
    ax_fitness.set_xlabel('Date')
    ax_fitness.set_ylabel('Fitness (standardized)', color='blue')
    ax_fitness.tick_params(axis='y', labelcolor='blue')
    ax_fitness.set_title(f'Latent Fitness State - {model_name}')
    ax_fitness.legend(loc='upper left')
    ax_fitness.grid(True, alpha=0.3)

    ax_intensity.set_ylabel('Workout intensity', color='orange')
    ax_intensity.tick_params(axis='y', labelcolor='orange')
    ax_intensity.legend(loc='upper right')

    # --- Panel 2: Weight expectations ---
    ax_weight = axes[1]

    # Check if this is a state-space model
    if 'fitness_stored' in idata.posterior and 'f_gp_stored' in idata.posterior:
        # State-space model: compute weight expectation from components
        weight_samples = _compute_state_space_weight_samples(idata, stan_data)

    else:
        # Standard GP model: use existing prediction variable
        pred_var = _select_prediction_var(idata, model_name=model_name)
        weight_samples = idata.posterior[pred_var].values

    # Compute mean and credible intervals (back-transformed)
    f_mean, f_lower, f_upper = _compute_stats(weight_samples, y_mean, y_sd)

    # Plot weight observations
    ax_weight.scatter(
        df_weight['timestamp'], df_weight['weight_lbs'],
        alpha=0.5, s=20, label='Observations', color='gray'
    )

    # Plot mean prediction
    ax_weight.plot(df_weight['timestamp'], f_mean, 'b-', linewidth=2,
                   label=f'{model_name} prediction')

    # Plot credible interval if requested
    if show_ci:
        ax_weight.fill_between(
            df_weight['timestamp'], f_lower, f_upper,
            alpha=0.3, color='blue', label='95% CI'
        )

    # Add horizontal line at mean weight for reference
    ax_weight.axhline(y=y_mean, color='k', linestyle='--', alpha=0.5,
                      label=f'Mean weight ({y_mean:.1f} lbs)')

    ax_weight.set_xlabel('Date')
    ax_weight.set_ylabel('Weight (lbs)')
    ax_weight.set_title(f'Weight Expectations - {model_name}')
    ax_weight.legend()
    ax_weight.grid(True, alpha=0.3)

    # Adjust layout
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'Saved state-space expectations plot with activity breakdown for {model_name} to {output_path}')

    return fig


def plot_state_space_spline_decomposition(
    idata,
    df_weight,
    df_intensity,
    stan_data,
    model_name: str,
    output_path: str = None,
    show_ci: bool = True,
):
    """Plot comprehensive decomposition of state-space model with daily spline.

    Creates a multi-panel figure showing:
    1. Weight decomposition into components (fitness effect, GP trend, daily spline, residual)
    2. Within-day pattern (hour-of-day effect from Fourier spline)
    3. Latent fitness and impulse states over time
    4. Workout intensity bars

    Args:
        idata: ArviZ InferenceData from state-space spline model fit
        df_weight: DataFrame with weight observations (columns: timestamp, weight_lbs)
        df_intensity: DataFrame with daily intensity (columns: date, intensity)
        stan_data: Stan data dictionary with scaling parameters
        model_name: Name of model for plot title
        output_path: Path to save plot (optional)
        show_ci: Whether to show 95% credible intervals

    Returns:
        matplotlib Figure object
    """
    # Back-transform scaling parameters
    y_mean = stan_data.get("_y_mean", 0.0)
    y_sd = stan_data.get("_y_sd", 1.0)
    intensity_mean = stan_data.get("intensity_mean", 0.0)
    intensity_std = stan_data.get("intensity_std", 1.0)

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax_weight_decomp = axes[0, 0]
    ax_daily_pattern = axes[0, 1]
    ax_fitness = axes[1, 0]
    ax_impulse = axes[1, 1]

    # --- Panel 1: Weight decomposition ---
    # Extract components
    if 'fitness_stored' not in idata.posterior:
        raise ValueError("Missing fitness_stored in posterior")
    if 'f_gp_stored' not in idata.posterior:
        raise ValueError("Missing f_gp_stored in posterior")
    if 'f_daily_stored' not in idata.posterior:
        raise ValueError("Missing f_daily_stored in posterior - model may not include daily spline")

    # Get gamma parameter (weight effect per fitness)
    if 'gamma' not in idata.posterior:
        raise ValueError("Missing gamma parameter in posterior")

    gamma_samples = idata.posterior['gamma'].values  # shape (chain, draw)
    gamma_mean = gamma_samples.mean()

    # Extract component samples
    fitness_samples = idata.posterior['fitness_stored'].values  # (chain, draw, D)
    f_gp_samples = idata.posterior['f_gp_stored'].values  # (chain, draw, N_weight)
    f_daily_samples = idata.posterior['f_daily_stored'].values  # (chain, draw, N_weight)

    # Map fitness to weight observation times via day_idx
    day_idx = stan_data['day_idx']  # length N_weight, values 1..D
    D = fitness_samples.shape[2]
    N_weight = len(day_idx)

    # Validate dimensions
    if fitness_samples.shape[2] != D:
        raise ValueError(f"fitness_samples.shape[2] ({fitness_samples.shape[2]}) != D ({D})")
    if f_gp_samples.shape[2] != N_weight:
        raise ValueError(f"f_gp_samples.shape[2] ({f_gp_samples.shape[2]}) != N_weight ({N_weight})")
    if f_daily_samples.shape[2] != N_weight:
        raise ValueError(f"f_daily_samples.shape[2] ({f_daily_samples.shape[2]}) != N_weight ({N_weight})")
    if day_idx.max() > D:
        raise ValueError(f"day_idx max ({day_idx.max()}) > D ({D})")
    if day_idx.min() < 1:
        raise ValueError(f"day_idx min ({day_idx.min()}) < 1")

    # Compute fitness effect at weight observation times
    # fitness_effect = gamma * fitness[day_idx]
    fitness_effect_samples = np.zeros_like(f_gp_samples)
    for chain in range(fitness_samples.shape[0]):
        for draw in range(fitness_samples.shape[1]):
            fitness_effect_samples[chain, draw, :] = gamma_samples[chain, draw] * fitness_samples[chain, draw, day_idx - 1]

    # Total prediction = fitness_effect + f_gp + f_daily
    total_samples = fitness_effect_samples + f_gp_samples + f_daily_samples

    # Compute means and credible intervals for each component (back-transformed)
    def back_transform(samples):
        mean = samples.mean(axis=(0, 1)) * y_sd + y_mean
        lower = np.percentile(samples, 2.5, axis=(0, 1)) * y_sd + y_mean
        upper = np.percentile(samples, 97.5, axis=(0, 1)) * y_sd + y_mean
        return mean, lower, upper

    total_mean, total_lower, total_upper = back_transform(total_samples)
    fitness_mean_bt, fitness_lower, fitness_upper = back_transform(fitness_effect_samples)
    gp_mean_bt, gp_lower, gp_upper = back_transform(f_gp_samples)
    daily_mean_bt, daily_lower, daily_upper = back_transform(f_daily_samples)

    # Get observation residuals (observed - predicted)
    obs_weight = df_weight['weight_lbs'].values
    residuals = obs_weight - total_mean

    # Plot stacked area or line decomposition
    timestamps = df_weight['timestamp']
    ax_weight_decomp.plot(timestamps, total_mean, 'k-', linewidth=2, label='Total prediction')
    if show_ci:
        ax_weight_decomp.fill_between(timestamps, total_lower, total_upper, alpha=0.2, color='gray', label='95% CI total')

    # Plot components as stacked or overlaid
    ax_weight_decomp.plot(timestamps, fitness_mean_bt, 'r-', linewidth=1.5, alpha=0.7, label='Fitness effect')
    ax_weight_decomp.plot(timestamps, gp_mean_bt, 'g-', linewidth=1.5, alpha=0.7, label='GP trend')
    ax_weight_decomp.plot(timestamps, daily_mean_bt, 'b-', linewidth=1.5, alpha=0.7, label='Daily spline')

    # Plot observations
    ax_weight_decomp.scatter(timestamps, obs_weight, alpha=0.3, s=10, color='gray', label='Observations')

    # Plot residuals
    ax_weight_decomp.scatter(timestamps, obs_weight - residuals, alpha=0.5, s=5, color='purple', label='Residuals (offset)')

    ax_weight_decomp.set_xlabel('Date')
    ax_weight_decomp.set_ylabel('Weight (lbs)')
    ax_weight_decomp.set_title(f'Weight Decomposition - {model_name}')
    ax_weight_decomp.legend(loc='upper left', fontsize='small')
    ax_weight_decomp.grid(True, alpha=0.3)

    # --- Panel 2: Within-day pattern ---
    # Create hour-of-day grid for visualization
    hour_grid = np.linspace(0, 24, 100)
    # Extract Fourier coefficients
    K = stan_data.get('K', 2)
    a_sin_samples = idata.posterior.get('a_sin', None)
    a_cos_samples = idata.posterior.get('a_cos', None)

    if a_sin_samples is not None and a_cos_samples is not None:
        # Convert to numpy arrays for easier indexing
        a_sin_values = a_sin_samples.values
        a_cos_values = a_cos_samples.values

        # Validate Fourier coefficient dimensions
        if a_sin_values.shape[2] != K:
            raise ValueError(
                f"a_sin dimension mismatch: a_sin_values.shape[2]={a_sin_values.shape[2]}, K={K}. "
                "Ensure stan_data K matches model output."
            )
        if a_cos_values.shape[2] != K:
            raise ValueError(
                f"a_cos dimension mismatch: a_cos_values.shape[2]={a_cos_values.shape[2]}, K={K}. "
                "Ensure stan_data K matches model output."
            )

        # Compute daily pattern samples
        daily_pattern_samples = []
        for chain in range(a_sin_values.shape[0]):
            for draw in range(a_sin_values.shape[1]):
                pattern = np.zeros(len(hour_grid))
                for k in range(K):
                    freq = 2.0 * np.pi * (k + 1)
                    hour_scaled = hour_grid / 24.0
                    pattern += (a_sin_values[chain, draw, k] * np.sin(freq * hour_scaled) +
                                a_cos_values[chain, draw, k] * np.cos(freq * hour_scaled))
                daily_pattern_samples.append(pattern)

        daily_pattern_samples = np.array(daily_pattern_samples)  # (n_samples, 100)
        daily_pattern_mean = daily_pattern_samples.mean(axis=0) * y_sd  # back-transform to lbs
        daily_pattern_lower = np.percentile(daily_pattern_samples, 2.5, axis=0) * y_sd
        daily_pattern_upper = np.percentile(daily_pattern_samples, 97.5, axis=0) * y_sd

        ax_daily_pattern.plot(hour_grid, daily_pattern_mean, 'b-', linewidth=2, label='Daily pattern')
        if show_ci:
            ax_daily_pattern.fill_between(hour_grid, daily_pattern_lower, daily_pattern_upper,
                                         alpha=0.3, color='blue', label='95% CI')

        # Add actual daily component values from observations
        hour_of_day = stan_data.get('hour_of_day', None)
        if hour_of_day is not None:
            # Validate dimension match
            if len(hour_of_day) != len(daily_mean_bt):
                raise ValueError(
                    f"hour_of_day length ({len(hour_of_day)}) does not match "
                    f"daily_mean_bt length ({len(daily_mean_bt)})"
                )
            # Get unique hour values (bin)
            ax_daily_pattern.scatter(hour_of_day, daily_mean_bt, alpha=0.3, s=10, color='red',
                                   label='Observed daily component')

        ax_daily_pattern.set_xlabel('Hour of day')
        ax_daily_pattern.set_ylabel('Weight effect (lbs)')
        ax_daily_pattern.set_title('Within-Day Pattern (Fourier Spline)')
        ax_daily_pattern.legend()
        ax_daily_pattern.grid(True, alpha=0.3)
        ax_daily_pattern.set_xlim(0, 24)
    else:
        ax_daily_pattern.text(0.5, 0.5, 'Fourier coefficients not available',
                            ha='center', va='center', transform=ax_daily_pattern.transAxes)
        ax_daily_pattern.set_title('Within-Day Pattern (data not available)')

    # --- Panel 3: Fitness state ---
    # Compute fitness mean and CI
    fitness_mean = fitness_samples.mean(axis=(0, 1))
    if show_ci:
        fitness_lower = np.percentile(fitness_samples, 2.5, axis=(0, 1))
        fitness_upper = np.percentile(fitness_samples, 97.5, axis=(0, 1))

    # Create date range for fitness states
    weight_dates = df_weight['timestamp'].dt.normalize()
    intensity_dates = df_intensity['date'].dt.normalize()
    start_date = min(weight_dates.min(), intensity_dates.min())
    days = pd.date_range(start=start_date, periods=D, freq='D')

    # Get intensity values
    if 'intensity' in stan_data:
        intensity_values = stan_data['intensity'] * intensity_std + intensity_mean
        if len(intensity_values) != D:
            # Ensure exact length D by truncating or padding with zeros
            if len(intensity_values) > D:
                intensity_values = intensity_values[:D]
                print(f"WARNING: intensity array truncated from {len(intensity_values)} to {D}")
            else:
                # Pad with zeros at the end
                pad_length = D - len(intensity_values)
                intensity_values = np.pad(intensity_values, (0, pad_length), mode='constant', constant_values=0.0)
                print(f"WARNING: intensity array padded with zeros from {len(intensity_values)-pad_length} to {D}")
    else:
        intensity_df = df_intensity.set_index('date').reindex(days, fill_value=0.0)
        intensity_values = intensity_df['intensity'].values

    # Plot fitness
    ax_fitness.plot(days, fitness_mean, 'b-', linewidth=2, label='Fitness state')
    if show_ci:
        ax_fitness.fill_between(days, fitness_lower, fitness_upper, alpha=0.3, color='blue', label='95% CI')

    # Add intensity bars
    ax_intensity_twin = ax_fitness.twinx()
    ax_intensity_twin.bar(days, intensity_values, width=1.0, alpha=0.3, color='orange', label='Workout intensity')

    ax_fitness.set_xlabel('Date')
    ax_fitness.set_ylabel('Fitness (standardized)', color='blue')
    ax_fitness.tick_params(axis='y', labelcolor='blue')
    ax_fitness.set_title('Latent Fitness State')
    ax_fitness.legend(loc='upper left')
    ax_fitness.grid(True, alpha=0.3)

    ax_intensity_twin.set_ylabel('Workout intensity', color='orange')
    ax_intensity_twin.tick_params(axis='y', labelcolor='orange')
    ax_intensity_twin.legend(loc='upper right')

    # --- Panel 4: Impulse state ---
    if 'impulse_stored' in idata.posterior:
        impulse_samples = idata.posterior['impulse_stored'].values
        impulse_mean = impulse_samples.mean(axis=(0, 1))
        if show_ci:
            impulse_lower = np.percentile(impulse_samples, 2.5, axis=(0, 1))
            impulse_upper = np.percentile(impulse_samples, 97.5, axis=(0, 1))

        ax_impulse.plot(days, impulse_mean, 'r-', linewidth=2, label='Impulse state')
        if show_ci:
            ax_impulse.fill_between(days, impulse_lower, impulse_upper, alpha=0.3, color='red', label='95% CI')

        # Add intensity bars (same as above)
        ax_impulse_twin = ax_impulse.twinx()
        ax_impulse_twin.bar(days, intensity_values, width=1.0, alpha=0.3, color='orange', label='Workout intensity')

        ax_impulse.set_xlabel('Date')
        ax_impulse.set_ylabel('Impulse (standardized)', color='red')
        ax_impulse.tick_params(axis='y', labelcolor='red')
        ax_impulse.set_title('Impulse State (Workout Accumulation)')
        ax_impulse.legend(loc='upper left')
        ax_impulse.grid(True, alpha=0.3)

        ax_impulse_twin.set_ylabel('Workout intensity', color='orange')
        ax_impulse_twin.tick_params(axis='y', labelcolor='orange')
        ax_impulse_twin.legend(loc='upper right')
    else:
        ax_impulse.text(0.5, 0.5, 'Impulse state not available',
                       ha='center', va='center', transform=ax_impulse.transAxes)
        ax_impulse.set_title('Impulse State (not available)')

    # Add overall title
    fig.suptitle(f'State-Space Model with Daily Spline: {model_name}', fontsize=16, y=1.02)

    # Adjust layout
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'Saved state-space spline decomposition plot to {output_path}')

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
    print("      plot_weekly_zoomed_predictions,")
    print("      plot_state_space_expectations")
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
    print("  plot_state_space_expectations() - Plot fitness and weight expectations with data overlays")