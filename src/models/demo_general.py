#!/usr/bin/env python3
"""General demo script for spline models with daily and/or weekly cycles.

This script lets you specify which model to use:
- 'daily': Optimized spline GP with daily Fourier cycles (trend + daily)
- 'weekly': Weekly spline GP with daily + weekly Fourier cycles (trend + daily + weekly)
- 'both': Run both models and compare them using WAIC/LOO and other metrics

Supports all standard options: sparse GP, prediction grids, zoomed plots, etc.

Usage:
    python -m src.models.demo_general --model daily [--output-dir output/daily] [--chains 4] ...
    python -m src.models.demo_general --model weekly [--weekly-harmonics 1] ...
    python -m src.models.demo_general --model both [--output-dir output/comparison] ...

Example:
    # Daily-only model with sparse GP and prediction grid
    python -m src.models.demo_general --model daily \
        --output-dir output/daily-demo \
        --chains 4 --iter-warmup 500 --iter-sampling 1500 \
        --fourier-harmonics 2 \
        --use-sparse --n-inducing-points 50 \
        --include-prediction-grid --prediction-hour 8.0 --prediction-step-days 1

    # Weekly model with both daily and weekly cycles
    python -m src.models.demo_general --model weekly \
        --output-dir output/weekly-demo \
        --chains 4 --iter-warmup 500 --iter-sampling 1500 \
        --fourier-harmonics 2 --weekly-harmonics 1 \
        --use-sparse --n-inducing-points 50 \
        --include-prediction-grid --prediction-hour 8.0 --prediction-step-days 1

    # Compare both models
    python -m src.models.demo_general --model both \
        --output-dir output/model-comparison \
        --chains 4 --iter-warmup 500 --iter-sampling 1500 \
        --fourier-harmonics 2 --weekly-harmonics 1 \
        --use-sparse --n-inducing-points 50 \
        --include-prediction-grid --prediction-hour 8.0 --prediction-step-days 1
"""

import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import arviz as az

from src.models.fit_weight import (
    fit_weight_model_spline_optimized,
    fit_weight_model_spline_weekly,
    extract_predictions,
)
from src.models.plot_cyclic import (
    plot_cyclic_components,
    plot_spline_daily_pattern,
    plot_model_full_expectation,
    plot_model_predictions,
    plot_hourly_predictions,
    plot_weekly_zoomed_predictions,
)


def should_show_plots() -> bool:
    """Return True if plots should be displayed interactively.

    Modified to always return False - plots are saved to disk only.
    """
    return False


def extract_all_parameters(idata, stan_data, model_name: str) -> Dict[str, Any]:
    """Extract all model parameters with summary statistics.

    Args:
        idata: ArviZ InferenceData object
        stan_data: Stan data dictionary
        model_name: 'daily' or 'weekly'

    Returns:
        Dictionary with parameter summaries organized by category
    """
    import numpy as np

    y_sd = stan_data["_y_sd"]
    stan_data["_y_mean"]

    def get_param_stats(param_name, scale_factor=1.0, transform_back=True):
        """Helper to extract parameter statistics."""
        if param_name not in idata.posterior:
            return None

        samples = idata.posterior[param_name].values
        # Reshape to (samples, ...)
        n_chains, n_draws = samples.shape[:2]
        samples_flat = samples.reshape(n_chains * n_draws, *samples.shape[2:])

        # Apply scaling if needed
        if transform_back and scale_factor != 1.0:
            samples_flat = samples_flat * scale_factor

        # Compute statistics
        if samples_flat.ndim == 1:
            # Scalar parameter
            mean = float(np.mean(samples_flat))
            sd = float(np.std(samples_flat))
            q2_5 = float(np.percentile(samples_flat, 2.5))
            q50 = float(np.percentile(samples_flat, 50))
            q97_5 = float(np.percentile(samples_flat, 97.5))
            return {
                "mean": mean,
                "sd": sd,
                "2.5%": q2_5,
                "50%": q50,
                "97.5%": q97_5
            }
        else:
            # Vector parameter - return summary statistics
            mean = float(np.mean(samples_flat))
            sd = float(np.std(samples_flat))
            return {
                "mean": mean,
                "sd": sd,
                "shape": samples_flat.shape
            }

    # Extract parameters based on model type
    params = {}

    # Common parameters for both models
    common_params = {
        "alpha_trend": get_param_stats("alpha_trend", scale_factor=y_sd),
        "rho_trend": get_param_stats("rho_trend"),
        "sigma": get_param_stats("sigma", scale_factor=y_sd),
        "nu": get_param_stats("nu"),
        "trend_change": get_param_stats("trend_change", scale_factor=y_sd),
        "daily_amplitude": get_param_stats("daily_amplitude", scale_factor=y_sd),
        "prop_variance_daily": get_param_stats("prop_variance_daily"),
    }

    # Raw Fourier coefficients (daily)
    if "a_sin_raw" in idata.posterior:
        a_sin_raw_stats = get_param_stats("a_sin_raw", scale_factor=y_sd)
        a_cos_raw_stats = get_param_stats("a_cos_raw", scale_factor=y_sd)
        common_params["a_sin_raw"] = a_sin_raw_stats
        common_params["a_cos_raw"] = a_cos_raw_stats

    # Transformed Fourier coefficients (daily)
    if "a_sin" in idata.posterior:
        a_sin_stats = get_param_stats("a_sin", scale_factor=y_sd)
        a_cos_stats = get_param_stats("a_cos", scale_factor=y_sd)
        common_params["a_sin"] = a_sin_stats
        common_params["a_cos"] = a_cos_stats

    params.update(common_params)

    # Model-specific parameters
    if model_name == "daily":
        params["sigma_fourier"] = get_param_stats("sigma_fourier", scale_factor=y_sd)

    elif model_name == "weekly":
        params["sigma_fourier_daily"] = get_param_stats("sigma_fourier_daily", scale_factor=y_sd)
        params["sigma_fourier_weekly"] = get_param_stats("sigma_fourier_weekly", scale_factor=y_sd)
        params["weekly_amplitude"] = get_param_stats("weekly_amplitude", scale_factor=y_sd)
        params["prop_variance_weekly"] = get_param_stats("prop_variance_weekly")

        # Raw weekly Fourier coefficients
        if "b_sin_raw" in idata.posterior:
            params["b_sin_raw"] = get_param_stats("b_sin_raw", scale_factor=y_sd)
            params["b_cos_raw"] = get_param_stats("b_cos_raw", scale_factor=y_sd)

        # Transformed weekly Fourier coefficients
        if "b_sin" in idata.posterior:
            params["b_sin"] = get_param_stats("b_sin", scale_factor=y_sd)
            params["b_cos"] = get_param_stats("b_cos", scale_factor=y_sd)

    # Raw GP parameters (eta_trend, eta_inducing)
    if "eta_trend" in idata.posterior:
        params["eta_trend"] = get_param_stats("eta_trend")

    if "eta_inducing" in idata.posterior and stan_data.get("use_sparse", 0) == 1:
        params["eta_inducing"] = get_param_stats("eta_inducing")

    return params


def write_model_report(
    output_dir: Path,
    model_name: str,
    idata,
    df,
    stan_data,
    sigma: float,
    daily_amplitude: float,
    prop_variance_daily: float,
    weekly_amplitude: Optional[float] = None,
    prop_variance_weekly: Optional[float] = None,
    adapt_delta: float = 0.99,
    max_treedepth: int = 12,
    fourier_harmonics: int = 2,
    weekly_harmonics: Optional[int] = None,
    use_sparse: bool = False,
    n_inducing_points: int = 50,
    chains: int = 4,
    iter_warmup: int = 500,
    iter_sampling: int = 1500,
    skip_plots: bool = False,
    include_prediction_grid: bool = False,
    prediction_hour: float = 8.0,
    prediction_hour_step: Optional[float] = None,
    prediction_step_days: int = 1,
    predictions: Optional[Dict] = None,
):
    """Write comprehensive markdown report for a single model.

    Args:
        output_dir: Directory containing generated images and report
        model_name: Name of the model ('daily', 'weekly')
        idata: ArviZ InferenceData object
        df: Original DataFrame
        stan_data: Stan data dictionary
        sigma: Measurement error (lbs)
        daily_amplitude: Daily amplitude (lbs)
        prop_variance_daily: Proportion of variance from daily component
        weekly_amplitude: Weekly amplitude (lbs), only for weekly model
        prop_variance_weekly: Proportion of variance from weekly component, only for weekly model
        adapt_delta: Adapt delta parameter used
        max_treedepth: Maximum tree depth used
        fourier_harmonics: Number of Fourier harmonics (K)
        weekly_harmonics: Number of weekly Fourier harmonics (L), only for weekly model
        use_sparse: Whether sparse GP was used
        n_inducing_points: Number of inducing points (if sparse)
        chains: Number of MCMC chains
        iter_warmup: Warmup iterations per chain
        iter_sampling: Sampling iterations per chain
        skip_plots: Whether to skip visualization section in report
        include_prediction_grid: Whether prediction grid was included
        prediction_hour: Hour of day for predictions
        prediction_hour_step: Step size in hours for multiple predictions per day
        prediction_step_days: Step size in days for prediction grid
        predictions: Dictionary of prediction results from extract_predictions
    """
    report_path = output_dir / f"{model_name}_report.md"

    # Extract all parameters
    all_params = extract_all_parameters(idata, stan_data, model_name)

    # Data statistics
    n_obs = stan_data["N"]
    date_range = f"{df['date'].min().date()} to {df['date'].max().date()}"
    hour_range = f"{df['timestamp'].dt.hour.min():02d}:00 - {df['timestamp'].dt.hour.max():02d}:00 GMT"
    days_with_multiple = (df.groupby('date').size() > 1).sum()
    total_days = df['date'].nunique()

    # Determine model description
    if model_name == "daily":
        model_desc = "Optimized spline GP (trend + Fourier spline for daily cycles)"
    else:  # weekly
        model_desc = "Weekly spline GP (trend + daily + weekly Fourier spline)"

    with open(report_path, 'w') as f:
        f.write(f"""# {model_desc.upper()} Model Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Model**: {model_desc}

## 1. Data Summary

- **Observations**: {n_obs} weight measurements
- **Date range**: {date_range}
- **Hour range**: {hour_range}
- **Days with multiple measurements**: {days_with_multiple}/{total_days} ({days_with_multiple/total_days:.1%})
- **Data sparsity note**: {'⚠ Sparse intraday data: daily components may capture residual variation rather than true daily cycles' if days_with_multiple/total_days < 0.1 else 'Sufficient intraday data for reliable daily cycle estimation'}

## 2. Model Specification

### 2.1 Model Components

This model decomposes weight measurements into:

1. **Long-term trend**: Gaussian Process with squared exponential kernel
   - Marginal standard deviation: `alpha_trend` ~ normal(0, 0.5)
   - Length scale: `rho_trend` ~ inv_gamma(5, 1)
   - Implementation: Stan's built-in `gp_exp_quad_cov` function

2. **Daily cycles**: Fourier basis expansion for hour-of-day effects
   - Fourier harmonics: K = {fourier_harmonics} (sin/cos pairs)
   - Coefficient scale: `sigma_fourier`{'`_daily`' if model_name == 'weekly' else ''} ~ exponential(2)
   - Non-centered parameterization: `a_sin_raw`, `a_cos_raw` ~ std_normal()

3. **Weekly cycles**{' (weekly model only)' if model_name == 'weekly' else ''}:
   - Fourier harmonics: L = {weekly_harmonics if model_name == 'weekly' else 'N/A'}
   {'   - Coefficient scale: `sigma_fourier_weekly` ~ exponential(2)' if model_name == 'weekly' else ''}
   {'   - Non-centered parameterization: `b_sin_raw`, `b_cos_raw` ~ std_normal()' if model_name == 'weekly' else ''}

4. **Observation noise**: Student-t distribution for robustness
   - Scale: `sigma` ~ exponential(3)
   - Degrees of freedom: `nu` ~ gamma(2, 0.1)

### 2.2 Computational Configuration

- **Sparse GP**: {'Enabled' if use_sparse else 'Disabled'} {f'({n_inducing_points} inducing points)' if use_sparse else ''}
- **Sampler parameters**:
  - Chains: {chains}
  - Warmup iterations: {iter_warmup}
  - Sampling iterations: {iter_sampling}
  - Adapt delta: {adapt_delta}
  - Max tree depth: {max_treedepth}

## 3. Parameter Estimates

### 3.1 Primary Parameters of Interest

| Parameter | Mean | SD | 2.5% | 50% | 97.5% | Interpretation |
|-----------|------|----|------|-----|-------|----------------|
| Measurement error (σ) | {sigma:.2f} lbs | {all_params.get('sigma', {}).get('sd', 0):.2f} | {all_params.get('sigma', {}).get('2.5%', 0):.2f} | {all_params.get('sigma', {}).get('50%', 0):.2f} | {all_params.get('sigma', {}).get('97.5%', 0):.2f} | Scale of random measurement noise |
| Daily amplitude | {daily_amplitude:.2f} lbs | {all_params.get('daily_amplitude', {}).get('sd', 0):.2f} | {all_params.get('daily_amplitude', {}).get('2.5%', 0):.2f} | {all_params.get('daily_amplitude', {}).get('50%', 0):.2f} | {all_params.get('daily_amplitude', {}).get('97.5%', 0):.2f} | Peak‑to‑peak variation within 24h |
| Prop. variance (daily) | {prop_variance_daily:.3f} | {all_params.get('prop_variance_daily', {}).get('sd', 0):.3f} | {all_params.get('prop_variance_daily', {}).get('2.5%', 0):.3f} | {all_params.get('prop_variance_daily', {}).get('50%', 0):.3f} | {all_params.get('prop_variance_daily', {}).get('97.5%', 0):.3f} | Fraction of total variance from daily cycle |
""")

        if weekly_amplitude is not None:
            f.write(f"""| Weekly amplitude | {weekly_amplitude:.2f} lbs | {all_params.get('weekly_amplitude', {}).get('sd', 0):.2f} | {all_params.get('weekly_amplitude', {}).get('2.5%', 0):.2f} | {all_params.get('weekly_amplitude', {}).get('50%', 0):.2f} | {all_params.get('weekly_amplitude', {}).get('97.5%', 0):.2f} | Peak‑to‑peak variation within 7 days |
| Prop. variance (weekly) | {prop_variance_weekly:.3f} | {all_params.get('prop_variance_weekly', {}).get('sd', 0):.3f} | {all_params.get('prop_variance_weekly', {}).get('2.5%', 0):.3f} | {all_params.get('prop_variance_weekly', {}).get('50%', 0):.3f} | {all_params.get('prop_variance_weekly', {}).get('97.5%', 0):.3f} | Fraction of total variance from weekly cycle |
""")

        f.write(f"""
### 3.2 Model Hyperparameters

| Parameter | Mean | SD | 2.5% | 50% | 97.5% | Interpretation |
|-----------|------|----|------|-----|-------|----------------|
| Trend GP std dev (α) | {all_params.get('alpha_trend', {}).get('mean', 0):.2f} lbs | {all_params.get('alpha_trend', {}).get('sd', 0):.2f} | {all_params.get('alpha_trend', {}).get('2.5%', 0):.2f} | {all_params.get('alpha_trend', {}).get('50%', 0):.2f} | {all_params.get('alpha_trend', {}).get('97.5%', 0):.2f} | Marginal standard deviation of trend component |
| Trend GP length scale (ρ) | {all_params.get('rho_trend', {}).get('mean', 0):.3f} | {all_params.get('rho_trend', {}).get('sd', 0):.3f} | {all_params.get('rho_trend', {}).get('2.5%', 0):.3f} | {all_params.get('rho_trend', {}).get('50%', 0):.3f} | {all_params.get('rho_trend', {}).get('97.5%', 0):.3f} | Characteristic time scale of trend changes (scaled 0-1) |
| Student-t degrees of freedom (ν) | {all_params.get('nu', {}).get('mean', 0):.1f} | {all_params.get('nu', {}).get('sd', 0):.1f} | {all_params.get('nu', {}).get('2.5%', 0):.1f} | {all_params.get('nu', {}).get('50%', 0):.1f} | {all_params.get('nu', {}).get('97.5%', 0):.1f} | Robustness parameter (ν→∞ → normal, ν small → heavy tails) |
| Trend change | {all_params.get('trend_change', {}).get('mean', 0):.2f} lbs | {all_params.get('trend_change', {}).get('sd', 0):.2f} | {all_params.get('trend_change', {}).get('2.5%', 0):.2f} | {all_params.get('trend_change', {}).get('50%', 0):.2f} | {all_params.get('trend_change', {}).get('97.5%', 0):.2f} | Total weight change from first to last observation |""")

        # Add Fourier coefficient scales
        if model_name == "daily":
            f.write(f"""
| Daily Fourier scale (σ_fourier) | {all_params.get('sigma_fourier', {}).get('mean', 0):.3f} lbs | {all_params.get('sigma_fourier', {}).get('sd', 0):.3f} | {all_params.get('sigma_fourier', {}).get('2.5%', 0):.3f} | {all_params.get('sigma_fourier', {}).get('50%', 0):.3f} | {all_params.get('sigma_fourier', {}).get('97.5%', 0):.3f} | Prior scale for daily Fourier coefficients |
""")
        else:  # weekly model
            f.write(f"""
| Daily Fourier scale (σ_fourier_daily) | {all_params.get('sigma_fourier_daily', {}).get('mean', 0):.3f} lbs | {all_params.get('sigma_fourier_daily', {}).get('sd', 0):.3f} | {all_params.get('sigma_fourier_daily', {}).get('2.5%', 0):.3f} | {all_params.get('sigma_fourier_daily', {}).get('50%', 0):.3f} | {all_params.get('sigma_fourier_daily', {}).get('97.5%', 0):.3f} | Prior scale for daily Fourier coefficients |
| Weekly Fourier scale (σ_fourier_weekly) | {all_params.get('sigma_fourier_weekly', {}).get('mean', 0):.3f} lbs | {all_params.get('sigma_fourier_weekly', {}).get('sd', 0):.3f} | {all_params.get('sigma_fourier_weekly', {}).get('2.5%', 0):.3f} | {all_params.get('sigma_fourier_weekly', {}).get('50%', 0):.3f} | {all_params.get('sigma_fourier_weekly', {}).get('97.5%', 0):.3f} | Prior scale for weekly Fourier coefficients |
""")

        f.write("""
### 3.3 Incidental Parameters (Summary Statistics)

*Note: These are raw/non-centered parameters used for computational efficiency.*

| Parameter | Mean | SD | Shape | Description |
|-----------|------|----|-------|-------------|""")

        # Add raw Fourier coefficients
        if 'a_sin_raw' in all_params:
            f.write(f"""
| Daily sine coefficients (a_sin_raw) | {all_params['a_sin_raw'].get('mean', 0):.3f} | {all_params['a_sin_raw'].get('sd', 0):.3f} | {all_params['a_sin_raw'].get('shape', 'N/A')} | Raw sine coefficients (non-centered) |
| Daily cosine coefficients (a_cos_raw) | {all_params['a_cos_raw'].get('mean', 0):.3f} | {all_params['a_cos_raw'].get('sd', 0):.3f} | {all_params['a_cos_raw'].get('shape', 'N/A')} | Raw cosine coefficients (non-centered) |""")

        if 'b_sin_raw' in all_params:
            f.write(f"""
| Weekly sine coefficients (b_sin_raw) | {all_params['b_sin_raw'].get('mean', 0):.3f} | {all_params['b_sin_raw'].get('sd', 0):.3f} | {all_params['b_sin_raw'].get('shape', 'N/A')} | Raw weekly sine coefficients (non-centered) |
| Weekly cosine coefficients (b_cos_raw) | {all_params['b_cos_raw'].get('mean', 0):.3f} | {all_params['b_cos_raw'].get('sd', 0):.3f} | {all_params['b_cos_raw'].get('shape', 'N/A')} | Raw weekly cosine coefficients (non-centered) |""")

        # Add transformed Fourier coefficients if available
        if 'a_sin' in all_params:
            f.write(f"""
| Daily sine coefficients (a_sin) | {all_params['a_sin'].get('mean', 0):.3f} lbs | {all_params['a_sin'].get('sd', 0):.3f} | {all_params['a_sin'].get('shape', 'N/A')} | Transformed sine coefficients (σ_fourier × a_sin_raw) |
| Daily cosine coefficients (a_cos) | {all_params['a_cos'].get('mean', 0):.3f} lbs | {all_params['a_cos'].get('sd', 0):.3f} | {all_params['a_cos'].get('shape', 'N/A')} | Transformed cosine coefficients (σ_fourier × a_cos_raw) |""")

        if 'b_sin' in all_params:
            f.write(f"""
| Weekly sine coefficients (b_sin) | {all_params['b_sin'].get('mean', 0):.3f} lbs | {all_params['b_sin'].get('sd', 0):.3f} | {all_params['b_sin'].get('shape', 'N/A')} | Transformed weekly sine coefficients (σ_fourier_weekly × b_sin_raw) |
| Weekly cosine coefficients (b_cos) | {all_params['b_cos'].get('mean', 0):.3f} lbs | {all_params['b_cos'].get('sd', 0):.3f} | {all_params['b_cos'].get('shape', 'N/A')} | Transformed weekly cosine coefficients (σ_fourier_weekly × b_cos_raw) |""")

        # Add GP parameters
        if 'eta_trend' in all_params:
            f.write(f"""
| Trend GP standardized values (η_trend) | {all_params['eta_trend'].get('mean', 0):.3f} | {all_params['eta_trend'].get('sd', 0):.3f} | {all_params['eta_trend'].get('shape', 'N/A')} | Standardized values for non-centered GP representation |""")

        if 'eta_inducing' in all_params:
            f.write(f"""
| Inducing point values (η_inducing) | {all_params['eta_inducing'].get('mean', 0):.3f} | {all_params['eta_inducing'].get('sd', 0):.3f} | {all_params['eta_inducing'].get('shape', 'N/A')} | Standardized values at inducing points (sparse GP) |""")

        f.write("""

### 3.4 Diagnostic Summary

""")

        # Add diagnostic information if available
        if hasattr(idata, 'sample_stats'):
            divergent = idata.sample_stats.get('divergent', None)
            if divergent is not None:
                n_divergent = divergent.values.sum()
                pct_divergent = 100 * n_divergent / (chains * iter_sampling) if chains * iter_sampling > 0 else 0
                f.write(f"- **Divergent transitions**: {n_divergent} ({pct_divergent:.1f}%)\n")

            treedepth = idata.sample_stats.get('treedepth__', None)
            if treedepth is not None:
                max_observed = treedepth.values.max()
                f.write(f"- **Maximum tree depth observed**: {max_observed}/{max_treedepth}\n")

        # Visualization section
        if not skip_plots:
            f.write("""
## 4. Visualizations

### 4.1 Cyclic Components
![Cyclic Components](cyclic_components.png)

Shows the decomposition of the fitted model into components.
""")

            if model_name == "weekly":
                f.write("The weekly model shows **trend** (slow changes), **daily** (24‑hour cycles), and **weekly** (7‑day cycles) components.\n")
            else:
                f.write("The daily model shows **trend** (slow changes) and **daily** (24‑hour cycles) components.\n")

            f.write(f"""
### 4.2 Spline Daily Pattern
![Spline Daily Pattern](spline_daily_pattern.png)

The estimated 24‑hour daily pattern (mean ± 2 SD) from the Fourier spline model with K={fourier_harmonics} harmonics.

### 4.3 Full Model Expectation
![Full Expectation](full_expectation.png)

Complete model prediction (trend + daily{' + weekly' if model_name == 'weekly' else ''}) with 95% credible interval. Observations are shown as points.
""")

            if predictions is not None:
                f.write("""
### 4.4 Predictions at Unobserved Days
![Predictions](predictions.png)

Model predictions for unobserved days across the full date range (generated with prediction grid). Shows the expected weight trajectory with 95% credible interval for days without measurements.
""")

                # Check for hourly predictions plot
                hourly_plot_path = output_dir / "hourly_predictions.png"
                if hourly_plot_path.exists():
                    f.write("""
### 4.5 Hourly Predictions
![Hourly Predictions](hourly_predictions.png)

Shows weight predictions at different hours of the day across all prediction days. The left panel aggregates predictions by hour (mean ± SD across days), the right panel shows hourly predictions for a single day. This visualization reveals the within‑day variation captured by the Fourier spline.
""")

                # Check for weekly zoomed predictions plot
                weekly_zoomed_path = output_dir / "weekly_zoomed_predictions.png"
                if weekly_zoomed_path.exists():
                    f.write("""
### 4.6 Weekly Zoomed Predictions
![Weekly Zoomed Predictions](weekly_zoomed_predictions.png)

Hourly predictions zoomed into a specific week. Each day is shown with a distinct color, with observed data points overlaid. This detailed view shows how the daily pattern repeats across consecutive days.
""")
        else:
            f.write("""
## 4. Visualizations

*Visualizations were skipped (--skip-plots used).*
""")

        # Interpretation and reproduction
        f.write(f"""
## 5. Interpretation

1. **Measurement error reduction**: The spline model separates true {'daily and weekly' if model_name == 'weekly' else 'daily'} variation from measurement noise.
2. **Daily cycle significance**: A substantial daily amplitude ({daily_amplitude:.2f} lbs) suggests regular within‑day weight fluctuations.
3. **Model fit**: The proportion of variance explained by the daily component ({prop_variance_daily:.3f}) indicates how much of the total variation is cyclic.
""")

        if weekly_amplitude is not None:
            f.write(f"""4. **Weekly cycle significance**: A weekly amplitude of {weekly_amplitude:.2f} lbs suggests day‑of‑week patterns in weight.
5. **Weekly contribution**: The weekly component explains {prop_variance_weekly:.3f} of total variance.
""")

        f.write(f"""6. **Fourier harmonics**: Using K={fourier_harmonics} harmonics allows the spline to capture more complex daily patterns than a simple periodic kernel.
""")

        if weekly_harmonics is not None:
            f.write(f"""   Using L={weekly_harmonics} harmonics for weekly cycles.
""")

        # Sparse approximation description
        if use_sparse:
            sparse_desc = f'Projected process (DIC) with {n_inducing_points} inducing points'
            sparse_cmd = f'--use-sparse --n-inducing-points {n_inducing_points}'
        else:
            sparse_desc = 'Full GP (no approximation)'
            sparse_cmd = ''

        # Prediction grid command line
        if include_prediction_grid:
            if prediction_hour_step is not None:
                prediction_cmd = f'--include-prediction-grid --prediction-hour {prediction_hour} --prediction-hour-step {prediction_hour_step} --prediction-step-days {prediction_step_days}'
            else:
                prediction_cmd = f'--include-prediction-grid --prediction-hour {prediction_hour} --prediction-step-days {prediction_step_days}'
        else:
            prediction_cmd = ''

        # Weekly harmonics command line
        weekly_harmonics_cmd = f'--weekly-harmonics {weekly_harmonics}' if weekly_harmonics is not None else ''

        f.write(f"""
## 6. Reproduction

To reproduce this analysis:

```bash
python -m src.models.demo_general --model {model_name} \\
  --chains {chains} \\
  --iter-warmup {iter_warmup} \\
  --iter-sampling {iter_sampling} \\
  --adapt-delta {adapt_delta} \\
  --max-treedepth {max_treedepth} \\
  --fourier-harmonics {fourier_harmonics} \\
  {weekly_harmonics_cmd} {sparse_cmd} {prediction_cmd} \\
  --output-dir {output_dir.absolute()}
```

## 7. Technical Details

- **Stan model**: `weight_gp_spline_{'weekly' if model_name == 'weekly' else 'optimized'}.stan`
- **Covariance function**: `gp_exp_quad_cov` (trend)
- **Daily representation**: Fourier spline with {fourier_harmonics} harmonics (sin/cos pairs)
""")

        if weekly_harmonics is not None:
            f.write(f"- **Weekly representation**: Fourier spline with {weekly_harmonics} harmonics (sin/cos pairs)\n")

        f.write(f"""- **Sparse approximation**: {sparse_desc}
- **Software**: CmdStanPy {chains} chains, ArviZ for diagnostics

---
*Report generated by `demo_general.py`*
""")

    print(f"✓ Markdown report saved to {report_path}")
    return report_path


def write_comparison_report(
    output_dir: Path,
    comparison_df: pd.DataFrame,
    daily_results: Dict[str, Any],
    weekly_results: Dict[str, Any],
    comparison_metrics: Dict[str, Any],
    adapt_delta: float = 0.99,
    max_treedepth: int = 12,
    fourier_harmonics: int = 2,
    weekly_harmonics: int = 1,
    use_sparse: bool = False,
    n_inducing_points: int = 50,
    chains: int = 4,
    iter_warmup: int = 500,
    iter_sampling: int = 1500,
):
    """Write comprehensive comparison report for both models.

    Args:
        output_dir: Directory for comparison outputs
        comparison_df: DataFrame from compare_models_waic_loo
        daily_results: Dictionary with daily model results
        weekly_results: Dictionary with weekly model results
        comparison_metrics: Dictionary with comparison metrics from compare_models_all
        adapt_delta: Adapt delta parameter used
        max_treedepth: Maximum tree depth used
        fourier_harmonics: Number of Fourier harmonics (K)
        weekly_harmonics: Number of weekly Fourier harmonics (L)
        use_sparse: Whether sparse GP was used
        n_inducing_points: Number of inducing points (if sparse)
        chains: Number of MCMC chains
        iter_warmup: Warmup iterations per chain
        iter_sampling: Sampling iterations per chain
    """
    report_path = output_dir / "comparison_report.md"

    # Extract key metrics
    sigma_daily = daily_results.get('sigma_lbs')
    sigma_weekly = weekly_results.get('sigma_lbs')
    daily_amp_daily = daily_results.get('daily_amplitude_lbs')
    daily_amp_weekly = weekly_results.get('daily_amplitude_lbs')
    weekly_amp = weekly_results.get('weekly_amplitude_lbs')
    prop_var_daily_daily = daily_results.get('prop_variance_daily')
    prop_var_daily_weekly = weekly_results.get('prop_variance_daily')
    prop_var_weekly = weekly_results.get('prop_variance_weekly')

    # WAIC/LOO weights
    waic_weight_daily = comparison_df.loc['daily', 'waic_weight'] if 'daily' in comparison_df.index else None
    waic_weight_weekly = comparison_df.loc['weekly', 'waic_weight'] if 'weekly' in comparison_df.index else None
    loo_weight_daily = comparison_df.loc['daily', 'loo_weight'] if 'daily' in comparison_df.index else None
    loo_weight_weekly = comparison_df.loc['weekly', 'loo_weight'] if 'weekly' in comparison_df.index else None

    # Determine best model by each criterion
    best_waic = comparison_df['waic_weight'].idxmax() if 'waic_weight' in comparison_df.columns and not comparison_df['waic_weight'].isna().all() else None
    best_loo = comparison_df['loo_weight'].idxmax() if 'loo_weight' in comparison_df.columns and not comparison_df['loo_weight'].isna().all() else None

    with open(report_path, 'w') as f:
        f.write(f"""# Model Comparison Report: Daily vs Weekly Spline Models

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. Comparison Overview

This report compares two Bayesian spline models for weight data:

1. **Daily model**: Optimized spline GP with trend + daily Fourier cycles
2. **Weekly model**: Weekly spline GP with trend + daily + weekly Fourier cycles

The comparison evaluates whether adding weekly cycles (day-of-week patterns) improves model fit and prediction.

### 1.1 Model Configurations

Both models share these settings:
- **Fourier harmonics (daily)**: K={fourier_harmonics}
- **Sparse GP**: {'Enabled' if use_sparse else 'Disabled'} {f'({n_inducing_points} inducing points)' if use_sparse else ''}
- **Sampler**: {chains} chains, {iter_warmup} warmup, {iter_sampling} sampling iterations
- **Adapt delta**: {adapt_delta}
- **Max tree depth**: {max_treedepth}

Weekly model additional setting:
- **Weekly Fourier harmonics**: L={weekly_harmonics}

## 2. Model Performance Comparison

### 2.1 Information Criteria (WAIC and LOO-CV)

Lower values indicate better predictive performance. Weights represent relative probability that each model is the best given the data.

""")

        # Add comparison table
        f.write(comparison_df.to_markdown(floatfmt=".1f"))
        f.write("\n\n")

        # Interpretation
        f.write("**Interpretation**:\n")
        if best_waic:
            f.write(f"- **Best model by WAIC**: {best_waic} (weight={waic_weight_daily if best_waic == 'daily' else waic_weight_weekly:.3f})\n")
        if best_loo:
            f.write(f"- **Best model by LOO-CV**: {best_loo} (weight={loo_weight_daily if best_loo == 'daily' else loo_weight_weekly:.3f})\n")

        # Rule of thumb for weight interpretation
        f.write(f"""
- **Weight > 0.9**: Strong evidence for that model
- **Weight 0.7‑0.9**: Moderate evidence
- **Weight 0.5‑0.7**: Weak evidence
- **Weight < 0.5**: Inconclusive (neither model clearly better)

### 2.2 Primary Parameter Comparison

| Metric | Daily Model | Weekly Model | Difference | Interpretation |
|--------|-------------|--------------|------------|----------------|
| Measurement error (σ) | {sigma_daily:.2f} lbs | {sigma_weekly:.2f} lbs | {sigma_weekly - sigma_daily:+.2f} lbs | {'Weekly model has lower measurement error' if sigma_weekly < sigma_daily else 'Daily model has lower measurement error'} |
| Daily amplitude | {daily_amp_daily:.2f} lbs | {daily_amp_weekly:.2f} lbs | {daily_amp_weekly - daily_amp_daily:+.2f} lbs | {'Weekly model estimates larger daily variation' if daily_amp_weekly > daily_amp_daily else 'Daily model estimates larger daily variation'} |
| Proportion of variance (daily) | {prop_var_daily_daily:.3f} | {prop_var_daily_weekly:.3f} | {prop_var_daily_weekly - prop_var_daily_daily:+.3f} | {'Weekly model attributes more variance to daily cycles' if prop_var_daily_weekly > prop_var_daily_daily else 'Daily model attributes more variance to daily cycles'} |
| Weekly amplitude | — | {weekly_amp:.2f} lbs | — | Peak‑to‑peak variation within a 7‑day period |
| Proportion of variance (weekly) | — | {prop_var_weekly:.3f} | — | Fraction of total variation explained by weekly cycle |

### 2.3 Hyperparameter Comparison

*Note: All values in original scale (lbs where applicable)*

| Hyperparameter | Daily Model | Weekly Model | Difference | Interpretation |
|----------------|-------------|--------------|------------|----------------|
| Trend GP std dev (α) | {daily_results.get('all_params', {}).get('alpha_trend', {}).get('mean', 0):.2f} lbs | {weekly_results.get('all_params', {}).get('alpha_trend', {}).get('mean', 0):.2f} lbs | {weekly_results.get('all_params', {}).get('alpha_trend', {}).get('mean', 0) - daily_results.get('all_params', {}).get('alpha_trend', {}).get('mean', 0):+.2f} lbs | Marginal std dev of trend component |
| Trend GP length scale (ρ) | {daily_results.get('all_params', {}).get('rho_trend', {}).get('mean', 0):.3f} | {weekly_results.get('all_params', {}).get('rho_trend', {}).get('mean', 0):.3f} | {weekly_results.get('all_params', {}).get('rho_trend', {}).get('mean', 0) - daily_results.get('all_params', {}).get('rho_trend', {}).get('mean', 0):+.3f} | Characteristic time scale (scaled 0-1) |
| Student-t degrees of freedom (ν) | {daily_results.get('all_params', {}).get('nu', {}).get('mean', 0):.1f} | {weekly_results.get('all_params', {}).get('nu', {}).get('mean', 0):.1f} | {weekly_results.get('all_params', {}).get('nu', {}).get('mean', 0) - daily_results.get('all_params', {}).get('nu', {}).get('mean', 0):+.1f} | Robustness (ν→∞ → normal) |
| Trend change | {daily_results.get('all_params', {}).get('trend_change', {}).get('mean', 0):.2f} lbs | {weekly_results.get('all_params', {}).get('trend_change', {}).get('mean', 0):.2f} lbs | {weekly_results.get('all_params', {}).get('trend_change', {}).get('mean', 0) - daily_results.get('all_params', {}).get('trend_change', {}).get('mean', 0):+.2f} lbs | Total weight change over period |
| Daily Fourier scale | {daily_results.get('all_params', {}).get('sigma_fourier', {}).get('mean', 0):.3f} lbs | {weekly_results.get('all_params', {}).get('sigma_fourier_daily', {}).get('mean', 0):.3f} lbs | {weekly_results.get('all_params', {}).get('sigma_fourier_daily', {}).get('mean', 0) - daily_results.get('all_params', {}).get('sigma_fourier', {}).get('mean', 0):+.3f} lbs | Prior scale for daily Fourier coefficients |
| Weekly Fourier scale | — | {weekly_results.get('all_params', {}).get('sigma_fourier_weekly', {}).get('mean', 0):.3f} lbs | — | Prior scale for weekly Fourier coefficients |

### 2.4 Model Selection Recommendations

""")

        # Decision logic
        if best_waic == best_loo and best_waic is not None:
            f.write(f"1. **Consensus best model**: Both WAIC and LOO-CV select the **{best_waic}** model.\n")
        elif best_waic != best_loo and best_waic is not None and best_loo is not None:
            f.write(f"1. **Inconsistent selection**: WAIC prefers **{best_waic}**, LOO-CV prefers **{best_loo}**.\n")
            f.write("   - Consider checking Pareto k diagnostics for LOO-CV reliability.\n")
        else:
            f.write("1. **Inconclusive selection**: Information criteria weights are similar.\n")

        # Check if weekly component is meaningful
        if weekly_amp is not None:
            if weekly_amp > 0.1:  # Arbitrary threshold
                f.write(f"2. **Weekly component meaningful**: Weekly amplitude of {weekly_amp:.2f} lbs suggests detectable day‑of‑week patterns.\n")
                if prop_var_weekly > 0.05:
                    f.write(f"   - Weekly component explains {prop_var_weekly:.1%} of total variance.\n")
            else:
                f.write(f"2. **Weekly component minimal**: Weekly amplitude of {weekly_amp:.2f} lbs is small; weekly patterns may not be meaningful.\n")

        # Check sigma reduction
        if sigma_weekly < sigma_daily:
            reduction_pct = 100 * (sigma_daily - sigma_weekly) / sigma_daily
            f.write(f"3. **Measurement error reduced**: Weekly model reduces σ by {reduction_pct:.1f}%.\n")
        elif sigma_weekly > sigma_daily:
            increase_pct = 100 * (sigma_weekly - sigma_daily) / sigma_daily
            f.write(f"3. **Measurement error increased**: Weekly model increases σ by {increase_pct:.1f}% (weekly component may be overfitting).\n")
        else:
            f.write("3. **No change in measurement error**: Weekly component doesn't affect σ estimate.\n")

        # Write visual comparison section up to prediction comparison
        f.write("""
## 3. Visual Comparison

### 3.1 Component Decomposition
![Component Comparison](component_comparison.png)

Side‑by‑side comparison of trend, daily, and weekly components. The weekly model shows an additional weekly component capturing day‑of‑week patterns.

### 3.2 Daily Pattern Comparison
![Daily Pattern Comparison](daily_pattern_comparison.png)

Comparison of estimated 24‑hour daily patterns from both models.

### 3.3 Weekly Pattern Visualization
![Weekly Pattern Comparison](weekly_pattern_comparison.png)

Visualization of the weekly component capturing day‑of‑week patterns (weekly model only). The left panel shows the estimated weekly pattern with uncertainty, the right panel shows detrended and dedaily residuals plotted against the weekly pattern.

### 3.4 Prediction Comparison
""")

        # Check if prediction comparison plot exists
        prediction_plot_path = output_dir / "prediction_comparison.png"
        if prediction_plot_path.exists():
            f.write("""![Prediction Comparison](prediction_comparison.png)

Side‑by‑side comparison of model predictions for unobserved days. The top panel shows daily model predictions, the bottom panel shows weekly model predictions.
""")
        else:
            f.write("""*Prediction comparison not generated because prediction grid was not enabled.*
To include prediction comparison, rerun with `--include-prediction-grid` flag.
""")

        # Continue with detailed results
        f.write(f"""
## 4. Detailed Results

### 4.1 Daily Model Results
See full report: [daily_report.md](daily/daily_report.md)

- **Sigma (σ)**: {sigma_daily:.2f} lbs
- **Daily amplitude**: {daily_amp_daily:.2f} lbs
- **Proportion of variance (daily)**: {prop_var_daily_daily:.3f}

### 4.2 Weekly Model Results
See full report: [weekly_report.md](weekly/weekly_report.md)

- **Sigma (σ)**: {sigma_weekly:.2f} lbs
- **Daily amplitude**: {daily_amp_weekly:.2f} lbs
- **Weekly amplitude**: {weekly_amp:.2f} lbs
- **Proportion of variance (daily)**: {prop_var_daily_weekly:.3f}
- **Proportion of variance (weekly)**: {prop_var_weekly:.3f}

## 5. Conclusions

""")

        # Generate conclusions
        conclusions = []

        # Based on information criteria
        if waic_weight_daily is not None and waic_weight_weekly is not None:
            if waic_weight_daily > 0.7:
                conclusions.append("**Daily model strongly preferred** by WAIC")
            elif waic_weight_weekly > 0.7:
                conclusions.append("**Weekly model strongly preferred** by WAIC")
            elif abs(waic_weight_daily - waic_weight_weekly) < 0.2:
                conclusions.append("**Inconclusive** - both models perform similarly by WAIC")
            else:
                conclusions.append(f"**{best_waic} model moderately preferred** by WAIC")

        # Based on weekly amplitude
        if weekly_amp is not None:
            if weekly_amp > 0.2:
                conclusions.append("**Weekly component substantial** ({weekly_amp:.2f} lbs amplitude)")
            elif weekly_amp > 0.05:
                conclusions.append("**Weekly component detectable but small** ({weekly_amp:.2f} lbs amplitude)")
            else:
                conclusions.append("**Weekly component negligible** ({weekly_amp:.2f} lbs amplitude)")

        # Based on sigma reduction
        if sigma_weekly < sigma_daily:
            reduction = sigma_daily - sigma_weekly
            conclusions.append(f"**Weekly model reduces measurement error** by {reduction:.2f} lbs")
        elif sigma_weekly > sigma_daily:
            increase = sigma_weekly - sigma_daily
            conclusions.append(f"**Weekly model increases measurement error** by {increase:.2f} lbs (possible overfitting)")

        # Write conclusions
        for i, conclusion in enumerate(conclusions, 1):
            f.write(f"{i}. {conclusion}\n")

        f.write(f"""
## 6. Reproduction

To reproduce this comparison:

```bash
python -m src.models.demo_general --model both \\
  --chains {chains} \\
  --iter-warmup {iter_warmup} \\
  --iter-sampling {iter_sampling} \\
  --adapt-delta {adapt_delta} \\
  --max-treedepth {max_treedepth} \\
  --fourier-harmonics {fourier_harmonics} \\
  --weekly-harmonics {weekly_harmonics} \\
  {'--use-sparse --n-inducing-points ' + str(n_inducing_points) if use_sparse else ''} \\
  --output-dir {output_dir.absolute()}
```

---
*Comparison report generated by `demo_general.py`*
""")

    print(f"✓ Comparison report saved to {report_path}")
    return report_path


def create_comparison_plots(
    output_dir: Path,
    daily_results: Dict[str, Any],
    weekly_results: Dict[str, Any],
    daily_predictions: Optional[Dict] = None,
    weekly_predictions: Optional[Dict] = None,
):
    """Create side-by-side comparison plots for both models.

    Args:
        output_dir: Directory for comparison outputs
        daily_results: Dictionary with daily model results (idata, df, stan_data)
        weekly_results: Dictionary with weekly model results (idata, df, stan_data)
        daily_predictions: Daily model predictions from extract_predictions
        weekly_predictions: Weekly model predictions from extract_predictions
    """
    # 1. Component decomposition comparison
    component_path = output_dir / "component_comparison.png"
    try:
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        # Back-transform parameters
        y_mean_daily = daily_results['stan_data']["_y_mean"]
        y_sd_daily = daily_results['stan_data']["_y_sd"]
        y_mean_weekly = weekly_results['stan_data']["_y_mean"]
        y_sd_weekly = weekly_results['stan_data']["_y_sd"]

        # Daily model components
        df_daily = daily_results['df']

        # Extract components for daily model
        if 'f_trend' in daily_results['idata'].posterior:
            f_trend_daily = daily_results['idata'].posterior["f_trend"].values.mean(axis=(0, 1)) * y_sd_daily + y_mean_daily
            f_daily_daily = daily_results['idata'].posterior["f_daily"].values.mean(axis=(0, 1)) * y_sd_daily + y_mean_daily

            axes[0, 0].plot(df_daily["date"], f_trend_daily, "b-", linewidth=2)
            axes[0, 0].set_title("Daily Model: Trend Component")
            axes[0, 0].set_ylabel("Weight (lbs)")
            axes[0, 0].grid(True, alpha=0.3)

            axes[1, 0].plot(df_daily["date"], f_daily_daily, "g-", linewidth=2)
            axes[1, 0].set_title("Daily Model: Daily Component")
            axes[1, 0].set_ylabel("Weight (lbs)")
            axes[1, 0].set_xlabel("Date")
            axes[1, 0].grid(True, alpha=0.3)

        # Weekly model components
        df_weekly = weekly_results['df']

        if 'f_trend' in weekly_results['idata'].posterior:
            f_trend_weekly = weekly_results['idata'].posterior["f_trend"].values.mean(axis=(0, 1)) * y_sd_weekly + y_mean_weekly
            f_daily_weekly = weekly_results['idata'].posterior["f_daily"].values.mean(axis=(0, 1)) * y_sd_weekly + y_mean_weekly
            f_weekly_weekly = weekly_results['idata'].posterior["f_weekly"].values.mean(axis=(0, 1)) * y_sd_weekly + y_mean_weekly

            axes[0, 1].plot(df_weekly["date"], f_trend_weekly, "b-", linewidth=2)
            axes[0, 1].set_title("Weekly Model: Trend Component")
            axes[0, 1].set_ylabel("Weight (lbs)")
            axes[0, 1].grid(True, alpha=0.3)

            axes[1, 1].plot(df_weekly["date"], f_daily_weekly, "g-", linewidth=2)
            axes[1, 1].set_title("Weekly Model: Daily Component")
            axes[1, 1].set_ylabel("Weight (lbs)")
            axes[1, 1].set_xlabel("Date")
            axes[1, 1].grid(True, alpha=0.3)

            axes[0, 2].plot(df_weekly["date"], f_weekly_weekly, "r-", linewidth=2)
            axes[0, 2].set_title("Weekly Model: Weekly Component")
            axes[0, 2].set_ylabel("Weight (lbs)")
            axes[0, 2].set_xlabel("Date")
            axes[0, 2].grid(True, alpha=0.3)

        # Leave one subplot empty for alignment
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.savefig(component_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"✓ Component comparison plot saved to {component_path}")
    except Exception as e:
        print(f"⚠ Could not create component comparison plot: {e}")

    # 2. Daily pattern comparison
    daily_pattern_path = output_dir / "daily_pattern_comparison.png"
    try:
        # Extract Fourier coefficients and compute daily patterns for both models
        y_sd_daily = daily_results['stan_data']["_y_sd"]
        y_sd_weekly = weekly_results['stan_data']["_y_sd"]

        # Daily model coefficients
        if "a_sin" in daily_results['idata'].posterior and "a_cos" in daily_results['idata'].posterior:
            a_sin_daily = daily_results['idata'].posterior["a_sin"].values  # (chain, draw, K)
            a_cos_daily = daily_results['idata'].posterior["a_cos"].values
            K_daily = daily_results['stan_data']["K"]
        else:
            raise ValueError("Daily model missing a_sin/a_cos parameters")

        # Weekly model daily coefficients
        if "a_sin" in weekly_results['idata'].posterior and "a_cos" in weekly_results['idata'].posterior:
            a_sin_weekly = weekly_results['idata'].posterior["a_sin"].values
            a_cos_weekly = weekly_results['idata'].posterior["a_cos"].values
            K_weekly = weekly_results['stan_data']["K"]
        else:
            raise ValueError("Weekly model missing a_sin/a_cos parameters")

        # Reshape to (samples, K)
        n_chains_d, n_draws_d, _ = a_sin_daily.shape
        n_samples_d = n_chains_d * n_draws_d
        a_sin_daily_flat = a_sin_daily.reshape(n_samples_d, K_daily) * y_sd_daily
        a_cos_daily_flat = a_cos_daily.reshape(n_samples_d, K_daily) * y_sd_daily

        n_chains_w, n_draws_w, _ = a_sin_weekly.shape
        n_samples_w = n_chains_w * n_draws_w
        a_sin_weekly_flat = a_sin_weekly.reshape(n_samples_w, K_weekly) * y_sd_weekly
        a_cos_weekly_flat = a_cos_weekly.reshape(n_samples_w, K_weekly) * y_sd_weekly

        # Create grid of hours (0-24)
        hours_grid = np.linspace(0, 24, 100)
        hours_scaled_grid = hours_grid / 24.0

        # Evaluate Fourier series for each sample
        def compute_pattern(a_sin_flat, a_cos_flat, hours_scaled_grid):
            n_samples, K = a_sin_flat.shape
            n_hours = len(hours_scaled_grid)
            pattern = np.zeros((n_samples, n_hours))
            for s in range(n_samples):
                for h_idx, hour_scaled in enumerate(hours_scaled_grid):
                    val = 0.0
                    for k in range(K):
                        freq = 2.0 * np.pi * (k + 1)
                        val += a_sin_flat[s, k] * np.sin(freq * hour_scaled) + a_cos_flat[s, k] * np.cos(freq * hour_scaled)
                    pattern[s, h_idx] = val
            return pattern

        pattern_daily = compute_pattern(a_sin_daily_flat, a_cos_daily_flat, hours_scaled_grid)
        pattern_weekly = compute_pattern(a_sin_weekly_flat, a_cos_weekly_flat, hours_scaled_grid)

        # Compute statistics
        def compute_stats(pattern):
            mean = pattern.mean(axis=0)
            lower = np.percentile(pattern, 2.5, axis=0)
            upper = np.percentile(pattern, 97.5, axis=0)
            return mean, lower, upper

        mean_daily, lower_daily, upper_daily = compute_stats(pattern_daily)
        mean_weekly, lower_weekly, upper_weekly = compute_stats(pattern_weekly)

        # Create comparison plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot daily model pattern with CI
        ax.plot(hours_grid, mean_daily, 'b-', linewidth=2, label='Daily model')
        ax.fill_between(hours_grid, lower_daily, upper_daily, alpha=0.3, color='blue', label='95% CI (daily)')

        # Plot weekly model daily pattern with CI
        ax.plot(hours_grid, mean_weekly, 'r-', linewidth=2, label='Weekly model (daily component)')
        ax.fill_between(hours_grid, lower_weekly, upper_weekly, alpha=0.3, color='red', label='95% CI (weekly)')

        ax.set_xlabel("Hour of day (GMT)")
        ax.set_ylabel("Weight deviation (lbs)")
        ax.set_title("Comparison of Daily Patterns: Daily vs Weekly Models")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 24)

        plt.tight_layout()
        plt.savefig(daily_pattern_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"✓ Daily pattern comparison plot saved to {daily_pattern_path}")
    except Exception as e:
        print(f"⚠ Could not create daily pattern comparison plot: {e}")

    # 3. Weekly pattern visualization (only for weekly model)
    weekly_pattern_path = output_dir / "weekly_pattern_comparison.png"
    try:
        # Check if weekly model has weekly Fourier coefficients
        if "b_sin" in weekly_results['idata'].posterior and "b_cos" in weekly_results['idata'].posterior:
            b_sin_weekly = weekly_results['idata'].posterior["b_sin"].values  # (chain, draw, L)
            b_cos_weekly = weekly_results['idata'].posterior["b_cos"].values
            L = weekly_results['stan_data']["L"]
            day_of_week_obs = np.array(weekly_results['stan_data']["day_of_week"])
            y_sd_weekly = weekly_results['stan_data']["_y_sd"]

            # Reshape to (samples, L)
            n_chains_w, n_draws_w, _ = b_sin_weekly.shape
            n_samples_w = n_chains_w * n_draws_w
            b_sin_weekly_flat = b_sin_weekly.reshape(n_samples_w, L) * y_sd_weekly
            b_cos_weekly_flat = b_cos_weekly.reshape(n_samples_w, L) * y_sd_weekly

            # Create grid of days (0-7 representing Monday-Sunday)
            days_grid = np.linspace(0, 7, 100)
            days_scaled_grid = days_grid / 7.0

            # Evaluate weekly Fourier series
            n_days = len(days_scaled_grid)
            pattern_weekly = np.zeros((n_samples_w, n_days))
            for s in range(n_samples_w):
                for d_idx, day_scaled in enumerate(days_scaled_grid):
                    val = 0.0
                    for l in range(L):
                        freq = 2.0 * np.pi * (l + 1)
                        val += b_sin_weekly_flat[s, l] * np.sin(freq * day_scaled) + b_cos_weekly_flat[s, l] * np.cos(freq * day_scaled)
                    pattern_weekly[s, d_idx] = val

            # Compute statistics
            mean_weekly = pattern_weekly.mean(axis=0)
            lower_weekly = np.percentile(pattern_weekly, 2.5, axis=0)
            upper_weekly = np.percentile(pattern_weekly, 97.5, axis=0)

            # Create plot
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Plot 1: Weekly pattern with uncertainty
            ax = axes[0]
            # Plot individual samples (first 50)
            for s in range(min(50, n_samples_w)):
                ax.plot(days_grid, pattern_weekly[s, :], alpha=0.05, color='blue', linewidth=0.5)
            ax.plot(days_grid, mean_weekly, 'r-', linewidth=2, label='Mean weekly pattern')
            ax.fill_between(days_grid, lower_weekly, upper_weekly, alpha=0.3, color='red', label='95% CI')
            ax.set_xlabel("Day of week (0=Monday, 7=Sunday)")
            ax.set_ylabel("Weight deviation (lbs)")
            ax.set_title(f"Weekly Pattern (L={L} Fourier harmonics)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 7)

            # Plot 2: Detrended & dedaily residuals vs weekly pattern
            ax = axes[1]
            # Extract trend and daily components
            y_scaled = np.array(weekly_results['stan_data']["y"])
            f_trend_samples = weekly_results['idata'].posterior["f_trend"].values
            f_daily_samples = weekly_results['idata'].posterior["f_daily"].values
            f_trend_mean = f_trend_samples.mean(axis=(0, 1))
            f_daily_mean = f_daily_samples.mean(axis=(0, 1))
            residuals_scaled = y_scaled - f_trend_mean - f_daily_mean
            residuals_original = residuals_scaled * y_sd_weekly

            ax.scatter(day_of_week_obs, residuals_original, alpha=0.6, s=30, color='blue',
                      label='Detrended & dedaily data', zorder=10)
            ax.plot(days_grid, mean_weekly, 'r-', linewidth=2, alpha=0.8, label='Estimated weekly pattern', zorder=5)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1, zorder=1)
            ax.set_xlabel("Day of week (0=Monday, 7=Sunday)")
            ax.set_ylabel("Weight deviation (lbs)")
            ax.set_title("Detrended & Dedaily Data vs. Weekly Pattern")
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 7)

            plt.tight_layout()
            plt.savefig(weekly_pattern_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"✓ Weekly pattern comparison plot saved to {weekly_pattern_path}")
        else:
            print("⚠ Weekly model does not have b_sin/b_cos parameters; skipping weekly pattern plot")
    except Exception as e:
        print(f"⚠ Could not create weekly pattern comparison plot: {e}")

    # 4. Prediction comparison (if predictions available)
    if daily_predictions is not None and weekly_predictions is not None:
        prediction_comparison_path = output_dir / "prediction_comparison.png"
        try:
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))

            # Plot daily model predictions
            if 'f_pred_mean' in daily_predictions:
                # Convert prediction times to datetime
                start_timestamp = daily_results['df']["timestamp"].min()
                t_pred_days = daily_predictions["t_pred"]
                hour_of_day_pred = daily_predictions.get("hour_of_day_pred")

                if hour_of_day_pred is not None and len(hour_of_day_pred) == len(t_pred_days):
                    dates_pred = [start_timestamp + pd.Timedelta(days=float(d)) + pd.Timedelta(hours=float(h))
                                 for d, h in zip(t_pred_days, hour_of_day_pred)]
                else:
                    dates_pred = [start_timestamp + pd.Timedelta(days=float(d)) for d in t_pred_days]

                axes[0].fill_between(dates_pred, daily_predictions["f_pred_lower"], daily_predictions["f_pred_upper"],
                                     alpha=0.3, color='blue', label='95% CI')
                axes[0].plot(dates_pred, daily_predictions["f_pred_mean"], 'b-', linewidth=2, label='Daily model')
                axes[0].set_title("Daily Model Predictions")
                axes[0].set_ylabel("Weight (lbs)")
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)

            # Plot weekly model predictions
            if 'f_pred_mean' in weekly_predictions:
                start_timestamp = weekly_results['df']["timestamp"].min()
                t_pred_days = weekly_predictions["t_pred"]
                hour_of_day_pred = weekly_predictions.get("hour_of_day_pred")

                if hour_of_day_pred is not None and len(hour_of_day_pred) == len(t_pred_days):
                    dates_pred = [start_timestamp + pd.Timedelta(days=float(d)) + pd.Timedelta(hours=float(h))
                                 for d, h in zip(t_pred_days, hour_of_day_pred)]
                else:
                    dates_pred = [start_timestamp + pd.Timedelta(days=float(d)) for d in t_pred_days]

                axes[1].fill_between(dates_pred, weekly_predictions["f_pred_lower"], weekly_predictions["f_pred_upper"],
                                     alpha=0.3, color='red', label='95% CI')
                axes[1].plot(dates_pred, weekly_predictions["f_pred_mean"], 'r-', linewidth=2, label='Weekly model')
                axes[1].set_title("Weekly Model Predictions")
                axes[1].set_ylabel("Weight (lbs)")
                axes[1].set_xlabel("Date")
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(prediction_comparison_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"✓ Prediction comparison plot saved to {prediction_comparison_path}")
        except Exception as e:
            print(f"⚠ Could not create prediction comparison plot: {e}")


def run_daily_model(
    args,
    output_dir: Path,
    skip_plots: bool = False,
) -> Dict[str, Any]:
    """Run the daily-only spline optimized model.

    Returns dictionary with model results.
    """
    print("\n" + "=" * 70)
    print("FITTING DAILY MODEL (trend + daily Fourier spline)")
    print("=" * 70)

    # Sparse GP parameters
    sparse_params = {}
    if args.use_sparse:
        sparse_params = {
            "use_sparse": True,
            "n_inducing_points": args.n_inducing_points,
            "inducing_point_method": "uniform",
        }

    # Fit daily model
    try:
        fit, idata, df, stan_data = fit_weight_model_spline_optimized(
            data_dir=args.data_dir,
            output_dir=output_dir,
            chains=args.chains,
            iter_warmup=args.iter_warmup,
            iter_sampling=args.iter_sampling,
            fourier_harmonics=args.fourier_harmonics,
            cache=True,
            force_refit=args.force_refit,
            include_prediction_grid=args.include_prediction_grid,
            prediction_hour=args.prediction_hour,
            prediction_hour_step=args.prediction_hour_step,
            prediction_step_days=args.prediction_step_days,
            **sparse_params,
        )
        print("✓ Daily model fitted successfully")
    except Exception as e:
        print(f"✗ Error fitting daily model: {e}")
        raise

    # Extract key metrics
    sigma = idata.posterior["sigma"].values.mean() * stan_data["_y_sd"]
    daily_amplitude = idata.posterior["daily_amplitude"].values.mean() * stan_data["_y_sd"]
    prop_variance_daily = idata.posterior["prop_variance_daily"].values.mean()

    print("\nDaily model summary:")
    print(f"  Measurement error (sigma): {sigma:.2f} lbs")
    print(f"  Daily amplitude: {daily_amplitude:.2f} lbs")
    print(f"  Proportion of variance from daily component: {prop_variance_daily:.3f}")

    # Extract predictions if prediction grid is enabled
    predictions = None
    if args.include_prediction_grid and stan_data.get('N_pred', 0) > 0:
        predictions = extract_predictions(idata, stan_data)
        print(f"  Extracted predictions for {len(predictions['t_pred'])} time points")

    # Generate plots
    if not skip_plots:
        print("\nGenerating daily model visualizations...")

        # 1. Cyclic components plot
        components_path = output_dir / "cyclic_components.png"
        try:
            plot_cyclic_components(
                idata,
                df,
                stan_data,
                output_path=str(components_path),
                show_trend_component=True,
                show_daily_component=True,
            )
            print(f"  ✓ Cyclic components plot saved to {components_path}")
        except Exception as e:
            print(f"  ⚠ Could not generate cyclic components plot: {e}")

        # 2. Spline daily pattern plot
        spline_daily_pattern_path = output_dir / "spline_daily_pattern.png"
        try:
            plot_spline_daily_pattern(
                idata,
                stan_data,
                output_path=str(spline_daily_pattern_path),
                n_hours_grid=100,
            )
            print(f"  ✓ Spline daily pattern plot saved to {spline_daily_pattern_path}")
        except ValueError as e:
            print(f"  ⚠ Could not generate spline daily pattern plot: {e}")

        # 3. Full expectation plot
        full_expectation_path = output_dir / "full_expectation.png"
        try:
            plot_model_full_expectation(
                idata,
                df,
                stan_data,
                output_path=str(full_expectation_path),
                model_name="Daily Spline",
            )
            print(f"  ✓ Full expectation plot saved to {full_expectation_path}")
        except Exception as e:
            print(f"  ⚠ Could not generate full expectation plot: {e}")

        # 4. Prediction plot (if prediction grid enabled)
        if predictions is not None:
            predictions_path = output_dir / "predictions.png"
            try:
                plot_model_predictions(
                    predictions=predictions,
                    df=df,
                    stan_data=stan_data,
                    model_name="Daily Spline",
                    output_path=str(predictions_path),
                    zoom_to=args.zoom_to,
                    zoom_reference_date=args.zoom_reference_date,
                )
                print(f"  ✓ Prediction plot saved to {predictions_path}")
            except Exception as e:
                print(f"  ⚠ Could not generate prediction plot: {e}")

            # 5. Hourly predictions plot (if multiple hours per day)
            unique_hours = np.unique(predictions["hour_of_day_pred"])
            if len(unique_hours) > 1:
                hourly_predictions_path = output_dir / "hourly_predictions.png"
                try:
                    plot_hourly_predictions(
                        predictions=predictions,
                        df=df,
                        stan_data=stan_data,
                        model_name="Daily Spline",
                        output_path=str(hourly_predictions_path),
                        show_all_days=True,
                    )
                    print(f"  ✓ Hourly predictions plot saved to {hourly_predictions_path}")
                except Exception as e:
                    print(f"  ⚠ Could not generate hourly predictions plot: {e}")

            # 6. Weekly zoomed predictions plot (if zoom range provided)
            if args.zoom_to is not None or (args.zoom_start and args.zoom_end):
                weekly_zoomed_path = output_dir / "weekly_zoomed_predictions.png"
                try:
                    # Determine target date
                    target_date = None
                    if args.zoom_start and args.zoom_end:
                        start_date = pd.Timestamp(args.zoom_start)
                        end_date = pd.Timestamp(args.zoom_end)
                        target_date = start_date + (end_date - start_date) / 2
                    elif args.zoom_to:
                        # Use middle of prediction range
                        if predictions is not None:
                            start_timestamp = df["timestamp"].min()
                            t_pred_days = predictions["t_pred"]
                            hour_of_day_pred = predictions.get("hour_of_day_pred")
                            if hour_of_day_pred is not None and len(hour_of_day_pred) == len(t_pred_days):
                                dates_pred = [start_timestamp + pd.Timedelta(days=float(d)) + pd.Timedelta(hours=float(h))
                                            for d, h in zip(t_pred_days, hour_of_day_pred)]
                            else:
                                dates_pred = [start_timestamp + pd.Timedelta(days=float(d)) for d in t_pred_days]
                            mid_idx = len(dates_pred) // 2
                            target_date = dates_pred[mid_idx]

                    if target_date is not None:
                        plot_weekly_zoomed_predictions(
                            predictions=predictions,
                            df=df,
                            stan_data=stan_data,
                            model_name="Daily Spline",
                            output_path=str(weekly_zoomed_path),
                            target_date=target_date,
                            show_ci=True,
                            show_observations=True,
                        )
                        print(f"  ✓ Weekly zoomed predictions plot saved to {weekly_zoomed_path}")
                except Exception as e:
                    print(f"  ⚠ Could not generate weekly zoomed predictions plot: {e}")

    # Write report
    report_path = write_model_report(
        output_dir=output_dir,
        model_name="daily",
        idata=idata,
        df=df,
        stan_data=stan_data,
        sigma=sigma,
        daily_amplitude=daily_amplitude,
        prop_variance_daily=prop_variance_daily,
        adapt_delta=args.adapt_delta,
        max_treedepth=args.max_treedepth,
        fourier_harmonics=args.fourier_harmonics,
        use_sparse=args.use_sparse,
        n_inducing_points=args.n_inducing_points,
        chains=args.chains,
        iter_warmup=args.iter_warmup,
        iter_sampling=args.iter_sampling,
        skip_plots=skip_plots,
        include_prediction_grid=args.include_prediction_grid,
        prediction_hour=args.prediction_hour,
        prediction_hour_step=args.prediction_hour_step,
        prediction_step_days=args.prediction_step_days,
        predictions=predictions,
    )

    # Extract all parameters for the return dictionary
    all_params = extract_all_parameters(idata, stan_data, "daily")

    return {
        'fit': fit,
        'idata': idata,
        'df': df,
        'stan_data': stan_data,
        'sigma': idata.posterior["sigma"].values.mean(),
        'sigma_lbs': sigma,
        'daily_amplitude': idata.posterior["daily_amplitude"].values.mean(),
        'daily_amplitude_lbs': daily_amplitude,
        'prop_variance_daily': prop_variance_daily,
        'predictions': predictions,
        'report_path': report_path,
        'all_params': all_params,  # Added for enhanced reporting
    }


def run_weekly_model(
    args,
    output_dir: Path,
    skip_plots: bool = False,
) -> Dict[str, Any]:
    """Run the weekly spline model (daily + weekly cycles).

    Returns dictionary with model results.
    """
    print("\n" + "=" * 70)
    print("FITTING WEEKLY MODEL (trend + daily + weekly Fourier spline)")
    print("=" * 70)

    # Sparse GP parameters
    sparse_params = {}
    if args.use_sparse:
        sparse_params = {
            "use_sparse": True,
            "n_inducing_points": args.n_inducing_points,
            "inducing_point_method": "uniform",
        }

    # Fit weekly model
    try:
        fit, idata, df, stan_data = fit_weight_model_spline_weekly(
            data_dir=args.data_dir,
            output_dir=output_dir,
            chains=args.chains,
            iter_warmup=args.iter_warmup,
            iter_sampling=args.iter_sampling,
            fourier_harmonics=args.fourier_harmonics,
            weekly_harmonics=args.weekly_harmonics,
            cache=True,
            force_refit=args.force_refit,
            include_prediction_grid=args.include_prediction_grid,
            prediction_hour=args.prediction_hour,
            prediction_hour_step=args.prediction_hour_step,
            prediction_step_days=args.prediction_step_days,
            **sparse_params,
        )
        print("✓ Weekly model fitted successfully")
    except Exception as e:
        print(f"✗ Error fitting weekly model: {e}")
        raise

    # Extract key metrics
    sigma = idata.posterior["sigma"].values.mean() * stan_data["_y_sd"]
    daily_amplitude = idata.posterior["daily_amplitude"].values.mean() * stan_data["_y_sd"]
    weekly_amplitude = idata.posterior["weekly_amplitude"].values.mean() * stan_data["_y_sd"]
    prop_variance_daily = idata.posterior["prop_variance_daily"].values.mean()
    prop_variance_weekly = idata.posterior["prop_variance_weekly"].values.mean()

    print("\nWeekly model summary:")
    print(f"  Measurement error (sigma): {sigma:.2f} lbs")
    print(f"  Daily amplitude: {daily_amplitude:.2f} lbs")
    print(f"  Weekly amplitude: {weekly_amplitude:.2f} lbs")
    print(f"  Proportion of variance from daily component: {prop_variance_daily:.3f}")
    print(f"  Proportion of variance from weekly component: {prop_variance_weekly:.3f}")

    # Extract predictions if prediction grid is enabled
    predictions = None
    if args.include_prediction_grid and stan_data.get('N_pred', 0) > 0:
        predictions = extract_predictions(idata, stan_data)
        print(f"  Extracted predictions for {len(predictions['t_pred'])} time points")

    # Generate plots
    if not skip_plots:
        print("\nGenerating weekly model visualizations...")

        # 1. Cyclic components plot
        components_path = output_dir / "cyclic_components.png"
        try:
            plot_cyclic_components(
                idata,
                df,
                stan_data,
                output_path=str(components_path),
                show_trend_component=True,
                show_daily_component=True,
                show_weekly_component=True,
            )
            print(f"  ✓ Cyclic components plot saved to {components_path}")
        except Exception as e:
            print(f"  ⚠ Could not generate cyclic components plot: {e}")

        # 2. Spline daily pattern plot
        spline_daily_pattern_path = output_dir / "spline_daily_pattern.png"
        try:
            plot_spline_daily_pattern(
                idata,
                stan_data,
                output_path=str(spline_daily_pattern_path),
                n_hours_grid=100,
            )
            print(f"  ✓ Spline daily pattern plot saved to {spline_daily_pattern_path}")
        except ValueError as e:
            print(f"  ⚠ Could not generate spline daily pattern plot: {e}")

        # 3. Full expectation plot
        full_expectation_path = output_dir / "full_expectation.png"
        try:
            plot_model_full_expectation(
                idata,
                df,
                stan_data,
                output_path=str(full_expectation_path),
                model_name="Weekly Spline",
            )
            print(f"  ✓ Full expectation plot saved to {full_expectation_path}")
        except Exception as e:
            print(f"  ⚠ Could not generate full expectation plot: {e}")

        # 4. Prediction plot (if prediction grid enabled)
        if predictions is not None:
            predictions_path = output_dir / "predictions.png"
            try:
                plot_model_predictions(
                    predictions=predictions,
                    df=df,
                    stan_data=stan_data,
                    model_name="Weekly Spline",
                    output_path=str(predictions_path),
                    zoom_to=args.zoom_to,
                    zoom_reference_date=args.zoom_reference_date,
                )
                print(f"  ✓ Prediction plot saved to {predictions_path}")
            except Exception as e:
                print(f"  ⚠ Could not generate prediction plot: {e}")

            # 5. Hourly predictions plot (if multiple hours per day)
            unique_hours = np.unique(predictions["hour_of_day_pred"])
            if len(unique_hours) > 1:
                hourly_predictions_path = output_dir / "hourly_predictions.png"
                try:
                    plot_hourly_predictions(
                        predictions=predictions,
                        df=df,
                        stan_data=stan_data,
                        model_name="Weekly Spline",
                        output_path=str(hourly_predictions_path),
                        show_all_days=True,
                    )
                    print(f"  ✓ Hourly predictions plot saved to {hourly_predictions_path}")
                except Exception as e:
                    print(f"  ⚠ Could not generate hourly predictions plot: {e}")

            # 6. Weekly zoomed predictions plot (if zoom range provided)
            if args.zoom_to is not None or (args.zoom_start and args.zoom_end):
                weekly_zoomed_path = output_dir / "weekly_zoomed_predictions.png"
                try:
                    # Determine target date
                    target_date = None
                    if args.zoom_start and args.zoom_end:
                        start_date = pd.Timestamp(args.zoom_start)
                        end_date = pd.Timestamp(args.zoom_end)
                        target_date = start_date + (end_date - start_date) / 2
                    elif args.zoom_to:
                        # Use middle of prediction range
                        if predictions is not None:
                            start_timestamp = df["timestamp"].min()
                            t_pred_days = predictions["t_pred"]
                            hour_of_day_pred = predictions.get("hour_of_day_pred")
                            if hour_of_day_pred is not None and len(hour_of_day_pred) == len(t_pred_days):
                                dates_pred = [start_timestamp + pd.Timedelta(days=float(d)) + pd.Timedelta(hours=float(h))
                                            for d, h in zip(t_pred_days, hour_of_day_pred)]
                            else:
                                dates_pred = [start_timestamp + pd.Timedelta(days=float(d)) for d in t_pred_days]
                            mid_idx = len(dates_pred) // 2
                            target_date = dates_pred[mid_idx]

                    if target_date is not None:
                        plot_weekly_zoomed_predictions(
                            predictions=predictions,
                            df=df,
                            stan_data=stan_data,
                            model_name="Weekly Spline",
                            output_path=str(weekly_zoomed_path),
                            target_date=target_date,
                            show_ci=True,
                            show_observations=True,
                        )
                        print(f"  ✓ Weekly zoomed predictions plot saved to {weekly_zoomed_path}")
                except Exception as e:
                    print(f"  ⚠ Could not generate weekly zoomed predictions plot: {e}")

    # Write report
    report_path = write_model_report(
        output_dir=output_dir,
        model_name="weekly",
        idata=idata,
        df=df,
        stan_data=stan_data,
        sigma=sigma,
        daily_amplitude=daily_amplitude,
        prop_variance_daily=prop_variance_daily,
        weekly_amplitude=weekly_amplitude,
        prop_variance_weekly=prop_variance_weekly,
        adapt_delta=args.adapt_delta,
        max_treedepth=args.max_treedepth,
        fourier_harmonics=args.fourier_harmonics,
        weekly_harmonics=args.weekly_harmonics,
        use_sparse=args.use_sparse,
        n_inducing_points=args.n_inducing_points,
        chains=args.chains,
        iter_warmup=args.iter_warmup,
        iter_sampling=args.iter_sampling,
        skip_plots=skip_plots,
        include_prediction_grid=args.include_prediction_grid,
        prediction_hour=args.prediction_hour,
        prediction_hour_step=args.prediction_hour_step,
        prediction_step_days=args.prediction_step_days,
        predictions=predictions,
    )

    # Extract all parameters for the return dictionary
    all_params = extract_all_parameters(idata, stan_data, "weekly")

    return {
        'fit': fit,
        'idata': idata,
        'df': df,
        'stan_data': stan_data,
        'sigma': idata.posterior["sigma"].values.mean(),
        'sigma_lbs': sigma,
        'daily_amplitude': idata.posterior["daily_amplitude"].values.mean(),
        'daily_amplitude_lbs': daily_amplitude,
        'weekly_amplitude': idata.posterior["weekly_amplitude"].values.mean(),
        'weekly_amplitude_lbs': weekly_amplitude,
        'prop_variance_daily': prop_variance_daily,
        'prop_variance_weekly': prop_variance_weekly,
        'predictions': predictions,
        'report_path': report_path,
        'all_params': all_params,  # Added for enhanced reporting
    }


def main():
    parser = argparse.ArgumentParser(
        description="General demo script for spline models with daily and/or weekly cycles",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        choices=["daily", "weekly", "both"],
        default="daily",
        help="Which model to run: 'daily' (trend+daily), 'weekly' (trend+daily+weekly), or 'both' (compare)",
    )

    # Data and output
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Path to data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/demo_general",
        help="Directory for output plots and reports",
    )

    # MCMC parameters
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
        default=1500,
        help="Sampling iterations per chain",
    )
    parser.add_argument(
        "--adapt-delta",
        type=float,
        default=0.99,
        help="Adapt delta parameter for NUTS sampler",
    )
    parser.add_argument(
        "--max-treedepth",
        type=int,
        default=12,
        help="Maximum tree depth for NUTS sampler",
    )

    # Model hyperparameters
    parser.add_argument(
        "--fourier-harmonics",
        type=int,
        default=2,
        help="Number of Fourier harmonics (K parameter) for daily cycles",
    )
    parser.add_argument(
        "--weekly-harmonics",
        type=int,
        default=2,
        help="Number of Fourier harmonics (L parameter) for weekly cycles (weekly model only)",
    )

    # Sparse GP
    parser.add_argument(
        "--no-sparse",
        action="store_true",
        help="Disable sparse GP approximation (sparse GP is default)",
    )
    parser.add_argument(
        "--n-inducing-points",
        type=int,
        default=50,
        help="Number of inducing points for sparse GP (unless --no-sparse)",
    )

    # Caching
    parser.add_argument(
        "--force-refit",
        action="store_true",
        help="Force refit even if cached results exist",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching (force refit)",
    )

    # Visualization
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip generating plots (only print summary)",
    )

    # Prediction grid
    parser.add_argument(
        "--include-prediction-grid",
        action="store_true",
        help="Include prediction grid for unobserved days (generates predictions at regular intervals)",
    )
    parser.add_argument(
        "--prediction-hour",
        type=float,
        default=8.0,
        help="Hour of day (0-24) for prediction points (default 8.0 = 8 AM)",
    )
    parser.add_argument(
        "--prediction-hour-step",
        type=float,
        default=None,
        help="Step size in hours for multiple prediction hours per day (default None = single hour)",
    )
    parser.add_argument(
        "--prediction-step-days",
        type=int,
        default=1,
        help="Step size in days for prediction grid (default 1 = daily)",
    )

    # Zoom arguments
    parser.add_argument(
        "--zoom-to",
        type=str,
        choices=["last_week", "last_month", "last_year", "all"],
        default=None,
        help="Zoom to preset date range for prediction plot",
    )
    parser.add_argument(
        "--zoom-reference-date",
        type=str,
        default=None,
        help="Reference date for zoom presets (YYYY-MM-DD). Defaults to latest prediction date.",
    )
    parser.add_argument(
        "--zoom-start",
        type=str,
        default=None,
        help="Start date for custom zoom range (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--zoom-end",
        type=str,
        default=None,
        help="End date for custom zoom range (YYYY-MM-DD)",
    )

    args = parser.parse_args()
    use_sparse = not args.no_sparse
    args.use_sparse = use_sparse

    # Validate arguments
    if args.model == "daily" and args.weekly_harmonics != 1:
        print("⚠ Warning: --weekly-harmonics ignored for daily-only model")

    # Process zoom arguments (similar to demo_spline_single)
    zoom_to = None
    zoom_reference_date = None

    if args.zoom_start and args.zoom_end:
        try:
            start = pd.Timestamp(args.zoom_start)
            end = pd.Timestamp(args.zoom_end)
            zoom_to = (start, end)
            print(f"Zoom: custom range {start.date()} to {end.date()}")
        except Exception as e:
            print(f"Warning: invalid zoom dates: {e}. Ignoring zoom.")
    elif args.zoom_to:
        zoom_to = args.zoom_to
        if args.zoom_reference_date:
            try:
                zoom_reference_date = pd.Timestamp(args.zoom_reference_date)
                print(f"Zoom: preset '{zoom_to}' with reference {zoom_reference_date.date()}")
            except Exception as e:
                print(f"Warning: invalid reference date: {e}. Using default.")
        else:
            print(f"Zoom: preset '{zoom_to}' (default reference)")

    # Store zoom arguments for later use
    args.zoom_to = zoom_to
    args.zoom_reference_date = zoom_reference_date

    # Cache setting
    if args.no_cache:
        args.force_refit = True
        print("Note: --no-cache implies --force-refit")

    # Create output directory structure
    base_output_dir = Path(args.output_dir)

    if args.model == "both":
        # Create separate directories for each model and comparison
        daily_dir = base_output_dir / "daily"
        weekly_dir = base_output_dir / "weekly"
        comparison_dir = base_output_dir / "comparison"

        daily_dir.mkdir(parents=True, exist_ok=True)
        weekly_dir.mkdir(parents=True, exist_ok=True)
        comparison_dir.mkdir(parents=True, exist_ok=True)

        print("\nOutput directories:")
        print(f"  Daily model: {daily_dir}")
        print(f"  Weekly model: {weekly_dir}")
        print(f"  Comparison: {comparison_dir}")
    else:
        # Single model - use base directory
        base_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nOutput directory: {base_output_dir}")

    print("\n" + "=" * 70)
    print("GENERAL DEMO SCRIPT - SPLINE MODELS WITH DAILY/WEEKLY CYCLES")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Chains: {args.chains}, Warmup: {args.iter_warmup}, Sampling: {args.iter_sampling}")
    print(f"Sampler: adapt_delta={args.adapt_delta}, max_treedepth={args.max_treedepth}")
    print(f"Fourier harmonics (daily): K={args.fourier_harmonics}")
    if args.model in ["weekly", "both"]:
        print(f"Weekly harmonics: L={args.weekly_harmonics}")
    print(f"Caching: {not args.no_cache}, Force refit: {args.force_refit}")
    if use_sparse:
        print(f"Sparse GP: Enabled ({args.n_inducing_points} inducing points)")
    else:
        print("Sparse GP: Disabled (full GP)")
    print()

    # Run selected model(s)
    if args.model == "daily":
        results = run_daily_model(args, base_output_dir, skip_plots=args.skip_plots)

        print("\n" + "=" * 70)
        print("DEMO COMPLETE - DAILY MODEL")
        print("=" * 70)
        print("✓ Model fitted successfully")
        print(f"✓ Output directory: {base_output_dir}")
        print(f"✓ Report: {results['report_path']}")

    elif args.model == "weekly":
        results = run_weekly_model(args, base_output_dir, skip_plots=args.skip_plots)

        print("\n" + "=" * 70)
        print("DEMO COMPLETE - WEEKLY MODEL")
        print("=" * 70)
        print("✓ Model fitted successfully")
        print(f"✓ Output directory: {base_output_dir}")
        print(f"✓ Report: {results['report_path']}")

    else:  # both
        print("\n" + "=" * 70)
        print("RUNNING BOTH MODELS FOR COMPARISON")
        print("=" * 70)

        # Run daily model
        daily_results = run_daily_model(args, daily_dir, skip_plots=args.skip_plots)

        # Run weekly model
        weekly_results = run_weekly_model(args, weekly_dir, skip_plots=args.skip_plots)

        print("\n" + "=" * 70)
        print("COMPARING MODELS")
        print("=" * 70)

        # Compare models using WAIC/LOO
        comparison_df = None
        try:
            idata_dict = {
                "daily": daily_results['idata'],
                "weekly": weekly_results['idata']
            }

            waic_results = {}
            loo_results = {}
            for name, idata in idata_dict.items():
                try:
                    waic = az.waic(idata)
                    waic_results[name] = waic
                except Exception as e:
                    print(f"⚠ WAIC computation failed for {name}: {e}")
                    waic_results[name] = None
                try:
                    loo = az.loo(idata)
                    loo_results[name] = loo
                except Exception as e:
                    print(f"⚠ LOO computation failed for {name}: {e}")
                    loo_results[name] = None

            # Create comparison DataFrame
            rows = []
            for name in idata_dict.keys():
                waic = waic_results.get(name)
                loo = loo_results.get(name)
                row = {
                    "model": name,
                    "waic": -2 * waic.elpd_waic if waic is not None else None,
                    "waic_se": 2 * waic.se if waic is not None else None,
                    "p_waic": waic.p_waic if waic is not None else None,
                    "loo": -2 * loo.elpd_loo if loo is not None else None,
                    "loo_se": 2 * loo.se if loo is not None else None,
                    "p_loo": loo.p_loo if loo is not None else None,
                }
                rows.append(row)
            df = pd.DataFrame(rows).set_index("model")

            # Compute model weights
            def compute_weights(values):
                valid_mask = pd.notna(values)
                valid_values = values[valid_mask]
                if len(valid_values) == 0:
                    return pd.Series([None] * len(values), index=values.index)
                min_val = valid_values.min()
                deltas = valid_values - min_val
                weights = np.exp(-0.5 * deltas)
                weights = weights / weights.sum()
                full_weights = pd.Series([None] * len(values), index=values.index)
                full_weights[valid_mask] = weights
                return full_weights

            if df["waic"].notna().any():
                df["waic_weight"] = compute_weights(df["waic"])
            if df["loo"].notna().any():
                df["loo_weight"] = compute_weights(df["loo"])

            comparison_df = df

            print_summary = True
            if print_summary:
                print("\nWAIC/LOO Model Comparison:")
                print(comparison_df.to_string(float_format=lambda x: f"{x:.1f}" if pd.notna(x) else "NaN"))

        except Exception as e:
            print(f"⚠ Could not compute WAIC/LOO comparison: {e}")

        # Compare using sigma and amplitude metrics
        comparison_metrics = None
        try:
            # Note: compare_models_all expects original and cyclic models, not our models
            # We'll create our own comparison
            sigma_daily = daily_results['sigma']
            sigma_weekly = weekly_results['sigma']
            sigma_reduction = sigma_daily - sigma_weekly
            sigma_reduction_pct = 100 * sigma_reduction / sigma_daily if sigma_daily > 0 else 0

            comparison_metrics = {
                'sigma_daily': sigma_daily,
                'sigma_weekly': sigma_weekly,
                'sigma_reduction': sigma_reduction,
                'sigma_reduction_pct': sigma_reduction_pct,
                'daily_amplitude_daily': daily_results['daily_amplitude'],
                'daily_amplitude_weekly': weekly_results['daily_amplitude'],
                'weekly_amplitude': weekly_results['weekly_amplitude'],
                'prop_variance_daily_daily': daily_results['prop_variance_daily'],
                'prop_variance_daily_weekly': weekly_results['prop_variance_daily'],
                'prop_variance_weekly': weekly_results['prop_variance_weekly'],
            }

            print("\nParameter comparison:")
            print(f"  Sigma reduction: {sigma_reduction:.4f} ({sigma_reduction_pct:.1f}%)")
            print(f"  Daily amplitude difference: {weekly_results['daily_amplitude'] - daily_results['daily_amplitude']:.4f}")
            print(f"  Weekly amplitude: {weekly_results['weekly_amplitude']:.4f}")

        except Exception as e:
            print(f"⚠ Could not compute parameter comparison: {e}")
            comparison_metrics = {}

        # Create comparison plots
        print("\nGenerating comparison plots...")
        create_comparison_plots(
            output_dir=comparison_dir,
            daily_results=daily_results,
            weekly_results=weekly_results,
            daily_predictions=daily_results.get('predictions'),
            weekly_predictions=weekly_results.get('predictions'),
        )

        # Write comparison report
        if comparison_df is not None and comparison_metrics is not None:
            write_comparison_report(
                output_dir=comparison_dir,
                comparison_df=comparison_df,
                daily_results=daily_results,
                weekly_results=weekly_results,
                comparison_metrics=comparison_metrics,
                adapt_delta=args.adapt_delta,
                max_treedepth=args.max_treedepth,
                fourier_harmonics=args.fourier_harmonics,
                weekly_harmonics=args.weekly_harmonics,
                use_sparse=args.use_sparse,
                n_inducing_points=args.n_inducing_points,
                chains=args.chains,
                iter_warmup=args.iter_warmup,
                iter_sampling=args.iter_sampling,
            )

        print("\n" + "=" * 70)
        print("COMPARISON COMPLETE")
        print("=" * 70)
        print("✓ Both models fitted successfully")
        print(f"✓ Daily model outputs: {daily_dir}")
        print(f"✓ Weekly model outputs: {weekly_dir}")
        print(f"✓ Comparison outputs: {comparison_dir}")
        if comparison_df is not None:
            print("✓ Model comparison computed (WAIC/LOO)")
        print("\nTo explore results:")
        print("  - Check individual model reports in daily/ and weekly/ subdirectories")
        print("  - View comparison plots and report in comparison/")

    # Show plots if not saved only
    if not args.skip_plots:
        print("\nDisplaying plots (close windows to exit)...")
        if should_show_plots():
            plt.show()
        else:
            print("Plots saved to disk (interactive display not available)")


if __name__ == "__main__":
    main()