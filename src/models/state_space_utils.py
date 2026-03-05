"""Utilities for state-space model analysis and interpolation."""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
import xarray as xr


def compute_fitness_expectation(
    alpha: float,
    beta: float,
    psi: float,
    intensity: np.ndarray,
    initial_fitness: float = 0.0,
    initial_impulse: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute deterministic fitness and impulse expectations given parameters.

    Computes the deterministic evolution of the state-space model:
    - impulse[t] = psi * impulse[t-1] + intensity[t]
    - fitness[t] = alpha * fitness[t-1] + beta * impulse[t-1]

    Args:
        alpha: Fitness persistence parameter (0 < alpha < 1)
        beta: Fitness gain per unit impulse
        psi: Impulse decay parameter (0 < psi < 1)
        intensity: Array of daily intensity values (length D)
        initial_fitness: Initial fitness value (default: 0)
        initial_impulse: Initial impulse value (default: intensity[0])

    Returns:
        Tuple of (fitness, impulse) arrays, each of length D
    """
    D = len(intensity)
    impulse = np.zeros(D)
    fitness = np.zeros(D)

    # Initialize
    impulse[0] = initial_impulse if initial_impulse is not None else intensity[0]
    fitness[0] = initial_fitness

    # Evolve states
    for t in range(1, D):
        impulse[t] = psi * impulse[t-1] + intensity[t]
        fitness[t] = alpha * fitness[t-1] + beta * impulse[t-1]

    return fitness, impulse


def interpolate_fitness_to_timestamps(
    fitness_daily: np.ndarray,
    date_range: pd.DatetimeIndex,
    target_timestamps: pd.DatetimeIndex,
    method: str = 'nearest'
) -> np.ndarray:
    """Interpolate daily fitness values to arbitrary timestamps.

    Args:
        fitness_daily: Array of fitness values for each day (length D)
        date_range: DatetimeIndex corresponding to fitness_daily (length D)
        target_timestamps: Timestamps to interpolate fitness to
        method: Interpolation method:
            - 'nearest': Use fitness from nearest day
            - 'linear': Linear interpolation between days
            - 'previous': Use fitness from previous day

    Returns:
        Array of fitness values at target_timestamps
    """
    if len(fitness_daily) != len(date_range):
        raise ValueError(f"fitness_daily length ({len(fitness_daily)}) must match date_range length ({len(date_range)})")

    # Convert to numeric for interpolation
    date_numeric = date_range.astype(np.int64) // 10**9  # seconds since epoch
    target_numeric = target_timestamps.astype(np.int64) // 10**9

    if method == 'nearest':
        # Find nearest day for each timestamp
        indices = np.abs(date_numeric.values[:, None] - target_numeric.values[None, :]).argmin(axis=0)
        return fitness_daily[indices]

    elif method == 'linear':
        # Linear interpolation between days
        from scipy import interpolate
        interp_func = interpolate.interp1d(
            date_numeric, fitness_daily,
            kind='linear',
            bounds_error=False,
            fill_value=(fitness_daily[0], fitness_daily[-1])
        )
        return interp_func(target_numeric)

    elif method == 'previous':
        # Use fitness from previous day (floor)
        indices = np.searchsorted(date_numeric, target_numeric, side='right') - 1
        indices = np.clip(indices, 0, len(fitness_daily) - 1)
        return fitness_daily[indices]

    else:
        raise ValueError(f"Unknown interpolation method: {method}")


def predict_weight_components(
    idata: xr.Dataset,
    stan_data: Dict[str, Any],
    target_timestamps: pd.DatetimeIndex,
    target_hours: Optional[np.ndarray] = None,
    include_ci: bool = False
) -> Dict[str, Any]:
    """Predict weight components for arbitrary timestamps using fitted state-space model.

    Computes predictions for:
    1. Fitness effect (gamma * fitness)
    2. GP trend component
    3. Daily spline component (if available)
    4. Total prediction

    Args:
        idata: ArviZ InferenceData from fitted state-space model
        stan_data: Stan data dictionary used for fitting
        target_timestamps: Timestamps to predict at
        target_hours: Optional hour-of-day values (0-24) for each timestamp
        include_ci: Whether to compute credible intervals

    Returns:
        Dictionary with predictions for each component
    """
    # Extract posterior samples
    posterior = idata.posterior

    # Get model parameters
    alpha_samples = posterior['alpha'].values if 'alpha' in posterior else None
    beta_samples = posterior['beta'].values if 'beta' in posterior else None
    psi_samples = posterior['psi'].values if 'psi' in posterior else None
    gamma_samples = posterior['gamma'].values if 'gamma' in posterior else None

    if any(s is None for s in [alpha_samples, beta_samples, psi_samples, gamma_samples]):
        raise ValueError("Missing required state-space parameters in posterior")

    # Get scaling parameters
    y_mean = stan_data.get('_y_mean', 0.0)
    y_sd = stan_data.get('_y_sd', 1.0)
    intensity_mean = stan_data.get('intensity_mean', 0.0)
    intensity_std = stan_data.get('intensity_std', 1.0)

    # Get intensity data
    intensity_standardized = stan_data.get('intensity', None)
    if intensity_standardized is None:
        raise ValueError("Missing intensity data in stan_data")

    # Unstandardize intensity
    intensity = intensity_standardized * intensity_std + intensity_mean

    # Create date range matching intensity data
    D = len(intensity)

    # Try to get start date from stan_data, fall back to default
    start_date_str = stan_data.get('_start_date', None)
    if start_date_str:
        start_date = pd.Timestamp(start_date_str)
    else:
        # If not available, use first weight observation date
        # This is a fallback - ideally _start_date should be in stan_data
        print("Warning: _start_date not found in stan_data, using default")
        start_date = pd.Timestamp('2020-01-01')

    date_range = pd.date_range(start=start_date, periods=D, freq='D')

    # Compute fitness expectations for each posterior sample
    n_chains, n_draws = alpha_samples.shape
    n_target = len(target_timestamps)

    # Initialize arrays
    fitness_effect_samples = np.zeros((n_chains, n_draws, n_target))
    gp_samples = np.zeros((n_chains, n_draws, n_target))
    daily_samples = np.zeros((n_chains, n_draws, n_target))

    # For each posterior sample, compute predictions
    for chain in range(n_chains):
        for draw in range(n_draws):
            # Compute fitness and impulse for this parameter set
            fitness_daily, _ = compute_fitness_expectation(
                alpha=alpha_samples[chain, draw],
                beta=beta_samples[chain, draw],
                psi=psi_samples[chain, draw],
                intensity=intensity_standardized
            )

            # Interpolate fitness to target timestamps
            fitness_target = interpolate_fitness_to_timestamps(
                fitness_daily, date_range, target_timestamps, method='nearest'
            )

            # Fitness effect
            fitness_effect_samples[chain, draw, :] = gamma_samples[chain, draw] * fitness_target

            # TODO: Add GP and daily component interpolation
            # For now, set to 0 (would need more complex interpolation)
            gp_samples[chain, draw, :] = 0.0
            daily_samples[chain, draw, :] = 0.0

    # Compute means and optionally credible intervals
    result = {
        'fitness_effect': {
            'mean': fitness_effect_samples.mean(axis=(0, 1)) * y_sd + y_mean
        },
        'gp_trend': {
            'mean': gp_samples.mean(axis=(0, 1)) * y_sd + y_mean
        },
        'daily_spline': {
            'mean': daily_samples.mean(axis=(0, 1)) * y_sd + y_mean
        }
    }

    # Total prediction
    total_mean = (result['fitness_effect']['mean'] +
                  result['gp_trend']['mean'] +
                  result['daily_spline']['mean'])
    result['total'] = {'mean': total_mean}

    if include_ci:
        # Compute 95% credible intervals
        for key, samples in [('fitness_effect', fitness_effect_samples),
                            ('gp_trend', gp_samples),
                            ('daily_spline', daily_samples)]:
            lower = np.percentile(samples, 2.5, axis=(0, 1)) * y_sd + y_mean
            upper = np.percentile(samples, 97.5, axis=(0, 1)) * y_sd + y_mean
            result[key]['lower'] = lower
            result[key]['upper'] = upper

        # Total CI (assuming independence, approximate)
        total_samples = fitness_effect_samples + gp_samples + daily_samples
        total_lower = np.percentile(total_samples, 2.5, axis=(0, 1)) * y_sd + y_mean
        total_upper = np.percentile(total_samples, 97.5, axis=(0, 1)) * y_sd + y_mean
        result['total']['lower'] = total_lower
        result['total']['upper'] = total_upper

    return result


def create_prediction_dataframe(
    predictions: Dict[str, Any],
    timestamps: pd.DatetimeIndex,
    include_ci: bool = False
) -> pd.DataFrame:
    """Convert prediction dictionary to DataFrame for easy analysis.

    Args:
        predictions: Dictionary from predict_weight_components
        timestamps: Timestamps corresponding to predictions
        include_ci: Whether predictions include credible intervals

    Returns:
        DataFrame with columns for each component
    """
    data = {
        'timestamp': timestamps,
        'total_prediction': predictions['total']['mean']
    }

    # Add component means
    for component in ['fitness_effect', 'gp_trend', 'daily_spline']:
        if component in predictions:
            data[f'{component}_mean'] = predictions[component]['mean']

    # Add credible intervals if available
    if include_ci:
        data['total_lower'] = predictions['total']['lower']
        data['total_upper'] = predictions['total']['upper']

        for component in ['fitness_effect', 'gp_trend', 'daily_spline']:
            if component in predictions and 'lower' in predictions[component]:
                data[f'{component}_lower'] = predictions[component]['lower']
                data[f'{component}_upper'] = predictions[component]['upper']

    return pd.DataFrame(data)