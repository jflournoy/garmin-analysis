#!/usr/bin/env python3
"""Quick test of prediction extraction."""
import sys
sys.path.insert(0, '.')

from src.models.fit_weight import fit_weight_model_spline_optimized, extract_predictions
from src.data.weight import load_weight_data, prepare_stan_data

# Load data
df = load_weight_data("data")
stan_data = prepare_stan_data(df, fourier_harmonics=2, use_sparse=False, include_prediction_grid=True, prediction_hour=8.0, prediction_step_days=1)
print(f"N_pred: {stan_data.get('N_pred', 0)}")
if stan_data.get('N_pred', 0) > 0:
    print(f"t_pred shape: {len(stan_data['t_pred'])}")
    print(f"hour_of_day_pred shape: {len(stan_data['hour_of_day_pred'])}")

# Fit model with minimal iterations
print("\nFitting model...")
fit, idata, df, stan_data = fit_weight_model_spline_optimized(
    data_dir="data",
    chains=1,
    iter_warmup=10,
    iter_sampling=10,
    fourier_harmonics=2,
    use_sparse=False,
    n_inducing_points=50,
    include_prediction_grid=True,
    prediction_hour=8.0,
    prediction_step_days=1,
    cache=False,
    force_refit=True
)

# Extract predictions
preds = extract_predictions(idata, stan_data)
if preds:
    print("\nPredictions extracted:")
    print(f"t_pred shape: {preds['t_pred'].shape}")
    print(f"f_pred_mean shape: {preds['f_pred_mean'].shape}")
    print(f"f_pred_mean range: {preds['f_pred_mean'].min():.2f} - {preds['f_pred_mean'].max():.2f} lbs")
    print(f"y_pred_mean range: {preds['y_pred_mean'].min():.2f} - {preds['y_pred_mean'].max():.2f} lbs")
else:
    print("No predictions extracted")