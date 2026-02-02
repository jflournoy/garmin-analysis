#!/usr/bin/env python3
"""Quick test of prediction extraction for flexible optimized model."""
import sys
sys.path.insert(0, '.')

from src.models.fit_weight import fit_weight_model_flexible_optimized, extract_predictions
from src.data.weight import load_weight_data, prepare_stan_data

# Load data
df = load_weight_data("data")
stan_data = prepare_stan_data(df, include_hour_info=False, use_sparse=False, include_prediction_grid=True, prediction_hour=8.0, prediction_step_days=1)
print(f"N_pred: {stan_data.get('N_pred', 0)}")
if stan_data.get('N_pred', 0) > 0:
    print(f"t_pred shape: {len(stan_data['t_pred'])}")
    # hour_of_day_pred not used in flexible model but still generated
    print(f"hour_of_day_pred shape: {len(stan_data['hour_of_day_pred'])}")

# Fit model with minimal iterations
print("\nFitting flexible optimized model...")
fit, idata, df, stan_data = fit_weight_model_flexible_optimized(
    data_dir="data",
    chains=1,
    iter_warmup=10,
    iter_sampling=10,
    alpha_prior_sd=1.0,
    rho_prior_shape=5.0,
    rho_prior_scale=1.0,
    sigma_prior_sd=0.5,
    use_sparse=False,
    n_inducing_points=50,
    inducing_point_method="uniform",
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
    # Check that predictions are within plausible range (roughly 120-150 lbs)
    assert preds['f_pred_mean'].min() > 120 and preds['f_pred_mean'].max() < 150, "Predictions outside plausible range"
    print("✓ Predictions within plausible range")
else:
    print("No predictions extracted")
    sys.exit(1)

print("\n✅ Flexible optimized prediction test passed!")