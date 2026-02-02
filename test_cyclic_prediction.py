import sys
sys.path.insert(0, 'src')

from models.fit_weight import fit_weight_model_cyclic_optimized, extract_predictions

print("Testing cyclic optimized model with prediction grid...")
result = fit_weight_model_cyclic_optimized(
    include_prediction_grid=True,
    prediction_hour=8.0,
    prediction_step_days=1,
    use_sparse=True,
    n_inducing_points=5,
    chains=1,
    iter_warmup=10,
    iter_sampling=10,
    force_refit=True,
    cache=False,
)
fit, idata, df, stan_data = result
print(f"Fit successful. N_pred = {stan_data.get('N_pred', 0)}")

# Extract predictions
predictions = extract_predictions(idata, stan_data)
if predictions:
    print(f"Predictions extracted. Shape f_pred_mean: {predictions['f_pred_mean'].shape}")
    print(f"Range: {predictions['f_pred_mean'].min():.2f} - {predictions['f_pred_mean'].max():.2f} lbs")
else:
    print("No predictions extracted")
    sys.exit(1)

print("Test passed.")