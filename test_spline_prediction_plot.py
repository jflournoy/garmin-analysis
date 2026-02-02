import sys
sys.path.insert(0, 'src')

from models.fit_weight import fit_weight_model_spline_optimized, extract_predictions
from models.plot_cyclic import plot_model_predictions

print("Testing spline optimized model predictions visualization...")
result = fit_weight_model_spline_optimized(
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
if not predictions:
    print("No predictions extracted")
    sys.exit(1)

print(f"Predictions extracted. Shape f_pred_mean: {predictions['f_pred_mean'].shape}")

# Plot predictions
fig = plot_model_predictions(
    predictions=predictions,
    df=df,
    stan_data=stan_data,
    model_name="Spline Optimized",
    output_path="output/spline_predictions.png",
    show_observations=True,
    show_ci=True,
)

print("Plot saved to output/spline_predictions.png")
# Optionally show plot
# plt.show()

print("Test passed.")