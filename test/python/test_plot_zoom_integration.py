"""Integration test for zoom functionality in plot_model_predictions."""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.fit_weight import fit_weight_model_spline_optimized, extract_predictions
from src.models.plot_cyclic import plot_model_predictions


def test_plot_model_predictions_with_zoom():
    """Test that zoom_to parameter works in plot_model_predictions."""
    print("Testing plot_model_predictions with zoom_to parameter...")

    # Fit model with minimal settings for speed
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
    predictions = extract_predictions(idata, stan_data)

    # Test 1: No zoom (default behavior)
    fig1 = plot_model_predictions(
        predictions=predictions,
        df=df,
        stan_data=stan_data,
        model_name="Spline Optimized",
        output_path=None,
        show_observations=True,
        show_ci=True,
        zoom_to=None,
    )
    # Should create figure without error
    assert fig1 is not None
    ax1 = fig1.axes[0]
    xlim1 = ax1.get_xlim()
    print(f"  Default xlim: {xlim1}")
    plt.close(fig1)

    # Test 2: Zoom to preset 'last_week'
    fig2 = plot_model_predictions(
        predictions=predictions,
        df=df,
        stan_data=stan_data,
        model_name="Spline Optimized",
        output_path=None,
        zoom_to='last_week',
    )
    ax2 = fig2.axes[0]
    xlim2 = ax2.get_xlim()
    print(f"  Zoom 'last_week' xlim: {xlim2}")
    # Check that xlim changed and range is approximately 7 days
    # (matplotlib date numbers, 1 unit = 1 day)
    range_width = xlim2[1] - xlim2[0]
    assert abs(range_width - 7) < 0.1, f"Expected ~7 day range, got {range_width}"
    plt.close(fig2)

    # Test 3: Zoom to preset 'last_month'
    fig3 = plot_model_predictions(
        predictions=predictions,
        df=df,
        stan_data=stan_data,
        model_name="Spline Optimized",
        output_path=None,
        zoom_to='last_month',
    )
    ax3 = fig3.axes[0]
    xlim3 = ax3.get_xlim()
    range_width = xlim3[1] - xlim3[0]
    assert abs(range_width - 30) < 0.1, f"Expected ~30 day range, got {range_width}"
    plt.close(fig3)

    # Test 4: Zoom to specific date range via tuple
    # Get date range from predictions
    start_timestamp = df["timestamp"].min()
    t_pred_days = predictions["t_pred"]
    dates_pred = [start_timestamp + pd.Timedelta(days=float(d)) for d in t_pred_days]
    # Pick middle week
    mid_idx = len(dates_pred) // 2
    start_date = dates_pred[mid_idx]
    end_date = start_date + pd.Timedelta(days=7)

    fig4 = plot_model_predictions(
        predictions=predictions,
        df=df,
        stan_data=stan_data,
        model_name="Spline Optimized",
        output_path=None,
        zoom_to=(start_date, end_date),
    )
    ax4 = fig4.axes[0]
    xlim4 = ax4.get_xlim()
    # Convert back to dates for comparison
    from matplotlib.dates import num2date
    lim_start = num2date(xlim4[0])
    lim_end = num2date(xlim4[1])
    # Check that limits match requested dates (within 1 day due to time of day)
    assert abs((lim_start.date() - start_date.date()).days) <= 1
    assert abs((lim_end.date() - end_date.date()).days) <= 1
    plt.close(fig4)

    print("✓ All zoom integration tests passed")


if __name__ == "__main__":
    test_plot_model_predictions_with_zoom()
    print("\nZoom integration test completed successfully!")