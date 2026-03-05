# Component Predictions Summary

Generated: 2026-03-04 21:31:00

## Model Information

- **Model**: `weight_state_space_training_decay_aerobic_symmetric_student_ar_spline_constrained.stan`
- **AR(1) component**: Excluded from predictions (measurement-time specific)
- **Prediction intervals**: 4-hour intervals (0, 4, 8, 12, 16, 20, 24 hours)
- **Time range**: 2023-07-12 to 2026-01-20 (924 days)

## Components Predicted

1. **Intercept**: Baseline weight level
2. **Strength Fitness Component**: `γ_s × strength_fitness[t]`
3. **Aerobic Fitness Component**: `γ_a × aerobic_fitness[t]`
4. **Daily Spline Component**: Fourier basis for intraday variations
5. **Total Prediction**: Sum of all components (intercept + strength + aerobic + spline)
6. **Full Model Prediction**: Direct output from Stan (for validation)

## Component Magnitudes (Mean Absolute Values)

- **Intercept**: 1.8680
- **Strength**: 1.5798
- **Aerobic**: 0.0415
- **Spline**: 0.1534
- **Total**: 0.9845

## Data Files Generated

### CSV Files
1. **Per-hour files**: `component_predictions_hour_XX.csv` (XX = 00, 04, 08, 12, 16, 20, 24)
   - Columns: date, hour, plus `{component}_mean`, `{component}_lower`, `{component}_upper`
2. **Combined file**: `all_component_predictions.csv`
   - Long format with all hours and days
   - Additional columns: day_index, hour_index

### Visualization Files
1. `component_time_series_noon.png` - Time series of each component at 12:00
2. `total_predictions_heatmap.png` - Heatmap of total predictions over time and hours
3. `component_contributions_sample_dates.png` - Component contributions at sample dates
4. `daily_patterns_analysis.png` - Daily patterns and variability

## Usage Notes

1. **Predictions are in standardized units** (mean=0, std=1 of original weight data)
2. **AR(1) component is excluded** as it models measurement-time residual correlation
3. **Credible intervals (95%)** provided for uncertainty quantification
4. **Components are additive**: Total = Intercept + Strength + Aerobic + Spline

## Key Statistics

- **Total prediction range**: 3.729 (min: -2.202, max: 1.526)
- **Average intercept**: -1.868
- **Average strength contribution**: 1.580
- **Average aerobic contribution**: -0.042
- **Average spline amplitude**: 0.153
- **Average daily spline range**: 0.456
