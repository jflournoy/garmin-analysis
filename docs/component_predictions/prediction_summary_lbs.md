# Component Predictions Summary

Generated: 2026-03-04 21:31:03

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

- **Intercept**: 130.0933
- **Strength**: 139.6273
- **Aerobic**: 135.1439
- **Spline**: 135.2215
- **Total**: 134.3098

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

1. **Predictions are in lbs scale** (converted from standardized units)
2. **Standardization parameters**: Mean = 135.26 lbs, Std = 2.77 lbs
3. 2. **AR(1) component is excluded** as it models measurement-time residual correlation
4. 3. **Credible intervals (95%)** provided for uncertainty quantification
5. 4. **Components are additive**: Total = Intercept + Strength + Aerobic + Spline

## Key Statistics

- **Total prediction range**: 10.311 (min: 129.168, max: 139.479)
- **Average intercept**: 130.093
- **Average strength contribution**: 139.627
- **Average aerobic contribution**: 135.144
- **Average spline amplitude**: 135.221
- **Average daily spline range**: 1.261
