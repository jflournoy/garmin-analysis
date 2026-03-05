# Component Predictions LBS Scale Update

## Summary of Changes

Updated `generate_component_predictions.py` to:
1. **Add actual weight data to plots** - Red scatter points showing actual weight measurements
2. **Convert predictions to raw lbs scale** - Dual output in both standardized units and lbs
3. **Create separate files for lbs scale** - All outputs have `_lbs` suffix for lbs scale versions

## Changes Made

### 1. Data Preparation (`prepare_data_with_4h_intervals`)
- Added return of `standardization_params` dictionary containing:
  - `weight_mean`, `weight_std`
  - `strength_mean`, `strength_std`
  - `aerobic_mean`, `aerobic_std`

### 2. Conversion Function (`convert_to_lbs_scale`)
- New function to convert component predictions from standardized units to lbs scale
- Formula: `lbs_value = standardized_value * weight_std + weight_mean`
- Preserves credible intervals

### 3. Updated Output Functions

#### `save_component_predictions`
- Added `lbs_scale` parameter (default: False)
- Files with `_lbs` suffix for lbs scale outputs
- Both standardized and lbs scale CSV files generated

#### `create_component_visualizations`
- Added `lbs_scale` parameter and `df_weight` parameter
- Actual weight data added as red scatter points to "total" and "full_model" plots
- Y-axis labels updated to "Weight (lbs)" for lbs scale
- Plot titles include "(lbs scale)" suffix
- Files with `_lbs` suffix for lbs scale images

#### `create_prediction_summary`
- Added `lbs_scale` and `standardization_params` parameters
- Updated usage notes to indicate lbs scale
- Files with `_lbs` suffix for lbs scale reports

### 4. Main Function Updates
- Generate both standardized and lbs scale outputs
- Dual CSV files, visualizations, and summary reports
- All lbs scale files have `_lbs` suffix

## File Naming Convention

### Standardized Units (original)
- `component_predictions_hour_XX.csv`
- `all_component_predictions.csv`
- `component_time_series_noon.png`
- `total_predictions_heatmap.png`
- `component_contributions_sample_dates.png`
- `daily_patterns_analysis.png`
- `prediction_summary.md`

### LBS Scale (new)
- `component_predictions_hour_XX_lbs.csv`
- `all_component_predictions_lbs.csv`
- `component_time_series_noon_lbs.png`
- `total_predictions_heatmap_lbs.png`
- `component_contributions_sample_dates_lbs.png`
- `daily_patterns_analysis_lbs.png`
- `prediction_summary_lbs.md`

## Usage

The updated script now generates:
1. **Standardized outputs** (original behavior) - for model analysis
2. **LBS scale outputs** (new) - for practical interpretation with actual weight data

Run the script as before:
```bash
uv run python generate_component_predictions.py
```

## Verification

- Conversion tested with mock data
- Syntax check passed
- Imports successfully
- File naming logic verified

## Next Steps

1. Run the full script to generate actual lbs scale outputs
2. Verify plots show actual weight data points
3. Check that lbs scale values are in expected range (~126-143 lbs based on historical data)