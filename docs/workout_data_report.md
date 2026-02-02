# Workout Data Report

## Overview

This report documents the workout data available from Garmin fitness exports, specifically focusing on strength training data for use in Bayesian time-series modeling. The data is loaded via the `src.data.workout` module, which processes Garmin's summarized activities JSON files.

**Key Statistics** (based on current dataset):
- **Total workouts**: 139 strength training sessions
- **Date range**: 2024-03-06 to 2026-01-12 (678 days)
- **Workout frequency**: 0.21 workouts/day (approximately 1.5 workouts per week)
- **Most common workout day**: Wednesday
- **Volume data quality**: 86.3% of workouts have zero volume recorded
- **Total reps**: 23,897 | **Total sets**: 2,164
- **Primary metrics**: Reps, sets, volume, duration, calories

## Data Fields

### Core Workout Metadata

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `activity_id` | int64 | Unique Garmin activity identifier | 14288027894 |
| `activity_type` | string | Type of activity (e.g., "strength_training") | "strength_training" |
| `name` | string | Workout name (user-defined) | "Morning Workout" |
| `start_time_gmt` | datetime | Start time in GMT (millisecond precision) | 2023-01-15 14:30:00 |
| `start_time_local` | datetime | Start time in local timezone | 2023-01-15 09:30:00 |
| `date` | datetime | Date extracted from local start time (day-level precision) | 2023-01-15 |

### Fitness Metrics

| Field | Type | Description | Notes |
|-------|------|-------------|-------|
| `duration` | float64 | Workout duration in seconds | 3600 = 1 hour |
| `calories` | float64 | Estimated calories burned | **Likely most reliable magnitude metric** - Garmin estimation based on HR, duration, user profile; consistently recorded |
| `avg_hr` | float64 | Average heart rate during workout | Optional, may be null |
| `max_hr` | float64 | Maximum heart rate during workout | Optional, may be null |
| `min_hr` | float64 | Minimum heart rate during workout | Optional, may be null |
| `steps` | float64 | Steps recorded during workout | May be zero for strength training |

### Strength-Specific Metrics

| Field | Type | Description | Notes |
|-------|------|-------------|-------|
| `total_reps` | int64 | Total repetitions across all exercises | **Potential counting errors** - may not capture effort intensity |
| `total_sets` | int64 | Total sets across all exercises | **Potential counting errors** - may not capture effort intensity |
| `active_sets` | int64 | Active sets (sets with recorded reps) | May equal total_sets |
| `total_volume` | int64 | Total weight lifted (volume) in arbitrary units | **Data quality issue**: Many zero values |

### Exercise Details

| Field | Type | Description | Structure |
|-------|------|-------------|-----------|
| `exercise_details` | list[dict] | Detailed breakdown of individual exercises | List of exercise records |

**Exercise Record Fields**:
- `category`: string - Exercise category (e.g., "BENCH_PRESS", "UNKNOWN")
- `sub_category`: string or None - More specific classification
- `reps`: int - Number of repetitions for this exercise
- `volume`: int - Weight lifted for this exercise (may be zero)
- `duration`: float - Duration of this exercise in seconds
- `sets`: int - Number of sets for this exercise
- `max_weight`: int or None - Maximum weight used

## Data Quality Assessment

### Missing Values
- `min_hr`: 79 out of 139 records missing (56.8%) - expected for strength training without continuous HR monitoring
- Other heart rate metrics (`avg_hr`, `max_hr`): Some missing but fewer than `min_hr`
- All other core fields have complete data

### Volume Data Issues
The `total_volume` field shows significant data quality problems:
- Many workouts have `total_volume = 0` despite having `total_reps` and `total_sets`
- Volume appears to be inconsistently recorded in Garmin exports
- **Recommendation**: Use `calories` as primary magnitude metric; `reps`/`sets` as alternatives with caution

### Metric Reliability Trade-offs
Different metrics capture different aspects of workout magnitude with varying reliability:

- **`calories`**: **Most reliable for magnitude** - consistently recorded, estimates energy expenditure based on heart rate, duration, and user profile. However, it's an estimation that may have individual errors.

- **`total_reps` / `total_sets`**: **Subject to counting errors** - may not accurately capture effort if reps are miscounted or sets are incomplete. Don't account for weight lifted.

- **`total_volume`**: **Poor data quality** - 86.3% of workouts have zero volume recorded.

- **`duration`**: **Doesn't capture intensity** - a short intense workout may have greater physiological impact than a long light one.

- **`count`**: **Simple frequency** - useful for workout occurrence but not magnitude.

For cross-lagged modeling of workout effects on weight, `calories` is recommended as the primary magnitude metric due to consistent recording and relevance to energy expenditure.

### Exercise Detail Completeness
- All workouts include `exercise_details` list (when `include_exercise_details=True`)
- Some exercises have `category: "UNKNOWN"` - Garmin may not classify all exercises
- `volume` field within exercise details often zero

## Aggregation Options

The `prepare_workout_aggregates()` function supports multiple aggregation strategies:

### Time Aggregation
- **Daily**: Aggregate metrics per calendar day
- **Weekly**: Aggregate metrics per week (starting Monday)

### Metric Aggregation
| Metric | Column Used | Description | Recommended |
|--------|-------------|-------------|-------------|
| `count` | None | Number of workouts per time period | Good for frequency analysis |
| `volume` | `total_volume` | Total weight lifted | **Not recommended** due to data quality |
| `reps` | `total_reps` | Total repetitions | **Use with caution** - potential counting errors, doesn't capture weight |
| `sets` | `total_sets` | Total sets | **Use with caution** - potential counting errors, doesn't capture weight |
| `calories` | `calories` | Total calories burned | **Primary recommendation** - most reliable magnitude metric, consistently recorded |
| `duration` | `duration` | Total workout duration | Useful for time commitment analysis |

### Handling Zero Days
When aggregating daily, many days have zero workouts. The `prepare_workout_for_stan()` function can fill missing days with zeros using `fill_missing=True`.

## Usage in Cross-Lagged GP Models

### Current Implementation
The cross-lagged Gaussian Process model (`weight_gp_crosslagged.stan`) uses:
- Fixed lag parameter (`lag_days`) specified as data
- Single lagged workout value affects weight at time `t`
- β parameter represents causal effect size
- Model tested with lags: 0, 1, 2, 3, 7 days

### Data Preparation Pipeline
```python
from src.data.workout import load_strength_training_data
from src.data.align import prepare_crosslagged_stan_data

# Load and aggregate workout data
df_workouts, stan_data_workout = load_strength_training_data(
    aggregation="daily",
    metric="calories"  # Most reliable magnitude metric
)

# Prepare cross-lagged data (requires weight data)
stan_data = prepare_crosslagged_stan_data(
    stan_data_weight,
    stan_data_workout,
    lag_days=2
)
```

### Recommendations for Modeling
1. **Primary metric**: Use `calories` as the most reliable magnitude metric (consistently recorded, estimates energy expenditure)
2. **Alternative metrics**: Consider `total_reps` or `total_sets` but be aware of potential counting errors
3. **Aggregation**: Daily aggregation with zero-fill for missing days
4. **Lag testing**: Test multiple lags (0-7 days) using WAIC/LOO comparison
5. **Sparse data**: Account for many zero days in workout time series

## Temporal Patterns

### Workout Frequency
- [To be populated from workout_report.py output]
- Average workouts per week: [ ]
- Most common workout days: [ ]
- Monthly trends: [ ]

### Exercise Distribution
- Top exercise categories by volume: [ ]
- Most frequent exercise types: [ ]
- Volume trends over time: [ ]

## Integration with Other Health Metrics

Workout data can be correlated with:
1. **Weight**: Muscle gain hypothesis (positive effect after 1-3 day lag)
2. **Sleep**: Impact of evening workouts on sleep quality
3. **Heart Rate**: Resting HR changes with training adaptation
4. **VO2 Max**: Cardiovascular fitness improvements

## Limitations and Considerations

### Garmin Export Limitations
1. **Volume data incomplete**: Many workouts lack weight information
2. **Exercise classification**: Some exercises marked "UNKNOWN"
3. **Timing precision**: Only start time recorded, not exercise timing within workout
4. **Missing metrics**: No RPE (rate of perceived exertion) or intensity measures

### Modeling Challenges
1. **Sparse time series**: Many days without workouts (zeros)
2. **Metric reliability trade-offs**:
   - `calories`: Most consistently recorded but estimated (HR-based)
   - `reps`/`sets`: May have counting errors, don't capture weight
   - `volume`: Often zero (86.3% of workouts)
   - `duration`: Doesn't capture intensity
3. **Variable workout intensity**: Reps/sets don't capture weight lifted
4. **Delayed effects**: Physiological responses may have distributed lags
5. **Individual variation**: Response to training varies by person

## Future Enhancements

### Data Improvements
1. Supplement with manual training log data
2. Estimate volume from exercise categories and typical weights
3. Add derived metrics (volume × reps, intensity index)

### Modeling Extensions
1. **Estimated lag models**: Treat lag as estimated parameter rather than fixed
2. **Distributed lag models**: Cumulative effects from multiple lag periods
3. **Threshold models**: Minimum workout intensity required for effect
4. **Interaction models**: Combined effects of workouts with sleep/nutrition

## References

- `src/data/workout.py`: Primary data loading module
- `src/analysis/workout_report.py`: Comprehensive analysis and visualization
- `stan/weight_gp_crosslagged.stan`: Cross-lagged GP model
- `src/data/align.py`: Data alignment for cross-lagged modeling

---
*Report generated: 2026-01-31*
*Data source: Garmin DI-Connect-Fitness export*
*Analysis code: `src/analysis/workout_report.py`*