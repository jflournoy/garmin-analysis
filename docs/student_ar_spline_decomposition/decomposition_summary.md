# Student-t AR Spline Model Decomposition Summary

Generated: 2026-03-04 16:16:28

## Model Overview

The Student-t AR Spline model decomposes weight variations into:
1. **Fitness states** (daily scale): Strength and aerobic fitness accumulation/decay
2. **Weight effects** (daily scale): Contributions from fitness to weight
3. **Daily spline** (intraday scale): 24h and 12h cycles in weight
4. **AR(1) process** (observation scale): Temporal correlation in residuals
5. **Student-t distribution** (observation scale): Robustness to outliers

## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| γ_s (strength effect) | 0.012 | Effect of strength fitness on weight |
| γ_a (aerobic effect) | -0.057 | Effect of aerobic fitness on weight |
| Weight intercept | -0.046 | Baseline weight level |
| ν (degrees of freedom) | 10.4 | Student-t tail heaviness |
| ρ (AR(1) correlation) | 0.975 | Temporal autocorrelation |
| σ_ε (innovation std) | 0.229 | AR(1) innovation scale |

## Fitness Decay Parameters

| Parameter | Strength | Aerobic |
|-----------|----------|---------|
| α_d (base decay) | 0.944 | 0.795 |
| α_m (training effect) | 0.502 | 0.498 |
| β (gain coefficient) | 0.224 | 0.261 |

## Component Magnitudes (in lbs)

| Component | Mean Abs Value | Std Dev |
|-----------|----------------|---------|
| Intercept | 0.127 | - |
| Strength Effect | 0.073 | - |
| Aerobic Effect | 0.086 | - |
| Daily Spline | 0.086 | - |
| AR(1) Innovation | 0.205 | - |

## Generated Plots

1. **Fitness States Over Time** (`fitness_states_over_time.png`)
   - Strength and aerobic fitness evolution
   - Training intensity inputs
   - Daily scale visualization

2. **Weight Effect Decomposition** (`weight_effect_decomposition.png`)
   - Strength effect on weight over time
   - Aerobic effect on weight over time
   - Total fitness effect
   - Daily scale visualization

3. **Daily Spline Component** (`daily_spline_decomposition.png`)
   - Spline vs hour of day
   - Spline by day of week
   - Fourier basis functions
   - Intraday scale visualization

4. **AR(1) Process Decomposition** (`ar_process_decomposition.png`)
   - AR(1) innovations over time
   - Autocorrelation function
   - Innovation distribution
   - Observation scale visualization

5. **Student-t Distribution Analysis** (`student_t_analysis.png`)
   - Residual distribution with Student-t fit
   - QQ-plot vs Student-t distribution
   - Degrees of freedom analysis
   - Observation scale visualization

6. **Complete Model Decomposition** (`complete_decomposition.png`)
   - Stacked component plot
   - Component magnitude comparison
   - Component correlation matrix
   - Observation scale visualization

7. **Decay Rate Analysis** (`decay_rate_analysis.png`)
   - Strength decay rates over time
   - Aerobic decay rates over time
   - Decay rate distributions
   - Daily scale visualization

8. **Cumulative Prediction Build-Up** (`cumulative_prediction_buildup.png`)
   - Intercept only (baseline)
   - Intercept + Strength effect
   - Intercept + Aerobic effect
   - Intercept + Strength + Aerobic
   - Intercept + Daily spline
   - Intercept + Strength + Aerobic + Daily spline
   - Shows how each component builds toward full prediction
   - Observation scale visualization

9. **Component Contribution Analysis** (`component_contribution_analysis.png`)
   - Prediction error reduction by component
   - Cumulative error reduction relative to intercept
   - Variance explained by each component
   - Prediction error distributions
   - Shows quantitative impact of each model piece
   - Observation scale visualization

