# Training-Dependent Decay Model: Summary

## Overview
We successfully implemented and tested a new state-space model where **training reduces fitness decay** rather than just providing fitness gain. This tests the hypothesis that workouts help preserve existing fitness in addition to building new fitness.

## Models Created

### 1. Strength-Only Training Decay Model
**File**: `stan/weight_state_space_training_decay_strength_only.stan`

**Equation**:
```
trained[t] = 1 if strength_intensity[t] > 0, else 0
fitness[t] = (alpha_d + alpha_m * trained[t-1]) * fitness[t-1] +
             beta * strength_intensity[t-1] * trained[t-1]
```

**Parameters**:
- `alpha_d`: Decay without training (0 < alpha_d < 1)
- `alpha_m`: Training reduces decay (0 < alpha_m < 1 - alpha_d)
- `beta`: Gain per unit intensity (beta > 0)
- `gamma`: Weight effect of fitness

### 2. Combined Aerobic + Strength Model
**File**: `stan/weight_state_space_training_decay.stan`

Same structure as above but with separate parameters for aerobic and strength fitness.

### 3. Simplified Model (for debugging)
**File**: `stan/weight_state_space_training_decay_simple.stan`

No GP, no daily cycle - just the core fitness model.

## Key Findings

### Strength-Only Model Results (simplified version)
From initial testing with 924 days of data (147 weight observations, 136 strength training days):

| Parameter | Mean ± SD | Interpretation |
|-----------|-----------|----------------|
| `alpha_d` | 0.478 ± 0.166 | Without training, 47.8% of fitness persists each day |
| `alpha_m` | 0.251 ± 0.131 | Training reduces decay by 25.1 percentage points |
| `beta` | 0.233 ± 0.231 | Gain per unit intensity |
| `gamma` | 0.182 ± 0.162 | Strength fitness increases weight (muscle mass) |
| `sigma_w` | 0.987 ± 0.058 | Residual weight variation |

### Interpretation

1. **Decay rates**:
   - **Without training**: Fitness decays to 47.8% of previous day
   - **With training**: Fitness decays to 47.8% - 25.1% = 22.7% of previous day
   - **Training reduces decay by 52.5%** (25.1/47.8)

2. **Training effect**: Workouts both preserve existing fitness (reduce decay) AND add new fitness (beta gain)

3. **Weight effect**: `gamma > 0` suggests strength training increases weight, likely through muscle mass

## Model Comparison

### vs. Diminishing Returns Model
The existing diminishing returns model (`weight_state_space_diminishing.stan`):
- Uses impulse accumulation: `impulse[t] = psi * impulse[t-1] + intensity[t]`
- Has diminishing returns: `gain = impulse * exp(-k * fitness[t-1])`
- Fitness evolution: `fitness[t] = alpha * fitness[t-1] + gain`

**Key differences**:
1. **Training decay model**: Explicitly models training reducing decay
2. **Diminishing returns model**: Models reduced gains at high fitness levels
3. **Training decay is simpler**: No impulse accumulation, no diminishing returns

## Technical Challenges & Solutions

### 1. Data Preparation
**Problem**: Existing `prepare_state_space_data` function didn't separate aerobic vs strength intensity.

**Solution**: Created `prepare_stan_data_for_training_decay_models()` function that:
- Loads intensity by activity type
- Combines walking + cycling into aerobic intensity
- Standardizes intensities: `(value - min)/std`
- Creates proper Stan data structure

### 2. Parameter Constraints
**Problem**: Need `alpha_d + alpha_m < 1` (total decay < 1).

**Solution**: Used inverse logit transformation:
```stan
real<lower=0, upper=1-alpha_d> alpha_m = (1 - alpha_d) * inv_logit(alpha_m_raw);
```
With prior: `alpha_m_raw ~ std_normal()`

### 3. Numerical Stability
**Problem**: Full model (with GP + Fourier) had `nan` values and divergent transitions.

**Solution**: Created simplified model without GP/daily cycle to test core fitness model first.

## Next Steps

### 1. Fix Full Model
Debug the full model (with GP + Fourier) to resolve `nan` issues.

### 2. Test Combined Model
Run the combined aerobic + strength model to compare effects.

### 3. Model Comparison
Compare training decay model with diminishing returns model using WAIC/LOO.

### 4. Add Aerobic-Only Model
Create aerobic-only version to test aerobic effects separately.

### 5. Incorporate into Analysis Pipeline
Integrate with existing `analyze_state_space.py` for comprehensive reporting.

## Code Files Created

1. **Stan models**:
   - `stan/weight_state_space_training_decay.stan` (combined)
   - `stan/weight_state_space_training_decay_strength_only.stan`
   - `stan/weight_state_space_training_decay_simple.stan`

2. **Test scripts**:
   - `test_training_decay_models.py` (comprehensive testing)
   - `test_simple_training_decay.py` (minimal configuration)
   - `test_simple_model_only.py` (simplified model test)

3. **Summary**: This document

## Conclusion

The training-dependent decay model successfully implements the hypothesis that workouts reduce fitness decay. Initial results show:
- Training reduces daily fitness decay by ~25 percentage points
- Without training: 47.8% decay per day
- With training: Only 22.7% decay per day
- Strength training appears to increase weight (likely muscle mass)

This provides a new perspective on how exercise affects fitness: not just by adding new fitness, but by preserving existing fitness through reduced decay.