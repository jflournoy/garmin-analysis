# Symmetric Model Outcome Distribution Analysis

## Summary

We compared three different outcome distributions for the symmetric aerobic model to address your concern that "Strength is continuous and positively bounded and probably not really normally distributed."

## Models Compared

1. **Normal distribution** (original): `y_weight ~ normal(mu, sigma_w)`
2. **Gamma distribution**: `y_weight ~ gamma(shape, shape / exp(mu))`
3. **Log-normal distribution**: `log(y_weight) ~ normal(mu, sigma_w)`

## Key Results

### Parameter Estimates

| Model | γ_s (strength) | γ_a (aerobic) | Log-Likelihood | WAIC |
|-------|----------------|---------------|----------------|------|
| Normal | 0.128 | -0.039 | -98.9 | 204.7 |
| Gamma | 0.110 | 0.022 | -159.7 | 326.7 |
| Log-Normal | 0.120 | 0.041 | -175.0 | 355.9 |

### Weight Effects in Original Units (lbs per fitness unit)

| Model | Strength Effect | Aerobic Effect |
|-------|----------------|----------------|
| Normal | 0.355 lbs | -0.107 lbs |
| Gamma | 0.305 lbs | 0.062 lbs |
| Log-Normal | 0.333 lbs | 0.113 lbs |

## Key Findings

1. **Normal model has best fit**: Based on WAIC (Watanabe-Akaike Information Criterion), the normal distribution has the best fit (lowest WAIC = 204.7).

2. **Different effect directions**:
   - Normal model suggests aerobic exercise **decreases** weight (γ_a = -0.039)
   - Gamma and log-normal models suggest aerobic exercise **increases** weight (γ_a = 0.022 and 0.041 respectively)

3. **Strength effects consistent**: All models show positive strength effects on weight, though magnitudes differ.

4. **Model diagnostics**:
   - Normal model had 1 divergent transition (0.3%)
   - Gamma model had 2 divergent transitions (0.7%) and numerical issues
   - Log-normal model had 3 divergent transitions (1.0%)

## Interpretation

### Why Normal Might Be Best Despite Data Characteristics

1. **Standardized data**: The weight data is standardized (mean=0, std=1), which reduces skewness
2. **Model simplicity**: Normal distribution is more numerically stable
3. **WAIC comparison**: Clear preference for normal distribution based on model fit

### Gamma vs Log-Normal Issues

The Gamma and log-normal models had numerical issues because:
- Weight data was shifted to ensure positivity (added 2.508 to all values)
- This artificial shift may distort the distribution
- Gamma shape parameter approaching zero causes numerical instability

## Recommendations

### 1. **Stick with Normal Distribution** (Recommended)
- Best model fit based on WAIC
- Numerically stable
- Consistent with current symmetric model approach
- Despite theoretical concerns, performs well empirically

### 2. **If Concerned About Positivity Constraint**
Consider these alternatives:

**Option A: Truncated Normal**
```stan
y_weight[i] ~ normal(mu, sigma_w) T[0, ];
```

**Option B: Scaled Beta Distribution**
- Scale weight to [0,1] range
- Use Beta distribution: `y_scaled[i] ~ beta(alpha, beta)`

**Option C: Student-t Distribution**
- More robust to outliers than normal
- `y_weight[i] ~ student_t(nu, mu, sigma_w)`

### 3. **Address Data Characteristics Directly**

If strength intensity data (not weight) is the concern:
- Consider zero-inflated models for strength intensity
- Use Gamma or log-normal for intensity, normal for weight
- Separate models for training vs. non-training days

## Next Steps

1. **Posterior predictive checks**: Verify which distribution best captures data patterns
2. **Sensitivity analysis**: Test with different priors for Gamma shape parameter
3. **Model expansion**: Consider separate distributions for strength vs. aerobic effects
4. **Hierarchical approach**: Allow distribution parameters to vary by fitness type

## Conclusion

While Gamma and log-normal distributions are theoretically more appropriate for positive continuous data, the normal distribution performs best empirically for this symmetric model with standardized weight data. The simplicity and numerical stability of the normal distribution outweigh theoretical concerns in this case.

**Recommendation**: Continue using the normal distribution for the symmetric model, but monitor posterior predictive checks for any systematic misfit.