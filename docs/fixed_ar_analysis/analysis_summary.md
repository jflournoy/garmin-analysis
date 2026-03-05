# Fixed AR(1) Model Analysis Summary

Generated: 2026-03-04 17:07:00

## Key Changes from Original Model

1. **Removed local shrinkage parameters** (`lambda[i]`) - AR(1) now has single `sigma_epsilon`
2. **Stronger priors on AR(1) parameters**:
   - `rho` transformed beta prior (centered near 0)
   - `sigma_epsilon ~ exponential(5)` (mean=0.2)
3. **Simplified AR(1) process**: Standard AR(1) without horseshoe complications

## AR(1) Parameter Results

- ρ (autocorrelation): 0.963 [0.923, 0.986]
- σ_ε (innovation std): 0.181 [0.120, 0.268]

## Fitness Effects

- γ_s (strength effect): 0.028 [-0.269, 0.294]
- γ_a (aerobic effect): -0.061 [-0.339, 0.236]

## Variance Decomposition

- Total variance in weight data: 0.9932
- Structural model (fitness + spline): 0.0210 (2.1%)
- AR(1) component: 0.8378 (84.4%)
- Unexplained (noise + interaction): 0.1343 (13.5%)

## ⚠️ WARNING

AR(1) component still explains 84.4% of variance - too high!
Consider even stronger priors on σ_ε or ρ.

## Generated Files

1. `ar_parameters.png` - Distributions of AR(1) parameters
2. `fitness_effects.png` - Distributions of fitness effects
3. `ar_innovations_time_series.png` - AR(1) innovations over time
4. `variance_decomposition.png` - Pie chart of variance proportions
5. `analysis_summary.md` - This summary file
