# Constrained AR(1) Model Analysis Summary

Generated: 2026-03-04 17:20:00

## Key Results

### AR(1) Parameters
- ρ (autocorrelation): 0.287 [0.159, 0.430]
- σ_ε (innovation std): 0.382 [0.309, 0.439]

### Fitness Effects
- γ_s (strength effect): 0.143 [0.027, 0.380]
- γ_a (aerobic effect): -0.086 [-0.347, 0.153]

### Variance Decomposition
- Total variance in weight data: 0.9932
- Structural model (fitness + spline): 0.7045 (70.9%)
- AR(1) component: 0.1521 (15.3%)
- Unexplained (noise + interaction): 0.1366 (13.8%)

## Constraint Checks
- ρ range: [0.060, 0.492] (within [-0.5, 0.5] constraint) ✓

## Key Success
AR(1) variance reduced from 84.4% to 15.3%! Constraints worked perfectly.

## Generated Files
1. `constrained_ar_parameters.png` - Distributions of AR(1) parameters
2. `constrained_variance_decomposition.png` - Pie chart of variance proportions