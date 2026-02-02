# Bayesian Gaussian Process Model for Weight Analysis

## Overview

This document explains the Bayesian Gaussian Process (GP) model used for analyzing weight trends from Garmin health data. The model is implemented in Stan and provides a flexible framework for capturing temporal patterns while quantifying uncertainty.

## 1. Data Preparation

### 1.1 Raw Data Loading

Weight measurements are loaded from Garmin's `userBioMetrics.json` export. Each entry contains:

- **Date**: Measurement timestamp
- **Weight**: Measurement in grams (converted to pounds)

```python
# Conversion: grams to pounds
GRAMS_TO_LBS = 0.00220462
weight_lbs = weight_grams * GRAMS_TO_LBS
```

### 1.2 Time Standardization

Time is measured in days since the first observation and scaled to $[0, 1]$:

$$
t_{\text{days}} = \text{days since first measurement}
$$

$$
t_{\text{scaled}} = \frac{t_{\text{days}}}{\max(t_{\text{days}})}
$$

This scaling improves numerical stability and helps the GP length scale parameter ($\rho$) interpret more consistently.

### 1.3 Weight Standardization

Weight values are centered and scaled to have zero mean and unit variance:

$$
y_{\text{original}} = \text{weight in pounds}
$$

$$
y_{\text{mean}} = \mathbb{E}[y_{\text{original}}]
$$

$$
y_{\text{sd}} = \text{SD}[y_{\text{original}}]
$$

$$
y_{\text{standardized}} = \frac{y_{\text{original}} - y_{\text{mean}}}{y_{\text{sd}}}
$$

**Note**: Scaling parameters are saved for back-transformation after model fitting.

## 2. Gaussian Process Model

### 2.1 Model Definition

The core model assumes weight measurements follow:

$$
y_i = f(t_i) + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma^2)
$$

where $f(t)$ is a Gaussian Process:

$$
f(t) \sim \mathcal{GP}(0, k(t, t'))
$$

### 2.2 Squared Exponential Kernel

The covariance between function values at times $t_i$ and $t_j$ is:

$$
k(t_i, t_j) = \alpha^2 \exp\left(-\frac{(t_i - t_j)^2}{2\rho^2}\right)
$$

**Parameters**:
- $\alpha$: Marginal standard deviation (amplitude)
- $\rho$: Length scale (how quickly correlations decay with time)
- $\sigma$: Observation noise standard deviation

### 2.3 Numerical Stability

A small "jitter" term is added to the diagonal for numerical stability:

$$
K_{ii} = \alpha^2 \left(1 + 10^{-4}\right) + 10^{-6}
$$

This prevents the covariance matrix from becoming numerically singular.

## 3. Stan Implementation

### 3.1 Data Block

```stan
data {
  int<lower=1> N;          // Number of observations
  array[N] real t;         // Time points (scaled to [0,1])
  vector[N] y;             // Weight observations (standardized)

  // For flexible model only:
  real<lower=0> alpha_prior_sd;
  real<lower=0> rho_prior_shape;
  real<lower=0> rho_prior_scale;
  real<lower=0> sigma_prior_sd;
}
```

### 3.2 Non-Centered Parameterization

Instead of sampling $f$ directly, we use a non-centered parameterization:

```stan
parameters {
  real<lower=0.01> alpha;  // GP amplitude
  real<lower=0.01> rho;    // Length scale
  real<lower=0.01> sigma;  // Observation noise
  vector[N] eta;           // Standardized values ~ N(0,1)
}

transformed parameters {
  vector[N] f;             // GP function values

  {
    matrix[N, N] K;
    matrix[N, N] L_K;

    // Build covariance matrix
    for (i in 1:N) {
      for (j in 1:N) {
        K[i, j] = square(alpha) * exp(-square(t[i] - t[j]) / (2 * square(rho)));
      }
      K[i, i] += square(alpha) * 1e-4 + 1e-6;  // Jitter
    }

    L_K = cholesky_decompose(K);
    f = L_K * eta;  // Transform: f = L * η ~ N(0, K)
  }
}
```

**Why non-centered?** This parameterization improves sampling efficiency by reducing posterior correlations between parameters.

### 3.3 Priors

#### Default Priors (Basic Model):
```stan
alpha ~ normal(0, 1);          // Reasonable amplitude
rho ~ inv_gamma(5, 1);         // Encourages smooth trends
sigma ~ normal(0, 0.5);        // Moderate noise
eta ~ std_normal();            // Non-centered transform
```

#### Flexible Priors:
```stan
alpha ~ normal(0, alpha_prior_sd);
rho ~ inv_gamma(rho_prior_shape, rho_prior_scale);
sigma ~ normal(0, sigma_prior_sd);
eta ~ std_normal();
```

**Inverse Gamma Distribution**: $p(\rho) \propto \rho^{-(\text{shape}+1)} \exp(-\text{scale}/\rho)$

### 3.4 Likelihood

```stan
model {
  y ~ normal(f, sigma);
}
```

Each observation $y_i$ is normally distributed around the GP function value $f(t_i)$ with noise $\sigma$.

### 3.5 Generated Quantities

```stan
generated quantities {
  vector[N] y_rep;            // Posterior predictive samples
  real<lower=0> trend_change; // Total change from first to last point

  for (n in 1:N) {
    y_rep[n] = normal_rng(f[n], sigma);
  }

  trend_change = abs(f[N] - f[1]);
}
```

**Posterior Predictive Checks**: Simulate new data $y^{\text{rep}}$ to check model fit.

**Trend Change**: Absolute difference between first and last GP function values.

## 4. Mathematical Details

### 4.1 Gaussian Process Definition

A Gaussian Process is a collection of random variables, any finite number of which have a joint Gaussian distribution:

$$
f(\mathbf{t}) \sim \mathcal{N}(\mathbf{0}, \mathbf{K})
$$

where $\mathbf{K}$ is the $N \times N$ covariance matrix with entries $K_{ij} = k(t_i, t_j)$.

### 4.2 Posterior Distribution

Given observed data $\mathbf{y}$, the posterior for $f$ is:

$$
p(f | \mathbf{y}, \mathbf{t}) \propto p(\mathbf{y} | f, \sigma^2) \cdot p(f | \alpha, \rho)
$$

### 4.3 Kernel Properties

**Squared Exponential (RBF) Kernel**:
- Infinitely differentiable (produces smooth functions)
- Stationary (depends only on time difference $|t_i - t_j|$)
- Positive definite (ensures valid covariance matrix)

### 4.4 Length Scale Interpretation

The length scale $\rho$ controls function smoothness:

- **Small $\rho$**: Rapid fluctuations, captures short-term variations
- **Large $\rho$**: Slow trends, smooth long-term patterns

**Rule of thumb**: Correlation drops to $\exp(-1) \approx 0.37$ at distance $\rho$.

## 5. Model Fitting and Inference

### 5.1 MCMC Sampling

The model is fitted using Hamiltonian Monte Carlo (HMC) via Stan:

```python
fit = model.sample(
    data=stan_data,
    chains=4,
    iter_warmup=500,
    iter_sampling=500,
    show_progress=True,
)
```

### 5.2 Diagnostics

Key diagnostics checked:

1. **R-hat**: $\hat{R} < 1.01$ indicates convergence
2. **ESS**: Effective Sample Size > 400 per chain
3. **Divergent transitions**: Should be minimal (< 1%)

### 5.3 Posterior Analysis

After sampling, we compute:

```python
# Extract posterior samples
f_samples = idata.posterior["f"].values  # Shape: (chain, draw, obs)

# Back-transform to original scale
f_mean = f_samples.mean(axis=(0, 1)) * y_sd + y_mean
f_std = f_samples.std(axis=(0, 1)) * y_sd
```

## 6. Practical Considerations

### 6.1 Prior Sensitivity

Different prior configurations affect model flexibility:

| Configuration | $\alpha$ prior | $\rho$ prior | Effect |
|---------------|----------------|--------------|---------|
| **Smooth** | normal(0, 1) | inv_gamma(5, 1) | Encourages smooth trends |
| **Wiggly** | normal(0, 2) | inv_gamma(2, 0.5) | Allows more short-term variation |
| **Very flexible** | normal(0, 3) | inv_gamma(1.5, 0.3) | Captures rapid fluctuations |

### 6.2 Computational Complexity

The GP covariance matrix is $O(N^2)$ in memory and $O(N^3)$ in computation (Cholesky decomposition). For large datasets ($N > 1000$), consider:

1. **Sparse approximations** (e.g., FITC, VFE)
2. **Kernel approximations** (e.g., Random Fourier Features)
3. **Time series models** (e.g., ARIMA, state space models)

### 6.3 Model Extensions

Potential improvements:

1. **Heteroskedastic noise**: $\sigma(t)$ varying with time
2. **Multiple outputs**: Joint modeling of weight, sleep, activity
3. **Changepoints**: Allowing trend changes at specific times
4. **Seasonality**: Adding periodic components

## 7. Example Results

### 7.1 Typical Output

For a 1.5-year weight dataset:
- **Trend change**: ~6.5 lbs (increase or decrease)
- **Average uncertainty**: ~0.23 lbs (standard deviation)
- **R-hat**: 1.00-1.01 (good convergence)

### 7.2 Visualization

The model produces:
1. **Expected trajectory**: Posterior mean of $f(t)$
2. **Credible intervals**: Uncertainty bands (e.g., 80%, 95% CI)
3. **Posterior predictive**: Simulated data vs. observed

## 8. Caching System

### 8.1 Cache Key Generation

Models are cached based on a SHA256 hash of:
- Data file modification time
- Stan file modification time
- Model hyperparameters
- MCMC configuration (chains, iterations)

### 8.2 Cache Contents

```bash
output/cache/{hash}/
├── metadata.json      # Configuration and metadata
├── idata.nc          # ArviZ InferenceData (NetCDF)
├── df.parquet        # Original DataFrame
├── stan_data.json    # Stan data dictionary
└── csv/              # MCMC CSV output files
```

### 8.3 Cache Invalidation

Cache is invalidated when:
1. **Data changes**: File modification time updates
2. **Model changes**: Stan file is modified
3. **Configuration changes**: Different hyperparameters or MCMC settings

## 9. Command Line Interface

The Gaussian Process weight analysis provides several command-line interfaces for fitting models and generating visualizations.

### 9.1 NPM Scripts (Recommended)

The project includes convenient npm scripts for common analysis tasks:

| Command | Description |
|---------|-------------|
| `npm run plot-weight` | Basic plot with 80% credible interval |
| `npm run plot-weight:stddev` | Plot with uncertainty (standard deviation) subplot |
| `npm run plot-weight:multi` | Plot with multiple credible intervals (50%, 80%, 95%) |
| `npm run plot-weight:flexible` | Use flexible model with custom priors |
| `npm run plot-weight:wiggly` | Use wiggly priors for more flexible fits |

These scripts use `uv` (Python package manager) internally and handle all dependencies automatically.

### 9.2 Python CLI Direct Usage

For more control, you can use the Python CLI directly:

```bash
# Basic usage
uv run python -m src.models.plot_weight_cli

# Show multiple credible intervals
uv run python -m src.models.plot_weight_cli --multiple-intervals --ci-levels 50 80 95

# Show uncertainty evolution
uv run python -m src.models.plot_weight_cli --show-stddev

# Use flexible model with custom priors
uv run python -m src.models.plot_weight_cli --use-flexible \
  --alpha-prior-sd 2 \
  --rho-prior-shape 2 \
  --rho-prior-scale 0.5 \
  --sigma-prior-sd 1

# Change MCMC settings
uv run python -m src.models.plot_weight_cli \
  --chains 4 \
  --iter-warmup 500 \
  --iter-sampling 1000
```

### 9.3 Command Line Options

Key command-line options:

| Option | Description | Default |
|--------|-------------|---------|
| `--output`, `-o` | Output file path | `output/weight_fit_enhanced.png` |
| `--ci-levels` | Credible interval levels | `80` |
| `--multiple-intervals` | Show multiple intervals simultaneously | `false` |
| `--show-stddev` | Show standard deviation subplot | `false` |
| `--use-flexible` | Use flexible Stan model | `false` |
| `--alpha-prior-sd` | Standard deviation for α prior | `1.0` |
| `--rho-prior-shape` | Shape parameter for ρ prior (inverse gamma) | `5.0` |
| `--rho-prior-scale` | Scale parameter for ρ prior | `1.0` |
| `--sigma-prior-sd` | Standard deviation for σ prior | `0.5` |
| `--chains` | Number of MCMC chains | `4` |
| `--iter-warmup` | Warmup iterations per chain | `500` |
| `--iter-sampling` | Sampling iterations per chain | `500` |
| `--data-dir` | Path to data directory | `data` |
| `--stan-file` | Path to Stan model file | `stan/weight_gp.stan` |

### 9.4 Model Fitting Examples

**Basic analysis**:
```bash
npm run plot-weight
```

**Enhanced uncertainty visualization**:
```bash
npm run plot-weight:stddev
```

**Multiple credible intervals with custom priors**:
```bash
npm run plot-weight:wiggly
```

**Direct Python usage with all options**:
```bash
uv run python -m src.models.plot_weight_cli \
  --multiple-intervals \
  --ci-levels 50 80 95 \
  --show-stddev \
  --use-flexible \
  --alpha-prior-sd 2 \
  --rho-prior-shape 2 \
  --rho-prior-scale 0.5 \
  --output output/weight_analysis_custom.png
```

### 9.5 Output and Results

The CLI generates:
1. **Plot visualization**: Saved to specified output path (default: `output/weight_fit_enhanced.png`)
2. **Console summary**: Key parameters, trend change, uncertainty statistics
3. **Cache files**: Model results cached for faster subsequent runs

## 10. References

1. Rasmussen & Williams (2006) *Gaussian Processes for Machine Learning*
2. Stan Development Team (2024) *Stan User's Guide*
3. Betancourt (2017) *A Conceptual Introduction to Hamiltonian Monte Carlo*
4. Gelman et al. (2013) *Bayesian Data Analysis*

## 11. Appendix: Mathematical Notation

| Symbol | Meaning | Range |
|--------|---------|-------|
| $N$ | Number of observations | $\mathbb{Z}^+$ |
| $t_i$ | Time point $i$ (scaled) | $[0, 1]$ |
| $y_i$ | Weight observation $i$ (standardized) | $\mathbb{R}$ |
| $f(t)$ | GP function value at time $t$ | $\mathbb{R}$ |
| $\alpha$ | GP amplitude (marginal SD) | $\mathbb{R}^+$ |
| $\rho$ | GP length scale | $\mathbb{R}^+$ |
| $\sigma$ | Observation noise SD | $\mathbb{R}^+$ |
| $\eta$ | Non-centered parameter | $\mathcal{N}(0, 1)$ |
| $K$ | Covariance matrix | $\mathbb{R}^{N \times N}$ |

---

*Last updated: 2026-01-23*
*Model version: Gaussian Process weight analysis v2.0*