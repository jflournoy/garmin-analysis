/*
 * Gaussian Process model for weight over time with trend + daily cyclic components
 *
 * This model separates:
 * 1. Long-term trend (squared exponential kernel)
 * 2. Daily cycles (periodic kernel with period = 24 hours)
 * 3. Observation noise (sigma)
 *
 * Goal: Better separate measurement error from intraday weight fluctuation
 */

data {
  int<lower=1> N;          // Number of observations
  array[N] real t;         // Time points (scaled to [0,1] based on days)
  vector[N] y;             // Weight observations (standardized)

  // Period for daily cycles (24 hours scaled to [0,1] time range)
  real<lower=0> period_daily;
}

parameters {
  // Trend GP parameters
  real<lower=0.01> alpha_trend;        // GP marginal standard deviation for trend
  real<lower=0.01> rho_trend;          // GP length scale for trend

  // Daily cyclic GP parameters
  real<lower=0.01> alpha_daily;        // GP amplitude for daily cycles
  real<lower=0.01> rho_daily;          // Length scale for daily cycles (smoothness)

  // Observation noise
  real<lower=0.01> sigma;              // Combined measurement error + residual variation

  // Non-centered parameterization for efficiency
  vector[N] eta_trend;                 // Standardized values for trend GP
  vector[N] eta_daily;                 // Standardized values for daily GP
}

transformed parameters {
  vector[N] f_trend;                   // Trend component
  vector[N] f_daily;                   // Daily cyclic component
  vector[N] f_total;                   // Combined function

  // Compute covariance matrices
  {
    matrix[N, N] K_trend;
    matrix[N, N] K_daily;
    matrix[N, N] L_trend;
    matrix[N, N] L_daily;

    // Trend: squared exponential kernel
    for (i in 1:N) {
      for (j in 1:N) {
        K_trend[i, j] = square(alpha_trend) * exp(-square(t[i] - t[j]) / (2 * square(rho_trend)));
      }
      K_trend[i, i] += square(alpha_trend) * 1e-4 + 1e-6;  // Jitter for numerical stability
    }

    // Daily: periodic kernel (manual implementation with numerical stability)
    // k(t, t') = α² exp(-2 sin²(π|t - t'|/p) / ℓ²)
    // For small period_daily, use careful computation to avoid numerical issues
    for (i in 1:N) {
      K_daily[i, i] = square(alpha_daily) + square(alpha_daily) * 1e-4 + 1e-6;  // Diagonal with jitter
      for (j in (i+1):N) {
        real d = t[i] - t[j];
        // Compute sin(π * d / period_daily) carefully for small period_daily
        real arg = pi() * d / period_daily;
        real sin_arg = sin(arg);
        real k_val = square(alpha_daily) * exp(-2 * square(sin_arg) / square(rho_daily));
        K_daily[i, j] = k_val;
        K_daily[j, i] = k_val;  // Ensure symmetry
      }
    }

    // Cholesky decomposition
    L_trend = cholesky_decompose(K_trend);
    L_daily = cholesky_decompose(K_daily);

    // Transform to function values
    f_trend = L_trend * eta_trend;
    f_daily = L_daily * eta_daily;
    f_total = f_trend + f_daily;
  }
}

model {
  // Priors
  // Trend: fairly smooth long-term changes
  alpha_trend ~ normal(0, 1);
  rho_trend ~ inv_gamma(5, 1);

  // Daily cycles:
  // - alpha_daily: amplitude of daily variation (likely smaller than trend)
  // - rho_daily: smoothness parameter for daily cycles
  //   With small period_daily, need informative prior to avoid identifiability issues
  alpha_daily ~ normal(0, 0.5);
  // More informative prior for rho_daily given small period_daily
  // rho_daily ~ inv_gamma(2, 0.1);  // Expect fairly smooth daily cycles
  rho_daily ~ inv_gamma(3, 0.5);

  // Observation noise: should be smaller than in original model
  // (since daily variation is now modeled separately)
  sigma ~ normal(0, 0.3);

  // Non-centered parameterization
  eta_trend ~ std_normal();
  eta_daily ~ std_normal();

  // Likelihood
  y ~ normal(f_total, sigma);
}

generated quantities {
  vector[N] y_rep;                     // Posterior predictive
  vector[N] log_lik;                   // Pointwise log likelihood for WAIC/LOO
  vector[N] f_trend_std;               // Trend component (for analysis)
  vector[N] f_daily_std;               // Daily component (for analysis)
  real<lower=0> trend_change;          // Total change in trend
  real<lower=0> daily_amplitude;       // Amplitude of daily variation
  real prop_variance_daily;            // Proportion of variance from daily component

  for (n in 1:N) {
    y_rep[n] = normal_rng(f_total[n], sigma);
    log_lik[n] = normal_lpdf(y[n] | f_total[n], sigma);
  }

  // Store components for analysis
  f_trend_std = f_trend;
  f_daily_std = f_daily;

  // Compute trend change from first to last observation
  trend_change = abs(f_trend[N] - f_trend[1]);

  // Estimate daily amplitude (max - min of daily component)
  daily_amplitude = max(f_daily) - min(f_daily);

  // Proportion of variance explained by daily component
  // (simplified: variance ratio at posterior mean)
  {
    real var_trend = variance(f_trend);
    real var_daily = variance(f_daily);
    real var_total = var_trend + var_daily + square(sigma);
    prop_variance_daily = var_daily / var_total;
  }
}