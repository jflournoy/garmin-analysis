/*
 * OPTIMIZED Gaussian Process model for weight over time with trend + daily cyclic components
 *
 * This model separates:
 * 1. Long-term trend (squared exponential kernel) - OPTIMIZED with gp_exp_quad_cov
 * 2. Daily cycles (periodic kernel with period = 24 hours) - OPTIMIZED with gp_periodic_cov
 * 3. Observation noise (sigma)
 *
 * OPTIMIZATIONS:
 * - Uses Stan's built-in gp_exp_quad_cov function for efficient trend covariance computation
 * - Uses Stan's built-in gp_periodic_cov function for efficient periodic covariance computation
 * - Supports optional sparse GP approximation via inducing points (FITC) for trend component
 * - Maintains backward compatibility with original cyclic model
 *
 * Goal: Better separate measurement error from intraday weight fluctuation
 * with improved computational performance.
 */

data {
  int<lower=1> N;          // Number of observations
  array[N] real t;         // Time points (scaled to [0,1] based on days)
  vector[N] y;             // Weight observations (standardized)

  // Period for daily cycles (24 hours scaled to [0,1] time range)
  real<lower=0> period_daily;

  // Sparse GP configuration for trend (optional)
  int<lower=0, upper=1> use_sparse;  // 0 = full GP (default), 1 = sparse GP approximation for trend
  int<lower=0> M;                    // Number of inducing points (if use_sparse=1)
  array[M] real t_inducing;          // Inducing point locations (scaled to [0,1], if use_sparse=1)

  // Prediction grid (optional)
  int<lower=0> N_pred;                     // Number of prediction points (0 for none)
  array[N_pred] real t_pred;               // Prediction time points (scaled to [0,1])
  vector[N_pred] hour_of_day_pred;         // Hour of day for prediction (0-24) - required for compatibility
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

  // Additional parameters for sparse GP (only used if use_sparse=1)
  vector[M] eta_inducing;              // Standardized values at inducing points
}

transformed parameters {
  vector[N] f_trend;                   // Trend component
  vector[N] f_daily;                   // Daily cyclic component
  vector[N] f_total;                   // Combined function
  vector[M] a_trend;                   // Coefficients for trend inducing points representation
  vector[N] a_daily;                   // Coefficients for daily GP representation

  // Compute trend covariance using inducing points representation
  // For full GP (use_sparse=0): inducing points = observed points (M = N, t_inducing = t)
  // For sparse GP (use_sparse=1): inducing points are subset
  matrix[M, M] K_uu = gp_exp_quad_cov(t_inducing, alpha_trend, rho_trend);
  matrix[N, M] K_fu = gp_exp_quad_cov(t, t_inducing, alpha_trend, rho_trend);

  // Add jitter to diagonal of K_uu for numerical stability
  for (i in 1:M) {
    K_uu[i, i] += square(alpha_trend) * 1e-4 + 1e-6;
  }

  // Cholesky decomposition of K_uu
  matrix[M, M] L_uu = cholesky_decompose(K_uu);

  // Compute u = L_uu * eta where eta depends on use_sparse
  vector[M] u;
  if (use_sparse == 0) {
    // Full GP: use eta_trend (which corresponds to observed points)
    // Note: when use_sparse=0, we have M = N and t_inducing = t
    u = L_uu * eta_trend;
  } else {
    // Sparse GP: use eta_inducing
    u = L_uu * eta_inducing;
  }

  // Compute a_trend = inv(K_uu) * u
  a_trend = mdivide_left_spd(K_uu, u);

  // Trend component at observed points
  f_trend = K_fu * a_trend;

  // Daily cyclic covariance using Stan's built-in gp_periodic_cov function
  // gp_periodic_cov(x, alpha, rho, period) = α² exp(-2 sin²(π|x - x'|/p) / ℓ²)
  matrix[N, N] K_daily = gp_periodic_cov(t, alpha_daily, rho_daily, period_daily);

  // Add jitter for numerical stability
  for (i in 1:N) {
    K_daily[i, i] += square(alpha_daily) * 1e-4 + 1e-6;
  }

  matrix[N, N] L_daily = cholesky_decompose(K_daily);
  f_daily = L_daily * eta_daily;

  // Compute a_daily = inv(K_daily) * f_daily for prediction
  a_daily = mdivide_left_spd(K_daily, f_daily);

  f_total = f_trend + f_daily;
}

model {
  // Priors
  // Trend: fairly smooth long-term changes
  alpha_trend ~ normal(0, 0.5);
  rho_trend ~ inv_gamma(5, 1);

  // Daily cycles:
  // - alpha_daily: amplitude of daily variation (likely smaller than trend)
  // - rho_daily: smoothness parameter for daily cycles
  alpha_daily ~ exponential(2);
  rho_daily ~ inv_gamma(3, 0.5);

  // Observation noise: should be smaller than in original model
  // (since daily variation is now modeled separately)
  sigma ~ exponential(3);

  // Non-centered parameterization
  eta_trend ~ std_normal();
  eta_daily ~ std_normal();

  // Priors for sparse GP parameters (always present, used depending on use_sparse)
  eta_inducing ~ std_normal();

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

  // Prediction at new time points (if N_pred > 0)
  vector[N_pred] f_trend_pred;         // Trend component at prediction points
  vector[N_pred] f_daily_pred;         // Daily component at prediction points
  vector[N_pred] f_pred;               // Combined function at prediction points
  vector[N_pred] y_pred;               // Posterior predictive at prediction points

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
  {
    real var_trend = variance(f_trend);
    real var_daily = variance(f_daily);
    real var_total = var_trend + var_daily + square(sigma);
    prop_variance_daily = var_daily / var_total;
  }

  // Compute predictions at new time points (if any)
  if (N_pred > 0) {
    // Trend component at prediction points using same inducing points representation
    matrix[N_pred, M] K_pred_u = gp_exp_quad_cov(t_pred, t_inducing, alpha_trend, rho_trend);
    f_trend_pred = K_pred_u * a_trend;

    // Daily component at prediction points
    matrix[N_pred, N] K_pred_daily = gp_periodic_cov(t_pred, t, alpha_daily, rho_daily, period_daily);
    f_daily_pred = K_pred_daily * a_daily;

    f_pred = f_trend_pred + f_daily_pred;
    for (n in 1:N_pred) {
      y_pred[n] = normal_rng(f_pred[n], sigma);
    }
  }
}