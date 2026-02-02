/*
 * OPTIMIZED Gaussian Process model for weight over time with trend + Fourier spline for daily cycles
 * Uses Student-t likelihood for robustness against outliers
 *
 * This model separates:
 * 1. Long-term trend (squared exponential kernel) - OPTIMIZED with gp_exp_quad_cov
 * 2. Daily cycles (Fourier basis expansion for hour-of-day effects)
 * 3. Observation noise (sigma) with Student-t degrees of freedom (nu)
 *
 * OPTIMIZATIONS:
 * - Uses Stan's built-in gp_exp_quad_cov function for efficient covariance computation
 * - Supports optional sparse GP approximation via inducing points (FITC)
 * - Maintains backward compatibility with original model
 * - Student-t likelihood for robust outlier handling
 *
 * Goal: Better separate measurement error from intraday weight fluctuation
 * using a simpler parametric cyclic model than periodic GP, with improved performance.
 */

data {
  int<lower=1> N;          // Number of observations
  array[N] real t;         // Time points (scaled to [0,1] based on days)
  vector[N] y;             // Weight observations (standardized)

  // Hour of day information for cyclic spline
  vector[N] hour_of_day;   // Hour of day (0-24) as float
  int<lower=1> K;          // Number of Fourier harmonics (K=1: 24h cycle, K=2: 12h + 24h, etc.)

  // Sparse GP configuration (optional - Phase 2)
  int<lower=0, upper=1> use_sparse;  // 0 = full GP (default), 1 = sparse GP approximation
  int<lower=0> M;                    // Number of inducing points (if use_sparse=1)
  array[M] real t_inducing;          // Inducing point locations (scaled to [0,1], if use_sparse=1)

  // Prediction grid (optional)
  int<lower=0> N_pred;                     // Number of prediction points (0 for none)
  array[N_pred] real t_pred;               // Prediction time points (scaled to [0,1])
  vector[N_pred] hour_of_day_pred;         // Hour of day for prediction (0-24)
}

parameters {
  // Trend GP parameters
  real<lower=0.01, upper=5> alpha_trend;        // GP marginal standard deviation for trend
  real<lower=0.01, upper=5> rho_trend;          // GP length scale for trend

  // Fourier coefficients for daily cycles (non-centered parameterization)
  vector[K] a_sin_raw;                 // Raw sine coefficients
  vector[K] a_cos_raw;                 // Raw cosine coefficients

  // Prior scale for Fourier coefficients
  real<lower=0.01, upper=2> sigma_fourier;      // Scale of Fourier coefficients

  // Observation noise (Student-t for robustness)
  real<lower=0.01, upper=2> sigma;              // Scale parameter for Student-t
  real<lower=1> nu;                             // Degrees of freedom for Student-t

  // Non-centered parameterization for trend GP
  vector[N] eta_trend;                 // Standardized values for trend GP

  // Additional parameters for sparse GP (only used if use_sparse=1)
  vector[M] eta_inducing;              // Standardized values at inducing points
}

transformed parameters {
  vector[N] f_trend;                   // Trend component
  vector[N] f_daily;                   // Daily cyclic component (Fourier)
  vector[N] f_total;                   // Combined function
  vector[M] a;                         // Coefficients for inducing points representation

  // Fourier coefficients (non-centered transformation)
  vector[K] a_sin = sigma_fourier * a_sin_raw;
  vector[K] a_cos = sigma_fourier * a_cos_raw;

  // Trend GP: squared exponential kernel using inducing points representation
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

  // Compute a = inv(K_uu) * u
  a = mdivide_left_spd(K_uu, u);

  // Trend component at observed points
  f_trend = K_fu * a;

  // Daily component: Fourier basis expansion (unchanged from original)
  // f_daily[n] = Σ_k [a_sin[k] * sin(2πk * hour_scaled[n]) + a_cos[k] * cos(2πk * hour_scaled[n])]
  // where hour_scaled = hour_of_day / 24.0 (maps 0-24 to 0-1)
  for (n in 1:N) {
    real hour_scaled = hour_of_day[n] / 24.0;
    f_daily[n] = 0.0;
    for (k in 1:K) {
      real freq = 2.0 * pi() * k;
      f_daily[n] += a_sin[k] * sin(freq * hour_scaled) + a_cos[k] * cos(freq * hour_scaled);
    }
  }

  f_total = f_trend + f_daily;
}

model {
  // Priors for trend GP
  alpha_trend ~ normal(0, 0.5);
  rho_trend ~ inv_gamma(5, 1);

  // Priors for Fourier coefficients (non-centered)
  sigma_fourier ~ exponential(2);
  a_sin_raw ~ std_normal();
  a_cos_raw ~ std_normal();

  // Observation noise prior (Student-t for robustness)
  sigma ~ exponential(3);
  // Degrees of freedom prior: gamma(2, 0.1) encourages values around 20 (near-normal)
  nu ~ gamma(2, 0.1);

  // Non-centered parameterization for trend GP
  eta_trend ~ std_normal();

  // Prior for inducing points parameters (used for both full and sparse GP)
  eta_inducing ~ std_normal();

  // Likelihood (Student-t for robustness against outliers)
  y ~ student_t(nu, f_total, sigma);
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

  // Posterior predictive samples and log likelihood
  for (n in 1:N) {
    y_rep[n] = student_t_rng(nu, f_total[n], sigma);
    log_lik[n] = student_t_lpdf(y[n] | nu, f_total[n], sigma);
  }

  // Store components for analysis
  f_trend_std = f_trend;
  f_daily_std = f_daily;

  // Compute trend change from first to last observation
  trend_change = abs(f_trend[N] - f_trend[1]);

  // Estimate daily amplitude (max - min of daily component)
  daily_amplitude = max(f_daily) - min(f_daily);

  // Proportion of variance explained by daily component
  // Note: For Student-t, observation variance is sigma^2 * nu/(nu-2) when nu > 2
  {
    real var_trend = variance(f_trend);
    real var_daily = variance(f_daily);
    real obs_variance = square(sigma);
    if (nu > 2) {
      obs_variance = square(sigma) * nu / (nu - 2);
    }
    real var_total = var_trend + var_daily + obs_variance;
    prop_variance_daily = var_daily / var_total;
  }

  // Compute predictions at new time points (if any)
  if (N_pred > 0) {
    // Trend component at prediction points using same inducing points representation
    matrix[N_pred, M] K_pred_u = gp_exp_quad_cov(t_pred, t_inducing, alpha_trend, rho_trend);
    f_trend_pred = K_pred_u * a;

    // Daily component at prediction points
    for (n in 1:N_pred) {
      real hour_scaled = hour_of_day_pred[n] / 24.0;
      f_daily_pred[n] = 0.0;
      for (k in 1:K) {
        real freq = 2.0 * pi() * k;
        f_daily_pred[n] += a_sin[k] * sin(freq * hour_scaled) + a_cos[k] * cos(freq * hour_scaled);
      }
    }

    f_pred = f_trend_pred + f_daily_pred;
    for (n in 1:N_pred) {
      y_pred[n] = student_t_rng(nu, f_pred[n], sigma);
    }
  }
}