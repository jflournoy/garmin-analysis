/*
 * OPTIMIZED Gaussian Process model for weight over time with flexible priors
 *
 * This version allows customization of prior hyperparameters via data input.
 * Uses Stan's built-in gp_exp_quad_cov function for efficient covariance computation.
 *
 * OPTIMIZATIONS:
 * - Uses Stan's built-in gp_exp_quad_cov function for efficient covariance computation
 * - Supports optional sparse GP approximation via inducing points (FITC)
 * - Maintains backward compatibility with original flexible model
 */

data {
  int<lower=1> N;          // Number of observations
  array[N] real t;         // Time points (scaled to [0,1])
  vector[N] y;             // Weight observations (standardized)

  // Hyperparameters for priors
  real<lower=0> alpha_prior_sd;      // Standard deviation for alpha ~ normal(0, sd)
  real<lower=0> rho_prior_shape;     // Shape parameter for rho ~ inv_gamma(shape, scale)
  real<lower=0> rho_prior_scale;     // Scale parameter for rho ~ inv_gamma(shape, scale)
  real<lower=0> sigma_prior_sd;      // Standard deviation for sigma ~ normal(0, sd)

  // Sparse GP configuration (optional)
  int<lower=0, upper=1> use_sparse;  // 0 = full GP (default), 1 = sparse GP approximation
  int<lower=0> M;                    // Number of inducing points (if use_sparse=1)
  array[M] real t_inducing;          // Inducing point locations (scaled to [0,1], if use_sparse=1)

  // Prediction grid (optional)
  int<lower=0> N_pred;                     // Number of prediction points (0 for none)
  array[N_pred] real t_pred;               // Prediction time points (scaled to [0,1])
  vector[N_pred] hour_of_day_pred;         // Hour of day for prediction (0-24) - required for compatibility
}

parameters {
  real<lower=0.01> alpha;        // GP marginal standard deviation (avoid zero)
  real<lower=0.01> rho;          // GP length scale (avoid extreme values)
  real<lower=0.01> sigma;        // Observation noise (avoid zero)
  vector[N] eta;                 // Standardized GP values (for non-centered param)

  // Additional parameters for sparse GP (only used if use_sparse=1)
  vector[M] eta_inducing;        // Standardized values at inducing points
}

transformed parameters {
  vector[N] f;                   // GP function values
  vector[M] a;                   // Coefficients for inducing points representation

  // Compute GP covariance using inducing points representation
  // For full GP (use_sparse=0): inducing points = observed points (M = N, t_inducing = t)
  // For sparse GP (use_sparse=1): inducing points are subset
  matrix[M, M] K_uu = gp_exp_quad_cov(t_inducing, alpha, rho);
  matrix[N, M] K_fu = gp_exp_quad_cov(t, t_inducing, alpha, rho);

  // Add jitter to diagonal of K_uu for numerical stability
  for (i in 1:M) {
    K_uu[i, i] += square(alpha) * 1e-4 + 1e-6;
  }

  // Cholesky decomposition of K_uu
  matrix[M, M] L_uu = cholesky_decompose(K_uu);

  // Compute u = L_uu * eta where eta depends on use_sparse
  vector[M] u;
  if (use_sparse == 0) {
    // Full GP: use eta (which corresponds to observed points)
    // Note: when use_sparse=0, we have M = N and t_inducing = t
    u = L_uu * eta;
  } else {
    // Sparse GP: use eta_inducing
    u = L_uu * eta_inducing;
  }

  // Compute a = inv(K_uu) * u
  a = mdivide_left_spd(K_uu, u);

  // GP function values at observed points
  f = K_fu * a;
}

model {
  // Priors with flexible hyperparameters
  alpha ~ normal(0, alpha_prior_sd);          // GP amplitude
  rho ~ inv_gamma(rho_prior_shape, rho_prior_scale);  // Length scale
  sigma ~ normal(0, sigma_prior_sd);          // Observation noise
  eta ~ std_normal();                         // Non-centered parameterization

  // Priors for sparse GP parameters (always present, used depending on use_sparse)
  eta_inducing ~ std_normal();

  // Likelihood
  y ~ normal(f, sigma);
}

generated quantities {
  vector[N] y_rep;               // Posterior predictive
  vector[N] log_lik;             // Pointwise log likelihood for WAIC/LOO
  real<lower=0> trend_change;    // Total change in trend over time period
  real<lower=0> avg_uncertainty; // Average posterior standard deviation
  vector[N] f_std;               // Standard deviation of f at each time point

  // Prediction at new time points (if N_pred > 0)
  vector[N_pred] f_pred;         // GP function values at prediction points
  vector[N_pred] y_pred;         // Posterior predictive at prediction points

  for (n in 1:N) {
    y_rep[n] = normal_rng(f[n], sigma);
    log_lik[n] = normal_lpdf(y[n] | f[n], sigma);
  }

  // Compute trend change from first to last observation
  trend_change = abs(f[N] - f[1]);

  // Compute average uncertainty (for monitoring)
  // Note: This is a placeholder - actual standard deviation would need
  // to be computed from posterior samples outside Stan
  avg_uncertainty = sigma;

  // Placeholder for f_std (would need to be computed from posterior)
  f_std = rep_vector(sigma, N);

  // Compute predictions at new time points (if any)
  if (N_pred > 0) {
    // GP function values at prediction points using same inducing points representation
    matrix[N_pred, M] K_pred_u = gp_exp_quad_cov(t_pred, t_inducing, alpha, rho);
    f_pred = K_pred_u * a;
    for (n in 1:N_pred) {
      y_pred[n] = normal_rng(f_pred[n], sigma);
    }
  }
}