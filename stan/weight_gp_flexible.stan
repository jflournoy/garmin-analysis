/*
 * Gaussian Process model for weight over time with flexible priors
 *
 * This version allows customization of prior hyperparameters via data input.
 * Uses a squared exponential kernel to capture smooth temporal trends.
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
}

parameters {
  real<lower=0.01> alpha;        // GP marginal standard deviation (avoid zero)
  real<lower=0.01> rho;          // GP length scale (avoid extreme values)
  real<lower=0.01> sigma;        // Observation noise (avoid zero)
  vector[N] eta;                 // Standardized GP values (for non-centered param)
}

transformed parameters {
  vector[N] f;                   // GP function values

  // Compute GP covariance and transform eta to f
  {
    matrix[N, N] K;
    matrix[N, N] L_K;

    // Squared exponential kernel
    for (i in 1:N) {
      for (j in 1:N) {
        K[i, j] = square(alpha) * exp(-square(t[i] - t[j]) / (2 * square(rho)));
      }
      K[i, i] += square(alpha) * 1e-4 + 1e-6;  // Relative jitter for numerical stability
    }

    L_K = cholesky_decompose(K);
    f = L_K * eta;
  }
}

model {
  // Priors with flexible hyperparameters
  alpha ~ normal(0, alpha_prior_sd);          // GP amplitude
  rho ~ inv_gamma(rho_prior_shape, rho_prior_scale);  // Length scale
  sigma ~ normal(0, sigma_prior_sd);          // Observation noise
  eta ~ std_normal();                         // Non-centered parameterization

  // Likelihood
  y ~ normal(f, sigma);
}

generated quantities {
  vector[N] y_rep;               // Posterior predictive
  vector[N] log_lik;             // Pointwise log likelihood for WAIC/LOO
  real<lower=0> trend_change;    // Total change in trend over time period
  real<lower=0> avg_uncertainty; // Average posterior standard deviation
  vector[N] f_std;               // Standard deviation of f at each time point

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
}