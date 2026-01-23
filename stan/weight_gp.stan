/*
 * Gaussian Process model for weight over time
 *
 * This is a proof-of-concept Bayesian model for analyzing
 * weight trends from Garmin data. Uses a squared exponential
 * kernel to capture smooth temporal trends.
 */

data {
  int<lower=1> N;          // Number of observations
  array[N] real t;         // Time points (scaled to [0,1])
  vector[N] y;             // Weight observations (standardized)
}

parameters {
  real<lower=0> alpha;           // GP marginal standard deviation
  real<lower=0> rho;             // GP length scale
  real<lower=0> sigma;           // Observation noise
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
      K[i, i] += 1e-6;  // Numerical stability
    }

    L_K = cholesky_decompose(K);
    f = L_K * eta;
  }
}

model {
  // Priors
  alpha ~ normal(0, 1);          // GP amplitude
  rho ~ inv_gamma(5, 1);         // Length scale (fairly smooth)
  sigma ~ normal(0, 0.5);        // Observation noise
  eta ~ std_normal();            // Non-centered parameterization

  // Likelihood
  y ~ normal(f, sigma);
}

generated quantities {
  vector[N] y_rep;               // Posterior predictive
  real<lower=0> trend_change;    // Total change in trend over time period

  for (n in 1:N) {
    y_rep[n] = normal_rng(f[n], sigma);
  }

  // Compute trend change from first to last observation
  trend_change = fabs(f[N] - f[1]);
}
