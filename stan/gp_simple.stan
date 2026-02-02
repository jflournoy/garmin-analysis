/*
 * Simple Gaussian Process model for scalar time series.
 * Squared exponential kernel, Student-t likelihood for robustness.
 * Sparse GP approximation via inducing points (optional).
 */

data {
  int<lower=1> N;          // Number of observations
  array[N] real t;         // Time points (scaled to [0,1])
  vector[N] y;             // Observations (standardized)

  // Sparse GP configuration (optional)
  int<lower=0, upper=1> use_sparse;  // 0 = full GP, 1 = sparse GP approximation
  int<lower=0> M;                    // Number of inducing points (if use_sparse=1)
  array[M] real t_inducing;          // Inducing point locations (scaled to [0,1])

  // Prediction grid (optional)
  int<lower=0> N_pred;               // Number of prediction points (0 for none)
  array[N_pred] real t_pred;         // Prediction time points (scaled to [0,1])
}

parameters {
  // GP parameters
  real<lower=0.01, upper=5> alpha;        // GP marginal standard deviation
  real<lower=0.01, upper=5> rho;          // GP length scale

  // Observation noise (Student-t for robustness)
  real<lower=0.01, upper=2> sigma;        // Scale parameter for Student-t
  real<lower=1> nu;                       // Degrees of freedom for Student-t

  // Non-centered parameterization for inducing points
  vector[M] eta_inducing_raw;             // Standard normal for inducing points
}

transformed parameters {
  vector[M] eta_inducing;                 // Transformed inducing points

  // Non-centered parameterization: u = L_uu * eta_inducing
  // where L_uu is Cholesky factor of K_uu
  // We'll compute this in model block after K_uu construction

  // GP covariance matrices
  matrix[M, M] K_uu = gp_exp_quad_cov(t_inducing, alpha, rho);
  matrix[N, M] K_fu = gp_exp_quad_cov(t, t_inducing, alpha, rho);

  // Add jitter to diagonal of K_uu for numerical stability
  for (i in 1:M) {
    K_uu[i, i] += square(alpha) * 1e-4 + 1e-6;
  }

  // Cholesky decomposition of K_uu
  matrix[M, M] L_uu = cholesky_decompose(K_uu);

  // Compute u = L_uu * eta_inducing_raw
  vector[M] u = L_uu * eta_inducing_raw;

  // Compute a = inv(K_uu) * u
  vector[M] a = mdivide_left_spd(K_uu, u);

  // Trend component at observed points
  vector[N] f_trend = K_fu * a;

  eta_inducing = eta_inducing_raw;  // for consistency with other models
}

model {
  // Priors for GP
  alpha ~ normal(0, 0.5);
  rho ~ inv_gamma(5, 1);

  // Observation noise priors
  sigma ~ exponential(3);
  nu ~ gamma(2, 0.1);  // encourages values around 20 (near-normal)

  // Prior for inducing points parameters
  eta_inducing_raw ~ std_normal();

  // Likelihood (Student-t for robustness against outliers)
  y ~ student_t(nu, f_trend, sigma);
}

generated quantities {
  vector[N] y_rep;                     // Posterior predictive
  vector[N] log_lik;                   // Pointwise log likelihood for WAIC/LOO
  vector[N] f_trend_std;               // Trend component (for analysis)

  // Prediction at new time points (if N_pred > 0)
  vector[N_pred] f_pred;               // Trend component at prediction points
  vector[N_pred] y_pred;               // Posterior predictive at prediction points

  // Posterior predictive samples and log likelihood
  for (n in 1:N) {
    y_rep[n] = student_t_rng(nu, f_trend[n], sigma);
    log_lik[n] = student_t_lpdf(y[n] | nu, f_trend[n], sigma);
  }

  // Store trend component for analysis
  f_trend_std = f_trend;

  // Compute predictions at new time points
  if (N_pred > 0) {
    matrix[N_pred, M] K_pred_u = gp_exp_quad_cov(t_pred, t_inducing, alpha, rho);
    f_pred = K_pred_u * a;  // using same 'a' from transformed parameters
    for (n in 1:N_pred) {
      y_pred[n] = student_t_rng(nu, f_pred[n], sigma);
    }
  } else {
    f_pred = rep_vector(0, N_pred);
    y_pred = rep_vector(0, N_pred);
  }
}