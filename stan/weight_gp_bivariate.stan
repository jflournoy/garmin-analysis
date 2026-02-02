/*
 * Bivariate Gaussian Process model for weight and another variable (e.g., resting heart rate).
 * Uses separable kernel: K(t, t') ⊗ B where B is a 2x2 coregionalization matrix.
 * Student-t likelihood for robustness.
 * Sparse GP approximation via inducing points (optional).
 *
 * Model:
 *   y_weight(t) = f_weight(t) + ε_weight
 *   y_other(t) = f_other(t) + ε_other
 *   where [f_weight(t), f_other(t)]^T ~ GP(0, B ⊗ k(t,t'))
 *   k: squared exponential kernel with length scale rho and marginal std alpha.
 *   B: 2x2 positive definite coregionalization matrix (parameterized via Cholesky factor).
 *   ε_i ~ Student-t(ν_i, σ_i)
 */

data {
  int<lower=1> N;          // Number of observations (same time points for both outputs)
  array[N] real t;         // Time points (scaled to [0,1] based on days)
  vector[N] y_weight;      // Weight observations (standardized)
  vector[N] y_other;       // Other variable observations (standardized)

  // Sparse GP configuration (optional)
  int<lower=0, upper=1> use_sparse;  // 0 = full GP, 1 = sparse GP approximation
  int<lower=0> M;                    // Number of inducing points (if use_sparse=1)
  array[M] real t_inducing;          // Inducing point locations (scaled to [0,1])

  // Prediction grid (optional)
  int<lower=0> N_pred;               // Number of prediction points (0 for none)
  array[N_pred] real t_pred;         // Prediction time points (scaled to [0,1])
}

parameters {
  // Trend GP parameters (shared kernel)
  real<lower=0.01, upper=5> alpha;        // GP marginal standard deviation
  real<lower=0.01, upper=5> rho;          // GP length scale

  // Coregionalization matrix B (2x2 positive definite)
  cholesky_factor_corr[2] L_corr;         // Cholesky factor of correlation matrix
  vector<lower=0>[2] sigma_B;             // Standard deviations for each output

  // Observation noise (Student-t for robustness)
  vector<lower=0.01, upper=2>[2] sigma;   // Scale parameters for Student-t
  vector<lower=1>[2] nu;                  // Degrees of freedom for Student-t

  // Non-centered parameterization for inducing points (size M x 2)
  matrix[M, 2] eta_inducing_raw;          // Standard normal for inducing points
}

transformed parameters {
  matrix[M, 2] eta_inducing;              // Transformed inducing points
  matrix[2, 2] L_B;                       // Cholesky factor of B = diag(sigma_B) * L_corr * diag(sigma_B)'
  matrix[2, 2] B;                         // Coregionalization matrix

  // Construct L_B = diag(sigma_B) * L_corr
  L_B = diag_pre_multiply(sigma_B, L_corr);
  B = multiply_lower_tri_self_transpose(L_B);  // B = L_B * L_B'

  // Non-centered parameterization for inducing points: u = L_B * eta_inducing_raw^T
  // Each column of eta_inducing_raw is standard normal for each output
  // We want u = L_B * eta_inducing_raw^T, so u^T = eta_inducing_raw * L_B'
  // Let's compute eta_inducing = eta_inducing_raw * L_B' (size M x 2)
  eta_inducing = eta_inducing_raw * L_B';

  // GP covariance matrices
  matrix[M, M] K_uu = gp_exp_quad_cov(t_inducing, alpha, rho);
  matrix[N, M] K_fu = gp_exp_quad_cov(t, t_inducing, alpha, rho);

  // Add jitter to diagonal of K_uu for numerical stability
  for (i in 1:M) {
    K_uu[i, i] += square(alpha) * 1e-4 + 1e-6;
  }

  // Cholesky decomposition of K_uu
  matrix[M, M] L_uu = cholesky_decompose(K_uu);

  // Compute a = inv(K_uu) * u where u = L_uu * eta_inducing
  // For each output dimension d: a[,d] = inv(K_uu) * u[,d]
  matrix[M, 2] a;
  for (d in 1:2) {
    vector[M] u_d = L_uu * eta_inducing[:, d];
    a[:, d] = mdivide_left_spd(K_uu, u_d);
  }

  // Trend component at observed points: f = K_fu * a (size N x 2)
  matrix[N, 2] f_trend = K_fu * a;
}

model {
  // Priors for trend GP
  alpha ~ normal(0, 0.5);
  rho ~ inv_gamma(5, 1);

  // Priors for coregionalization matrix
  sigma_B ~ exponential(1);
  L_corr ~ lkj_corr_cholesky(2);  // weakly informative for 2x2

  // Observation noise priors
  sigma ~ exponential(3);
  nu ~ gamma(2, 0.1);  // encourages values around 20 (near-normal)

  // Prior for inducing points parameters
  to_vector(eta_inducing_raw) ~ std_normal();

  // Likelihood (independent Student-t for each output)
  for (n in 1:N) {
    y_weight[n] ~ student_t(nu[1], f_trend[n, 1], sigma[1]);
    y_other[n] ~ student_t(nu[2], f_trend[n, 2], sigma[2]);
  }
}

generated quantities {
  // Posterior predictive
  vector[N] y_weight_rep;
  vector[N] y_other_rep;
  matrix[N, 2] log_lik;  // column 1: weight, column 2: other

  // Coregionalization correlation
  corr_matrix[2] corr_B = multiply_lower_tri_self_transpose(L_corr);
  real correlation = corr_B[1, 2];  // correlation between latent processes

  // Prediction at new time points (if N_pred > 0)
  matrix[N_pred, 2] f_pred;
  matrix[N_pred, 2] y_pred;

  // Posterior predictive and log likelihood
  for (n in 1:N) {
    y_weight_rep[n] = student_t_rng(nu[1], f_trend[n, 1], sigma[1]);
    y_other_rep[n] = student_t_rng(nu[2], f_trend[n, 2], sigma[2]);
    log_lik[n, 1] = student_t_lpdf(y_weight[n] | nu[1], f_trend[n, 1], sigma[1]);
    log_lik[n, 2] = student_t_lpdf(y_other[n] | nu[2], f_trend[n, 2], sigma[2]);
  }

  // Compute predictions at new time points
  if (N_pred > 0) {
    matrix[N_pred, M] K_pred_u = gp_exp_quad_cov(t_pred, t_inducing, alpha, rho);
    f_pred = K_pred_u * a;  // N_pred x 2
    for (n in 1:N_pred) {
      y_pred[n, 1] = student_t_rng(nu[1], f_pred[n, 1], sigma[1]);
      y_pred[n, 2] = student_t_rng(nu[2], f_pred[n, 2], sigma[2]);
    }
  } else {
    f_pred = rep_matrix(0, N_pred, 2);
    y_pred = rep_matrix(0, N_pred, 2);
  }
}