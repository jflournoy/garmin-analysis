/*
 * Cross-lagged Gaussian Process model for weight and workouts.
 * Weight depends on lagged workouts: weight(t) = β·workouts(t-τ) + weight_intrinsic(t)
 * Separate GP parameters for weight and workout processes.
 * Student-t likelihood for robustness.
 * Sparse GP approximation via inducing points.
 *
 * Model:
 *   y_weight(t) = f_weight(t) + ε_weight
 *   y_workout(t) = f_workout(t) + ε_workout
 *   where:
 *     f_weight(t) = β·f_workout(t-τ) + g_weight(t)
 *     g_weight(t) ~ GP(0, k_weight(t,t'))  [intrinsic weight dynamics]
 *     f_workout(t) ~ GP(0, k_workout(t,t'))  [workout process]
 *     ε_i ~ Student-t(ν_i, σ_i)
 *
 * τ (lag) is specified as data parameter (fixed for model comparison).
 */

data {
  // Weight data (frequent measurements)
  int<lower=1> N_weight;
  array[N_weight] real t_weight;      // Time points scaled to [0,1]
  vector[N_weight] y_weight;          // Weight observations (standardized)

  // Workout data (daily aggregates, may have zeros)
  int<lower=1> N_workout;
  array[N_workout] real t_workout;    // Time points scaled to [0,1]
  vector[N_workout] y_workout;        // Workout metric (standardized)

  // Lag specification (in scaled time units, 0-1)
  real<lower=0> lag_scaled;           // τ in scaled time units

  // Sparse GP configuration
  int<lower=0, upper=1> use_sparse;   // 0 = full GP, 1 = sparse GP approximation
  int<lower=0> M;                     // Number of inducing points
  array[M] real t_inducing;           // Inducing point locations (scaled to [0,1])

  // Prediction grid (optional)
  int<lower=0> N_pred;                // Number of prediction points (0 for none)
  array[N_pred] real t_pred;          // Prediction time points (scaled to [0,1])
}

parameters {
  // Separate GP parameters for weight and workouts
  real<lower=0.01, upper=5> alpha_weight;     // GP marginal std for weight
  real<lower=0.01, upper=5> rho_weight;       // GP length scale for weight
  real<lower=0.01, upper=5> alpha_workout;    // GP marginal std for workouts
  real<lower=0.01, upper=5> rho_workout;      // GP length scale for workouts

  // Causal effect: workouts → weight
  real beta;                                 // Effect size (weight ~ beta * lagged(workouts))

  // Observation noise (Student-t for robustness)
  real<lower=0.01, upper=2> sigma_weight;    // Scale parameter for weight
  real<lower=0.01, upper=2> sigma_workout;   // Scale parameter for workouts
  real<lower=1> nu_weight;                   // Degrees of freedom for weight
  real<lower=1> nu_workout;                  // Degrees of freedom for workouts

  // Non-centered parameterization for inducing points (separate for each process)
  matrix[M, 2] eta_inducing_raw;             // Standard normal for inducing points
}

transformed parameters {
  // Transformed inducing points for each process
  matrix[M, 2] eta_inducing;
  matrix[M, M] K_uu_weight;
  matrix[M, M] K_uu_workout;
  matrix[M, M] L_uu_weight;
  matrix[M, M] L_uu_workout;
  vector[M] a_weight;
  vector[M] a_workout;

  // Separate inducing points for weight and workouts
  eta_inducing[:, 1] = eta_inducing_raw[:, 1];  // Weight process
  eta_inducing[:, 2] = eta_inducing_raw[:, 2];  // Workout process

  // Weight GP covariance matrices
  K_uu_weight = gp_exp_quad_cov(t_inducing, alpha_weight, rho_weight);
  for (i in 1:M) {
    K_uu_weight[i, i] += square(alpha_weight) * 1e-4 + 1e-6;
  }
  L_uu_weight = cholesky_decompose(K_uu_weight);

  // Workout GP covariance matrices
  K_uu_workout = gp_exp_quad_cov(t_inducing, alpha_workout, rho_workout);
  for (i in 1:M) {
    K_uu_workout[i, i] += square(alpha_workout) * 1e-4 + 1e-6;
  }
  L_uu_workout = cholesky_decompose(K_uu_workout);

  // Compute a vectors: a = inv(K_uu) * (L_uu * eta_inducing)
  a_weight = mdivide_left_spd(K_uu_weight, L_uu_weight * eta_inducing[:, 1]);
  a_workout = mdivide_left_spd(K_uu_workout, L_uu_workout * eta_inducing[:, 2]);

  // Workout latent function at observed times
  vector[N_workout] f_workout;
  {
    matrix[N_workout, M] K_fu_workout = gp_exp_quad_cov(t_workout, t_inducing, alpha_workout, rho_workout);
    f_workout = K_fu_workout * a_workout;
  }

  // Weight latent function with lagged workout effect
  vector[N_weight] f_weight;
  {
    matrix[N_weight, M] K_fu_weight = gp_exp_quad_cov(t_weight, t_inducing, alpha_weight, rho_weight);

    for (i in 1:N_weight) {
      real t = t_weight[i];
      real t_lag = t - lag_scaled;

      // Predict workout at lagged time using inducing points
      real f_workout_lag = 0;
      if (t_lag >= 0 && t_lag <= 1) {
        // Compute covariance between lagged time and inducing points
        row_vector[M] K_lag_u;
        for (j in 1:M) {
          K_lag_u[j] = square(alpha_workout) * exp(-0.5 * square(t_lag - t_inducing[j]) / square(rho_workout));
        }
        f_workout_lag = K_lag_u * a_workout;
      } else {
        // If lagged time outside [0,1], treat as zero (outside observation window)
        f_workout_lag = 0;
      }

      // Weight intrinsic dynamics at time t
      real f_weight_intrinsic = 0;
      for (j in 1:M) {
        f_weight_intrinsic += K_fu_weight[i, j] * a_weight[j];
      }

      // Combined: lagged workout effect + intrinsic dynamics
      f_weight[i] = beta * f_workout_lag + f_weight_intrinsic;
    }
  }
}

model {
  // Priors for GP parameters
  alpha_weight ~ normal(0, 0.5);
  alpha_workout ~ normal(0, 0.5);
  rho_weight ~ inv_gamma(5, 1);
  rho_workout ~ inv_gamma(5, 1);

  // Prior for causal effect
  beta ~ normal(0, 1);  // weakly informative (standardized units)

  // Observation noise priors
  sigma_weight ~ exponential(3);
  sigma_workout ~ exponential(3);
  nu_weight ~ gamma(2, 0.1);    // encourages values around 20 (near-normal)
  nu_workout ~ gamma(2, 0.1);

  // Priors for inducing points
  to_vector(eta_inducing_raw) ~ std_normal();

  // Likelihoods
  y_weight ~ student_t(nu_weight, f_weight, sigma_weight);
  y_workout ~ student_t(nu_workout, f_workout, sigma_workout);
}

generated quantities {
  // Posterior predictive
  vector[N_weight] y_weight_rep;
  vector[N_workout] y_workout_rep;

  // Log likelihood for model comparison
  vector[N_weight] log_lik_weight;
  vector[N_workout] log_lik_workout;

  // Causal effect in original units (if scaling factors provided externally)
  real beta_scaled;  // Will be computed externally in Python

  // Store latent functions for analysis
  vector[N_weight] f_weight_stored = f_weight;
  vector[N_workout] f_workout_stored = f_workout;

  // Posterior predictive samples
  for (i in 1:N_weight) {
    y_weight_rep[i] = student_t_rng(nu_weight, f_weight[i], sigma_weight);
    log_lik_weight[i] = student_t_lpdf(y_weight[i] | nu_weight, f_weight[i], sigma_weight);
  }

  for (i in 1:N_workout) {
    y_workout_rep[i] = student_t_rng(nu_workout, f_workout[i], sigma_workout);
    log_lik_workout[i] = student_t_lpdf(y_workout[i] | nu_workout, f_workout[i], sigma_workout);
  }

  // Prediction at new time points (if requested)
  matrix[N_pred, 2] f_pred;
  matrix[N_pred, 2] y_pred;

  if (N_pred > 0) {
    // Workout predictions at grid points
    matrix[N_pred, M] K_pred_u_workout = gp_exp_quad_cov(t_pred, t_inducing, alpha_workout, rho_workout);
    vector[N_pred] f_workout_pred = K_pred_u_workout * a_workout;

    // Weight predictions (need to account for lag)
    matrix[N_pred, M] K_pred_u_weight = gp_exp_quad_cov(t_pred, t_inducing, alpha_weight, rho_weight);

    for (i in 1:N_pred) {
      real t = t_pred[i];
      real t_lag = t - lag_scaled;

      // Predict workout at lagged time
      real f_workout_lag = 0;
      if (t_lag >= 0 && t_lag <= 1) {
        row_vector[M] K_lag_u;
        for (j in 1:M) {
          K_lag_u[j] = square(alpha_workout) * exp(-0.5 * square(t_lag - t_inducing[j]) / square(rho_workout));
        }
        f_workout_lag = K_lag_u * a_workout;
      }

      // Weight prediction: beta * lagged workout + intrinsic
      real f_weight_pred = 0;
      for (j in 1:M) {
        f_weight_pred += K_pred_u_weight[i, j] * a_weight[j];
      }
      f_weight_pred += beta * f_workout_lag;

      // Store predictions
      f_pred[i, 1] = f_weight_pred;
      f_pred[i, 2] = f_workout_pred[i];

      // Generate posterior predictive samples
      y_pred[i, 1] = student_t_rng(nu_weight, f_weight_pred, sigma_weight);
      y_pred[i, 2] = student_t_rng(nu_workout, f_workout_pred[i], sigma_workout);
    }
  } else {
    f_pred = rep_matrix(0, N_pred, 2);
    y_pred = rep_matrix(0, N_pred, 2);
  }
}