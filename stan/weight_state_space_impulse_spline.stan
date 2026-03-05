/*
 * State-space model for weight and workout intensity with impulse-response fitness
 * and daily variance spline.
 *
 * Impulse state evolves daily:
 *   impulse[t] = psi·impulse[t-1] + intensity[t]  (impulse accumulates and decays)
 *
 * Fitness state evolves daily (deterministic function of impulse):
 *   fitness[t] = alpha·fitness[t-1] + beta·impulse[t-1]
 *
 * Weight depends on fitness, intrinsic dynamics, and daily cycles:
 *   weight[t] = baseline + gamma·fitness[day(t)] + GP(t) + f_daily[t] + ε_w[t], ε_w[t] ~ N(0, σ_w)
 *
 * Daily component uses Fourier basis expansion:
 *   f_daily[n] = Σ_k [a_sin[k] * sin(2πk * hour_scaled[n]) + a_cos[k] * cos(2πk * hour_scaled[n])]
 *   where hour_scaled = hour_of_day[n] / 24.0
 *
 * Workout intensity is observed (precomputed from HR, duration):
 *   intensity[t] = duration[t] × (avg_hr[t] - resting_hr) / (max_hr - resting_hr)
 *
 * Separate GP for intrinsic weight variations (sparse approximation).
 */

functions {
  // Define exponential covariance function for GP
  matrix gp_exp_quad_cov_custom(array[] real x1,
                         array[] real x2,
                         real alpha,
                         real rho) {
    int N1 = size(x1);
    int N2 = size(x2);
    matrix[N1, N2] result;
    real alpha_sq = square(alpha);
    real neg_half_inv_rho_sq = -0.5 / square(rho);

    for (i in 1:N1) {
      for (j in 1:N2) {
        result[i, j] = alpha_sq * exp(neg_half_inv_rho_sq * square(x1[i] - x2[j]));
      }
    }
    return result;
  }
}

data {
  // Daily fitness states
  int<lower=1> D;                     // number of days
  vector[D] intensity;                // workout intensity (standardized)

  // Weight observations
  int<lower=1> N_weight;
  array[N_weight] real t_weight;      // time points scaled to [0,1]
  vector[N_weight] y_weight;          // weight observations (standardized)
  array[N_weight] int<lower=1, upper=D> day_idx;  // day index for each weight obs

  // Hour of day information for daily spline
  vector[N_weight] hour_of_day;       // Hour of day (0-24) as float
  int<lower=1> K;                     // Number of Fourier harmonics (K=1: 24h cycle, K=2: 12h + 24h, etc.)

  // Sparse GP configuration
  int<lower=0, upper=1> use_sparse;
  int<lower=0> M;                     // number of inducing points
  array[M] real t_inducing;           // inducing point locations [0,1]

  // Prediction grid (optional)
  int<lower=0> N_pred;
  array[N_pred] real t_pred;
  vector[N_pred] hour_of_day_pred;    // Hour of day for prediction (0-24)
}

parameters {
  // State-space parameters
  real<lower=0, upper=1> alpha;       // fitness decay (0 < α < 1)
  real<lower=0, upper=1> psi;         // impulse decay (0 < ψ < 1)
  real<lower=0> beta;                 // fitness gain per unit impulse (positive)
  real gamma;                         // weight effect per unit fitness

  // Observation noise
  real<lower=0.01> sigma_w;           // weight observation noise

  // GP parameters for intrinsic weight dynamics
  real<lower=0.01, upper=5> alpha_gp; // GP marginal std
  real<lower=0.01, upper=5> rho_gp;   // GP length scale

  // Fourier coefficients for daily cycles (non-centered parameterization)
  vector[K] a_sin_raw;                // Raw sine coefficients
  vector[K] a_cos_raw;                // Raw cosine coefficients

  // Prior scale for Fourier coefficients
  real<lower=0.01> sigma_fourier;     // Scale of Fourier coefficients

  // Non-centered parameterization for inducing points
  vector[M] eta_inducing_raw;         // standard normal for inducing points
}

transformed parameters {
  // Impulse and fitness states
  vector[D] impulse;
  vector[D] fitness;

  // GP covariance matrices
  matrix[M, M] K_uu;
  matrix[M, M] L_uu;
  vector[M] a;                        // a = inv(K_uu) * (L_uu * eta_inducing)

  // Fourier coefficients (non-centered transformation)
  vector[K] a_sin = sigma_fourier * a_sin_raw;
  vector[K] a_cos = sigma_fourier * a_cos_raw;

  // GP latent function at weight times
  vector[N_weight] f_gp;

  // Daily cyclic component (Fourier)
  vector[N_weight] f_daily;

  // Compute impulse states (deterministic given intensity and psi)
  impulse[1] = intensity[1];          // start with first day's intensity
  for (t in 2:D) {
    impulse[t] = psi * impulse[t-1] + intensity[t];
  }

  // Compute fitness states (deterministic function of impulse)
  fitness[1] = 0;  // Start at baseline fitness
  for (t in 2:D) {
    fitness[t] = alpha * fitness[t-1] + beta * impulse[t-1];
  }

  // GP covariance at inducing points
  K_uu = gp_exp_quad_cov_custom(t_inducing, t_inducing, alpha_gp, rho_gp);
  for (i in 1:M) {
    K_uu[i, i] += square(alpha_gp) * 1e-4 + 1e-6;  // add jitter
  }
  L_uu = cholesky_decompose(K_uu);

  // Compute a vector
  a = mdivide_left_spd(K_uu, L_uu * eta_inducing_raw);

  // Compute GP at weight observation times
  {
    matrix[N_weight, M] K_fu = gp_exp_quad_cov_custom(t_weight, t_inducing, alpha_gp, rho_gp);
    f_gp = K_fu * a;
  }

  // Daily component: Fourier basis expansion
  // f_daily[n] = Σ_k [a_sin[k] * sin(2πk * hour_scaled[n]) + a_cos[k] * cos(2πk * hour_scaled[n])]
  // where hour_scaled = hour_of_day[n] / 24.0 (maps 0-24 to 0-1)
  for (n in 1:N_weight) {
    real hour_scaled = hour_of_day[n] / 24.0;
    f_daily[n] = 0.0;
    for (k in 1:K) {
      real freq = 2.0 * pi() * k;
      f_daily[n] += a_sin[k] * sin(freq * hour_scaled) + a_cos[k] * cos(freq * hour_scaled);
    }
  }
}

model {
  // Priors for state-space parameters - informative based on successful impulse-response model
  // alpha: fitness persistence, should be high (0.8-0.95) since fitness persists
  alpha ~ beta(8, 2);                 // favors values around 0.8 (mean 0.8)
  // psi: impulse decay, should be moderate (0.5-0.8) since workout effects fade
  psi ~ beta(5, 2);                   // favors values around 0.7 (mean 0.714)
  // beta: fitness gain per impulse, should be positive but moderate
  beta ~ exponential(3.33);           // mean = 0.3, mode = 0 (positive only)
  // gamma: weight effect per fitness, should be negative (fitness reduces weight)
  gamma ~ normal(-0.5, 0.2);          // negative, stronger prior based on physiology

  // Observation noise prior
  sigma_w ~ exponential(1);           // weakly informative, mean=1, mode=0

  // GP priors - strongly constrain to prevent capturing fitness signal
  // alpha_gp: marginal std of GP, smaller values mean GP explains less variance
  alpha_gp ~ exponential(5);          // strongly favors small values (mean=0.2, mode=0)
  // rho_gp: length scale, longer values mean smoother variations
  rho_gp ~ inv_gamma(8, 1);           // favors longer length scales (mean=0.143, mode=0.111)

  // Priors for Fourier coefficients (non-centered)
  sigma_fourier ~ exponential(1);     // weakly informative (mean=1)
  a_sin_raw ~ std_normal();
  a_cos_raw ~ std_normal();

  // Priors for non-centered parameters
  eta_inducing_raw ~ std_normal();

  // Likelihood for weight observations
  for (i in 1:N_weight) {
    y_weight[i] ~ normal(gamma * fitness[day_idx[i]] + f_gp[i] + f_daily[i], sigma_w);
  }
}

generated quantities {
  // Posterior predictive for weight
  vector[N_weight] y_weight_rep;

  // Log likelihood for model comparison
  vector[N_weight] log_lik_weight;

  // Store states for analysis
  vector[D] impulse_stored = impulse;
  vector[D] fitness_stored = fitness;

  // GP predictions at weight times
  vector[N_weight] f_gp_stored = f_gp;

  // Daily component stored
  vector[N_weight] f_daily_stored = f_daily;

  // Proportion of variance from daily component
  real prop_variance_daily;

  // Amplitude of daily variation
  real daily_amplitude;

  // Posterior predictive samples
  for (i in 1:N_weight) {
    real mu = gamma * fitness[day_idx[i]] + f_gp[i] + f_daily[i];
    y_weight_rep[i] = normal_rng(mu, sigma_w);
    log_lik_weight[i] = normal_lpdf(y_weight[i] | mu, sigma_w);
  }

  // Compute daily variance metrics
  {
    real var_fitness = variance(gamma * fitness[day_idx]);
    real var_gp = variance(f_gp);
    real var_daily = variance(f_daily);
    real var_total = var_fitness + var_gp + var_daily + square(sigma_w);
    prop_variance_daily = var_daily / var_total;
    daily_amplitude = max(f_daily) - min(f_daily);
  }

  // Prediction at new time points (if requested)
  matrix[N_pred, 3] f_pred;
  matrix[N_pred, 3] y_pred;

  if (N_pred > 0) {
    // Compute GP at prediction points
    matrix[N_pred, M] K_pred_u = gp_exp_quad_cov_custom(t_pred, t_inducing, alpha_gp, rho_gp);
    vector[N_pred] f_gp_pred = K_pred_u * a;

    // Compute daily component at prediction points
    vector[N_pred] f_daily_pred;
    for (i in 1:N_pred) {
      real hour_scaled = hour_of_day_pred[i] / 24.0;
      f_daily_pred[i] = 0.0;
      for (k in 1:K) {
        real freq = 2.0 * pi() * k;
        f_daily_pred[i] += a_sin[k] * sin(freq * hour_scaled) + a_cos[k] * cos(freq * hour_scaled);
      }
    }

    // Need to map prediction times to day indices
    // For simplicity, assume prediction times align with days (could be extended)
    // Here we'll just use the nearest day (floor(t_pred * D))
    for (i in 1:N_pred) {
      int day_idx_pred = 1 + to_int(floor(t_pred[i] * D));  // approximate mapping
      if (day_idx_pred > D) day_idx_pred = D;
      if (day_idx_pred < 1) day_idx_pred = 1;

      real mu_pred = gamma * fitness[day_idx_pred] + f_gp_pred[i] + f_daily_pred[i];

      f_pred[i, 1] = gamma * fitness[day_idx_pred];
      f_pred[i, 2] = f_gp_pred[i];
      f_pred[i, 3] = f_daily_pred[i];

      y_pred[i, 1] = normal_rng(mu_pred, sigma_w);
      y_pred[i, 2] = normal_rng(f_gp_pred[i] + f_daily_pred[i], sigma_w);  // GP + daily component alone
      y_pred[i, 3] = normal_rng(f_daily_pred[i], sigma_w);  // daily component alone
    }
  } else {
    f_pred = rep_matrix(0, N_pred, 3);
    y_pred = rep_matrix(0, N_pred, 3);
  }
}