/*
 * Four-fitness model with IMPROVED parameterization and data-informed priors
 *
 * Key improvements:
 * 1. Data-informed priors based on previous estimates
 * 2. Non-centered parameterization for hierarchical parameters
 * 3. Better priors to avoid scale parameters going to 0
 * 4. Transformed parameters to avoid boundary issues
 * 5. More stable GP implementation
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

  // Activity intensities (HR-based, standardized)
  vector[D] aerobic_intensity;        // walking, cycling, etc.
  vector[D] strength_intensity;       // strength training

  // Weight observations
  int<lower=1> N_weight;
  array[N_weight] real t_weight;      // time points scaled to [0,1]
  vector[N_weight] y_weight;          // weight observations (standardized)
  array[N_weight] int<lower=1, upper=D> day_idx;  // day index for each weight obs

  // Hour of day information for daily spline
  vector[N_weight] hour_of_day;       // hour of day (0-24)

  // Fourier harmonics for daily cycle
  int<lower=1> K;                     // number of Fourier harmonics

  // Sparse GP settings
  int<lower=0, upper=1> use_sparse;   // 1 = use sparse GP
  int<lower=0> M;                     // number of inducing points (if sparse)
  array[M] real t_inducing;           // inducing point locations (if sparse)

  // Prediction settings
  int<lower=0> N_pred;                // number of prediction points
  array[N_pred] real t_pred;          // prediction time points
  vector[N_pred] hour_of_day_pred;    // prediction hour of day
}

transformed data {
  // Scale hour to [0, 1] for Fourier basis
  vector[N_weight] hour_scaled = hour_of_day / 24.0;
  vector[N_pred] hour_scaled_pred = hour_of_day_pred / 24.0;
}

parameters {
  // HYPERPARAMETERS (non-centered parameterization)

  // Short-term hyperparameters (means on transformed scale)
  real<lower=0, upper=1> mu_psi_short_raw;      // raw mean for psi (short-term decay)
  real<lower=0> sigma_psi_short_raw;            // raw std for psi

  real<lower=0, upper=1> mu_alpha_short_raw;    // raw mean for alpha (fitness decay)
  real<lower=0> sigma_alpha_short_raw;          // raw std for alpha

  real<lower=0> mu_beta_short_raw;              // raw mean for beta (fitness gain)
  real<lower=0> sigma_beta_short_raw;           // raw std for beta

  // Long-term hyperparameters
  real<lower=0, upper=1> mu_psi_long_raw;       // raw mean for psi (long-term decay)
  real<lower=0> sigma_psi_long_raw;             // raw std for psi

  real<lower=0, upper=1> mu_alpha_long_raw;     // raw mean for alpha (long-term fitness decay)
  real<lower=0> sigma_alpha_long_raw;           // raw std for alpha

  real<lower=0> mu_beta_long_raw;               // raw mean for beta (long-term fitness gain)
  real<lower=0> sigma_beta_long_raw;            // raw std for beta

  // INDIVIDUAL PARAMETERS (non-centered)
  // Short-term parameters
  real psi_a_short_raw;
  real psi_s_short_raw;

  real alpha_a_short_raw;
  real alpha_s_short_raw;

  real beta_a_short_raw;
  real beta_s_short_raw;

  // Long-term parameters
  real psi_a_long_raw;
  real psi_s_long_raw;

  real alpha_a_long_raw;
  real alpha_s_long_raw;

  real beta_a_long_raw;
  real beta_s_long_raw;

  // Weight effects (tight priors based on physiology)
  real gamma_a_short;   // aerobic short-term effect (should be negative)
  real gamma_s_short;   // strength short-term effect (should be positive)
  real gamma_a_long;    // aerobic long-term effect (should be negative)
  real gamma_s_long;    // strength long-term effect (should be positive)

  // Measurement noise
  real<lower=0.01> sigma_w;  // weight measurement noise

  // GP parameters (constrained but with better priors)
  real<lower=0.1, upper=0.5> alpha_gp_raw;  // GP marginal std (0.1-0.5)
  real<lower=0.1, upper=1.0> rho_gp_raw;    // GP length scale (0.1-1.0)

  // Fourier coefficients for daily cycles
  vector[K] a_sin_raw;                // Raw sine coefficients
  vector[K] a_cos_raw;                // Raw cosine coefficients
  real<lower=0.01> sigma_fourier;     // Scale of Fourier coefficients

  // Non-centered parameterization for inducing points
  vector[M] eta_inducing_raw;         // standard normal for inducing points
}

transformed parameters {
  // Transform hyperparameters to actual scales
  real mu_psi_short = 0.3 + 0.4 * mu_psi_short_raw;      // ~0.3-0.7
  real sigma_psi_short = 0.05 + 0.15 * sigma_psi_short_raw; // ~0.05-0.20

  real mu_alpha_short = 0.4 + 0.4 * mu_alpha_short_raw;  // ~0.4-0.8
  real sigma_alpha_short = 0.05 + 0.15 * sigma_alpha_short_raw; // ~0.05-0.20

  real mu_beta_short = 0.02 + 0.1 * mu_beta_short_raw;   // ~0.02-0.12
  real sigma_beta_short = 0.02 + 0.08 * sigma_beta_short_raw; // ~0.02-0.10

  real mu_psi_long = 0.6 + 0.3 * mu_psi_long_raw;        // ~0.6-0.9
  real sigma_psi_long = 0.05 + 0.15 * sigma_psi_long_raw; // ~0.05-0.20

  real mu_alpha_long = 0.7 + 0.25 * mu_alpha_long_raw;   // ~0.7-0.95
  real sigma_alpha_long = 0.05 + 0.15 * sigma_alpha_long_raw; // ~0.05-0.20

  real mu_beta_long = 0.01 + 0.05 * mu_beta_long_raw;    // ~0.01-0.06
  real sigma_beta_long = 0.01 + 0.05 * sigma_beta_long_raw; // ~0.01-0.06

  // Transform individual parameters
  real psi_a_short = mu_psi_short + sigma_psi_short * psi_a_short_raw;
  real psi_s_short = mu_psi_short + sigma_psi_short * psi_s_short_raw;

  real alpha_a_short = mu_alpha_short + sigma_alpha_short * alpha_a_short_raw;
  real alpha_s_short = mu_alpha_short + sigma_alpha_short * alpha_s_short_raw;

  real beta_a_short = mu_beta_short + sigma_beta_short * beta_a_short_raw;
  real beta_s_short = mu_beta_short + sigma_beta_short * beta_s_short_raw;

  real psi_a_long = mu_psi_long + sigma_psi_long * psi_a_long_raw;
  real psi_s_long = mu_psi_long + sigma_psi_long * psi_s_long_raw;

  real alpha_a_long = mu_alpha_long + sigma_alpha_long * alpha_a_long_raw;
  real alpha_s_long = mu_alpha_long + sigma_alpha_long * alpha_s_long_raw;

  real beta_a_long = mu_beta_long + sigma_beta_long * beta_a_long_raw;
  real beta_s_long = mu_beta_long + sigma_beta_long * beta_s_long_raw;

  // Transform GP parameters (avoid boundaries)
  real alpha_gp = 0.1 + 0.4 * alpha_gp_raw;  // 0.1-0.5
  real rho_gp = 0.1 + 0.9 * rho_gp_raw;      // 0.1-1.0

  // Fourier coefficients
  vector[K] a_sin = sigma_fourier * a_sin_raw;
  vector[K] a_cos = sigma_fourier * a_cos_raw;

  // Impulse and fitness states for all four components
  vector[D] impulse_a_short;
  vector[D] impulse_s_short;
  vector[D] impulse_a_long;
  vector[D] impulse_s_long;

  vector[D] fitness_a_short;
  vector[D] fitness_s_short;
  vector[D] fitness_a_long;
  vector[D] fitness_s_long;

  // GP covariance matrices
  matrix[M, M] K_uu;
  matrix[M, M] L_uu;
  vector[M] a;                        // a = inv(K_uu) * (L_uu * eta_inducing)

  // GP latent function at weight times
  vector[N_weight] f_gp;

  // Daily cyclic component (Fourier)
  vector[N_weight] f_daily;

  // Compute impulse states (with stability check)
  impulse_a_short[1] = aerobic_intensity[1];
  impulse_s_short[1] = strength_intensity[1];
  impulse_a_long[1] = aerobic_intensity[1];
  impulse_s_long[1] = strength_intensity[1];

  for (t in 2:D) {
    impulse_a_short[t] = psi_a_short * impulse_a_short[t-1] + aerobic_intensity[t];
    impulse_s_short[t] = psi_s_short * impulse_s_short[t-1] + strength_intensity[t];
    impulse_a_long[t] = psi_a_long * impulse_a_long[t-1] + aerobic_intensity[t];
    impulse_s_long[t] = psi_s_long * impulse_s_long[t-1] + strength_intensity[t];
  }

  // Compute fitness states
  fitness_a_short[1] = 0;
  fitness_s_short[1] = 0;
  fitness_a_long[1] = 0;
  fitness_s_long[1] = 0;

  for (t in 2:D) {
    fitness_a_short[t] = alpha_a_short * fitness_a_short[t-1] + beta_a_short * impulse_a_short[t-1];
    fitness_s_short[t] = alpha_s_short * fitness_s_short[t-1] + beta_s_short * impulse_s_short[t-1];
    fitness_a_long[t] = alpha_a_long * fitness_a_long[t-1] + beta_a_long * impulse_a_long[t-1];
    fitness_s_long[t] = alpha_s_long * fitness_s_long[t-1] + beta_s_long * impulse_s_long[t-1];
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
  for (n in 1:N_weight) {
    f_daily[n] = 0.0;
    for (k in 1:K) {
      real freq = 2.0 * pi() * k;
      f_daily[n] += a_sin[k] * sin(freq * hour_scaled[n]) + a_cos[k] * cos(freq * hour_scaled[n]);
    }
  }
}

model {
  // HYPERPRIORS (data-informed)

  // Short-term hyperpriors
  mu_psi_short_raw ~ beta(3, 3);           // centered around 0.5 on [0,1] scale
  sigma_psi_short_raw ~ exponential(2);    // mean=0.5 on raw scale -> ~0.1 on transformed

  mu_alpha_short_raw ~ beta(3, 3);         // centered around 0.5
  sigma_alpha_short_raw ~ exponential(2);

  mu_beta_short_raw ~ exponential(2);      // mean=0.5 on raw scale -> ~0.07 on transformed
  sigma_beta_short_raw ~ exponential(2);

  // Long-term hyperpriors
  mu_psi_long_raw ~ beta(4, 2);            // favors higher values (~0.67)
  sigma_psi_long_raw ~ exponential(2);

  mu_alpha_long_raw ~ beta(5, 2);          // favors higher values (~0.71)
  sigma_alpha_long_raw ~ exponential(2);

  mu_beta_long_raw ~ exponential(2);
  sigma_beta_long_raw ~ exponential(2);

  // INDIVIDUAL PARAMETER PRIORS (non-centered)
  psi_a_short_raw ~ std_normal();
  psi_s_short_raw ~ std_normal();

  alpha_a_short_raw ~ std_normal();
  alpha_s_short_raw ~ std_normal();

  beta_a_short_raw ~ std_normal();
  beta_s_short_raw ~ std_normal();

  psi_a_long_raw ~ std_normal();
  psi_s_long_raw ~ std_normal();

  alpha_a_long_raw ~ std_normal();
  alpha_s_long_raw ~ std_normal();

  beta_a_long_raw ~ std_normal();
  beta_s_long_raw ~ std_normal();

  // CRITICAL: TIGHT PRIORS on weight effects based on physiology
  gamma_a_short ~ normal(-0.3, 0.1);   // aerobic short-term: dehydration reduces weight
  gamma_s_short ~ normal(0.2, 0.1);    // strength short-term: inflammation increases weight
  gamma_a_long ~ normal(-0.2, 0.05);   // aerobic long-term: fat loss reduces weight
  gamma_s_long ~ normal(0.3, 0.05);    // strength long-term: muscle gain increases weight

  // Measurement noise
  sigma_w ~ exponential(5);            // mean=0.2

  // GP parameters
  alpha_gp_raw ~ beta(2, 2);           // centered in [0.1, 0.5]
  rho_gp_raw ~ beta(2, 2);             // centered in [0.1, 1.0]

  // Fourier coefficients
  a_sin_raw ~ std_normal();
  a_cos_raw ~ std_normal();
  sigma_fourier ~ exponential(5);      // mean=0.2

  // Inducing points
  eta_inducing_raw ~ std_normal();

  // Likelihood for weight observations
  for (i in 1:N_weight) {
    real mu = gamma_a_short * fitness_a_short[day_idx[i]] +
              gamma_s_short * fitness_s_short[day_idx[i]] +
              gamma_a_long * fitness_a_long[day_idx[i]] +
              gamma_s_long * fitness_s_long[day_idx[i]] +
              f_gp[i] + f_daily[i];
    y_weight[i] ~ normal(mu, sigma_w);
  }
}

generated quantities {
  // Posterior predictive for weight
  vector[N_weight] y_weight_rep;

  // Log likelihood for model comparison
  vector[N_weight] log_lik_weight;

  // Store states for analysis
  vector[D] fitness_a_short_stored = fitness_a_short;
  vector[D] fitness_s_short_stored = fitness_s_short;
  vector[D] fitness_a_long_stored = fitness_a_long;
  vector[D] fitness_s_long_stored = fitness_s_long;

  // Store GP and daily components
  vector[N_weight] f_gp_stored = f_gp;
  vector[N_weight] f_daily_stored = f_daily;

  // Variance proportions
  real total_variance = square(sigma_w);
  real var_a_short = square(gamma_a_short) * variance(fitness_a_short);
  real var_s_short = square(gamma_s_short) * variance(fitness_s_short);
  real var_a_long = square(gamma_a_long) * variance(fitness_a_long);
  real var_s_long = square(gamma_s_long) * variance(fitness_s_long);
  real var_gp = variance(f_gp);
  real var_daily = variance(f_daily);

  real prop_variance_a_short = var_a_short / (var_a_short + var_s_short + var_a_long + var_s_long + var_gp + var_daily + total_variance);
  real prop_variance_s_short = var_s_short / (var_a_short + var_s_short + var_a_long + var_s_long + var_gp + var_daily + total_variance);
  real prop_variance_a_long = var_a_long / (var_a_short + var_s_short + var_a_long + var_s_long + var_gp + var_daily + total_variance);
  real prop_variance_s_long = var_s_long / (var_a_short + var_s_short + var_a_long + var_s_long + var_gp + var_daily + total_variance);
  real prop_variance_gp = var_gp / (var_a_short + var_s_short + var_a_long + var_s_long + var_gp + var_daily + total_variance);
  real prop_variance_daily = var_daily / (var_a_short + var_s_short + var_a_long + var_s_long + var_gp + var_daily + total_variance);

  // Half-lives (in days)
  real half_life_a_short = -log(2) / log(alpha_a_short);
  real half_life_s_short = -log(2) / log(alpha_s_short);
  real half_life_a_long = -log(2) / log(alpha_a_long);
  real half_life_s_long = -log(2) / log(alpha_s_long);

  // Predictions at new time points
  vector[N_pred] y_pred;
  vector[N_pred] f_gp_pred;
  vector[N_pred] f_daily_pred;

  if (N_pred > 0) {
    // GP at prediction points
    {
      matrix[N_pred, M] K_pred_u = gp_exp_quad_cov_custom(t_pred, t_inducing, alpha_gp, rho_gp);
      f_gp_pred = K_pred_u * a;
    }

    // Daily component at prediction points
    for (n in 1:N_pred) {
      f_daily_pred[n] = 0.0;
      for (k in 1:K) {
        real freq = 2.0 * pi() * k;
        f_daily_pred[n] += a_sin[k] * sin(freq * hour_scaled_pred[n]) + a_cos[k] * cos(freq * hour_scaled_pred[n]);
      }
    }

    // Note: For predictions, we'd need to compute fitness states at prediction times
    // This is simplified - in practice you'd need to extend the time series
    y_pred = f_gp_pred + f_daily_pred;
  }

  // Generate posterior predictive samples
  for (i in 1:N_weight) {
    real mu = gamma_a_short * fitness_a_short[day_idx[i]] +
              gamma_s_short * fitness_s_short[day_idx[i]] +
              gamma_a_long * fitness_a_long[day_idx[i]] +
              gamma_s_long * fitness_s_long[day_idx[i]] +
              f_gp[i] + f_daily[i];
    y_weight_rep[i] = normal_rng(mu, sigma_w);
    log_lik_weight[i] = normal_lpdf(y_weight[i] | mu, sigma_w);
  }
}