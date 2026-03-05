/*
 * Four-fitness model with CONSTRAINED GP - GP for intrinsic weight variations
 * but constrained to not absorb fitness effects.
 *
 * Key changes:
 * 1. KEEP GP component (weight has intrinsic variations)
 * 2. TIGHTER GP priors (smaller variance, longer length scale)
 * 3. Hierarchical priors for fitness parameters
 * 4. Tight priors on weight effects (gamma parameters)
 *
 * Philosophy: Weight = Fitness Effects + Intrinsic Variations (GP) + Daily + Noise
 * GP should capture slow, smooth intrinsic variations, not fitness patterns.
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
  vector[N_weight] hour_of_day;       // Hour of day (0-24) as float
  int<lower=1> K;                     // Number of Fourier harmonics

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
  // HIERARCHICAL HYPERPARAMETERS

  // Short-term hyperparameters
  real<lower=0, upper=1> mu_psi_short;      // mean impulse decay for short-term
  real<lower=0> sigma_psi_short;            // std of impulse decay for short-term

  real<lower=0, upper=1> mu_alpha_short;    // mean fitness decay for short-term
  real<lower=0> sigma_alpha_short;          // std of fitness decay for short-term

  real<lower=0> mu_beta_short;              // mean fitness gain for short-term
  real<lower=0> sigma_beta_short;           // std of fitness gain for short-term

  // Long-term hyperparameters
  real<lower=0, upper=1> mu_psi_long;       // mean impulse decay for long-term
  real<lower=0> sigma_psi_long;             // std of impulse decay for long-term

  real<lower=0, upper=1> mu_alpha_long;     // mean fitness decay for long-term
  real<lower=0> sigma_alpha_long;           // std of fitness decay for long-term

  real<lower=0> mu_beta_long;               // mean fitness gain for long-term
  real<lower=0> sigma_beta_long;            // std of fitness gain for long-term

  // INDIVIDUAL PARAMETERS WITH HIERARCHICAL STRUCTURE

  // Short-term impulse decay parameters
  real<lower=0, upper=1> psi_a_short;  // aerobic short-term impulse decay
  real<lower=0, upper=1> psi_s_short;  // strength short-term impulse decay

  // Long-term impulse decay parameters
  real<lower=0, upper=1> psi_a_long;   // aerobic long-term impulse decay
  real<lower=0, upper=1> psi_s_long;   // strength long-term impulse decay

  // Fitness decay parameters
  real<lower=0, upper=1> alpha_a_short; // aerobic short-term fitness decay
  real<lower=0, upper=1> alpha_s_short; // strength short-term fitness decay
  real<lower=0, upper=1> alpha_a_long;  // aerobic long-term fitness decay
  real<lower=0, upper=1> alpha_s_long;  // strength long-term fitness decay

  // Fitness gain parameters
  real<lower=0> beta_a_short;          // aerobic short-term fitness gain
  real<lower=0> beta_s_short;          // strength short-term fitness gain
  real<lower=0> beta_a_long;           // aerobic long-term fitness gain
  real<lower=0> beta_s_long;           // strength long-term fitness gain

  // Weight effects - TIGHT PRIORS (key to preventing GP absorption)
  real gamma_a_short;                  // weight effect per unit short-term aerobic fitness
  real gamma_s_short;                  // weight effect per unit short-term strength fitness
  real gamma_a_long;                   // weight effect per unit long-term aerobic fitness
  real gamma_s_long;                   // weight effect per unit long-term strength fitness

  // Observation noise
  real<lower=0.01> sigma_w;           // weight observation noise

  // GP parameters for intrinsic weight dynamics - CONSTRAINED
  real<lower=0.001, upper=0.5> alpha_gp; // GP marginal std (TIGHT: max 0.5)
  real<lower=0.05, upper=1.0> rho_gp;    // GP length scale (LONG: 0.05-1.0)

  // Fourier coefficients for daily cycles
  vector[K] a_sin_raw;                // Raw sine coefficients
  vector[K] a_cos_raw;                // Raw cosine coefficients
  real<lower=0.01> sigma_fourier;     // Scale of Fourier coefficients

  // Non-centered parameterization for inducing points
  vector[M] eta_inducing_raw;         // standard normal for inducing points
}

transformed parameters {
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

  // Fourier coefficients
  vector[K] a_sin = sigma_fourier * a_sin_raw;
  vector[K] a_cos = sigma_fourier * a_cos_raw;

  // GP latent function at weight times
  vector[N_weight] f_gp;

  // Daily cyclic component (Fourier)
  vector[N_weight] f_daily;

  // Compute impulse states
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
    real hour_scaled = hour_of_day[n] / 24.0;
    f_daily[n] = 0.0;
    for (k in 1:K) {
      real freq = 2.0 * pi() * k;
      f_daily[n] += a_sin[k] * sin(freq * hour_scaled) + a_cos[k] * cos(freq * hour_scaled);
    }
  }
}

model {
  // HYPERPRIORS

  // Short-term hyperpriors
  mu_psi_short ~ beta(3, 5);           // favors ~0.375 (fast decay)
  sigma_psi_short ~ exponential(10);   // tight: mean=0.1

  mu_alpha_short ~ beta(4, 4);         // favors ~0.5 (moderate decay)
  sigma_alpha_short ~ exponential(10); // tight: mean=0.1

  mu_beta_short ~ exponential(5);      // mean=0.2, mode=0
  sigma_beta_short ~ exponential(10);  // tight: mean=0.1

  // Long-term hyperpriors
  mu_psi_long ~ beta(5, 2);            // favors ~0.714 (slow decay)
  sigma_psi_long ~ exponential(10);    // tight: mean=0.1

  mu_alpha_long ~ beta(8, 2);          // favors ~0.8 (slow decay)
  sigma_alpha_long ~ exponential(10);  // tight: mean=0.1

  mu_beta_long ~ exponential(10);      // mean=0.1, mode=0 (smaller gain)
  sigma_beta_long ~ exponential(10);   // tight: mean=0.1

  // HIERARCHICAL PRIORS FOR INDIVIDUAL PARAMETERS

  // Short-term parameters
  psi_a_short ~ normal(mu_psi_short, sigma_psi_short);
  psi_s_short ~ normal(mu_psi_short, sigma_psi_short);

  alpha_a_short ~ normal(mu_alpha_short, sigma_alpha_short);
  alpha_s_short ~ normal(mu_alpha_short, sigma_alpha_short);

  beta_a_short ~ normal(mu_beta_short, sigma_beta_short);
  beta_s_short ~ normal(mu_beta_short, sigma_beta_short);

  // Long-term parameters
  psi_a_long ~ normal(mu_psi_long, sigma_psi_long);
  psi_s_long ~ normal(mu_psi_long, sigma_psi_long);

  alpha_a_long ~ normal(mu_alpha_long, sigma_alpha_long);
  alpha_s_long ~ normal(mu_alpha_long, sigma_alpha_long);

  beta_a_long ~ normal(mu_beta_long, sigma_beta_long);
  beta_s_long ~ normal(mu_beta_long, sigma_beta_long);

  // CRITICAL: TIGHT PRIORS on weight effects - prevents GP absorption
  gamma_a_short ~ normal(-0.3, 0.05);   // tight: dehydration reduces weight
  gamma_s_short ~ normal(0.2, 0.05);    // tight: inflammation/water retention increases weight
  gamma_a_long ~ normal(-0.2, 0.03);    // very tight: fat loss reduces weight
  gamma_s_long ~ normal(0.3, 0.05);     // tight: muscle gain increases weight

  // Observation noise prior
  sigma_w ~ exponential(2);            // moderate: mean=0.5

  // CONSTRAINED GP priors - prevents absorbing fitness effects
  // alpha_gp has tight bounds in parameter declaration (0.001-0.5)
  // rho_gp has bounds favoring long length scales (0.05-1.0)
  // Additional prior to favor small values
  alpha_gp ~ exponential(10);          // strongly favors small values (mean=0.1)
  rho_gp ~ beta(5, 2);                 // favors longer length scales (~0.714)

  // Priors for Fourier coefficients
  sigma_fourier ~ exponential(1);     // weakly informative (mean=1)
  a_sin_raw ~ std_normal();
  a_cos_raw ~ std_normal();

  // Priors for non-centered parameters
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

  // Store components
  vector[N_weight] f_gp_stored = f_gp;
  vector[N_weight] f_daily_stored = f_daily;

  // Hyperparameters stored
  real mu_psi_short_stored = mu_psi_short;
  real mu_alpha_short_stored = mu_alpha_short;
  real mu_beta_short_stored = mu_beta_short;
  real mu_psi_long_stored = mu_psi_long;
  real mu_alpha_long_stored = mu_alpha_long;
  real mu_beta_long_stored = mu_beta_long;

  // Proportion of variance from each component
  real prop_variance_a_short;
  real prop_variance_s_short;
  real prop_variance_a_long;
  real prop_variance_s_long;
  real prop_variance_daily;
  real prop_variance_gp;

  // Compute half-lives for interpretability
  real half_life_a_short = -log(0.5) / (-log(alpha_a_short + 1e-10));
  real half_life_s_short = -log(0.5) / (-log(alpha_s_short + 1e-10));
  real half_life_a_long = -log(0.5) / (-log(alpha_a_long + 1e-10));
  real half_life_s_long = -log(0.5) / (-log(alpha_s_long + 1e-10));

  // Posterior predictive samples
  for (i in 1:N_weight) {
    real mu = gamma_a_short * fitness_a_short[day_idx[i]] +
              gamma_s_short * fitness_s_short[day_idx[i]] +
              gamma_a_long * fitness_a_long[day_idx[i]] +
              gamma_s_long * fitness_s_long[day_idx[i]] +
              f_gp[i] + f_daily[i];
    y_weight_rep[i] = normal_rng(mu, sigma_w);
    log_lik_weight[i] = normal_lpdf(y_weight[i] | mu, sigma_w);
  }

  // Compute variance decomposition
  {
    real var_a_short = variance(gamma_a_short * fitness_a_short[day_idx]);
    real var_s_short = variance(gamma_s_short * fitness_s_short[day_idx]);
    real var_a_long = variance(gamma_a_long * fitness_a_long[day_idx]);
    real var_s_long = variance(gamma_s_long * fitness_s_long[day_idx]);
    real var_gp = variance(f_gp);
    real var_daily = variance(f_daily);
    real var_total = var_a_short + var_s_short + var_a_long + var_s_long +
                     var_gp + var_daily + square(sigma_w);

    prop_variance_a_short = var_a_short / var_total;
    prop_variance_s_short = var_s_short / var_total;
    prop_variance_a_long = var_a_long / var_total;
    prop_variance_s_long = var_s_long / var_total;
    prop_variance_gp = var_gp / var_total;
    prop_variance_daily = var_daily / var_total;
  }

  // Prediction at new time points (if requested)
  matrix[N_pred, 6] f_pred;
  matrix[N_pred, 6] y_pred;

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
    for (i in 1:N_pred) {
      int day_idx_pred = 1 + to_int(floor(t_pred[i] * D));  // approximate mapping
      if (day_idx_pred > D) day_idx_pred = D;
      if (day_idx_pred < 1) day_idx_pred = 1;

      real mu_pred = gamma_a_short * fitness_a_short[day_idx_pred] +
                     gamma_s_short * fitness_s_short[day_idx_pred] +
                     gamma_a_long * fitness_a_long[day_idx_pred] +
                     gamma_s_long * fitness_s_long[day_idx_pred] +
                     f_gp_pred[i] + f_daily_pred[i];

      f_pred[i, 1] = gamma_a_short * fitness_a_short[day_idx_pred];
      f_pred[i, 2] = gamma_s_short * fitness_s_short[day_idx_pred];
      f_pred[i, 3] = gamma_a_long * fitness_a_long[day_idx_pred];
      f_pred[i, 4] = gamma_s_long * fitness_s_long[day_idx_pred];
      f_pred[i, 5] = f_gp_pred[i];
      f_pred[i, 6] = f_daily_pred[i];

      y_pred[i, 1] = normal_rng(mu_pred, sigma_w);
      y_pred[i, 2] = normal_rng(gamma_s_short * fitness_s_short[day_idx_pred] +
                               gamma_s_long * fitness_s_long[day_idx_pred] +
                               f_gp_pred[i] + f_daily_pred[i], sigma_w);
      y_pred[i, 3] = normal_rng(gamma_a_short * fitness_a_short[day_idx_pred] +
                               gamma_a_long * fitness_a_long[day_idx_pred] +
                               f_gp_pred[i] + f_daily_pred[i], sigma_w);
      y_pred[i, 4] = normal_rng(gamma_s_long * fitness_s_long[day_idx_pred] +
                               f_gp_pred[i] + f_daily_pred[i], sigma_w);
      y_pred[i, 5] = normal_rng(gamma_a_long * fitness_a_long[day_idx_pred] +
                               f_gp_pred[i] + f_daily_pred[i], sigma_w);
      y_pred[i, 6] = normal_rng(f_gp_pred[i] + f_daily_pred[i], sigma_w);
    }
  } else {
    f_pred = rep_matrix(0, N_pred, 6);
    y_pred = rep_matrix(0, N_pred, 6);
  }
}