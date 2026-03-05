/*
 * BALANCED Four-fitness state-space model - Middle ground between original and tight
 *
 * Key features:
 * 1. MINIMAL GP component - Simple, low-dimensional (not flexible enough to absorb fitness effects)
 * 2. Hierarchical priors with NON-CENTERED parameterization (better convergence)
 * 3. Slightly looser observation noise prior
 * 4. Higher adapt_delta recommended (0.99)
 *
 * Short-term effects (hours to days): water weight, inflammation, glycogen
 * Long-term effects (weeks to months): muscle gain, fat loss
 */

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

  // Minimal GP: small number of basis functions
  int<lower=1> M_gp;                  // Number of GP basis functions (small, e.g., 10-20)
  matrix[N_weight, M_gp] X_gp;        // GP basis matrix (e.g., cubic B-splines)

  // Prediction grid (optional)
  int<lower=0> N_pred;
  array[N_pred] real t_pred;
  vector[N_pred] hour_of_day_pred;    // Hour of day for prediction (0-24)
  matrix[N_pred, M_gp] X_gp_pred;     // GP basis for prediction
}

parameters {
  // HIERARCHICAL HYPERPARAMETERS (NON-CENTERED)

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

  // NON-CENTERED individual parameters
  real psi_a_short_raw;
  real psi_s_short_raw;
  real psi_a_long_raw;
  real psi_s_long_raw;

  real alpha_a_short_raw;
  real alpha_s_short_raw;
  real alpha_a_long_raw;
  real alpha_s_long_raw;

  real beta_a_short_raw;
  real beta_s_short_raw;
  real beta_a_long_raw;
  real beta_s_long_raw;

  // Weight effects (key physiological parameters) - Reasonable priors
  real gamma_a_short;                  // weight effect per unit short-term aerobic fitness
  real gamma_s_short;                  // weight effect per unit short-term strength fitness
  real gamma_a_long;                   // weight effect per unit long-term aerobic fitness
  real gamma_s_long;                   // weight effect per unit long-term strength fitness

  // Observation noise - Reasonable prior
  real<lower=0.01> sigma_w;           // weight observation noise

  // Fourier coefficients for daily cycles (non-centered parameterization)
  vector[K] a_sin_raw;                // Raw sine coefficients
  vector[K] a_cos_raw;                // Raw cosine coefficients
  real<lower=0.01> sigma_fourier;     // Scale of Fourier coefficients

  // Minimal GP coefficients (low-dimensional, not flexible)
  vector[M_gp] beta_gp_raw;           // Raw GP coefficients
  real<lower=0.01> sigma_gp;          // Scale of GP coefficients
}

transformed parameters {
  // Transform non-centered parameters
  real<lower=0, upper=1> psi_a_short = inv_logit(logit(mu_psi_short) + sigma_psi_short * psi_a_short_raw);
  real<lower=0, upper=1> psi_s_short = inv_logit(logit(mu_psi_short) + sigma_psi_short * psi_s_short_raw);
  real<lower=0, upper=1> psi_a_long = inv_logit(logit(mu_psi_long) + sigma_psi_long * psi_a_long_raw);
  real<lower=0, upper=1> psi_s_long = inv_logit(logit(mu_psi_long) + sigma_psi_long * psi_s_long_raw);

  real<lower=0, upper=1> alpha_a_short = inv_logit(logit(mu_alpha_short) + sigma_alpha_short * alpha_a_short_raw);
  real<lower=0, upper=1> alpha_s_short = inv_logit(logit(mu_alpha_short) + sigma_alpha_short * alpha_s_short_raw);
  real<lower=0, upper=1> alpha_a_long = inv_logit(logit(mu_alpha_long) + sigma_alpha_long * alpha_a_long_raw);
  real<lower=0, upper=1> alpha_s_long = inv_logit(logit(mu_alpha_long) + sigma_alpha_long * alpha_s_long_raw);

  real<lower=0> beta_a_short = exp(log(mu_beta_short) + sigma_beta_short * beta_a_short_raw);
  real<lower=0> beta_s_short = exp(log(mu_beta_short) + sigma_beta_short * beta_s_short_raw);
  real<lower=0> beta_a_long = exp(log(mu_beta_long) + sigma_beta_long * beta_a_long_raw);
  real<lower=0> beta_s_long = exp(log(mu_beta_long) + sigma_beta_long * beta_s_long_raw);

  // Impulse and fitness states for all four components
  vector[D] impulse_a_short;
  vector[D] impulse_s_short;
  vector[D] impulse_a_long;
  vector[D] impulse_s_long;

  vector[D] fitness_a_short;
  vector[D] fitness_s_short;
  vector[D] fitness_a_long;
  vector[D] fitness_s_long;

  // Fourier coefficients
  vector[K] a_sin = sigma_fourier * a_sin_raw;
  vector[K] a_cos = sigma_fourier * a_cos_raw;

  // Minimal GP component
  vector[M_gp] beta_gp = sigma_gp * beta_gp_raw;
  vector[N_weight] f_gp = X_gp * beta_gp;

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
  sigma_psi_short ~ exponential(5);    // moderate: mean=0.2

  mu_alpha_short ~ beta(4, 4);         // favors ~0.5 (moderate decay)
  sigma_alpha_short ~ exponential(5);  // moderate: mean=0.2

  mu_beta_short ~ exponential(5);      // mean=0.2, mode=0
  sigma_beta_short ~ exponential(5);   // moderate: mean=0.2

  // Long-term hyperpriors
  mu_psi_long ~ beta(5, 2);            // favors ~0.714 (slow decay)
  sigma_psi_long ~ exponential(5);     // moderate: mean=0.2

  mu_alpha_long ~ beta(8, 2);          // favors ~0.8 (slow decay)
  sigma_alpha_long ~ exponential(5);   // moderate: mean=0.2

  mu_beta_long ~ exponential(10);      // mean=0.1, mode=0 (smaller gain)
  sigma_beta_long ~ exponential(5);    // moderate: mean=0.2

  // Priors for non-centered parameters
  psi_a_short_raw ~ std_normal();
  psi_s_short_raw ~ std_normal();
  psi_a_long_raw ~ std_normal();
  psi_s_long_raw ~ std_normal();

  alpha_a_short_raw ~ std_normal();
  alpha_s_short_raw ~ std_normal();
  alpha_a_long_raw ~ std_normal();
  alpha_s_long_raw ~ std_normal();

  beta_a_short_raw ~ std_normal();
  beta_s_short_raw ~ std_normal();
  beta_a_long_raw ~ std_normal();
  beta_s_long_raw ~ std_normal();

  // Reasonable priors: Weight effects based on physiology
  gamma_a_short ~ normal(-0.3, 0.1);   // dehydration reduces weight
  gamma_s_short ~ normal(0.2, 0.1);    // inflammation/water retention increases weight
  gamma_a_long ~ normal(-0.2, 0.05);   // fat loss reduces weight
  gamma_s_long ~ normal(0.3, 0.1);     // muscle gain increases weight

  // Observation noise prior
  sigma_w ~ exponential(2);            // moderate: mean=0.5

  // Priors for Fourier coefficients
  sigma_fourier ~ exponential(1);      // weakly informative (mean=1)
  a_sin_raw ~ std_normal();
  a_cos_raw ~ std_normal();

  // Priors for minimal GP
  sigma_gp ~ exponential(2);           // moderate: mean=0.5
  beta_gp_raw ~ std_normal();

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
    vector[N_pred] f_gp_pred = X_gp_pred * beta_gp;

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