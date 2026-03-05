/*
 * State-space model with training-dependent decay for STRENGTH and AEROBIC fitness.
 * WITH INTERCEPT TERM in weight model.
 * FIXED VERSION with better priors to prevent divergent transitions.
 *
 * Two independent fitness states:
 *   strength_fitness[t] = (alpha_d_s + alpha_m_s * trained_s[t-1]) * strength_fitness[t-1] +
 *                         beta_s * strength_intensity[t-1] * trained_s[t-1]
 *
 *   aerobic_fitness[t] = (alpha_d_a + alpha_m_a * trained_a[t-1]) * aerobic_fitness[t-1] +
 *                        beta_a * aerobic_intensity[t-1] * trained_a[t-1]
 *
 * Where alpha_d and alpha_m are transformed from logit scale for each fitness type.
 *
 * Weight model WITH INTERCEPT:
 *   weight[t] = weight_intercept + gamma_s * strength_fitness[day(t)] +
 *               gamma_a * aerobic_fitness[day(t)] + ε_w[t]
 */

data {
  // Daily data
  int<lower=1> D;                     // number of days
  vector[D] strength_intensity;       // strength intensity (standardized)
  vector[D] aerobic_intensity;        // aerobic intensity (standardized)

  // Weight observations
  int<lower=1> N_weight;
  vector[N_weight] y_weight;          // weight observations (standardized)
  array[N_weight] int<lower=1, upper=D> day_idx;  // day index for each weight obs
}

transformed data {
  // Training indicators (1 if intensity > 0)
  array[D] int<lower=0, upper=1> trained_s;
  array[D] int<lower=0, upper=1> trained_a;

  for (t in 1:D) {
    trained_s[t] = strength_intensity[t] > 0 ? 1 : 0;
    trained_a[t] = aerobic_intensity[t] > 0 ? 1 : 0;
  }
}

parameters {
  // Strength fitness decay parameters on logit scale
  real alpha_d_s_logit;               // logit of decay without training
  real alpha_m_s_logit;               // logit of training effect

  // Aerobic fitness decay parameters on logit scale
  real alpha_d_a_logit;               // logit of decay without training
  real alpha_m_a_logit;               // logit of training effect

  // Fitness gain per unit intensity (with tighter priors)
  real<lower=0> beta_s;               // strength gain coefficient
  real<lower=0> beta_a;               // aerobic gain coefficient

  // Weight model parameters
  real weight_intercept;              // INTERCEPT TERM
  real gamma_s;                       // strength effect on weight
  real gamma_a;                       // aerobic effect on weight

  // Measurement noise
  real<lower=0.01> sigma_w;           // weight measurement noise
}

transformed parameters {
  // Transform from logit to (0,1) scale
  real<lower=0, upper=1> alpha_d_s = inv_logit(alpha_d_s_logit);
  real<lower=0, upper=1> alpha_m_s = inv_logit(alpha_m_s_logit);
  real<lower=0, upper=1> alpha_d_a = inv_logit(alpha_d_a_logit);
  real<lower=0, upper=1> alpha_m_a = inv_logit(alpha_m_a_logit);

  // Fitness states
  vector[D] strength_fitness;
  vector[D] aerobic_fitness;

  // Compute strength fitness state with training-dependent decay
  strength_fitness[1] = 0;

  for (t in 2:D) {
    // Total decay rate depends on whether trained yesterday
    // alpha_m is the ADDITIONAL retention when trained
    real alpha_total_s = alpha_d_s + (1 - alpha_d_s) * alpha_m_s * trained_s[t-1];

    // Gain only if trained yesterday
    real gain_s = beta_s * strength_intensity[t-1] * trained_s[t-1];

    strength_fitness[t] = alpha_total_s * strength_fitness[t-1] + gain_s;
  }

  // Compute aerobic fitness state with training-dependent decay
  aerobic_fitness[1] = 0;

  for (t in 2:D) {
    // Total decay rate depends on whether trained yesterday
    real alpha_total_a = alpha_d_a + (1 - alpha_d_a) * alpha_m_a * trained_a[t-1];

    // Gain only if trained yesterday
    real gain_a = beta_a * aerobic_intensity[t-1] * trained_a[t-1];

    aerobic_fitness[t] = alpha_total_a * aerobic_fitness[t-1] + gain_a;
  }
}

model {
  // TIGHTER PRIORS to prevent divergent transitions

  // Priors on logit scale for strength
  // Centered around slower decay (0.95-0.99 retention)
  alpha_d_s_logit ~ normal(2.9, 0.5);     // logit(0.95) ≈ 2.9, tighter prior
  alpha_m_s_logit ~ normal(0, 0.5);       // logit(0.5) = 0, tighter prior

  // Priors on logit scale for aerobic
  // Aerobic fitness decays faster than strength
  alpha_d_a_logit ~ normal(1.4, 0.5);     // logit(0.8) ≈ 1.4, tighter prior
  alpha_m_a_logit ~ normal(0, 0.5);       // logit(0.5) = 0, tighter prior

  // Tighter priors for gain coefficients to prevent explosion
  beta_s ~ exponential(2);                // mean=0.5 (tighter)
  beta_a ~ exponential(2);                // mean=0.5 (tighter)

  // Prior for weight intercept (centered around 0 since weight is standardized)
  weight_intercept ~ normal(0, 0.5);      // Tighter prior

  // Prior for weight effects with tighter constraints
  gamma_s ~ normal(0.2, 0.1);             // strength might increase weight
  gamma_a ~ normal(-0.1, 0.1);            // aerobic might decrease weight

  // Measurement noise with tighter prior
  sigma_w ~ exponential(10);              // mean=0.1 (tighter)

  // Likelihood for weight observations
  for (i in 1:N_weight) {
    real mu = weight_intercept + gamma_s * strength_fitness[day_idx[i]] +
              gamma_a * aerobic_fitness[day_idx[i]];
    y_weight[i] ~ normal(mu, sigma_w);
  }
}

generated quantities {
  // Posterior predictive for weight
  vector[N_weight] y_weight_rep;

  // Log likelihood for model comparison
  vector[N_weight] log_lik_weight;

  // Store states for analysis
  vector[D] strength_fitness_stored = strength_fitness;
  vector[D] aerobic_fitness_stored = aerobic_fitness;
  array[D] int trained_s_stored = trained_s;
  array[D] int trained_a_stored = trained_a;

  // Store total decay rates for interpretation
  vector[D] alpha_total_s;
  vector[D] alpha_total_a;

  for (t in 1:D) {
    alpha_total_s[t] = alpha_d_s + (1 - alpha_d_s) * alpha_m_s * trained_s[t];
    alpha_total_a[t] = alpha_d_a + (1 - alpha_d_a) * alpha_m_a * trained_a[t];
  }

  // Generate posterior predictive samples
  for (i in 1:N_weight) {
    real mu = weight_intercept + gamma_s * strength_fitness[day_idx[i]] +
              gamma_a * aerobic_fitness[day_idx[i]];
    y_weight_rep[i] = normal_rng(mu, sigma_w);
    log_lik_weight[i] = normal_lpdf(y_weight[i] | mu, sigma_w);
  }
}