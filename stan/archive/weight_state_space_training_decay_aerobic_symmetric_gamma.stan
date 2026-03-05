/*
 * State-space model with training-dependent decay for STRENGTH and AEROBIC fitness.
 * WITH INTERCEPT TERM in weight model.
 * SYMMETRIC PRIORS VERSION - same priors for strength and aerobic effects.
 * GAMMA OUTCOME DISTRIBUTION for positive continuous weight data.
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
 * Weight model WITH INTERCEPT and GAMMA likelihood:
 *   weight[t] ~ gamma(shape, shape / exp(linear_predictor))
 *   linear_predictor = weight_intercept + gamma_s * strength_fitness[day(t)] +
 *                      gamma_a * aerobic_fitness[day(t)]
 *
 * Using Gamma distribution parameterized by shape (alpha) and rate (beta = shape / mean)
 * where mean = exp(linear_predictor) to ensure positivity.
 */

data {
  // Daily data
  int<lower=1> D;                     // number of days
  vector[D] strength_intensity;       // strength intensity (standardized)
  vector[D] aerobic_intensity;        // aerobic intensity (standardized)

  // Weight observations - MUST BE POSITIVE for Gamma distribution
  int<lower=1> N_weight;
  vector<lower=0>[N_weight] y_weight; // weight observations (standardized, > 0)
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

  // Fitness gain per unit intensity
  real<lower=0> beta_s;               // strength gain coefficient
  real<lower=0> beta_a;               // aerobic gain coefficient

  // Weight model parameters - SYMMETRIC PRIORS
  real weight_intercept;              // INTERCEPT TERM
  real gamma_s;                       // strength effect on weight
  real gamma_a;                       // aerobic effect on weight

  // Gamma shape parameter (must be > 0)
  real<lower=0> gamma_shape;          // shape parameter for Gamma distribution
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

  // Linear predictors for weight (on log scale for Gamma mean)
  vector[N_weight] linear_predictor;

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

  // Compute linear predictors for each weight observation
  for (i in 1:N_weight) {
    linear_predictor[i] = weight_intercept +
                          gamma_s * strength_fitness[day_idx[i]] +
                          gamma_a * aerobic_fitness[day_idx[i]];
  }
}

model {
  // Priors on logit scale for strength
  alpha_d_s_logit ~ normal(2.9, 0.5);     // logit(0.95) ≈ 2.9
  alpha_m_s_logit ~ normal(0, 0.5);       // logit(0.5) = 0

  // Priors on logit scale for aerobic
  alpha_d_a_logit ~ normal(1.4, 0.5);     // logit(0.8) ≈ 1.4 (still faster decay)
  alpha_m_a_logit ~ normal(0, 0.5);       // logit(0.5) = 0

  // Priors for gain coefficients
  beta_s ~ exponential(2);                // mean=0.5
  beta_a ~ exponential(2);                // mean=0.5

  // Prior for weight intercept
  weight_intercept ~ normal(0, 0.5);

  // SYMMETRIC PRIORS for weight effects - same for both!
  gamma_s ~ normal(0, 0.2);               // Centered at 0, symmetric
  gamma_a ~ normal(0, 0.2);               // Centered at 0, symmetric

  // Prior for Gamma shape parameter
  // Exponential(1) gives mean=1, encourages moderate dispersion
  gamma_shape ~ exponential(1);

  // Likelihood for weight observations using Gamma distribution
  // Parameterization: gamma(y | shape, rate) where rate = shape / mean
  // mean = exp(linear_predictor) to ensure positivity
  for (i in 1:N_weight) {
    real mean_weight = exp(linear_predictor[i]);
    // Ensure mean_weight is not too small to avoid numerical issues
    real mean_weight_safe = fmax(mean_weight, 1e-6);
    real rate = gamma_shape / mean_weight_safe;
    y_weight[i] ~ gamma(gamma_shape, rate);
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

  // Store linear predictors and means
  vector[N_weight] linear_predictor_stored = linear_predictor;
  vector[N_weight] mean_weight;

  for (t in 1:D) {
    alpha_total_s[t] = alpha_d_s + (1 - alpha_d_s) * alpha_m_s * trained_s[t];
    alpha_total_a[t] = alpha_d_a + (1 - alpha_d_a) * alpha_m_a * trained_a[t];
  }

  for (i in 1:N_weight) {
    mean_weight[i] = exp(linear_predictor[i]);
  }

  // Generate posterior predictive samples
  for (i in 1:N_weight) {
    real mean_val = exp(linear_predictor[i]);
    // Ensure mean_val is not too small to avoid numerical issues
    real mean_val_safe = fmax(mean_val, 1e-6);
    real rate = gamma_shape / mean_val_safe;
    y_weight_rep[i] = gamma_rng(gamma_shape, rate);
    log_lik_weight[i] = gamma_lpdf(y_weight[i] | gamma_shape, rate);
  }
}