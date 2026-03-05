/*
 * State-space model with training-dependent decay for STRENGTH fitness only.
 * Parameterized on logit scale for better sampling.
 *
 * Fitness state evolution:
 *   trained[t] = 1 if strength_intensity[t] > 0, else 0
 *   fitness[t] = (alpha_d + alpha_m * trained[t-1]) * fitness[t-1] +
 *                beta * strength_intensity[t-1] * trained[t-1]
 *
 * Where alpha_d and alpha_m are transformed from logit scale.
 */

data {
  // Daily data
  int<lower=1> D;                     // number of days
  vector[D] strength_intensity;       // strength intensity (standardized)

  // Weight observations
  int<lower=1> N_weight;
  vector[N_weight] y_weight;          // weight observations (standardized)
  array[N_weight] int<lower=1, upper=D> day_idx;  // day index for each weight obs
}

transformed data {
  // Training indicators (1 if intensity > 0)
  array[D] int<lower=0, upper=1> trained;

  for (t in 1:D) {
    trained[t] = strength_intensity[t] > 0 ? 1 : 0;
  }
}

parameters {
  // Fitness decay parameters on logit scale
  real alpha_d_logit;                 // logit of decay without training
  real alpha_m_logit;                 // logit of training effect

  // Fitness gain per unit intensity
  real<lower=0> beta;                 // gain coefficient

  // Weight effect
  real gamma;                         // strength effect on weight

  // Measurement noise
  real<lower=0.01> sigma_w;           // weight measurement noise
}

transformed parameters {
  // Transform from logit to (0,1) scale
  real<lower=0, upper=1> alpha_d = inv_logit(alpha_d_logit);
  real<lower=0, upper=1> alpha_m = inv_logit(alpha_m_logit);

  // Fitness state
  vector[D] fitness;

  // Compute fitness state with training-dependent decay
  fitness[1] = 0;

  for (t in 2:D) {
    // Total decay rate depends on whether trained yesterday
    // alpha_m is the ADDITIONAL retention when trained
    real alpha_total = alpha_d + (1 - alpha_d) * alpha_m * trained[t-1];

    // Gain only if trained yesterday
    real gain = beta * strength_intensity[t-1] * trained[t-1];

    fitness[t] = alpha_total * fitness[t-1] + gain;
  }
}

model {
  // Priors on logit scale
  // alpha_d_logit ~ normal(logit(0.9), 1) centers around 0.9
  alpha_d_logit ~ normal(2.2, 1);     // logit(0.9) ≈ 2.2

  // alpha_m_logit ~ normal(logit(0.5), 1) centers around 0.5
  alpha_m_logit ~ normal(0, 1);       // logit(0.5) = 0

  // Prior for gain coefficient
  beta ~ exponential(1);              // mean=1.0

  // Prior for weight effect
  gamma ~ normal(0.2, 0.2);           // strength might increase weight (muscle)

  // Measurement noise
  sigma_w ~ exponential(5);           // mean=0.2

  // Likelihood for weight observations
  for (i in 1:N_weight) {
    real mu = gamma * fitness[day_idx[i]];
    y_weight[i] ~ normal(mu, sigma_w);
  }
}

generated quantities {
  // Posterior predictive for weight
  vector[N_weight] y_weight_rep;

  // Log likelihood for model comparison
  vector[N_weight] log_lik_weight;

  // Store states for analysis
  vector[D] fitness_stored = fitness;
  array[D] int trained_stored = trained;

  // Store total decay rates for interpretation
  vector[D] alpha_total;

  for (t in 1:D) {
    alpha_total[t] = alpha_d + (1 - alpha_d) * alpha_m * trained[t];
  }

  // Generate posterior predictive samples
  for (i in 1:N_weight) {
    real mu = gamma * fitness[day_idx[i]];
    y_weight_rep[i] = normal_rng(mu, sigma_w);
    log_lik_weight[i] = normal_lpdf(y_weight[i] | mu, sigma_w);
  }
}