/*
 * SIMPLE State-space model with training-dependent decay for STRENGTH fitness only.
 * No GP, no daily cycle - just the core fitness model.
 *
 * Fitness state evolution:
 *   trained[t] = 1 if strength_intensity[t] > 0, else 0
 *   fitness[t] = (alpha_d + alpha_m * trained[t-1]) * fitness[t-1] +
 *                beta * strength_intensity[t-1] * trained[t-1]
 *
 * Where:
 *   0 < alpha_d < 1              (decay without training)
 *   0 < alpha_m < 1 - alpha_d    (training reduces decay, total < 1)
 *   beta > 0                     (gain per unit intensity)
 *
 * Weight model:
 *   weight[t] = gamma * fitness[day(t)] + ε_w[t]
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
  // Fitness decay parameters
  real<lower=0, upper=1> alpha_d;     // decay without training

  // Training reduces decay (raw logit scale)
  real alpha_m_raw;                   // raw training effect on logit scale

  // Fitness gain per unit intensity
  real<lower=0> beta;                 // gain coefficient

  // Weight effect
  real gamma;                         // strength effect on weight

  // Measurement noise
  real<lower=0.01> sigma_w;           // weight measurement noise
}

transformed parameters {
  // Constrained training effect (ensure total decay < 1)
  // Use inverse logit transformation to constrain alpha_m between 0 and 1-alpha_d
  real<lower=0, upper=1-alpha_d> alpha_m = (1 - alpha_d) * inv_logit(alpha_m_raw);

  // Fitness state
  vector[D] fitness;

  // Compute fitness state with training-dependent decay
  fitness[1] = 0;

  for (t in 2:D) {
    // Total decay rate depends on whether trained yesterday
    real alpha_total = alpha_d + alpha_m * trained[t-1];

    // Gain only if trained yesterday
    real gain = beta * strength_intensity[t-1] * trained[t-1];

    fitness[t] = alpha_total * fitness[t-1] + gain;
  }
}

model {
  // Prior for decay without training
  alpha_d ~ beta(4, 4);               // centered around 0.5

  // Prior for training effect on decay (raw logit scale)
  alpha_m_raw ~ std_normal();

  // Prior for gain coefficient
  beta ~ exponential(2);              // mean=0.5

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
    alpha_total[t] = alpha_d + alpha_m * trained[t];
  }

  // Generate posterior predictive samples
  for (i in 1:N_weight) {
    real mu = gamma * fitness[day_idx[i]];
    y_weight_rep[i] = normal_rng(mu, sigma_w);
    log_lik_weight[i] = normal_lpdf(y_weight[i] | mu, sigma_w);
  }
}