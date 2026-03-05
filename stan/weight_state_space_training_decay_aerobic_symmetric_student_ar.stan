/*
 * State-space model with training-dependent decay for STRENGTH and AEROBIC fitness.
 * WITH INTERCEPT TERM in weight model.
 * SYMMETRIC PRIORS VERSION - same priors for strength and aerobic effects.
 * STUDENT-T LIKELIHOOD for robust regression + AR(1) CORRELATION STRUCTURE.
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
 * Weight model WITH INTERCEPT, STUDENT-T LIKELIHOOD, and AR(1) CORRELATION:
 *   weight[t] ~ student_t(nu, mu[t], sigma_w)
 *   mu[t] = weight_intercept + gamma_s * strength_fitness[day(t)] +
 *           gamma_a * aerobic_fitness[day(t)] + epsilon[t]
 *   epsilon[t] ~ normal(rho * epsilon[t-1], sigma_epsilon)  # AR(1) process
 *
 * Using Student-t distribution for robust regression (handles outliers better)
 * and AR(1) process to model temporal correlation in residuals.
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

  // AR(1) process needs time ordering - assume day_idx is sorted by time
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

  // Student-t degrees of freedom (must be > 2 for finite variance)
  real<lower=2> nu;                   // degrees of freedom for Student-t

  // AR(1) process parameters
  real<lower=-1, upper=1> rho;        // autocorrelation coefficient
  real<lower=0.01> sigma_epsilon;     // innovation standard deviation

  // AR(1) innovations (one per weight observation)
  vector[N_weight] epsilon_raw;       // standardized innovations
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

  // Linear predictors without AR component
  vector[N_weight] mu_no_ar;

  // AR(1) innovations (scaled by sigma_epsilon)
  vector[N_weight] epsilon;

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

  // Compute linear predictors without AR component
  for (i in 1:N_weight) {
    mu_no_ar[i] = weight_intercept +
                  gamma_s * strength_fitness[day_idx[i]] +
                  gamma_a * aerobic_fitness[day_idx[i]];
  }

  // AR(1) process: epsilon[i] = rho * epsilon[i-1] + sigma_epsilon * epsilon_raw[i]
  // For the first observation, assume stationary distribution
  epsilon[1] = sigma_epsilon / sqrt(1 - rho^2) * epsilon_raw[1];

  for (i in 2:N_weight) {
    epsilon[i] = rho * epsilon[i-1] + sigma_epsilon * epsilon_raw[i];
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

  // Prior for Student-t degrees of freedom
  // Exponential(0.1) gives mean=10, encourages moderate heaviness
  nu ~ exponential(0.1);

  // Prior for AR(1) coefficient - centered at 0 with moderate variance
  rho ~ normal(0, 0.5);

  // Prior for innovation standard deviation
  sigma_epsilon ~ exponential(10);        // mean=0.1

  // Standard normal prior for standardized innovations
  epsilon_raw ~ std_normal();

  // Likelihood for weight observations using Student-t distribution
  // with AR(1) correlated errors
  for (i in 1:N_weight) {
    real mu_total = mu_no_ar[i] + epsilon[i];
    y_weight[i] ~ student_t(nu, mu_total, sigma_epsilon);
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

  // Store linear predictors and residuals
  vector[N_weight] mu_no_ar_stored = mu_no_ar;
  vector[N_weight] epsilon_stored = epsilon;
  vector[N_weight] mu_total;  // mu_no_ar + epsilon
  vector[N_weight] residual;  // y - mu_total

  for (t in 1:D) {
    alpha_total_s[t] = alpha_d_s + (1 - alpha_d_s) * alpha_m_s * trained_s[t];
    alpha_total_a[t] = alpha_d_a + (1 - alpha_d_a) * alpha_m_a * trained_a[t];
  }

  for (i in 1:N_weight) {
    mu_total[i] = mu_no_ar[i] + epsilon[i];
    residual[i] = y_weight[i] - mu_total[i];
  }

  // Generate posterior predictive samples
  for (i in 1:N_weight) {
    real mu_pred = mu_no_ar[i] + epsilon[i];
    y_weight_rep[i] = student_t_rng(nu, mu_pred, sigma_epsilon);
    log_lik_weight[i] = student_t_lpdf(y_weight[i] | nu, mu_pred, sigma_epsilon);
  }
}