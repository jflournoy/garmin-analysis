/*
 * State-space model with training-dependent decay for STRENGTH and AEROBIC fitness.
 * WITH INTERCEPT TERM in weight model.
 * SYMMETRIC PRIORS VERSION - same priors for strength and aerobic effects.
 * STUDENT-T LIKELIHOOD for robust regression + AR(1) CORRELATION STRUCTURE.
 * DAILY SPLINE for intraday weight variations (Fourier basis).
 * SIMPLE AR(1) with strong priors to prevent overfitting.
 * GENERATES PREDICTIONS FOR ALL DAYS.
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
 * Weight model WITH INTERCEPT, STUDENT-T LIKELIHOOD, AR(1) CORRELATION, and DAILY SPLINE:
 *   weight[t] ~ student_t(nu, mu[t], sigma_w)
 *   mu[t] = weight_intercept + gamma_s * strength_fitness[day(t)] +
 *           gamma_a * aerobic_fitness[day(t)] + f_daily[t] + epsilon[t]
 *   epsilon[t] ~ normal(rho * epsilon[t-1], sigma_epsilon)  # SIMPLE AR(1) process
 *
 * Daily spline using Fourier basis:
 *   f_daily[t] = Σ_k [a_sin[k] * sin(2πk * hour_scaled[t]) + a_cos[k] * cos(2πk * hour_scaled[t])]
 *   where hour_scaled = hour_of_day / 24.0 (maps 0-24 to 0-1)
 *
 * Using Student-t distribution for robust regression (handles outliers better)
 * SIMPLE AR(1) process with strong priors to prevent overfitting
 * Daily spline to capture intraday weight variations
 * Generates predictions for all days (not just days with weight data)
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

  // Hour of day information for daily spline
  vector[N_weight] hour_of_day;       // Hour of day (0-24) as float
  int<lower=1> K;                     // Number of Fourier harmonics (K=1: 24h cycle, K=2: 12h + 24h, etc.)

  // For predictions at multiple hours
  int<lower=1> H;                     // Number of hours to predict at per day
  vector<lower=0, upper=1>[H] pred_hours_scaled;  // Hours to predict at (scaled to [0,1])

  // AR(1) process needs time ordering - assume data is sorted by time
}

transformed data {
  // Training indicators (1 if intensity > 0)
  array[D] int<lower=0, upper=1> trained_s;
  array[D] int<lower=0, upper=1> trained_a;

  // Scale hour of day to [0, 1] for Fourier basis
  vector[N_weight] hour_scaled;

  for (t in 1:D) {
    trained_s[t] = strength_intensity[t] > 0 ? 1 : 0;
    trained_a[t] = aerobic_intensity[t] > 0 ? 1 : 0;
  }

  for (i in 1:N_weight) {
    hour_scaled[i] = hour_of_day[i] / 24.0;
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

  // SIMPLE AR(1) process parameters with strong priors
  real<lower=-1, upper=1> rho;        // autocorrelation coefficient
  real<lower=0> sigma_epsilon;        // innovation standard deviation

  // AR(1) innovations
  vector[N_weight] epsilon_raw;       // Standard normal innovations

  // Fourier coefficients for daily spline (non-centered parameterization)
  vector[K] a_sin_raw;                // Raw sine coefficients
  vector[K] a_cos_raw;                // Raw cosine coefficients

  // Prior scale for Fourier coefficients
  real<lower=0.01> sigma_fourier;     // Scale of Fourier coefficients

  // Observation noise
  real<lower=0> sigma_w;              // Student-t scale parameter
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

  // Fourier coefficients (non-centered transformation)
  vector[K] a_sin = sigma_fourier * a_sin_raw;
  vector[K] a_cos = sigma_fourier * a_cos_raw;

  // Daily spline component for weight observations
  vector[N_weight] f_daily;

  // Linear predictors without AR component for weight observations
  vector[N_weight] mu_no_ar;

  // SIMPLE AR(1) innovations (no local shrinkage)
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

  // Compute daily spline component using Fourier basis for weight observations
  for (i in 1:N_weight) {
    f_daily[i] = 0.0;
    for (k in 1:K) {
      real freq = 2.0 * pi() * k;
      f_daily[i] += a_sin[k] * sin(freq * hour_scaled[i]) + a_cos[k] * cos(freq * hour_scaled[i]);
    }
  }

  // Compute linear predictors without AR component (includes daily spline)
  for (i in 1:N_weight) {
    mu_no_ar[i] = weight_intercept +
                  gamma_s * strength_fitness[day_idx[i]] +
                  gamma_a * aerobic_fitness[day_idx[i]] +
                  f_daily[i];
  }

  // SIMPLE AR(1) process: epsilon[i] = rho * epsilon[i-1] + sigma_epsilon * epsilon_raw[i]
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

  // STRONG PRIOR for AR(1) coefficient - beta distribution centered at 0
  // Beta(2, 2) gives mean=0.5, but we want mean near 0
  // Use transformed beta: rho_scaled = 2*rho_beta - 1 where rho_beta ~ beta(2, 5)
  // This gives mean around 0.2 after transformation
  {
    real rho_beta = (rho + 1) / 2;  // Transform from [-1, 1] to [0, 1]
    target += beta_lpdf(rho_beta | 2, 5);
  }

  // STRONG PRIOR for AR(1) innovation scale - small values
  sigma_epsilon ~ exponential(5);         // mean=0.2

  // Prior for AR(1) innovations
  epsilon_raw ~ std_normal();

  // Priors for Fourier coefficients (non-centered)
  a_sin_raw ~ std_normal();
  a_cos_raw ~ std_normal();

  // Prior for Fourier coefficient scale
  sigma_fourier ~ exponential(1);         // mean=1

  // Prior for observation noise
  sigma_w ~ exponential(2);               // mean=0.5

  // Likelihood for weight observations using Student-t distribution
  // with AR(1) correlated errors and daily spline
  for (i in 1:N_weight) {
    real mu_total = mu_no_ar[i] + epsilon[i];
    y_weight[i] ~ student_t(nu, mu_total, sigma_w);
  }
}

generated quantities {
  // Posterior predictive for weight observations
  vector[N_weight] y_weight_rep;

  // Predictions for ALL DAYS at MULTIPLE HOURS
  // Array of predictions for each day at each hour
  array[D] vector[H] y_pred_all_days;      // Predictions for each day at each hour
  array[D] vector[H] mu_pred_all_days_no_ar;  // Without AR component
  array[D] vector[H] mu_pred_all_days;        // With AR component (same as no_ar for daily predictions)

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

  // Store spline and linear predictors for weight observations
  vector[N_weight] f_daily_stored = f_daily;
  vector[N_weight] mu_no_ar_stored = mu_no_ar;
  vector[N_weight] epsilon_stored = epsilon;
  vector[N_weight] mu_total;  // mu_no_ar + epsilon
  vector[N_weight] residual;  // y - mu_total

  // Store Fourier coefficients for interpretation
  vector[K] a_sin_stored = a_sin;
  vector[K] a_cos_stored = a_cos;

  // Store AR(1) parameters
  real sigma_epsilon_stored = sigma_epsilon;
  real rho_stored = rho;

  for (t in 1:D) {
    alpha_total_s[t] = alpha_d_s + (1 - alpha_d_s) * alpha_m_s * trained_s[t];
    alpha_total_a[t] = alpha_d_a + (1 - alpha_d_a) * alpha_m_a * trained_a[t];
  }

  for (i in 1:N_weight) {
    mu_total[i] = mu_no_ar[i] + epsilon[i];
    residual[i] = y_weight[i] - mu_total[i];
  }

  // Generate posterior predictive samples for weight observations
  for (i in 1:N_weight) {
    real mu_pred = mu_no_ar[i] + epsilon[i];
    y_weight_rep[i] = student_t_rng(nu, mu_pred, sigma_w);
    log_lik_weight[i] = student_t_lpdf(y_weight[i] | nu, mu_pred, sigma_w);
  }

  // Generate predictions for ALL DAYS at MULTIPLE HOURS
  // For predictions on all days, we omit the AR(1) component
  // because it's specific to the measurement times and correlation structure
  // The AR(1) process models residual correlation between weight measurements,
  // not a daily process. So for daily predictions, we use only the structural components.

  for (t in 1:D) {
    for (h in 1:H) {
      // Compute daily spline component for this specific hour
      real f_daily_hour = 0.0;
      for (k in 1:K) {
        real freq = 2.0 * pi() * k;
        f_daily_hour += a_sin[k] * sin(freq * pred_hours_scaled[h]) + a_cos[k] * cos(freq * pred_hours_scaled[h]);
      }

      // Linear predictor without AR component (structural model only)
      mu_pred_all_days_no_ar[t][h] = weight_intercept +
                                    gamma_s * strength_fitness[t] +
                                    gamma_a * aerobic_fitness[t] +
                                    f_daily_hour;

      // For daily predictions, we use the structural model without AR component
      // The AR component is measurement-time specific residual correlation
      mu_pred_all_days[t][h] = mu_pred_all_days_no_ar[t][h];

      // Generate predictive distribution (with measurement error)
      y_pred_all_days[t][h] = student_t_rng(nu, mu_pred_all_days[t][h], sigma_w);
    }
  }
}