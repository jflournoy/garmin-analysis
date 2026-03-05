/*
 * State-space model with training-dependent decay for STRENGTH fitness only.
 *
 * Key features:
 * 1. Training reduces fitness decay: alpha_total = alpha_d + alpha_m * trained[t-1]
 * 2. Training provides fitness gain: beta * intensity[t-1] * trained[t-1]
 * 3. No impulse accumulation (simpler than impulse models)
 * 4. No diminishing returns (focus on linear gains)
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
 *   weight[t] = gamma * fitness[day(t)] + GP(t) + daily_cycle(t) + ε_w[t]
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
  // Daily data
  int<lower=1> D;                     // number of days
  vector[D] strength_intensity;       // strength intensity (standardized)

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
}

transformed data {
  // Scale hour to [0, 1] for Fourier basis
  vector[N_weight] hour_scaled = hour_of_day / 24.0;

  // Training indicators (1 if intensity > 0)
  array[D] int<lower=0, upper=1> trained;

  for (t in 1:D) {
    trained[t] = strength_intensity[t] > 0 ? 1 : 0;
  }
}

parameters {
  // Fitness decay parameters
  real<lower=0, upper=1> alpha_d;     // decay without training

  // Training reduces decay
  real<lower=0> alpha_m_raw;          // raw training effect

  // Fitness gain per unit intensity
  real<lower=0> beta;                 // gain coefficient

  // Weight effect
  real gamma;                         // strength effect on weight

  // Measurement noise
  real<lower=0.01> sigma_w;           // weight measurement noise

  // GP parameters
  real<lower=0.1, upper=1.0> alpha_gp;    // GP marginal std
  real<lower=0.1, upper=1.0> rho_gp;      // GP length scale

  // Fourier coefficients for daily cycles
  vector[K] a_sin_raw;                // Raw sine coefficients
  vector[K] a_cos_raw;                // Raw cosine coefficients
  real<lower=0.01> sigma_fourier;     // Scale of Fourier coefficients

  // Non-centered parameterization for inducing points
  vector[M] eta_inducing_raw;         // standard normal for inducing points
}

transformed parameters {
  // Fourier coefficients
  vector[K] a_sin = sigma_fourier * a_sin_raw;
  vector[K] a_cos = sigma_fourier * a_cos_raw;

  // Constrained training effect (ensure total decay < 1)
  // Use inverse logit transformation to constrain alpha_m between 0 and 1-alpha_d
  real<lower=0, upper=1-alpha_d> alpha_m = (1 - alpha_d) * inv_logit(alpha_m_raw);

  // Fitness state
  vector[D] fitness;

  // GP covariance matrices
  matrix[M, M] K_uu;
  matrix[M, M] L_uu;
  vector[M] a;                        // a = inv(K_uu) * (L_uu * eta_inducing)

  // GP latent function at weight times
  vector[N_weight] f_gp;

  // Daily cyclic component (Fourier)
  vector[N_weight] f_daily;

  // Compute fitness state with training-dependent decay
  fitness[1] = 0;

  for (t in 2:D) {
    // Total decay rate depends on whether trained yesterday
    real alpha_total = alpha_d + alpha_m * trained[t-1];

    // Gain only if trained yesterday
    real gain = beta * strength_intensity[t-1] * trained[t-1];

    fitness[t] = alpha_total * fitness[t-1] + gain;
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
  // Prior for decay without training
  alpha_d ~ beta(4, 4);               // centered around 0.5

  // Prior for training effect on decay (raw logit scale)
  // Using normal(0, 1) on logit scale gives reasonable distribution after transformation
  alpha_m_raw ~ std_normal();

  // Prior for gain coefficient
  beta ~ exponential(2);              // mean=0.5

  // Prior for weight effect
  gamma ~ normal(0.2, 0.2);           // strength might increase weight (muscle)

  // Measurement noise
  sigma_w ~ exponential(5);           // mean=0.2

  // GP parameters
  alpha_gp ~ beta(2, 3);              // favors smaller values
  rho_gp ~ beta(2, 2);                // centered

  // Fourier coefficients
  a_sin_raw ~ std_normal();
  a_cos_raw ~ std_normal();
  sigma_fourier ~ exponential(5);     // mean=0.2

  // Inducing points
  eta_inducing_raw ~ std_normal();

  // Likelihood for weight observations
  for (i in 1:N_weight) {
    real mu = gamma * fitness[day_idx[i]] + f_gp[i] + f_daily[i];
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

  // Store GP and daily components
  vector[N_weight] f_gp_stored = f_gp;
  vector[N_weight] f_daily_stored = f_daily;

  // Generate posterior predictive samples
  for (i in 1:N_weight) {
    real mu = gamma * fitness[day_idx[i]] + f_gp[i] + f_daily[i];
    y_weight_rep[i] = normal_rng(mu, sigma_w);
    log_lik_weight[i] = normal_lpdf(y_weight[i] | mu, sigma_w);
  }
}