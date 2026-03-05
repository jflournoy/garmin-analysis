/*
 * State-space model with training-dependent decay rates.
 *
 * Key idea: Fitness decays faster when not training, slower when training.
 *
 * Model:
 *   trained_today = 1 if intensity > 0
 *   alpha[t] = alpha_fast if trained_today == 0, else alpha_slow
 *   fitness[t] = alpha[t] * fitness[t-1] + gain[t]
 *   gain[t] = beta * intensity[t] / (1 + k * fitness[t-1])  # diminishing returns
 *
 * This captures:
 * 1. Fast decay when not training (detraining)
 * 2. Slow decay + possible gain when training
 * 3. Diminishing returns on gain
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
  vector[D] aerobic_intensity;        // aerobic intensity (standardized: (value-min)/std)
  vector[D] strength_intensity;       // strength intensity (standardized: (value-min)/std)

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

  // Binary training indicators
  array[D] int<lower=0, upper=1> trained_aerobic = (aerobic_intensity > 0) ? 1 : 0;
  array[D] int<lower=0, upper=1> trained_strength = (strength_intensity > 0) ? 1 : 0;
}

parameters {
  // Decay rates: fast (no training) vs slow (training)
  real<lower=0, upper=1> alpha_fast_a;   // aerobic decay when not training
  real<lower=0, upper=1> alpha_slow_a;   // aerobic decay when training (alpha_slow > alpha_fast)
  real<lower=0, upper=1> alpha_fast_s;   // strength decay when not training
  real<lower=0, upper=1> alpha_slow_s;   // strength decay when training

  // Impulse decay (for accumulating intensity)
  real<lower=0, upper=1> psi_a;          // aerobic impulse decay
  real<lower=0, upper=1> psi_s;          // strength impulse decay

  // Fitness gain parameters
  real<lower=0> beta_a;                  // aerobic gain coefficient
  real<lower=0> beta_s;                  // strength gain coefficient
  real<lower=0> k_a;                     // aerobic diminishing returns
  real<lower=0> k_s;                     // strength diminishing returns

  // Weight effects
  real gamma_a;                          // aerobic effect on weight
  real gamma_s;                          // strength effect on weight

  // Measurement noise
  real<lower=0.01> sigma_w;              // weight measurement noise

  // GP parameters
  real<lower=0.1, upper=1.0> alpha_gp;   // GP marginal std
  real<lower=0.1, upper=1.0> rho_gp;     // GP length scale

  // Fourier coefficients for daily cycles
  vector[K] a_sin_raw;                   // Raw sine coefficients
  vector[K] a_cos_raw;                   // Raw cosine coefficients
  real<lower=0.01> sigma_fourier;        // Scale of Fourier coefficients

  // Non-centered parameterization for inducing points
  vector[M] eta_inducing_raw;            // standard normal for inducing points
}

transformed parameters {
  // Fourier coefficients
  vector[K] a_sin = sigma_fourier * a_sin_raw;
  vector[K] a_cos = sigma_fourier * a_cos_raw;

  // Impulse states (accumulated intensity)
  vector[D] impulse_a;
  vector[D] impulse_s;

  // Fitness states with training-dependent decay
  vector[D] fitness_a;
  vector[D] fitness_s;

  // GP covariance matrices
  matrix[M, M] K_uu;
  matrix[M, M] L_uu;
  vector[M] a;                           // a = inv(K_uu) * (L_uu * eta_inducing)

  // GP latent function at weight times
  vector[N_weight] f_gp;

  // Daily cyclic component (Fourier)
  vector[N_weight] f_daily;

  // Compute impulse states
  impulse_a[1] = aerobic_intensity[1];
  impulse_s[1] = strength_intensity[1];

  for (t in 2:D) {
    impulse_a[t] = psi_a * impulse_a[t-1] + aerobic_intensity[t];
    impulse_s[t] = psi_s * impulse_s[t-1] + strength_intensity[t];
  }

  // Compute fitness states with training-dependent decay
  fitness_a[1] = 0;
  fitness_s[1] = 0;

  for (t in 2:D) {
    // Choose decay rate based on training
    real alpha_a_t = trained_aerobic[t-1] ? alpha_slow_a : alpha_fast_a;
    real alpha_s_t = trained_strength[t-1] ? alpha_slow_s : alpha_fast_s;

    // Compute gain with diminishing returns (only if trained)
    real gain_a = trained_aerobic[t-1] ?
                  beta_a * impulse_a[t-1] / (1 + k_a * fitness_a[t-1]) : 0;
    real gain_s = trained_strength[t-1] ?
                  beta_s * impulse_s[t-1] / (1 + k_s * fitness_s[t-1]) : 0;

    fitness_a[t] = alpha_a_t * fitness_a[t-1] + gain_a;
    fitness_s[t] = alpha_s_t * fitness_s[t-1] + gain_s;
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
  // Priors for decay rates (fast < slow)
  alpha_fast_a ~ beta(3, 5);            // favors ~0.375 (fast decay)
  alpha_slow_a ~ beta(6, 2);            // favors ~0.75 (slow decay)
  alpha_fast_s ~ beta(3, 5);
  alpha_slow_s ~ beta(6, 2);

  // Constraint: slow decay > fast decay
  target += alpha_slow_a > alpha_fast_a ? 0 : negative_infinity();
  target += alpha_slow_s > alpha_fast_s ? 0 : negative_infinity();

  // Priors for impulse decay
  psi_a ~ beta(3, 5);
  psi_s ~ beta(3, 5);

  // Priors for gain parameters
  beta_a ~ exponential(2);              // mean=0.5
  beta_s ~ exponential(2);
  k_a ~ exponential(2);                 // mean=0.5
  k_s ~ exponential(2);

  // Priors for weight effects
  gamma_a ~ normal(-0.3, 0.2);          // aerobic reduces weight
  gamma_s ~ normal(0.2, 0.2);           // strength increases weight

  // Measurement noise
  sigma_w ~ exponential(5);             // mean=0.2

  // GP parameters
  alpha_gp ~ beta(2, 3);
  rho_gp ~ beta(2, 2);

  // Fourier coefficients
  a_sin_raw ~ std_normal();
  a_cos_raw ~ std_normal();
  sigma_fourier ~ exponential(5);

  // Inducing points
  eta_inducing_raw ~ std_normal();

  // Likelihood for weight observations
  for (i in 1:N_weight) {
    real mu = gamma_a * fitness_a[day_idx[i]] +
              gamma_s * fitness_s[day_idx[i]] +
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
  vector[D] fitness_a_stored = fitness_a;
  vector[D] fitness_s_stored = fitness_s;
  vector[D] impulse_a_stored = impulse_a;
  vector[D] impulse_s_stored = impulse_s;

  // Store decay rates for each day
  vector[D] alpha_a_daily;
  vector[D] alpha_s_daily;
  for (t in 1:D) {
    alpha_a_daily[t] = trained_aerobic[t] ? alpha_slow_a : alpha_fast_a;
    alpha_s_daily[t] = trained_strength[t] ? alpha_slow_s : alpha_fast_s;
  }

  // Store GP and daily components
  vector[N_weight] f_gp_stored = f_gp;
  vector[N_weight] f_daily_stored = f_daily;

  // Generate posterior predictive samples
  for (i in 1:N_weight) {
    real mu = gamma_a * fitness_a[day_idx[i]] +
              gamma_s * fitness_s[day_idx[i]] +
              f_gp[i] + f_daily[i];
    y_weight_rep[i] = normal_rng(mu, sigma_w);
    log_lik_weight[i] = normal_lpdf(y_weight[i] | mu, sigma_w);
  }

  // Compute half-lives (in days)
  real half_life_fast_a = -log(2) / log(alpha_fast_a);
  real half_life_slow_a = -log(2) / log(alpha_slow_a);
  real half_life_fast_s = -log(2) / log(alpha_fast_s);
  real half_life_slow_s = -log(2) / log(alpha_slow_s);
}