/*
 * State-space model with diminishing returns for fitness accumulation.
 *
 * Key features:
 * 1. Fixed beta = 1.0 (impulse converts 1:1 to fitness when fitness=0)
 * 2. Diminishing returns: gain = impulse * exp(-k * current_fitness)
 * 3. Only estimate: decay rate (alpha), diminishing returns (k), weight effect (gamma)
 *
 * Fitness state evolution:
 *   fitness[t] = alpha * fitness[t-1] + impulse[t-1] * exp(-k * fitness[t-1])
 *
 * Where impulse accumulates intensity:
 *   impulse[t] = psi * impulse[t-1] + intensity[t]
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
  vector[D] aerobic_intensity;        // aerobic intensity (standardized)
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
}

parameters {
  // Impulse decay rates
  real<lower=0, upper=1> psi_a;       // aerobic impulse decay
  real<lower=0, upper=1> psi_s;       // strength impulse decay

  // Fitness decay rates
  real<lower=0, upper=1> alpha_a;     // aerobic fitness decay
  real<lower=0, upper=1> alpha_s;     // strength fitness decay

  // Diminishing returns parameters
  real<lower=0> k_a;                  // aerobic diminishing returns
  real<lower=0> k_s;                  // strength diminishing returns

  // Weight effects
  real gamma_a;                       // aerobic effect on weight
  real gamma_s;                       // strength effect on weight

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

  // Impulse and fitness states
  vector[D] impulse_a;
  vector[D] impulse_s;
  vector[D] fitness_a;
  vector[D] fitness_s;

  // GP covariance matrices
  matrix[M, M] K_uu;
  matrix[M, M] L_uu;
  vector[M] a;                        // a = inv(K_uu) * (L_uu * eta_inducing)

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

  // Compute fitness states with diminishing returns
  // Fixed beta = 1.0: impulse converts 1:1 to fitness when fitness=0
  fitness_a[1] = 0;
  fitness_s[1] = 0;

  for (t in 2:D) {
    // Diminishing returns: gain = impulse * exp(-k * current_fitness)
    real gain_a = impulse_a[t-1] * exp(-k_a * fitness_a[t-1]);
    real gain_s = impulse_s[t-1] * exp(-k_s * fitness_s[t-1]);

    fitness_a[t] = alpha_a * fitness_a[t-1] + gain_a;
    fitness_s[t] = alpha_s * fitness_s[t-1] + gain_s;
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
  // Priors for impulse decay
  psi_a ~ beta(3, 5);                 // favors ~0.375 (fast decay)
  psi_s ~ beta(3, 5);

  // Priors for fitness decay
  alpha_a ~ beta(4, 4);               // favors ~0.5 (moderate decay)
  alpha_s ~ beta(4, 4);

  // Priors for diminishing returns
  k_a ~ exponential(2);               // mean=0.5, reasonable diminishing returns
  k_s ~ exponential(2);

  // Priors for weight effects (centered but with physiological direction)
  gamma_a ~ normal(-0.3, 0.2);        // aerobic fitness reduces weight
  gamma_s ~ normal(0.2, 0.2);         // strength might increase weight (muscle)

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

  // Compute effective gains (for interpretation)
  vector[D] effective_gain_a;
  vector[D] effective_gain_s;

  effective_gain_a[1] = 0;
  effective_gain_s[1] = 0;

  for (t in 2:D) {
    effective_gain_a[t] = impulse_a[t-1] * exp(-k_a * fitness_a[t-1]);
    effective_gain_s[t] = impulse_s[t-1] * exp(-k_s * fitness_s[t-1]);
  }
}