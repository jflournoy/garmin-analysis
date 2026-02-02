/*
 * Gaussian Process model for weight over time with trend + Fourier spline for daily cycles
 *
 * This model separates:
 * 1. Long-term trend (squared exponential kernel)
 * 2. Daily cycles (Fourier basis expansion for hour-of-day effects)
 * 3. Observation noise (sigma)
 *
 * Goal: Better separate measurement error from intraday weight fluctuation
 * using a simpler parametric cyclic model than periodic GP
 */

data {
  int<lower=1> N;          // Number of observations
  array[N] real t;         // Time points (scaled to [0,1] based on days)
  vector[N] y;             // Weight observations (standardized)

  // Hour of day information for cyclic spline
  vector[N] hour_of_day;   // Hour of day (0-24) as float
  int<lower=1> K;          // Number of Fourier harmonics (K=1: 24h cycle, K=2: 12h + 24h, etc.)
}

parameters {
  // Trend GP parameters
  real<lower=0.01> alpha_trend;        // GP marginal standard deviation for trend
  real<lower=0.01> rho_trend;          // GP length scale for trend

  // Fourier coefficients for daily cycles (non-centered parameterization)
  vector[K] a_sin_raw;                 // Raw sine coefficients
  vector[K] a_cos_raw;                 // Raw cosine coefficients

  // Prior scale for Fourier coefficients
  real<lower=0.01> sigma_fourier;      // Scale of Fourier coefficients

  // Observation noise
  real<lower=0.01> sigma;              // Combined measurement error + residual variation

  // Non-centered parameterization for trend GP
  vector[N] eta_trend;                 // Standardized values for trend GP
}

transformed parameters {
  vector[N] f_trend;                   // Trend component
  vector[N] f_daily;                   // Daily cyclic component (Fourier)
  vector[N] f_total;                   // Combined function

  // Fourier coefficients (non-centered transformation)
  vector[K] a_sin = sigma_fourier * a_sin_raw;
  vector[K] a_cos = sigma_fourier * a_cos_raw;

  // Trend GP: squared exponential kernel
  {
    matrix[N, N] K_trend;
    matrix[N, N] L_trend;

    for (i in 1:N) {
      for (j in 1:N) {
        K_trend[i, j] = square(alpha_trend) * exp(-square(t[i] - t[j]) / (2 * square(rho_trend)));
      }
      K_trend[i, i] += square(alpha_trend) * 1e-4 + 1e-6;  // Jitter for numerical stability
    }

    L_trend = cholesky_decompose(K_trend);
    f_trend = L_trend * eta_trend;
  }

  // Daily component: Fourier basis expansion
  // f_daily[n] = Σ_k [a_sin[k] * sin(2πk * hour_scaled[n]) + a_cos[k] * cos(2πk * hour_scaled[n])]
  // where hour_scaled = hour_of_day / 24.0 (maps 0-24 to 0-1)
  for (n in 1:N) {
    real hour_scaled = hour_of_day[n] / 24.0;
    f_daily[n] = 0.0;
    for (k in 1:K) {
      real freq = 2.0 * pi() * k;
      f_daily[n] += a_sin[k] * sin(freq * hour_scaled) + a_cos[k] * cos(freq * hour_scaled);
    }
  }

  f_total = f_trend + f_daily;
}

model {
  // Priors for trend GP
  alpha_trend ~ normal(0, 0.5);
  rho_trend ~ inv_gamma(5, 1);

  // Priors for Fourier coefficients (non-centered)
  sigma_fourier ~ exponential(2);
  a_sin_raw ~ std_normal();
  a_cos_raw ~ std_normal();

  // Observation noise prior (should be smaller than in original model)
  sigma ~ exponential(3);

  // Non-centered parameterization for trend GP
  eta_trend ~ std_normal();

  // Likelihood
  y ~ normal(f_total, sigma);
}

generated quantities {
  vector[N] y_rep;                     // Posterior predictive
  vector[N] log_lik;                   // Pointwise log likelihood for WAIC/LOO
  vector[N] f_trend_std;               // Trend component (for analysis)
  vector[N] f_daily_std;               // Daily component (for analysis)
  real<lower=0> trend_change;          // Total change in trend
  real<lower=0> daily_amplitude;       // Amplitude of daily variation
  real prop_variance_daily;            // Proportion of variance from daily component

  // Posterior predictive samples and log likelihood
  for (n in 1:N) {
    y_rep[n] = normal_rng(f_total[n], sigma);
    log_lik[n] = normal_lpdf(y[n] | f_total[n], sigma);
  }

  // Store components for analysis
  f_trend_std = f_trend;
  f_daily_std = f_daily;

  // Compute trend change from first to last observation
  trend_change = abs(f_trend[N] - f_trend[1]);

  // Estimate daily amplitude (max - min of daily component)
  daily_amplitude = max(f_daily) - min(f_daily);

  // Proportion of variance explained by daily component
  {
    real var_trend = variance(f_trend);
    real var_daily = variance(f_daily);
    real var_total = var_trend + var_daily + square(sigma);
    prop_variance_daily = var_daily / var_total;
  }
}