data {
  int<lower=1> N;
  array[N] real x;
  real alpha;
  real rho;
  real period;
}
transformed parameters {
  matrix[N, N] K = cov_periodic(x, alpha, rho, period);
}
model {
}