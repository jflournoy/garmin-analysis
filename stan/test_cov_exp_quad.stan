data {
  int<lower=1> N;
  array[N] real x;
  real alpha;
  real rho;
}
transformed parameters {
  matrix[N, N] K = cov_exp_quad(x, alpha, rho);
}
model {
}