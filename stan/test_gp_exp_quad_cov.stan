data {
  int<lower=1> N;
  array[N] real x;
  real sigma;
  real length_scale;
}
transformed parameters {
  matrix[N, N] K = gp_exp_quad_cov(x, sigma, length_scale);
}
model {
}