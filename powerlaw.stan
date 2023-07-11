//
functions {
  /* Horseshoe prior
   * https://projecteuclid.org/euclid.ejs/1513306866
   *   z: standardized population-level coefficients
   *   lambda: local shrinkage parameters
   *   tau: global shrinkage parameter
   *   c2: slap regularization parameter
   */
  vector horseshoe(vector z, vector lambda, real tau, real c2) {
    int K = rows(z);
    vector[K] lambda2 = square(lambda);
    vector[K] lambda_tilde = sqrt(c2 * lambda2 ./ (c2 + tau^2 * lambda2));
    return z .* lambda_tilde * tau;
  }
  
  // integer to real
  real int_to_real(int x) {
    real y;
    if(x == 0)
    y = 0;
    else
    y = 1;
    return(y);
    }

  /* integer sequence of values
   *   start: starting integer
   *   end: ending integer
   */ 
  int[] sequence(int start, int end) { 
    int seq[end - start + 1];
    for (n in 1:num_elements(seq)) {
      seq[n] = n + start - 1;
    }
    return seq; 
  } 
  // compute partial sums of the log-likelihood
  real partial_log_lik(int[] seq, int start, int end, vector Y, matrix X_INT, vector b_INT, matrix X_POW, vector b_POW, vector C_1, real sigma) {
    real ptarget = 0;
    int N = end - start + 1;
    // initialize linear predictor term
    vector[N] nlp_INT = X_INT[start:end] * b_INT;
    // initialize linear predictor term
    vector[N] nlp_POW = X_POW[start:end] * b_POW;
    // initialize non-linear predictor term
    vector[N] mu;
    for (n in 1:N) {
      int nn = n + start - 1;
      // compute non-linear predictor values
      mu[n] = nlp_INT[n] * C_1[nn] ^ nlp_POW[n];
    }
    ptarget += normal_lpdf(Y[start:end] | mu, sigma);
    return ptarget;
  }
}
data {
  int<lower=1> N;  // total number of observations
  int<lower=1> N_f;  // total number of forecasts
  vector[N] Y;  // response variable
  // int<lower=1> s[N];  // group
  int<lower=1> K_INT;  // number of population-level effects
  matrix[N, K_INT] X_INT;  // population-level design matrix
  // data for the horseshoe 
  real<lower=0> hs_df_INT;  // local degrees of freedom
  real<lower=0> hs_df_global_INT;  // global degrees of freedom
  real<lower=0> hs_df_slab_INT;  // slab degrees of freedom
  real<lower=0> hs_scale_global_INT;  // global prior scale
  real<lower=0> hs_scale_slab_INT;  // slab prior scale
  int<lower=1> K_POW;  // number of population-level effects
  matrix[N, K_POW] X_POW;  // population-level design matrix
  // data for the horseshoe 
  real<lower=0> hs_df_POW;  // local degrees of freedom
  real<lower=0> hs_df_global_POW;  // global degrees of freedom
  real<lower=0> hs_df_slab_POW;  // slab degrees of freedom
  real<lower=0> hs_scale_global_POW;  // global prior scale
  real<lower=0> hs_scale_slab_POW;  // slab prior scale
  // covariate vectors for non-linear functions
  vector[N] C_1;
  vector[N_f] C_1_f;
  
  int grainsize;  // grainsize for threading
  int prior_only;  // should the likelihood be ignored?
  
  vector[N_f] p_an_tmean_4m;
  vector[N_f] p_bn_tmean_4m;
  vector[N_f] p_an_tmean_3m;
  vector[N_f] p_bn_tmean_3m;
  vector[N_f] p_an_tmean_2m;
  vector[N_f] p_bn_tmean_2m;
  
  vector[N_f] p_an_pptn_4m;
  vector[N_f] p_bn_pptn_4m;
  vector[N_f] p_an_pptn_3m;
  vector[N_f] p_bn_pptn_3m;
  vector[N_f] p_an_pptn_2m;
  vector[N_f] p_bn_pptn_2m;
  
  vector[N_f] POW_f_11;
  vector[N_f] POW_f_10;
  vector[N_f] POW_f_9;
  vector[N_f] POW_f_8;
  vector[N_f] POW_f_7;
  vector[N_f] POW_f_6;
  vector[N_f] POW_f_5;
  vector[N_f] POW_f_4;
  vector[N_f] POW_f_3;
  vector[N_f] POW_f_2;
  vector[N_f] INT_f_3;
  vector[N_f] INT_f_2;


}
transformed data {
  int seq[N] = sequence(1, N);
}
parameters {
  // local parameters for horseshoe prior
  vector[K_INT] zb_INT;
  vector<lower=0>[K_INT] hs_local_INT;
  // horseshoe shrinkage parameters
  real<lower=0> hs_global_INT;  // global shrinkage parameters
  real<lower=0> hs_slab_INT;  // slab regularization parameter
  // local parameters for horseshoe prior
  vector[K_POW] zb_POW;
  vector<lower=0>[K_POW] hs_local_POW;
  // horseshoe shrinkage parameters
  real<lower=0> hs_global_POW;  // global shrinkage parameters
  real<lower=0> hs_slab_POW;  // slab regularization parameter
  real<lower=0> sigma;  // residual SD
}
transformed parameters {
  vector[K_INT] b_INT;  // population-level effects
  vector[K_POW] b_POW;  // population-level effects
  // compute actual regression coefficients
  b_INT = horseshoe(zb_INT, hs_local_INT, hs_global_INT, hs_scale_slab_INT^2 * hs_slab_INT);
  // compute actual regression coefficients
  b_POW = horseshoe(zb_POW, hs_local_POW, hs_global_POW, hs_scale_slab_POW^2 * hs_slab_POW);
}
model {
  // likelihood including all constants
  target += reduce_sum(partial_log_lik, seq, grainsize, Y, X_INT, b_INT, X_POW, b_POW, C_1, sigma);
  // priors including all constants
  target += std_normal_lpdf(zb_INT);
  target += student_t_lpdf(hs_local_INT | hs_df_INT, 0, 1)
    - rows(hs_local_INT) * log(0.5);
  target += student_t_lpdf(hs_global_INT | hs_df_global_INT, 0, hs_scale_global_INT * sigma)
    - 1 * log(0.5);
  target += inv_gamma_lpdf(hs_slab_INT | 0.5 * hs_df_slab_INT, 0.5 * hs_df_slab_INT);
  target += std_normal_lpdf(zb_POW);
  target += student_t_lpdf(hs_local_POW | hs_df_POW, 0, 1)
    - rows(hs_local_POW) * log(0.5);
  target += student_t_lpdf(hs_global_POW | hs_df_global_POW, 0, hs_scale_global_POW * sigma)
    - 1 * log(0.5);
  target += inv_gamma_lpdf(hs_slab_POW | 0.5 * hs_df_slab_POW, 0.5 * hs_df_slab_POW);
  target += student_t_lpdf(sigma | 3, 0, 2.5)
    - 1 * student_t_lccdf(0 | 3, 0, 2.5);
    // target += gamma_lpdf(sigma | 3,1);
}

generated quantities {
  // forecasts
  vector[N_f] y_f;
  vector[N_f] mu_f;
  matrix[N_f, K_INT] X_INT_f;
  matrix[N_f, K_POW] X_POW_f;
  vector[N_f] nlp_INT_f;  
  vector[N_f] nlp_POW_f;  

  int an_tmean_4m[N_f];
  int bn_tmean_4m[N_f];
  int an_tmean_3m[N_f];
  int bn_tmean_3m[N_f];
  int an_tmean_2m[N_f];
  int bn_tmean_2m[N_f];

  int an_pptn_4m[N_f];
  int bn_pptn_4m[N_f];
  int an_pptn_3m[N_f];
  int bn_pptn_3m[N_f];
  int an_pptn_2m[N_f];
  int bn_pptn_2m[N_f];
  
  for (i in 1:N_f) {
  an_pptn_4m[i] = bernoulli_rng(p_an_pptn_4m[i]);
  bn_pptn_4m[i] = bernoulli_rng(p_bn_pptn_4m[i]);
  an_pptn_3m[i] = bernoulli_rng(p_an_pptn_3m[i]);
  bn_pptn_3m[i] = bernoulli_rng(p_bn_pptn_3m[i]);
  an_pptn_2m[i] = bernoulli_rng(p_an_pptn_2m[i]);
  bn_pptn_2m[i] = bernoulli_rng(p_bn_pptn_2m[i]);
  
  an_tmean_4m[i] = bernoulli_rng(p_an_tmean_4m[i]);
  bn_tmean_4m[i] = bernoulli_rng(p_bn_tmean_4m[i]);
  an_tmean_3m[i] = bernoulli_rng(p_an_tmean_3m[i]);
  bn_tmean_3m[i] = bernoulli_rng(p_bn_tmean_3m[i]);
  an_tmean_2m[i] = bernoulli_rng(p_an_tmean_2m[i]);
  bn_tmean_2m[i] = bernoulli_rng(p_bn_tmean_2m[i]);

  }   

  X_INT_f[,3] = INT_f_3;//crm;
  X_INT_f[,2] = INT_f_2;//srad;
  X_INT_f[,1] = rep_vector(1,N_f);
  
  for (i in 1:N_f) {

  X_POW_f[i,12] = int_to_real(an_tmean_2m[i]);
  if (X_POW_f[i,12]<0.5) {
    X_POW_f[i,13] = int_to_real(bn_tmean_2m[i])/(1-X_POW_f[i,12]);
  } else {
    X_POW_f[i,13] = 0;
    }
    
  X_POW_f[i,14] = int_to_real(an_tmean_3m[i]);
  if (X_POW_f[i,14]<0.5) {
    X_POW_f[i,15] = int_to_real(bn_tmean_3m[i])/(1-X_POW_f[i,14]);
  } else {
    X_POW_f[i,15] = 0;
    }
    
  X_POW_f[i,16] = int_to_real(an_tmean_4m[i]);
  if (X_POW_f[i,16]<0.5) {
    X_POW_f[i,17] = int_to_real(bn_tmean_4m[i])/(1-X_POW_f[i,16]);
  } else {
    X_POW_f[i,17] = 0;
    }

// TEMP 
  X_POW_f[i,18] = int_to_real(an_pptn_2m[i]);
  if (X_POW_f[i,18]<0.5) {
    X_POW_f[i,19] = int_to_real(bn_pptn_2m[i])/(1-X_POW_f[i,18]);
  } else {
    X_POW_f[i,19] = 0;
    }
    
  X_POW_f[i,20] = int_to_real(an_pptn_3m[i]);
  if (X_POW_f[i,20]<0.5) {
    X_POW_f[i,21] = int_to_real(bn_pptn_3m[i])/(1-X_POW_f[i,20]);
  } else {
    X_POW_f[i,21] = 0;
    }
    
    X_POW_f[i,22] = int_to_real(an_pptn_4m[i]);
  if (X_POW_f[i,22]<0.5) {
    X_POW_f[i,23] = int_to_real(bn_pptn_4m[i])/(1-X_POW_f[i,22]);
  } else {
    X_POW_f[i,23] = 0;
    }
    }

  X_POW_f[,11] = POW_f_11; // tmean_hist;
  X_POW_f[,10] = POW_f_10; // susm_planting;
  X_POW_f[,9] = POW_f_9; // susm_2weeks;
  X_POW_f[,8] = POW_f_8; // susm_1month;
  X_POW_f[,7] = POW_f_7; // ppt_4m_post_hist
  X_POW_f[,6] = POW_f_6; // ppt_3m_post_hist;
  X_POW_f[,5] = POW_f_5; // ppt_2m_post_hist;
  X_POW_f[,4] = POW_f_4; // pptn_hist;
  X_POW_f[,3] = POW_f_3; // soil_whc;
  X_POW_f[,2] = POW_f_2; // crm;
  X_POW_f[,1] = rep_vector(1,N_f);
  
  nlp_INT_f = X_INT_f * b_INT; 
  nlp_POW_f = X_POW_f * b_POW;
  
  for (n in 1:N_f) {
    mu_f[n] = (nlp_INT_f[n] * C_1_f[n]^nlp_POW_f[n]);
    y_f[n] = normal_rng(mu_f[n], sigma);
    }
    
    }
    
    
    
