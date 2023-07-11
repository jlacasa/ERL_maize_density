library(rstan);library(brms)

load("/homes/lacasa/data.RData")

data_basis$hs_df_INT <- 7
data_basis$hs_df_POW <- 10
data_basis$hs_df_slab_INT <- 4

m1 <- stan("/homes/lacasa/powerlaw.stan",
    data = data_basis, 
    cores = 16,
    control = list(adapt_delta = 0.98, max_treedepth=12),
    chains=4, 
    iter = 15000 , 
    warmup = 13000 , 
    seed = 531, 
    thin=1, 
    save_warmup=F)

saveRDS(m1, "/homes/lacasa/m1.RData") 
