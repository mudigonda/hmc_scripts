from mjhmc.search.objective import obj_func
from mjhmc.samplers.markov_jump_hmc import MarkovJumpHMC
from mjhmc.misc.distributions import ProductOfT
from mjhmc.misc.autocor import autocorrelation
from mjhmc.misc.autocor import sample_to_df
from mjhmc.fast import hmc
from scipy.sparse import rand
import numpy as np
import theano.tensor as T
import matplotlib.pyplot as plt

np.random.seed(2016)

nbasis=36
ndims=12
n_steps=1000
half_window=False
rand_val = rand(ndims,nbasis/2,density=0.1)
W = np.concatenate([rand_val.toarray(), -rand_val.toarray()],axis=1)
logalpha = np.random.randn(nbasis,1)

PoT_instance = ProductOfT(nbatch=100,ndims=ndims,nbasis=nbasis,W=W,logalpha=logalpha)
simulate = hmc.wrapper_hmc(s_rng=s_rng,energy_fn=rw.E_val,dim=dim)
a = np.zeros([1,10,n_samples])
b = np.zeros([2,10,n_samples])
for ii in np.arange(n_steps):
    a[:,:,ii],b[:,:,ii] = simulate()
    print "Acceptance Rate"
    print a[:,:,ii]
    print "Samples"
    print b[:,:,ii]

ac_df = autocorrelation(df,half_window) 

#Now, we can run the same autocorrelation from the data we will extract from the data frame
#but with the theano function

ac= hmc.normed_autocorrelation(df)
#We can compare the two plots individually and on a single plot

#Drop Mic and leave.
n_grad_evals = ac_df['num grad'].astype(int)
