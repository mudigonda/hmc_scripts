from mjhmc.search.objective import obj_func
from mjhmc.samplers.markov_jump_hmc import MarkovJumpHMC
from mjhmc.misc.distributions import ProductOfT,Gaussian
from mjhmc.misc.autocor import autocorrelation,slow_autocorrelation
from mjhmc.misc.autocor import sample_to_df
from mjhmc.fast import hmc
from scipy.sparse import rand
import numpy as np
import theano.tensor as T
import matplotlib.pyplot as plt
import time

np.random.seed(2015)

nbasis=36
ndims=12
n_steps=30000
half_window=False
rand_val = rand(ndims,nbasis/2,density=0.1)
W = np.concatenate([rand_val.toarray(), -rand_val.toarray()],axis=1)
logalpha = np.random.randn(nbasis,1)

#PoT_instance = ProductOfT(nbatch=100,ndims=ndims,nbasis=nbasis,W=W,lognu=logalpha)
df = sample_to_df(MarkovJumpHMC,Gaussian(ndims=10),num_steps=n_steps)
tm1 = time.time()
ac_df = slow_autocorrelation(df,half_window) 
tm2 = time.time()
print tm2-tm1

#Now, we can run the same autocorrelation from the data we will extract from the data frame
#but with the theano function
tm1 = time.time()
ac= autocorrelation(df)
tm2 = time.time()
print tm2-tm1
#We can compare the two plots individually and on a single plot

#Drop Mic and leave.
n_grad_evals = ac_df['num grad'].astype(int)
fig = plt.figure()
plt.plot(n_grad_evals,ac['autocorrelation'],'r')
plt.plot(n_grad_evals,ac_df['autocorrelation'],'g')
fig.savefig('tmp_autocorr.png')
