#Temp script to test the fast HMC
from mjhmc.fast import hmc
from mjhmc.fast.distributions_T import RoughWell
import theano.tensor as T
import numpy as np
import time

tm1 = time.time()
seed = 1234
s_rng = T.shared_randomstreams.RandomStreams(seed)
ndims  = 64
nbatch = 250 
rw = RoughWell(nbatch=nbatch)
simulate = hmc.wrapper_hmc(s_rng=s_rng,energy_fn=rw.E_val,dim=[ndims,nbatch])
n_samples = 1000 
a = np.zeros([1,nbatch,n_samples])
b = np.zeros([ndims,nbatch,n_samples])
for ii in np.arange(n_samples):
    a[:,:,ii],b[:,:,ii] = simulate()
    print "Acceptance Rate"
    print a[:,:,ii]
    print "Samples"
    print b[:,:,ii]

tm2 = time.time()
print tm2-tm1
