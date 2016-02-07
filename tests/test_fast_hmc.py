#Temp script to test the fast HMC
from mjhmc.fast import hmc
from mjhmc.fast.distributions_T import RoughWell
import theano.tensor as T
import numpy as np

seed = 1234
s_rng = T.shared_randomstreams.RandomStreams(seed)
dim = np.array([2,10])
rw = RoughWell(nbatch=dim[1])
simulate = hmc.wrapper_hmc(s_rng=s_rng,energy_fn=rw.E_val,dim=dim)
n_samples = 300 
a = np.zeros([1,10,n_samples])
b = np.zeros([2,10,n_samples])
for ii in np.arange(n_samples):
    a[:,:,ii],b[:,:,ii] = simulate()
    print "Acceptance Rate"
    print a[:,:,ii]
    print "Samples"
    print b[:,:,ii]
