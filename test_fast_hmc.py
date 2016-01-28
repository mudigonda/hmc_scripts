#Temp script to test the fast HMC
from mjhmc.fast import hmc
from mjhmc.fast.distributions_T import RoughWell
import theano.tensor as T
import numpy as np

seed = 1234
s_rng = T.shared_randomstreams.RandomStreams(seed)
dim = np.array([2,10])
rw = RoughWell(nbatch=dim[1])
hmc.wrapper_hmc(s_rng=s_rng,energy_fn=rw.E_val,dim=dim)
