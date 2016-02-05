from mjhmc.fast import hmc
from mjhmc.fast.distributions_T import RoughWell
import theano.tensor as T
import numpy as np
import time

seed = 1234
s_rng = T.shared_randomstreams.RandomStreams(seed)
dim = np.array([2,10])
rw = RoughWell(nbatch=dim[1])
X = np.random.randn(2,100,5000).astype('float32')
t1 = time.time()
theano_ac = hmc.autocorrelation()
t2 = time.time()
print "elapsed time {}".format(t2 - t1)
ac= theano_ac(X)
print ac
print "ac shape: {}".format(ac.shape)