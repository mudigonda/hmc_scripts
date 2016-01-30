from mjhmc.fast import hmc
from mjhmc.fast.distributions_T import RoughWell
import theano.tensor as T
import numpy as np

seed = 1234
s_rng = T.shared_randomstreams.RandomStreams(seed)
dim = np.array([2,10])
rw = RoughWell(nbatch=dim[1])
X = np.random.randn(2,100,50000).astype('float32')
theano_ac = hmc.autocorrelation()
ac= theano_ac(X)
print ac
