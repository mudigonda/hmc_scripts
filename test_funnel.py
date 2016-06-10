from mjhmc.misc.distributions import Funnel
from mjhmc.samplers.markov_jump_hmc import MarkovJumpHMC
import numpy as np
import theano
import theano.tensor as T



X = np.random.randn(10,5).astype('float32')

f = Funnel(nbatch=15,scale=1.0)
print X
print f.E_val(X)
print f.dEdX_val(X)
print f.nbatch
print f.scale
mjhmc = MarkovJumpHMC(distribution=f)
print("Testing if it can generate a few samples")
for ii in range(5):
    print(mjhmc.sample())
