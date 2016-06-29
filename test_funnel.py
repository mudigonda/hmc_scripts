from mjhmc.misc.distributions import Funnel
from mjhmc.samplers.markov_jump_hmc import MarkovJumpHMC
from mjhmc.misc.plotting import hist_2d
import numpy as np
import theano
import theano.tensor as T


nbatch = 15
scale = 1.0
ndims = 2
X = np.random.randn(ndims,nbatch).astype('float32')

f = Funnel(nbatch=nbatch,scale=scale,ndims=ndims)
print X
print f.E_val(X)
print f.E_val(X).shape
print f.dEdX_val(X)
print f.dEdX_val(X).shape
print f.nbatch
print f.scale
mjhmc = MarkovJumpHMC(distribution=f)
'''
print("Testing if it can generate a few samples")
for ii in range(5):
    print(mjhmc.sample())
'''
fig = hist_2d(f,10000)
import IPython; IPython.embed()
