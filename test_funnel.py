from mjhmc.misc.distributions import Funnel
from mjhmc.samplers.markov_jump_hmc import MarkovJumpHMC
import numpy as np
import theano
import theano.tensor as T

def E_val(X):
        """
        Energy function for a 10 Dimensional funnel distribution
        where the first dimenion sets the mean for the other dimensions
        which are all sampled from a Gaussian
        """
        term1 = (1/3)*(1/(X[0,:]**9))*(1/(T.sqrt(2*np.pi)))
        term2 = T.exp((-X[0,:]**2)/(2*(3**2)))
        term3 = T.sum(T.exp((-X[1:,:]**2)/(2*(X[0,:]**2))),axis=0)
        return T.log(term1+term2+term3) 


state = T.matrix()
energy = E_val(state)
Energy = theano.function([state],energy,allow_input_downcast=True)

X = np.random.randn(10,5).astype('float32')

f = Funnel(nbatch=15)
print X
print f.E_val(X)
print f.dEdX_val(X)
print f.nbatch
print Energy(X)
mjhmc = MarkovJumpHMC(distribution=f)
print("Testing if it can generate a few samples")
for ii in range(5):
    mjhmc.sample()
