from mjhmc.samplers.markov_jump_hmc import MarkovJumpHMC
from mjhmc.misc.autocor import calculate_autocorrelation
from mjhmc.misc.distributions import ProductOfT
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


np.random.seed(2015)
BURN_IN_STEPS = int(1E+3)
BATCH_SIZE = int(1E+1)
nbasis = 36
ndims = 36
W_half = np.random.randn(ndims,nbasis/2)
W = np.concatenate((W_half.todense(),-W_half.todense()),axis=1)

#Not doing this makes it bork crazily. Poor implementation of inheritance
PoT = ProductOfT(nbasis=nbasis,ndims=ndims,W=W)
ac_df = calculate_autocorrelation(MarkovJumpHMC,PoT,num_steps=1000)
import IPython; IPython.embed()
plt.plot(ac_df['autocorrelation'])
plt.title('Random_W_AC')
plt.savefig('ac_rand_poe_w.png')
