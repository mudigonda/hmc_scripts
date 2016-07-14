from mjhmc.samplers.markov_jump_hmc import MarkovJumpHMC
from mjhmc.misc.autocor import calculate_autocorrelation
from mjhmc.misc.distributions import ProductOfT
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse import rand


np.random.seed(2015)
BURN_IN_STEPS = int(1E+3)
BATCH_SIZE = int(1E+1)
nbasis = 36
ndims = 36
W_half = rand(ndims,nbasis/2,density=0.35)
W = np.concatenate((W_half.todense(),-W_half.todense()),axis=1)

#Not doing this makes it bork crazily. Poor implementation of inheritance
PoT = ProductOfT(nbasis=nbasis,ndims=ndims,W=W)
ac_df = calculate_autocorrelation(MarkovJumpHMC,PoT,num_steps=1000)
import IPython; IPython.embed()
plt.plot(ac_df['autocorrelation'])
plt.title('Random_Sparse_W_AC')
plt.savefig('ac_rand_sparse_poe_w.png')
