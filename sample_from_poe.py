from mjhmc.samplers.markov_jump_hmc import MarkovJumpHMC
from mjhmc.misc.autocor import calculate_autocorrelation
from mjhmc.misc.distributions import ProductOfT
import numpy as np


BURN_IN_STEPS = int(1E+3)
BATCH_SIZE = int(1E+2)

#Not doing this makes it bork crazily. Poor implementation of inheritance
mjhmc = MarkovJumpHMC(distribution=ProductOfT(nbatch=BATCH_SIZE))
mjhmc.sampling_iteration()
