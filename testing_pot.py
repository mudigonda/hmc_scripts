#script to transform samples from student t to poe

import numpy as np
from scipy.sparse import rand

#This is the random seed we set for repeatability
np.random.seed(2016)
#Number of dimensions
ndims = 36
#Number of experts
nbasis = 72
#Batch size
nbatch = 25

#We can set this to any arbitrary full rank Matrix, for now we set it to eye
W = np.eye(ndims,ndims) 
#you could use this as well or not, doesn't really matter
logalpha = np.random.randn(nbasis,1)
#Storing both the intermeditatory steps and the final set of samples
samples_init = np.zeros((ndims,nbatch,nbasis))
samples = np.ones((ndims,nbatch))

for ii in range(nbasis):
  #Sample from a student-T distribution with an arbitrary degree if we wish 
  samples_init[:,:,ii] = np.random.standard_t(df=1,size=(ndims,nbatch))
  #Multiply them together, we can basically multiply as many as we like (If I understand this correctly!)
  samples = samples* samples_init[:,:,ii]

#Now samples is from *a* product of experts, just like we did in the Gaussian case!!!
print "Our initial samples are"
#Tada!
print samples
#Save them 
samples.save('PoE_pkl')
