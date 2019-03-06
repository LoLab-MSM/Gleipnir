"""
Implementation of the 2-dimensional eggbox problem adapted from the
pymultinest_demo.py at:
https://github.com/JohannesBuchner/PyMultiNest/blob/master/pymultinest_demo.py
"""
import numpy as np
from numpy import exp, log, pi
from scipy.stats import uniform
import matplotlib.pyplot as plt
from gleipnir.sampled_parameter import SampledParameter
from gleipnir.multinest import MultiNestNestedSampling



# Number of paramters to sample is 1
ndim = 2
# Set up the list of sampled parameters: the prior is Uniform(0:10*pi) --
# we are using a fixed uniform prior from scipy.stats
sampled_parameters = [SampledParameter(name=i, prior=uniform(loc=0.0,scale=10.0*np.pi)) for i in range(ndim)]

# Define the loglikelihood function
def loglikelihood(sampled_parameter_vector):
    chi = (np.cos(sampled_parameter_vector)).prod()
    return (2. + chi)**5

population_size = 100000

# Setup the Nested Sampling run
n_params = len(sampled_parameters)
print("Sampling a total of {} parameters".format(n_params))
#population_size = 10
print("Will use NS population size of {}".format(population_size))
# Construct the Nested Sampler
MNNS = MultiNestNestedSampling(sampled_parameters=sampled_parameters,
                               loglikelihood=loglikelihood,
                               population_size=population_size)
#print(PCNS.likelihood(np.array([1.0])))
#quit()
# run it
MNNS.run(verbose=True)
# Print the output -- logZ should be approximately 236
print(MNNS.output)
