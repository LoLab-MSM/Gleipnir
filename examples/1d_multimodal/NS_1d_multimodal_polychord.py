"""
Implementation of the One-dimensional multi-modal problem (Example1) from the
PyMultiNest tutorial at:
http://johannesbuchner.github.io/pymultinest-tutorial/example1.html
"""
import numpy as np
from numpy import exp, log, pi
from scipy.stats import uniform
import matplotlib.pyplot as plt
from Gleipnir.sampled_parameter import SampledParameter
from Gleipnir.polychord import PolyChordNestedSampling

# Number of paramters to sample is 1
ndim = 1
# Set up the list of sampled parameters: the prior is Uniform(0:2) --
# we are using a fixed uniform prior from scipy.stats
sampled_parameters = [SampledParameter(name=i, prior=uniform(loc=0.0,scale=2.0)) for i in range(ndim)]
# Mode positions for the multi-modal likelihood
positions = np.array([0.1, 0.2, 0.5, 0.55, 0.9, 1.1])
# its width
width = 0.01

# Define the loglikelihood function
def loglikelihood(sampled_parameter_vector):
    diff = sampled_parameter_vector[0] - positions
    diff_scale = diff / width
    l = np.exp(-0.5 * diff_scale**2) / (2.0*pi*width**2)**0.5
    logl = log(l.mean())
    if logl < -1000.0:
        logl = -1000.0
    return logl


# Construct the Nested Sampler
PCNS = PolyChordNestedSampling(sampled_parameters=sampled_parameters,
                    loglikelihood=loglikelihood, population_size=500)
#print(PCNS.likelihood(np.array([1.0])))
#quit()
# run it
PCNS.run()
# Print the output
print(PCNS.output)
# Evidence should be 1/2
