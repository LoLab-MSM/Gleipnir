"""
Implementation of the One-dimensional multi-modal problem (Example1) from the
PyMultiNest tutorial at:
http://johannesbuchner.github.io/pymultinest-tutorial/example1.html
"""
import numpy as np
from numpy import exp, log, pi
from scipy.stats import uniform
import matplotlib.pyplot as plt
from gleipnir.samplers import MetropolisComponentWiseHardNSRejection
from gleipnir.sampled_parameter import SampledParameter
from gleipnir.stopping_criterion import NumberOfIterations
from gleipnir.nested_sampling import NestedSampling

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
    return log(l.mean())

# Setup the sampler to use when updated points during the NS run --
# Here we are using an implementation of the Metropolis Monte Carlo algorithm
# with component-wise trial moves and augmented acceptance criteria that adds a
# hard rejection constraint for the NS likelihood boundary.
sampler = MetropolisComponentWiseHardNSRejection(iterations=500, burn_in=100)
# Setup the stopping criterion for the NS run -- We'll use a fixed number of
# iterations: 10*population_size
stopping_criterion = NumberOfIterations(1000)
# Construct the Nested Sampler
NS = NestedSampling(sampled_parameters=sampled_parameters,
                    loglikelihood=loglikelihood, sampler=sampler,
                    population_size=500,
                    stopping_criterion=stopping_criterion)
# run it
NS.run()
# Retrieve the evidence
evidence = NS.evidence()
# Evidence should be 1/2
print("evidence: ",evidence)
print("log_evidence: ", np.log(evidence))
