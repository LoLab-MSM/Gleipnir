"""
Implementation of the 2-dimensional eggbox problem adapted from the
pymultinest_demo.py at:
https://github.com/JohannesBuchner/PyMultiNest/blob/master/pymultinest_demo.py
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
ndim = 2
# Set up the list of sampled parameters: the prior is Uniform(0:10*pi) --
# we are using a fixed uniform prior from scipy.stats
sampled_parameters = [SampledParameter(name=i, prior=uniform(loc=0.0,scale=10.0*np.pi)) for i in range(ndim)]

# Define the loglikelihood function
def loglikelihood(sampled_parameter_vector):
    chi = (np.cos(sampled_parameter_vector)).prod()
    return (2. + chi)**5

population_size = 100
# Setup the sampler to use when updated points during the NS run --
# Here we are using an implementation of the Metropolis Monte Carlo algorithm
# with component-wise trial moves and augmented acceptance criteria that adds a
# hard rejection constraint for the NS likelihood boundary.
sampler = MetropolisComponentWiseHardNSRejection(iterations=100, burn_in=0)
# Setup the stopping criterion for the NS run -- We'll use a fixed number of
# iterations: 10*population_size
stopping_criterion = NumberOfIterations(500)
# Construct the Nested Sampler
NS = NestedSampling(sampled_parameters=sampled_parameters,
                    loglikelihood=loglikelihood, sampler=sampler,
                    population_size=population_size,
                    stopping_criterion=stopping_criterion)
# run it
NS.run(verbose=True)
# Retrieve the evidence
evidence = NS.evidence()
error = NS.evidence_error()

print("evidence: {} +- {}".format(evidence, error))
# log Evidence (lnZ) should be approximately 236
print("log_evidence: {} +- {} ".format(np.log(evidence), np.log(error)))
