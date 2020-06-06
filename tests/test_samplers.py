import gleipnir.nestedsampling.samplers
from gleipnir.nestedsampling.samplers import MetropolisSampler
from gleipnir.sampled_parameter import SampledParameter
from gleipnir.nestedsampling import NestedSampling
from gleipnir.nestedsampling.stopping_criterion import NumberOfIterations
from scipy.stats import uniform
import numpy as np

# Number of paramters to sample is 5
ndim = 5
# Set up the list of sampled parameters: the prior is Uniform(-5:5) --
# we are using a fixed uniform prior from scipy.stats
sampled_parameters = [SampledParameter(name=i, prior=uniform(loc=-5.0,scale=10.0)) for i in range(ndim)]
# Set the active point population size
population_size = 20
stopping_criterion = NumberOfIterations(1)
# Define the loglikelihood function
def loglikelihood(sampled_parameter_vector):
    const = -0.5*np.log(2*np.pi)
    return -0.5*np.sum(sampled_parameter_vector**2) + ndim * const


def test_metropolissampler_initialization():
    s = MetropolisSampler(iterations=10, tuning_cycles=2)

def test_metropolissampler_attributes():
    s = MetropolisSampler(iterations=10, tuning_cycles=2)
    iterations = s.iterations
    assert s.iterations == 10
    burn_in = s.burn_in
    assert s.burn_in == 0
    tuning_cycles = s.tuning_cycles
    assert s.tuning_cycles == 2
    proposal = s.proposal
    assert s.proposal == 'uniform'

def test_metropolissampler_func_call():
    #sps = list([SampledParameter('test', norm(0.,1.))])
    s = MetropolisSampler(iterations=10, tuning_cycles=2)
    NS = NestedSampling(sampled_parameters=sampled_parameters,
                        loglikelihood=loglikelihood,
                        sampler = sampler,
                        population_size=population_size,
                        stopping_criterion=stopping_criterion)
    NS.run()

if __name__ == '__main__':
    test_metropolissampler_initialization()
    test_metropolissampler_attributes()
    test_metropolissampler_func_call()
