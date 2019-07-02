import gleipnir.nestedsampling.samplers
from gleipnir.nestedsampling.samplers import MetropolisComponentWiseHardNSRejection
from gleipnir.sampled_parameter import SampledParameter
from scipy.stats import norm
import numpy as np

def test_metropoliscomponentwisehardnsrejection_initialization():
    s = MetropolisComponentWiseHardNSRejection(iterations=10, tuning_cycles=2)

def test_metropoliscomponentwisehardnsrejection_attributes():
    s = MetropolisComponentWiseHardNSRejection(iterations=10, tuning_cycles=2)
    iterations = s.iterations
    assert s.iterations == 10
    burn_in = s.burn_in
    assert s.burn_in == 0
    tuning_cycles = s.tuning_cycles
    assert s.tuning_cycles == 2
    proposal = s.proposal
    assert s.proposal == 'uniform'

def test_metropoliscomponentwisehardnsrejection_func_call():
    sps = list([SampledParameter('test', norm(0.,1.))])
    s = MetropolisComponentWiseHardNSRejection(iterations=10, tuning_cycles=2)
    def loglikelihood(point):
        return 1.
    new_point, log_l = s(sps, loglikelihood, np.array([0.5]), 2.)


if __name__ == '__main__':
    test_metropoliscomponentwisehardnsrejection_initialization()
    test_metropoliscomponentwisehardnsrejection_attributes()
    test_metropoliscomponentwisehardnsrejection_func_call()
