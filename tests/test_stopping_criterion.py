import gleipnir.stopping_criterion
from gleipnir.nestedsampling.stopping_criterion import NumberOfIterations, RemainingPriorMass
from gleipnir.nestedsampling import NestedSampling
from gleipnir.sampled_parameter import SampledParameter
from gleipnir.nestedsampling.samplers import MetropolisComponentWiseHardNSRejection
from scipy.stats import norm
import numpy as np

sps = list([SampledParameter('test', norm(0.,1.))])
sampler = MetropolisComponentWiseHardNSRejection(iterations=100)
def loglikelihood(point):
    pass

def test_numberofiterations_initialization():
    noi = NumberOfIterations(10)

def test_numberofiterations_attributes():
    noi = NumberOfIterations(10)
    n_iterations = noi.n_iterations
    assert noi.n_iterations == 10

def test_numberofiterations_func_call():
    noi = NumberOfIterations(10)
    NS = NestedSampling(sps, loglikelihood, 10, sampler=sampler,
                        stopping_criterion=noi)
    fail = noi(NS)
    assert fail == False

def test_remainingpriormass_initialization():
    rpm = RemainingPriorMass(0.01)

def test_remainingpriormass_attributes():
    rpm = RemainingPriorMass(0.01)
    cutoff = rpm.cutoff
    assert np.isclose(rpm.cutoff, 0.01)

def test_remainingpriormass_func_call():
    rpm = RemainingPriorMass(0.01)
    NS = NestedSampling(sps, loglikelihood, 10, sampler=sampler,
                        stopping_criterion=rpm)
    fail = rpm(NS)
    assert fail == False

if __name__ == '__main__':
    test_numberofiterations_initialization()
    test_numberofiterations_attributes()
    test_numberofiterations_func_call()
    test_remainingpriormass_initialization()
    test_remainingpriormass_attributes()
    test_remainingpriormass_func_call()
