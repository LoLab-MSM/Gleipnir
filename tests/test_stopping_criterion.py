import gleipnir.nestedsampling.stopping_criterion
from gleipnir.nestedsampling.stopping_criterion import NumberOfIterations, RemainingPriorMass, RelativeEvidenceThreshold
from gleipnir.nestedsampling import NestedSampling
from gleipnir.sampled_parameter import SampledParameter
from gleipnir.nestedsampling.samplers import MetropolisSampler
from scipy.stats import norm
import numpy as np

sps = list([SampledParameter('test', norm(0.,1.))])
sampler = MetropolisSampler(iterations=100)
def loglikelihood(point):
    return np.random.random()

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
    threshold = rpm.threshold
    assert np.isclose(rpm.threshold, 0.01)

def test_remainingpriormass_func_call():
    rpm = RemainingPriorMass(0.01)
    NS = NestedSampling(sps, loglikelihood, 10, sampler=sampler,
                        stopping_criterion=rpm)
    fail = rpm(NS)
    assert fail == False

def test_relativeevidencethreshold_initialization():
    ret = RelativeEvidenceThreshold(0.01)

def test_relavtiveevidencethreshold_attributes():
    ret = RelativeEvidenceThreshold(0.01)
    threshold = ret.threshold
    assert np.isclose(ret.threshold, 0.01)

def test_relativeevidencethreshold_func_call():
    ret = RelativeEvidenceThreshold(1.e-6)
    NS = NestedSampling(sps, loglikelihood, 10, sampler=sampler,
                        stopping_criterion=NumberOfIterations(2))
    fail = ret(NS)
    assert fail == False
    NS.run()
    fail = ret(NS)
    assert fail == False

if __name__ == '__main__':
    test_numberofiterations_initialization()
    test_numberofiterations_attributes()
    test_numberofiterations_func_call()
    test_remainingpriormass_initialization()
    test_remainingpriormass_attributes()
    test_remainingpriormass_func_call()
    test_relativeevidencethreshold_initialization()
    test_relavtiveevidencethreshold_attributes()
    test_relativeevidencethreshold_func_call()
