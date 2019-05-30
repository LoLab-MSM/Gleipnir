import gleipnir.sampled_parameter as sampled_parameter
from gleipnir.sampled_parameter import SampledParameter
from scipy.stats import norm
import numpy as np

def test_initialization():
    sp = SampledParameter('sample', norm(0.0,1.0))
    return

def test_attributes():
    sp = SampledParameter('sample', norm(0.0,1.0))
    name = sp.name
    assert sp.name == 'sample'
    prior = sp.prior_dist
    n = norm(0.0,1.0)
    assert isinstance(sp.prior_dist, type(n))

def test_func_rvs():
    sp = SampledParameter('sample', norm(0.0,1.0))
    rvs = sp.rvs(10)
    assert len(rvs) == 10

def test_func_logprior():
    sp = SampledParameter('sample', norm(0.0,1.0))
    logprior = sp.logprior(0.5)
    assert np.isclose(logprior, -1.0439385332046727)

def test_func_prior():
    sp = SampledParameter('sample', norm(0.0,1.0))
    prior = sp.prior(0.5)
    assert np.isclose(prior, 0.3520653267642995)

def test_func_invcdf():
    sp = SampledParameter('sample', norm(0.0,1.0))
    invcdf = sp.invcdf(0.5)
    assert np.isclose(invcdf, 0.0)



if __name__ == '__main__':
    test_initialization()
    test_attributes()
    test_func_rvs()
    test_func_prior()
    test_func_logprior()
    test_func_invcdf()            
