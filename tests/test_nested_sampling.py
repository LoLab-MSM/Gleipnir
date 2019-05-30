"""
Tests using an implementation of a 5-dimensional Gaussian problem and its
Nested Sampling using via Gleipnir's built-in Nested Sampler.

Adapted from the DNest4 python gaussian example:
https://github.com/eggplantbren/DNest4/blob/master/python/examples/gaussian/gaussian.py
"""

import pytest
import numpy as np
from numpy import exp, log, pi
from scipy.stats import uniform
from scipy.special import erf
from gleipnir.sampled_parameter import SampledParameter
from gleipnir.nested_sampling import NestedSampling
from gleipnir.samplers import MetropolisComponentWiseHardNSRejection
from gleipnir.stopping_criterion import NumberOfIterations
import os
import glob


# Number of paramters to sample is 5
ndim = 5
# Set up the list of sampled parameters: the prior is Uniform(-5:5) --
# we are using a fixed uniform prior from scipy.stats
sampled_parameters = [SampledParameter(name=i, prior=uniform(loc=-5.0,scale=10.0)) for i in range(ndim)]
# Set the active point population size
population_size = 20

sampler = MetropolisComponentWiseHardNSRejection(iterations=10, tuning_cycles=1)
stopping_criterion = NumberOfIterations(120)

# Define the loglikelihood function
def loglikelihood(sampled_parameter_vector):
    const = -0.5*np.log(2*np.pi)
    return -0.5*np.sum(sampled_parameter_vector**2) + ndim * const

width = 10.0
def analytic_log_evidence(ndim, width):
      lZ = (ndim * np.log(erf(0.5*width/np.sqrt(2)))) - (ndim * np.log(width))
      return lZ


shared = {'NS': None}

def test_initialization():
    NS = NestedSampling(sampled_parameters=sampled_parameters,
                        loglikelihood=loglikelihood,
                        sampler = sampler,
                        population_size=population_size,
                        stopping_criterion=stopping_criterion)
    shared['NS'] = NS

def test_attributes():
    NS = shared['NS']
    sp = NS.sampled_parameters
    assert sp == sampled_parameters
    lnl = NS.loglikelihood
    spv = np.array([5.,5.,5.,5.,5.])
    assert lnl(spv) == loglikelihood(spv)
    pop = NS.population_size
    assert pop == population_size

def test_func_run():
    NS = shared['NS']
    log_evidence, log_evidence_error = NS.run(verbose=False)
    analytic = analytic_log_evidence(ndim, width)
    print(analytic, log_evidence)
    assert np.isclose(log_evidence, analytic, rtol=1.)
    shared['NS'] = NS

def test_properties():
    NS = shared['NS']
    analytic = analytic_log_evidence(ndim, width)
    lnZ = NS.log_evidence
    assert np.isclose(lnZ, analytic, rtol=1.)
    lnZ_err = NS.log_evidence_error
    Z = NS.evidence
    Z_err = NS.evidence_error
    H = NS.information

def test_func_posteriors():
    NS = shared['NS']
    posteriors = NS.posteriors()
    keys = list(posteriors.keys())
    assert len(keys) == len(sampled_parameters)

def test_func_akaike_ic():
    NS = shared['NS']
    aic = NS.akaike_ic()

def test_func_bayesian_ic():
    NS = shared['NS']
    bic = NS.bayesian_ic(n_data=5)

def test_func_deviance_ic():
    NS = shared['NS']
    dic = NS.deviance_ic()


if __name__ == '__main__':
    test_initialization()
    test_attributes()
    test_func_run()
    test_properties()
    test_func_posteriors()
    test_func_akaike_ic()
    test_func_bayesian_ic()
    test_func_deviance_ic()
