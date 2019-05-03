"""
Tests using an implementation of a 5-dimensional Gaussian problem and its
Nested Sampling using MultiNest via Gleipnir.

Adapted from the DNest4 python gaussian example:
https://github.com/eggplantbren/DNest4/blob/master/python/examples/gaussian/gaussian.py
"""

import pytest
import numpy as np
from numpy import exp, log, pi
from scipy.stats import uniform
from scipy.special import erf
from gleipnir.sampled_parameter import SampledParameter
from gleipnir.multinest import MultiNestNestedSampling
import os
import glob


# Number of paramters to sample is 5
ndim = 5
# Set up the list of sampled parameters: the prior is Uniform(-5:5) --
# we are using a fixed uniform prior from scipy.stats
sampled_parameters = [SampledParameter(name=i, prior=uniform(loc=-5.0,scale=10.0)) for i in range(ndim)]
# Set the active point population size
population_size = 100

# Define the loglikelihood function
def loglikelihood(sampled_parameter_vector):
    const = -0.5*np.log(2*np.pi)
    return -0.5*np.sum(sampled_parameter_vector**2) + ndim * const

width = 10.0
def analytic_log_evidence(ndim, width):
      lZ = (ndim * np.log(erf(0.5*width/np.sqrt(2)))) - (ndim * np.log(width))
      return lZ


shared = {'MNNS': None}

def test_initialization():
    MNNS = MultiNestNestedSampling(sampled_parameters=sampled_parameters,
                                   loglikelihood=loglikelihood,
                                   population_size=population_size)
    shared['MNNS'] = MNNS

def test_attributes():
    MNNS = shared['MNNS']
    sp = MNNS.sampled_parameters
    assert sp == sampled_parameters
    lnl = MNNS.loglikelihood
    spv = np.array([5.,5.,5.,5.,5.])
    assert lnl(spv) == loglikelihood(spv)
    pop = MNNS.population_size
    assert pop == population_size

def test_func_run():
    MNNS = shared['MNNS']
    log_evidence, log_evidence_error = MNNS.run(verbose=False)
    analytic = analytic_log_evidence(ndim, width)
    print(analytic, log_evidence)
    assert np.isclose(log_evidence, analytic, rtol=1.e-1)
    shared['MNNS'] = MNNS

def test_properties():
    MNNS = shared['MNNS']
    analytic = analytic_log_evidence(ndim, width)
    lnZ = MNNS.log_evidence
    assert np.isclose(lnZ, analytic, rtol=1.e-1)
    lnZ_err = MNNS.log_evidence_error
    Z = MNNS.evidence
    Z_err = MNNS.evidence_error
    H = MNNS.information
    assert H is None

def test_func_posteriors():
    MNNS = shared['MNNS']
    posteriors = MNNS.posteriors()
    keys = list(posteriors.keys())
    assert len(keys) == len(sampled_parameters)

def test_func_akaike_ic():
    MNNS = shared['MNNS']
    aic = MNNS.akaike_ic()

def test_func_bayesian_ic():
    MNNS = shared['MNNS']
    bic = MNNS.bayesian_ic(n_data=5)

def test_func_deviance_ic():
    MNNS = shared['MNNS']
    dic = MNNS.deviance_ic()

def test_cleanup():
    # Clean-up the MultiNest output files
    for f in glob.glob("./multinest_run*"):
        os.remove(f)

if __name__ == '__main__':
    test_initialization()
    test_attributes()
    test_func_run()
    test_properties()
    test_func_posteriors()
    test_func_akaike_ic()
    test_func_bayesian_ic()
    test_func_deviance_ic()
    test_cleanup()
