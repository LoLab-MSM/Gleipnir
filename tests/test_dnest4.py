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
from gleipnir.dnest4 import DNest4NestedSampling
import os
import glob


# Number of paramters to sample is 5
ndim = 5
# Set up the list of sampled parameters: the prior is Uniform(-5:5) --
# we are using a fixed uniform prior from scipy.stats
sampled_parameters = [SampledParameter(name=i, prior=uniform(loc=-5.0,scale=10.0)) for i in range(ndim)]
# Set the active point population size
population_size = 100
# Number of iterations -- num_steps
num_steps = 200
# Number of iterations between creation of each diffusive level. num_per_step
num_per_step = 100
# Number of diffusive levels
n_levels = 10

# Define the loglikelihood function
def loglikelihood(sampled_parameter_vector):
    const = -0.5*np.log(2*np.pi)
    return -0.5*np.sum(sampled_parameter_vector**2) + ndim * const

width = 10.0
def analytic_log_evidence(ndim, width):
      lZ = (ndim * np.log(erf(0.5*width/np.sqrt(2)))) - (ndim * np.log(width))
      return lZ


shared = {'DNS': None}

def test_initialization():
    DNS = DNest4NestedSampling(sampled_parameters=sampled_parameters,
                                   loglikelihood=loglikelihood,
                                   population_size=population_size,
                                   n_diffusive_levels=n_levels,
                                   num_steps=num_steps,
                                   num_per_step=num_per_step)
    shared['DNS'] = DNS

def test_attributes():
    DNS = shared['DNS']
    sp = DNS.sampled_parameters
    assert sp == sampled_parameters
    lnl = DNS.loglikelihood
    spv = np.array([5.,5.,5.,5.,5.])
    assert lnl(spv) == loglikelihood(spv)
    pop = DNS.population_size
    assert pop == population_size
    diff_lev = DNS.n_diffusive_levels
    assert diff_lev == 10
    dn4back = DNS.dnest4_backend
    assert dn4back == 'memory'
    nst = DNS.dnest4_kwargs['num_steps']
    assert nst == num_steps
    nps = DNS.dnest4_kwargs['num_per_step']
    assert nps == num_per_step

def test_func_run():
    DNS = shared['DNS']
    log_evidence, log_evidence_error = DNS.run(verbose=False)
    analytic = analytic_log_evidence(ndim, width)
    print(analytic, log_evidence)
    assert np.isclose(log_evidence, analytic, rtol=1.e-1)
    shared['DNS'] = DNS

def test_properties():
    DNS = shared['DNS']
    analytic = analytic_log_evidence(ndim, width)
    lnZ = DNS.log_evidence
    assert np.isclose(lnZ, analytic, rtol=1.e-1)
    lnZ_err = DNS.log_evidence_error
    Z = DNS.evidence
    Z_err = DNS.evidence_error
    H = DNS.information
    assert H is not None

def test_func_posteriors():
    DNS = shared['DNS']
    posteriors = DNS.posteriors()
    keys = list(posteriors.keys())
    assert len(keys) == len(sampled_parameters)

def test_func_akaike_ic():
    DNS = shared['DNS']
    aic = DNS.akaike_ic()

def test_func_bayesian_ic():
    DNS = shared['DNS']
    bic = DNS.bayesian_ic(n_data=5)

def test_func_deviance_ic():
    DNS = shared['DNS']
    dic = DNS.deviance_ic()


if __name__ == '__main__':
    test_initialization()
    test_attributes()
    test_func_run()
    test_properties()
    test_func_posteriors()
    test_func_akaike_ic()
    test_func_bayesian_ic()
    test_func_deviance_ic()
