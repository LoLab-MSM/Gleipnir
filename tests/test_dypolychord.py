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
from gleipnir.dypolychord import dyPolyChordNestedSampling
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


shared = {'dyPCNS': None}

def test_initialization():
    dyPCNS = dyPolyChordNestedSampling(sampled_parameters=sampled_parameters,
                                   loglikelihood=loglikelihood,
                                   population_size=population_size)
    shared['dyPCNS'] = dyPCNS

def test_attributes():
    dyPCNS = shared['dyPCNS']
    sp = dyPCNS.sampled_parameters
    assert sp == sampled_parameters
    lnl = dyPCNS.loglikelihood
    spv = np.array([5.,5.,5.,5.,5.])
    assert lnl(spv) == loglikelihood(spv)
    pop = dyPCNS.population_size
    assert pop == population_size

def test_func_run():
    dyPCNS = shared['dyPCNS']
    log_evidence, log_evidence_error = dyPCNS.run()
    analytic = analytic_log_evidence(ndim, width)
    print(analytic, log_evidence)
    assert np.isclose(log_evidence, analytic, rtol=1.e-1)
    shared['dyPCNS'] = dyPCNS

def test_properties():
    dyPCNS = shared['dyPCNS']
    analytic = analytic_log_evidence(ndim, width)
    lnZ = dyPCNS.log_evidence
    assert np.isclose(lnZ, analytic, rtol=1.e-1)
    lnZ_err = dyPCNS.log_evidence_error
    Z = dyPCNS.evidence
    Z_err = dyPCNS.evidence_error
    H = dyPCNS.information
    assert H is None

def test_func_posteriors():
    dyPCNS = shared['dyPCNS']
    posteriors = dyPCNS.posteriors()
    keys = list(posteriors.keys())
    assert len(keys) == len(sampled_parameters)

def test_func_akaike_ic():
    dyPCNS = shared['dyPCNS']
    aic = dyPCNS.akaike_ic()

def test_func_bayesian_ic():
    dyPCNS = shared['dyPCNS']
    bic = dyPCNS.bayesian_ic(n_data=5)

def test_func_deviance_ic():
    dyPCNS = shared['dyPCNS']
    dic = dyPCNS.deviance_ic()

def test_cleanup():
    # Clean-up the MultiNest output files
    for f in glob.glob("./dypolychord_chains/dypolychord_run*"):
        os.remove(f)
    os.rmdir('dypolychord_chains/clusters')
    os.rmdir('dypolychord_chains')

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
