"""Implementation on top of PolyChord via its python wrapper pypolychord.

This module defines the class for Nested Sampling using using the PolyChord
(i.e., PolyChordLite) program via its Python wrapper pypolychord. Note that
pypolychord has to be built and installed separately (from gleipnir) before
this module can be used.

PolyChordLite: https://github.com/PolyChord/PolyChordLite

References:
    1. Handley, W. J., M. P. Hobson, and A. N. Lasenby. "PolyChord: nested
        sampling for cosmology." Monthly Notices of the Royal Astronomical
        Society: Letters 450.1 (2015): L61-L65.
    2. Handley, W. J., M. P. Hobson, and A. N. Lasenby. "POLYCHORD:
        next-generation nested sampling." Monthly Notices of the Royal
        Astronomical Society 453.4 (2015): 4384-4398.

"""

import numpy as np
import scipy
import warnings
from .nsbase import NestedSamplingBase

try:
    import pypolychord
    from pypolychord.settings import PolyChordSettings
except ImportError as err:
    #print(err)
    raise err

class PolyChordNestedSampling(NestedSamplingBase):
    """Nested Sampling using PolyChord.
    PolyChord and pypolychord: https://github.com/PolyChord/PolyChordLites
    Attributes:
        sampled_parameters (list of :obj:gleipnir.sampled_parameter.SampledParameter):
            The parameters that are being sampled during the Nested Sampling
            run.
        loglikelihood (function): The log-likelihood function to use for
            assigning a likelihood to parameter vectors during the sampling.
        population_size (int): The number of points to use in the Nested
            Sampling active population.
    References:
        1. Handley, W. J., M. P. Hobson, and A. N. Lasenby. "PolyChord: nested
            sampling for cosmology." Monthly Notices of the Royal Astronomical
            Society: Letters 450.1 (2015): L61-L65.
        2. Handley, W. J., M. P. Hobson, and A. N. Lasenby. "POLYCHORD:
            next-generation nested sampling." Monthly Notices of the Royal
            Astronomical Society 453.4 (2015): 4384-4398.
    """

    def __init__(self, sampled_parameters, loglikelihood, population_size):
        """Initialize the PolyChord Nested Sampler."""
        self.sampled_parameters = sampled_parameters
        self.loglikelihood = loglikelihood
        self.population_size = population_size

        self._nDims = len(sampled_parameters)
        self._nDerived = 0
        self._post_eval = False
        self._posteriors = None
        #if self.population_size is None:
        #    self.population_size = 25*self._nDims
        # make the likelihood function for polychord
        def likelihood(theta):
            r2 = 0
            return loglikelihood(theta), [r2]
        self._likelihood = likelihood
        # make the prior for polychord
        def prior(hypercube):
            return np.array([self.sampled_parameters[i].invcdf(value) for i,value in enumerate(hypercube)])

        self._prior = prior
        # PolyChord settings object
        self._settings = PolyChordSettings(self._nDims, self._nDerived,
                                           nlive=self.population_size)
        self._settings.file_root = 'polychord_run' #string
        self._settings.do_clustering = True
        self._settings.read_resume = False
        # Make the polychord dumper function
        # param : array, array, array, float, float
        def dumper(live, dead, logweights, logZ, logZerr):
            print("Last dead point:", dead[-1]) # prints last element of dead (wich is an array)

        self._dumper = dumper
        return


    def run(self, verbose=False):
        """Initiate the PolyChord Nested Sampling run."""
        output = pypolychord.run_polychord(self._likelihood, self._nDims,
                                           self._nDerived, self._settings,
                                           self._prior, self._dumper)
        self._output = output
        return output.logZ, output.logZerr

    @property
    def evidence(self):
        """float: Estimate of the Bayesian evidence, or Z."""
        return np.exp(self._output.logZ)
    @evidence.setter
    def evidence(self, value):
        warnings.warn("evidence is not settable")

    @property
    def evidence_error(self):
        """float: Estimate (rough) of the error in the evidence, or Z."""
        return np.exp(self._output.logZerr)
    @evidence_error.setter
    def evidence_error(self, value):
        warnings.warn("evidence_error is not settable")

    @property
    def log_evidence(self):
        """float: Estimate of the natural logarithm of the Bayesian evidence, or ln(Z).
        """
        return self._output.logZ
    @log_evidence.setter
    def log_evidence(self, value):
        warnings.warn("log_evidence is not settable")

    @property
    def log_evidence_error(self):
        """float: Estimate of the error in the natural logarithm of the evidence.
        """
        return self._output.logZerr
    @log_evidence_error.setter
    def log_evidence_error(self, value):
        warnings.warn("log_evidence_error is not settable")

    @property
    def information(self):
        """None: Not implemented yet->Estimate of the Bayesian information, or H."""
        return None
    @information.setter
    def information(self, value):
        warnings.warn("information is not settable")

    @property
    def polychord_file_root(self):
        """str: The file root used by Polychord output files."""
        return self._settings.file_root
    @polychord_file_root.setter
    def polychord_file_root(self, value):
        self._settings.file_root = value
        return

    def posteriors(self, nbins=None):
        """Estimates of the posterior marginal probability distributions of each parameter.
        Returns:
            dict of tuple of (numpy.ndarray, numpy.ndarray, numpy.ndarray): The
                histogram estimates of the posterior marginal probability
                distributions. The returned dict is keyed by the sampled
                parameter names and each element is a tuple with
                (marginal_weights, bin_edges, bin_centers).
        """
        # Lazy evaluation at first call of the function and store results
        # so that subsequent calls don't have to recompute.
        if not self._post_eval:
            # Here the samples are samples directly from the posterior
            # (i.e. equal weights). - The samples from the PolyChord output
            # is a pandas DataFrame.
            samples = self._output.samples
            log_likelihoods = samples['loglike'].to_numpy()
            parms = samples.columns[2:]
            # Rice bin count selection
            if nbins is None:
                nbins = 2 * int(np.cbrt(len(log_likelihoods)))
            self._posteriors = dict()
            for ii,parm in enumerate(parms):
                marginal, edge = np.histogram(samples[parm], density=True, bins=nbins)
                center = (edge[:-1] + edge[1:])/2.
                self._posteriors[self.sampled_parameters[ii].name] = (marginal, edge, center)
            self._post_eval = True

        return self._posteriors

    def max_loglikelihood(self):
        samples = self._output.samples
        mx = samples.max()
        ml = mx['loglike']
        return ml

    def deviance_ic(self):
        """Estimate Deviance Information Criterion.
        This function estimates the Deviance Information Criterion (DIC) for the
        model simulated with Nested Sampling (NS). It does so by using the
        posterior distribution estimates computed from the NS outputs.
        The DIC formula is given by:
            DIC = p_D + D_bar,
        where p_D = D_bar - D(theta_bar), D_bar is the posterior average of
        the deviance D(theta)= -2*ln(L(theta)) with L(theta) the likelihood
        of parameter set theta, and theta_bar is posterior average parameter set.

        Returns:
            float: The DIC estimate.
        """
        samples = self._output.samples
        log_likelihoods = samples['loglike'].to_numpy()
        # likelihoods = np.exp(log_likelihoods)
        parms = samples.columns[2:]
        params = samples[parms]
        D_of_theta = -2.*log_likelihoods
        D_bar = np.average(D_of_theta)
        theta_bar = np.average(params, axis=0)
        print(theta_bar)
        D_of_theta_bar = -2. * self.loglikelihood(theta_bar)
        p_D = D_bar - D_of_theta_bar
        return p_D + D_bar

    def best_fit_likelihood(self):
        """Parameter vector with the maximum likelihood.
        Returns:
            numpy.array: The parameter vector.
        """
        samples = self._output.samples
        midx = np.argmax(samples['loglike'].values)
        ml = samples.values[midx][2:]
        return ml
