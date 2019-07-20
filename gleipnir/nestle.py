"""Implementation on top of Nestle.

This module defines the class for Nested Sampling using the Nestle
program (a Python library).

Nestle: https://github.com/kbarbary/nestle
Nestle docs: http://kylebarbary.com/nestle/

References:
    None

"""

import numpy as np
import warnings
from .nsbase import NestedSamplingBase

try:
    import nestle
except ImportError as err:
    #print(err)
    raise err

class NestleNestedSampling(NestedSamplingBase):
    """Nested Sampling using Nestle.
    Nestle: https://github.com/kbarbary/nestle

    Attributes:
        sampled_parameters (list of :obj:gleipnir.sampled_parameter.SampledParameter):
            The parameters that are being sampled during the Nested Sampling
            run.
        loglikelihood (function): The log-likelihood function to use for
            assigning a likelihood to parameter vectors during the sampling.
        population_size (int): The number of points to use in the Nested
            Sampling active population.
        nestle_kwargs (dict): Additional keyword arguments that should be
            passed to the to nestle.sample. Available options are:
                method (str): Method to use to generate new sample points.
                    Options are 'classic' (MCMC), 'single' (single-ellipsoidal),
                    and 'multi' (multi-ellipsoidal). Default: 'single'
                update_interval (int): Only update the new point generator
                    every this many likelihood calls. Default: round(0.6*npoints)
	            maxiter (int): Set a maximum number of nested iterations that
                    are allowed to be run before terminating.
                    Default: None -> no limit
                maxcall (int): Set a maximum number of likelihood evaluations that
                    are allowed to be run before terminating the nested sampling
                    run. Default: None -> no limit
                dlogz (float): If specified, set a termination threshold for
                    the contribution of the remaining prior volume to the
                    log-evidence. Terminates when the following condition is
                    true: log(Z + Z_remain) - log(Z) < dlogz
                    to output files. True is required for additional
                    analysis. Default: None -> If neither this nor
                    decline_factor are set, then dlogz will be set to
                    dlogz=0.5.
                decline_factor (float): If specified, set a termination
                    threshold based on the weight (prior mass times likelihood)
                    of the most recent dead points. Terminates if the weight
                    has been declining for decline_factor * nsamples iterations.
                    Nestle documentation suggests a value of 1.0 works well.
                    Mutually exclusive from dlogz. Default: None -> If neither
                    this nor dlogz are set, then dlogz will be set to
                    dlogz=0.5.

    References:
        None
    """

    def __init__(self, sampled_parameters, loglikelihood, population_size,
                 **nestle_kwargs):
        """Initialize the Nestle Nested Sampler."""
        self.sampled_parameters = sampled_parameters
        self.loglikelihood = loglikelihood
        self.population_size = population_size
        self.nestle_kwargs = nestle_kwargs

        self._nDims = len(sampled_parameters)
        self._nDerived = 0
        self._output = None
        self._post_eval = False
        #if self.population_size is None:
        #    self.population_size = 25*self._nDims

        # Make the prior_transform function for Nest.
        def prior_transform(hypercube):
            return np.array([self.sampled_parameters[i].invcdf(value) for i,value in enumerate(hypercube)])

        self._prior_transform = prior_transform
        # multinest settings
        #self._file_root = 'multinest_run_' #string

        return


    def run(self, verbose=False):
        """Initiate the Nestle Nested Sampling run."""
        callback = None
        if verbose:
            callback = nestle.print_progress
        output = nestle.sample(self.loglikelihood, self._prior_transform,
                               self._nDims,
                               npoints=self.population_size,
                               callback = callback,
                               **self.nestle_kwargs)
        if verbose:
            output.summary()
        self._output = output
        return self.log_evidence, self.log_evidence_error

    @property
    def evidence(self):
        """float: Estimate of the Bayesian evidence, or Z."""
        return np.exp(self._output.logz)
    @evidence.setter
    def evidence(self, value):
        warnings.warn("evidence is not settable")

    @property
    def evidence_error(self):
        """float: Estimate (rough) of the error in the evidence, or Z."""
        return np.exp(self._output.logzerr)
    @evidence_error.setter
    def evidence_error(self, value):
        warnings.warn("evidence_error is not settable")

    @property
    def log_evidence(self):
        """float: Estimate of the natural logarithm of the Bayesian evidence, or ln(Z).
        """
        return self._output.logz
    @log_evidence.setter
    def log_evidence(self, value):
        warnings.warn("log_evidence is not settable")

    @property
    def log_evidence_error(self):
        """float: Estimate of the error in the natural logarithm of the evidence.
        """
        return self._output.logzerr
    @log_evidence_error.setter
    def log_evidence_error(self, value):
        warnings.warn("log_evidence_error is not settable")

    @property
    def information(self):
        """Estimate of the Bayesian information, or H."""
        return self._output.h
    @information.setter
    def information(self, value):
        warnings.warn("information is not settable")


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
        #print('nbins', nbins)
        if not self._post_eval:
            # Here the samples are samples directly from the posterior
            # (i.e. equal weights).
            samples = self._output.samples
            weights = self._output.weights
            # Rice bin count selection
            if nbins is None:
                nbins = 2 * int(np.cbrt(len(samples)))
            # print('nbins', nbins)
            nd = samples.shape[1]
            self._posteriors = dict()
            for ii in range(nd):
                marginal, edge = np.histogram(samples[:,ii], density=True, bins=nbins, weights=weights)
                center = (edge[:-1] + edge[1:])/2.
                self._posteriors[self.sampled_parameters[ii].name] = (marginal, edge, center)
            self._post_eval = True

        return self._posteriors

    def max_loglikelihood(self):
        log_ls = self._output.logl
        ml = log_ls.max()
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
        params = self._output.samples
        log_likelihoods = self._output.logl
        norm_weights = self._output.weights
        nw_mask = np.isnan(norm_weights)
        if  np.any(nw_mask):
            return np.inf
        D_of_theta = -2.*log_likelihoods
        D_bar = np.average(D_of_theta, weights=norm_weights)
        theta_bar = np.average(params, axis=0, weights=norm_weights)
        D_of_theta_bar = -2. * self.loglikelihood(theta_bar)
        p_D = D_bar - D_of_theta_bar
        return p_D + D_bar

    def best_fit_likelihood(self):
        """Parameter vector with the maximum likelihood.
        Returns:
            numpy.array: The parameter vector.
        """
        samples = self._output.samples
        log_ls = self._output.logl
        midx = np.argmax(log_ls)
        ml = samples[midx][:]
        return ml
