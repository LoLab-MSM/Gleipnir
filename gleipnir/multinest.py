"""Implementation on top of MultiNest via PyMultiNest.

This module defines the class for Nested Sampling using the MultiNest
program via its Python wrapper PyMultiNest. Note that PyMultiNest and MultiNest
have to be built and installed separately (from gleipnir) before this module
can be used.

PyMultiNest: https://github.com/JohannesBuchner/PyMultiNest
MultiNest: https://github.com/JohannesBuchner/MultiNest

References:
    MultiNest:
    1. Feroz, Farhan, and M. P. Hobson. "Multimodal nested sampling: an
        efficient and robust alternative to Markov Chain Monte Carlo
        methods for astronomical data analyses." Monthly Notices of the
        Royal Astronomical Society 384.2 (2008): 449-463.
    2. Feroz, F., M. P. Hobson, and M. Bridges. "MultiNest: an efficient
        and robust Bayesian inference tool for cosmology and particle
        physics." Monthly Notices of the Royal Astronomical Society 398.4
        (2009): 1601-1614.
    3. Feroz, F., et al. "Importance nested sampling and the MultiNest
        algorithm." arXiv preprint arXiv:1306.2144 (2013).
    PyMultiNest:
    4. Buchner, J., et al. "X-ray spectral modelling of the AGN obscuring
        region in the CDFS: Bayesian model selection and catalogue."
        Astronomy & Astrophysics 564 (2014): A125.

"""

import numpy as np
import warnings

try:
    import pymultinest
    from pymultinest.solve import solve
    from pymultinest.analyse import Analyzer
except ImportError as err:
    #print(err)
    raise err

class MultiNestNestedSampling(object):
    """Nested Sampling using MultiNest.
    PyMultiNest: https://github.com/JohannesBuchner/PyMultiNest
    MultiNest: https://github.com/JohannesBuchner/MultiNest

    Attributes:
        sampled_parameters (list of :obj:gleipnir.sampled_parameter.SampledParameter):
            The parameters that are being sampled during the Nested Sampling
            run.
        loglikelihood (function): The log-likelihood function to use for
            assigning a likelihood to parameter vectors during the sampling.
        population_size (int): The number of points to use in the Nested
            Sampling active population. Default: None -> gets set to
            25*(number of sampled parameters) if left at default.
        multinest_kwargs (dict): Additional keyword arguments that should be
            passed to the PyMultiNest MultiNest solver.
    References:
        1. Feroz, Farhan, and M. P. Hobson. "Multimodal nested sampling: an
            efficient and robust alternative to Markov Chain Monte Carlo
            methods for astronomical data analyses." Monthly Notices of the
            Royal Astronomical Society 384.2 (2008): 449-463.
        2. Feroz, F., M. P. Hobson, and M. Bridges. "MultiNest: an efficient
            and robust Bayesian inference tool for cosmology and particle
            physics." Monthly Notices of the Royal Astronomical Society 398.4
            (2009): 1601-1614.
        3. Feroz, F., et al. "Importance nested sampling and the MultiNest
            algorithm." arXiv preprint arXiv:1306.2144 (2013).
        4. Buchner, J., et al. "X-ray spectral modelling of the AGN obscuring
            region in the CDFS: Bayesian model selection and catalogue."
            Astronomy & Astrophysics 564 (2014): A125.
    """

    def __init__(self, sampled_parameters, loglikelihood, population_size=None,
                 **multinest_kwargs):
        """Initialize the MultiNest Nested Sampler."""
        self.sampled_parameters = sampled_parameters
        self.loglikelihood = loglikelihood
        self.population_size = population_size
        self.multinest_kwargs = multinest_kwargs

        self._nDims = len(sampled_parameters)
        self._nDerived = 0
        self._output = None
        self._post_eval = False
        if self.population_size is None:
            self.population_size = 25*self._nDims

        # Make the prior function for PyMultiNest.
        def prior(hypercube):
            return np.array([self.sampled_parameters[i].invcdf(value) for i,value in enumerate(hypercube)])

        self._prior = prior
        # multinest settings
        self._file_root = 'multinest_run' #string

        return


    def run(self, verbose=False):
        """Initiate the MultiNest Nested Sampling run."""
        output = solve(LogLikelihood=self.loglikelihood, Prior=self._prior,
                       n_dims = self._nDims,
                       n_live_points=self.population_size,
                       outputfiles_basename=self._file_root,
                       verbose=verbose,
                       **self.multinest_kwargs)
        self._output = output
        return self.log_evidence, self.log_evidence_error

    @property
    def evidence(self):
        """float: Estimate of the Bayesian evidence, or Z."""
        return np.exp(self._output['logZ'])
    @evidence.setter
    def evidence(self, value):
        warnings.warn("evidence is not settable")

    @property
    def evidence_error(self):
        """float: Estimate (rough) of the error in the evidence, or Z."""
        return np.exp(self._output['logZerr'])
    @evidence_error.setter
    def evidence_error(self, value):
        warnings.warn("evidence_error is not settable")

    @property
    def log_evidence(self):
        """float: Estimate of the natural logarithm of the Bayesian evidence, or ln(Z).
        """
        return self._output['logZ']
    @log_evidence.setter
    def log_evidence(self, value):
        warnings.warn("log_evidence is not settable")

    @property
    def log_evidence_error(self):
        """float: Estimate of the error in the natural logarithm of the evidence.
        """
        return self._output['logZerr']
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

    def posteriors(self, nbins=None):
        """Estimates of the posterior marginal probability distributions of each parameter.
        Returns:
            dict of tuple of (numpy.ndarray, numpy.ndarray): The histogram
                estimates of the posterior marginal probability distributions.
                The returned dict is keyed by the sampled parameter names and
                each element is a tuple with (marginal_weights, bin_centers).
        """
        # Lazy evaluation at first call of the function and store results
        # so that subsequent calls don't have to recompute.
        print('nbins', nbins)
        if not self._post_eval:
            # Here the samples are samples directly from the posterior
            # (i.e. equal weights).
            samples = self._output['samples']
            # Rice bin count selection
            if nbins is None:
                nbins = 2 * int(np.cbrt(len(samples)))
            print('nbins', nbins)
            nd = samples.shape[1]
            self._posteriors = dict()
            for ii in range(nd):
                marginal, edge = np.histogram(samples[:,ii], density=True, bins=nbins)
                center = (edge[:-1] + edge[1:])/2.
                self._posteriors[self.sampled_parameters[ii].name] = (marginal, center)
            self._post_eval = True

        return self._posteriors

    def akaike_ic(self):
        mn_data = Analyzer(len(self.sampled_parameters), self._file_root, verbose=False).get_data()
        log_ls = -0.5*mn_data[:,1]
        ml = log_ls.max()
        k = len(self.sampled_parameters)
        return  2.*k - 2.*ml
