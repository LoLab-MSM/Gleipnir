"""Implementation on top of dyPolyChord.

This module defines the class for Nested Sampling using using the dyPolyChord
program, which implements Dynamic Nested Sampling using PolyChord. Note that
dyPolyChord requires PolyChord (i.e., pypolychord), which
has to be built and installed separately (from gleipnir) before
this module can be used. dyPolyChord also requires nestcheck (installed
automatically if you pip install dyPolyChord) to post-process
the PolyChord outputs for dynamic Nested Sampling.

dyPolyChord: https://github.com/ejhigson/dyPolyChord
PolyChordLite/pypolychord: https://github.com/PolyChord/PolyChordLite
nestcheck:

References:
    dyPolyChord
    1. Higson, (2018). dyPolyChord: dynamic nested sampling with PolyChord.
        Journal of Open Source Software, 3(29), 965,
        https://doi.org/10.21105/joss.00965
    2. Higson, E., Handley, W., Hobson, M. et al. Dynamic nested sampling: an
        improved algorithm for parameter estimation and evidence calculation.
        Stat Comput (2018).
        https://doi.org/10.1007/s11222-018-9844-0
    PolyChord
    3. Handley, W. J., M. P. Hobson, and A. N. Lasenby. "PolyChord: nested
        sampling for cosmology." Monthly Notices of the Royal Astronomical
        Society: Letters 450.1 (2015): L61-L65.
    4. Handley, W. J., M. P. Hobson, and A. N. Lasenby. "POLYCHORD:
        next-generation nested sampling." Monthly Notices of the Royal
        Astronomical Society 453.4 (2015): 4384-4398.
    nestcheck
    5. Edward Higson, Will Handley, Michael Hobson, Anthony Lasenby,
        nestcheck: diagnostic tests for nested sampling calculations,
        Monthly Notices of the Royal Astronomical Society, Volume 483, Issue 2,
        February 2019, Pages 2044-2056,
        https://doi.org/10.1093/mnras/sty3090

"""

import numpy as np
import scipy
import warnings
try:
    import pypolychord
    from pypolychord.settings import PolyChordSettings
except ImportError as err:
    #print(err)
    raise err
try:
    import dyPolyChord
    import dyPolyChord.pypolychord_utils
except ImportError as err:
    raise err
try:
    import nestcheck
    import nestcheck.data_processing as ncheck_dp
    import nestcheck.estimators as ncheck_e
except ImportError as err:
    raise err

class dyPolyChordNestedSampling(object):
    """Nested Sampling using dyPolyChord.
    dyPolyChord: https://github.com/ejhigson/dyPolyChord
    PolyChord and pypolychord: https://github.com/PolyChord/PolyChordLite
    Attributes:
        sampled_parameters (list of :obj:gleipnir.sampled_parameter.SampledParameter):
            The parameters that are being sampled during the Nested Sampling
            run.
        loglikelihood (function): The log-likelihood function to use for
            assigning a likelihood to parameter vectors during the sampling.
        population_size (int): The number of points to use in the Nested
            Sampling active population.
    References:
        dyPolyChord
        1. Higson, (2018). dyPolyChord: dynamic nested sampling with PolyChord.
            Journal of Open Source Software, 3(29), 965,
            https://doi.org/10.21105/joss.00965
        2. Higson, E., Handley, W., Hobson, M. et al. Dynamic nested sampling: an
            improved algorithm for parameter estimation and evidence calculation.
            Stat Comput (2018).
            https://doi.org/10.1007/s11222-018-9844-0
        PolyChord
        3. Handley, W. J., M. P. Hobson, and A. N. Lasenby. "PolyChord: nested
            sampling for cosmology." Monthly Notices of the Royal Astronomical
            Society: Letters 450.1 (2015): L61-L65.
        4. Handley, W. J., M. P. Hobson, and A. N. Lasenby. "POLYCHORD:
            next-generation nested sampling." Monthly Notices of the Royal
            Astronomical Society 453.4 (2015): 4384-4398.
        nestcheck
        5. Edward Higson, Will Handley, Michael Hobson, Anthony Lasenby,
            nestcheck: diagnostic tests for nested sampling calculations,
            Monthly Notices of the Royal Astronomical Society, Volume 483, Issue 2,
            February 2019, Pages 2044-2056,
            https://doi.org/10.1093/mnras/sty3090
    """

    def __init__(self, sampled_parameters, loglikelihood, population_size,
                 initial_population_size=None, dynamic_goal=0.5):
        """Initialize the PolyChord Nested Sampler."""
        self.sampled_parameter = sampled_parameters
        self.loglikelihood = loglikelihood
        self.population_size = population_size
        if initial_population_size is None:
            self.initial_population_size = int(population_size/2)
        else:
            self.initial_population_size = initial_population_size
        self.dynamic_goal = dynamic_goal

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
            return np.array([self.sampled_parameter[i].invcdf(value) for i,value in enumerate(hypercube)])

        self._prior = prior
        # PolyChord settings object
        #self._settings = PolyChordSettings(self._nDims, self._nDerived,
        #                                   nlive=self.population_size)
        #self._settings.file_root = 'polychord_run' #string
        #self._settings.do_clustering = True
        #self._settings.read_resume = False
        # dyPolyChord settings dict
        self._settings_dict = {'file_root': 'dypolychord_run',
                         'base_dir': 'dypolychord_chains',
                         'do_clustering': True,
                         'seed': 1,
                         'read_resume': False}
        # Make the polychord dumper function
        # param : array, array, array, float, float
        #def dumper(live, dead, logweights, logZ, logZerr):
        #    print("Last dead point:", dead[-1]) # prints last element of dead (wich is an array)

        #self._dumper = dumper
        return


    def run(self, verbose=False):
        """Initiate the dyPolyChord Nested Sampling run."""
        #output = pypolychord.run_polychord(self._likelihood, self._nDims,
        #                                   self._nDerived, self._settings,
        #                                   self._prior, self._dumper)
        # Make the dyPolyChord callable for running PolyChord
        dypc_callable = dyPolyChord.pypolychord_utils.RunPyPolyChord(self._likelihood,
                                                                     self._prior,
                                                                     self._nDims)
        # Run dyPolyChord
        dyPolyChord.run_dypolychord(dypc_callable, self.dynamic_goal,
                                    self._settings_dict,
                                    ninit=self.initial_population_size,
                                    nlive_const=self.population_size)
        # load the dyPolyChord run output into nestcheck
        run = nestcheck.data_processing.process_polychord_run('dypolychord_run',  # = settings['file_root']
                                                              'dypolychord_chains')    # = settings['base_dir']
        self._run = run
        self._logZ = ncheck_e.logz(run)
        self._Z = ncheck_e.evidence(run)
        return self._logZ, None

    @property
    def evidence(self):
        """float: Estimate of the Bayesian evidence, or Z."""
        return self._Z
    @evidence.setter
    def evidence(self, value):
        warnings.warn("evidence is not settable")

    @property
    def evidence_error(self):
        """float: Not returned by dyPolyChord->Estimate (rough) of the error in the evidence, or Z.
        The dyPolyChord error could be estimated from multiple runs using a
        bootstrap method provided by nestcheck, but Gleipnir currently doesn't
        bridge this functionality.
        """
        return None
    @evidence_error.setter
    def evidence_error(self, value):
        warnings.warn("evidence_error is not settable")

    @property
    def log_evidence(self):
        """float: Estimate of the natural logarithm of the Bayesian evidence, or ln(Z).
        """
        return self._logZ
    @log_evidence.setter
    def log_evidence(self, value):
        warnings.warn("log_evidence is not settable")

    @property
    def log_evidence_error(self):
        """float: Not returned by dyPolyChord->Estimate of the error in the natural logarithm of the evidence.
        The dyPolyChord error could be estimated from multiple runs using a
        bootstrap method provided by nestcheck, but Gleipnir currently doesn't
        bridge this functionality.
        """
        return None
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
    def dypolychord_file_root(self):
        """str: The file root used by dyPolychord output files."""
        return self._settings_dict['file_root']
    @dypolychord_file_root.setter
    def dypolychord_file_root(self, value):
        self._settings_dict['file_root'] = value
        return

    def posteriors(self):
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
            #samples = self._run['theta']
            #log_likelihoods = self._run['logl']
            parms = self._run['theta']
            # The log posterior weight
            logpw = nestcheck.ns_run_utils.get_logw(self._run)
            # Rice bin count selection
            nbins = 2 * int(np.cbrt(len(logpw)))
            self._posteriors = dict()
            for ii,parm in enumerate(parms[0]):
                marginal, edge = np.histogram(parms[:,ii], weights=logpw, density=True, bins=nbins)
                center = (edge[:-1] + edge[1:])/2.
                self._posteriors[self.sampled_parameter[ii].name] = (marginal, edge, center)
            self._post_eval = True

        return self._posteriors

    def posterior_moments(self):
        """Get the first 4 moments of each marginal distribution.
        Returns:
            dict of tuple of (float, float, float, float): The first 4 moments
                (mean, var, skew, kurtosis) for each parameter's marginal
                posterior distribution. The dict is keyed to parameter names.
        """
        post = self.posteriors()
        moments = dict()
        for parm in post.keys():
            marginal, edges, centers = post[parm]
            width = edges[1] - edges[0]
            # resample from the distribution
            samples = np.random.choice(centers, size=10000, p=marginal/(marginal.sum())) + (width*(np.random.random(10000)-0.5))
            mean = np.mean(samples)
            var = np.var(samples)
            skew = scipy.stats.skew(samples)
            kurtosis = scipy.stats.kurtosis(samples)
            moments[parm] = tuple((mean, var, skew, kurtosis))
        return moments

    def akaike_ic(self):
        """Estimate Akaike Information Criterion.
        This function estimates the Akaike Information Criterion (AIC) for the
        model simulated with Nested Sampling (NS). It does so by using the
        largest likelihood value found during the NS run and using that as
        the maximum likelihood estimate. The AIC formula is given by:
            AIC = 2k - 2ML,
        where k is number of sampled parameters and ML is maximum likelihood
        estimate.

        Returns:
            float: The AIC estimate.
        """
        log_likelihoods = self._run['logl']
        ml = log_likelihoods.max()
        k = len(self.sampled_parameter)
        return  2.*k - 2.*ml

    def bayesian_ic(self, n_data):
        """Estimate Bayesian Information Criterion.
        This function estimates the Bayesian Information Criterion (BIC) for the
        model simulated with Nested Sampling (NS). It does so by using the
        largest likelihood value found during the NS run and taking that as
        the maximum likelihood estimate. The BIC formula is given by:
            BIC = ln(n_data)k - 2ML,
        where n_data is the number of data points used in computing the likelihood
        function fitting, k is number of sampled parameters, and ML is maximum
        likelihood estimate.

        Args:
            n_data (int): The number of data points used when comparing to data
                in the likelihood function.

        Returns:
            float: The BIC estimate.
        """
        log_likelihoods = self._run['logl']
        ml = log_likelihoods.max()
        k = len(self.sampled_parameter)
        return  np.log(n_data)*k - 2.*ml

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
        params = self._run['theta']
        log_likelihoods = self._run['logl']
        weights = nestcheck.ns_run_utils.get_logw(self._run)
        D_of_theta = -2.*log_likelihoods
        D_bar = np.average(D_of_theta, weights=weights)
        theta_bar = np.average(params, axis=0, weights=weights)
        print(theta_bar)
        D_of_theta_bar = -2. * self.loglikelihood(theta_bar)
        p_D = D_bar - D_of_theta_bar
        return p_D + D_bar


    def best_fit_likelihood(self):
        """Parameter vector with the maximum likelihood.
        Returns:
            numpy.array: The parameter vector.
        """
        samples = self._run['theta']
        midx = np.argmax(self._run['logl'])
        ml = samples[midx][:]
        return ml

    def best_fit_posterior(self):
        """Parameter vector with the maximum posterior weight.
        The parameter vector is estimated by first estimating the posterior
        distributions via histogramming. Then the parameters with the
        highest posterior probability are determined.
        Returns:
            numpy.array, numpy.array: The parameter vector and the error
                associated with the histogram bin widths.
        """
        post = self.posteriors()
        mparms = list()
        errors = list()
        for parm in post.keys():
            marginal, edge, center = post[parm]
            midx = np.argmax(marginal)
            mparm = center[midx]
            mparms.append(mparm)
            errors.append(edge[1]-edge[0])
        return np.array(mparms), np.array(errors)/2.0
