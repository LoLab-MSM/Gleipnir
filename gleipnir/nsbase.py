import numpy as np
import scipy
from abc import ABC, abstractmethod

class NestedSamplingBase(ABC):
    """Abstract base class for Nested Samplers."""

    @abstractmethod
    def __init__(self, sampled_parameters, loglikelihood, population_size):
        self.sampled_parameters = sampled_parameters
        self.loglikelihood = loglikelihood
        self.population_size = population_size

        self._post_eval = False
        self._posteriors = None
        return

    @abstractmethod
    def run(self, verbose=False):
        pass

    @property
    @abstractmethod
    def evidence(self):
        """float: Estimate of the Bayesian evidence, or Z."""
        pass

    @property
    @abstractmethod
    def evidence_error(self):
        """float: Estimate (rough) of the error in the evidence, or Z."""
        pass

    @property
    @abstractmethod
    def log_evidence(self):
        """float: Estimate of the natural logarithm of the Bayesian evidence, or ln(Z).
        """
        pass

    @property
    @abstractmethod
    def log_evidence_error(self):
        """float: Estimate of the error in the natural logarithm of the evidence.
        """
        pass

    @property
    @abstractmethod
    def information(self):
        """None: Not implemented yet->Estimate of the Bayesian information, or H."""
        pass

    @abstractmethod
    def posteriors(self, nbins=None):
        """Estimates of the posterior marginal probability distributions of each parameter.
        Returns:
            dict of tuple of (numpy.ndarray, numpy.ndarray, numpy.ndarray): The
                histogram estimates of the posterior marginal probability
                distributions. The returned dict is keyed by the sampled
                parameter names and each element is a tuple with
                (marginal_weights, bin_edges, bin_centers).
        """
        pass

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

    @abstractmethod
    def max_loglikelihood(self):
        """Get the maximum likelihood value found during the NS run.
        """
        pass

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
        ml = self.max_loglikelihood()
        k = len(self.sampled_parameters)
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
        ml = self.max_loglikelihood()
        k = len(self.sampled_parameters)
        return  np.log(n_data)*k - 2.*ml

    @abstractmethod
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
        pass

    @abstractmethod
    def best_fit_likelihood(self):
        """Parameter vector with the maximum likelihood.
        Returns:
            numpy.array: The parameter vector.
        """
        pass

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
