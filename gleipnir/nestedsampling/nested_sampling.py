"""Implementation of Nested Sampling.

This module defines the class used for Nested Sampling.

References:
    1. Skilling, John. "Nested sampling." AIP Conference Proceedings. Vol.
        735. No. 1. AIP, 2004.
    2. Skilling, John. "Nested sampling for general Bayesian computation."
        Bayesian analysis 1.4 (2006): 833-859.
    3. Skilling, John. "Nested sampling’s convergence." AIP Conference
        Proceedings. Vol. 1193. No. 1. AIP, 2009.
"""

import numpy as np
import pandas as pd
import warnings
from ..nsbase import NestedSamplingBase
from .samplers import MetropolisComponentWiseHardNSRejection
from .stopping_criterion import NumberOfIterations


class NestedSampling(NestedSamplingBase):
    """A Nested Sampler.
    This class is an implementation of the outer layer of the classic Nested
    Sampling algorithm.

    Attributes:
        sampled_parameters (list of :obj:gleipnir.sampled_parameter.SampledParameter):
            The parameters that are being sampled during the Nested Sampling
            run.
        loglikelihood (function): The log-likelihood function to use for
            assigning a likelihood to parameter vectors during the sampling.
        population_size (int): The number of points to use in the Nested
            Sampling active population.
        sampler (obj from gleipnir.samplers, optional): The sampling scheme to
            be used when updating sample points.
            Default: MetropolisComponentWiseHardNSRejection(10, tuning_cycles=1)
        stopping_criterion (obj from gleipnir.stopping_criterion, optional):
            The criterion that should be used to determine when to stop the
            Nested Sampling run. Default: NumberOfIterations(1000)
    References:
        1. Skilling, John. "Nested sampling." AIP Conference Proceedings. Vol.
            735. No. 1. AIP, 2004.
        2. Skilling, John. "Nested sampling for general Bayesian computation."
            Bayesian analysis 1.4 (2006): 833-859.
        3. Skilling, John. "Nested sampling’s convergence." AIP Conference
            Proceedings. Vol. 1193. No. 1. AIP, 2009.
    """

    def __init__(self, sampled_parameters, loglikelihood, population_size,
                 sampler=MetropolisComponentWiseHardNSRejection(10, tuning_cycles=1),
                 stopping_criterion=NumberOfIterations(1000)):
        """Initialize the Nested Sampler."""
        # stor inputs
        self.sampled_parameters = sampled_parameters
        # Make a dictionary version of the sampled parameters
        self._sampled_parameters_dict = {sp.name:sp for sp in sampled_parameters}
        self.loglikelihood = loglikelihood
        self.sampler = sampler
        self.population_size = population_size
        self.stopping_criterion = stopping_criterion

        # estimate of NS constriction factor
        self._alpha = population_size/(population_size+1)

        # NS accumulators and other private attributes
        self._evidence = 0.0
        self._evidence_error = 0.0
        self._logZ_err = 0.0
        self._log_evidence = 0.0
        self._information = 0.0
        self._H = 0.0
        self._previous_evidence = 0.0
        self._current_weights = 1.0
        self._previous_weight = 1.0
        self._n_iterations = 0
        self._dead_points = list()
        self._live_points = None
        self._post_eval = False
        self._posteriors = None
        return

    def run(self, verbose=False):
        """Initiate the Nested Sampling run.
        Returns:
            tuple of (float, float): Tuple containing the natural logarithm
            of the evidence and its error estimate as computed from the
            Nested Sampling run: (log_evidence, log_evidence_error)
        """
        # Zeroth iteration -- generate all the random samples
        if verbose:
            print("Generating the initial set of live points with population size {}...".format(self.population_size))
        live_points = dict()
        for i in range(self.population_size):
            for sampled_parameter_name in self._sampled_parameters_dict:
                name = sampled_parameter_name
                rs = self._sampled_parameters_dict[sampled_parameter_name].rvs(1)[0]
                if name not in live_points.keys():
                    live_points[name] = list([rs])
                else:
                    live_points[name].append(rs)

        self._live_points = pd.DataFrame(live_points)

        # Evaulate the log likelihood function for each live point
        if verbose:
            print("Evaluating the loglikelihood function for each live point...")
        log_likelihoods = np.array([self.loglikelihood(sampled_parameter_vector) for sampled_parameter_vector in self._live_points.values])

        # first iteration
        self._n_iterations += 1
        self._current_weights = 1.0 - self._alpha**self._n_iterations

        # Get the lowest likelihood live point
        ndx = np.argmin(log_likelihoods)
        log_l = log_likelihoods[ndx]
        param_vec = self._live_points.values[ndx]
        dZ = self._current_weights*np.exp(log_l)
        self._evidence += dZ
        # Accumulate the information
        dH = dZ*log_l
        if np.isnan(dH): dH = 0.0
        self._H += dH
        if self._evidence > 0.0:
            self._information = -np.log(self._evidence)+self._H/self._evidence

        self._previous_weight = self._current_weights
        # Add the lowest likelihood live point to dead points -- use dict that
        # that can be easily converted to pandas DataFrame.
        dpd = dict({'log_l': log_l, 'weight':self._current_weights})
        for k,val in enumerate(param_vec):
            dpd[self.sampled_parameters[k].name] = val
        self._dead_points.append(dpd)

        if verbose:
            print("Iteration: {} Evidence estimate: {} Remaining prior mass: {}".format(self._n_iterations, self._evidence, self._alpha**self._n_iterations))
            print("Dead Point:")
            print(self._dead_points[-1])

        # subseqent iterations
        while not self._stopping_criterion():
            self._n_iterations += 1
            self._current_weights = self._alpha**(self._n_iterations-1.0) - self._alpha**self._n_iterations

            # Replace the dead point with a modified survivor.
            # Choose at random from the survivors.
            r_p_ndx = int(np.random.random(1)*self.population_size)
            while r_p_ndx == ndx:
                r_p_ndx = int(np.random.random(1)*self.population_size)
            # Now make a new point from the survivor via the sampler.
            r_p_param_vec = self._live_points.values[r_p_ndx]
            updated_point_param_vec, u_log_l = self.sampler(self.sampled_parameters, self.loglikelihood, r_p_param_vec, log_l)
            log_likelihoods[ndx] = u_log_l
            self._live_points.values[ndx] = updated_point_param_vec
            # Get the lowest likelihood live point.
            ndx = np.argmin(log_likelihoods)
            log_l = log_likelihoods[ndx]
            param_vec = self._live_points.values[ndx]
            # Accumulate the evidence.
            dZ = self._current_weights*np.exp(log_l)
            self._evidence += dZ
            # Accumulate the information.
            dH = dZ*log_l
            if np.isnan(dH): dH = 0.0
            self._H += dH
            if self._evidence > 0.0:
                self._information = -np.log(self._evidence)+self._H/self._evidence

            # Add the lowest likelihood live point to dead points
            dpd = dict({'log_l': log_l, 'weight':self._current_weights})
            for k,val in enumerate(param_vec):
                dpd[self.sampled_parameters[k].name] = val
            self._dead_points.append(dpd)

            self._previous_weight = self._current_weights
            if verbose and (self._n_iterations%10==0):
                logZ_err = np.sqrt(self._information/self.population_size)
                ev_err = np.exp(logZ_err)
                print("Iteration: {} Evidence estimate: {} +- {} Remaining prior mass: {}".format(self._n_iterations, self._evidence, ev_err, self._alpha**self._n_iterations))
                print("Dead Point:")
                print(self._dead_points[-1])

        # Accumulate the final bit for remaining surviving points.
        weight = self._alpha**(self._n_iterations)
        likelihoods = np.exp(log_likelihoods)
        likelihoods_surv = np.array([likelihood for i,likelihood in enumerate(likelihoods) if i != ndx])
        l_m = likelihoods_surv.mean()
        self._evidence += weight*l_m
        # Accumulate the information.
        dH = weight*l_m*np.log(l_m)
        if np.isnan(dH): dH = 0.0
        self._H += dH
        if self._evidence > 0.0:
            self._information = -np.log(self._evidence)+self._H/self._evidence
        n_left = len(likelihoods_surv)
        a_weight = weight/n_left
        # Add the final survivors to the dead points.
        for i,l_likelihood in enumerate(log_likelihoods):
            if i != ndx:
                dpd = dict({'log_l': l_likelihood, 'weight':a_weight})
                for k,val in enumerate(self._live_points.values[i]):
                    dpd[self.sampled_parameters[k].name] = val
                self._dead_points.append(dpd)

        logZ_err = np.sqrt(self._information/self.population_size)
        self._logZ_err = logZ_err
        ev_err = np.exp(logZ_err)
        self._evidence_error = ev_err
        self._log_evidence = np.log(self._evidence)
        # Convert the dead points dict to a pandas DataFrame.
        self._dead_points = pd.DataFrame(self._dead_points)

        return self._log_evidence, logZ_err

    def _stopping_criterion(self):
        """Wrapper function for the stopping criterion."""
        return self.stopping_criterion(self)

    @property
    def evidence(self):
        """float: Estimate of the Bayesian evidence, or Z.
        """
        return self._evidence
    @evidence.setter
    def evidence(self, value):
        warnings.warn("evidence is not settable")

    @property
    def evidence_error(self):
        """float: Estimate (rough) of the error in the evidence, or Z.

        The error in the evidence is computed as the approximation:
            exp(sqrt(information/population_size))
        """
        return self._evidence_error
    @evidence_error.setter
    def evidence_error(self, value):
        warnings.warn("evidence_error is not settable")

    @property
    def log_evidence(self):
        """float: Estimate of the natural logarithm of the Bayesian evidence, or ln(Z).
        """
        return self._log_evidence
    @log_evidence.setter
    def log_evidence(self, value):
        warnings.warn("log_evidence is not settable")

    @property
    def log_evidence_error(self):
        """float: Estimate (rough) of the error in the natural logarithm of the evidence.
        """
        return self._logZ_err
    @log_evidence_error.setter
    def log_evidence_error(self, value):
        warnings.warn("log_evidence_error is not settable")

    @property
    def information(self):
        """float: Estimate of the Bayesian information, or H."""
        return self._information
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
        if not self._post_eval:
            log_likelihoods = self._dead_points['log_l'].to_numpy()
            weights = self._dead_points['weight'].to_numpy()
            likelihoods = np.exp(log_likelihoods)
            norm_weights = (weights*likelihoods)/self.evidence
            gt_mask = norm_weights > 0.0
            parms = self._dead_points.columns[2:]
            # Rice bin count selection
            if nbins is None:
                nbins = 2 * int(np.cbrt(len(norm_weights[gt_mask])))
            self._posteriors = dict()
            for parm in parms:
                marginal, edge = np.histogram(self._dead_points[parm][gt_mask], weights=norm_weights[gt_mask], density=True, bins=nbins)
                center = (edge[:-1] + edge[1:])/2.
                self._posteriors[parm] = (marginal, edge, center)
            self._post_eval = True
        return self._posteriors

    def max_loglikelihood(self):
        mx = self._dead_points.max()
        ml = mx['log_l']
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
        log_likelihoods = self._dead_points['log_l'].to_numpy()
        weights = self._dead_points['weight'].to_numpy()
        likelihoods = np.exp(log_likelihoods)
        norm_weights = (weights*likelihoods)/self.evidence
        gt_mask = norm_weights > 0.0
        parms = self._dead_points.columns[2:]
        params = self._dead_points[parms]
        D_of_theta = -2.*log_likelihoods[gt_mask]
        D_bar = np.average(D_of_theta, weights=norm_weights[gt_mask])
        theta_bar = np.average(params[gt_mask], weights=norm_weights[gt_mask], axis=0)
        D_of_theta_bar = -2. * self.loglikelihood(theta_bar)
        p_D = D_bar - D_of_theta_bar
        return p_D + D_bar

    def best_fit_likelihood(self):
        """Parameter vector with the maximum likelihood.
        Returns:
            numpy.array: The parameter vector.
        """
        midx = np.argmax(self._dead_points.values[:,0])
        # print(mx)
        ml = self._dead_points.values[midx][2:]
        return ml

    @property
    def dead_points(self):
        """The set of dead points collected during the Nested Sampling run."""
        return self._dead_points
    @dead_points.setter
    def dead_points(self, value):
            warnings.warn("dead_points is not settable")
    # @property
    # def samples(self):
    #     return self._dead_points
    # @samples.setter
    # def samples(self, value):
    #     warnings.warn("samples is not settable")
