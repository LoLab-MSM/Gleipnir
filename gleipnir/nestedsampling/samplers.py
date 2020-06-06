"""Samplers used to update points from Nested Sampling.

This module defines the classes for the samplers used to update a chosen
survivor point in order to replace the dead point during the
Nested Sampling runs via the gleipnir.nested_sampling.NestedSampling class.


"""

import numpy as np
from scipy.stats import uniform, norm

class MetropolisSampler(object):
    """Markov Chain Monte Carlo sampler based on Metropolis algorithm.
    This sampler uses Metropolis sampling, a type of Markov chain Monte Carlo,
    sampling from the prior to update a sample point. Trial moves are generated
    component-wise (looped over in order) proportional to the parameter priors
    and the typical acceptance/rejection criterion is augmented with an extra
    hard rejection criterion for the Nested Sampling likelihood level (i.e.,
    trial moves  are always rejected if the likelihood of the trial move is
    lower than that of the current nested likelihood level).

    Attributes:
        iterations (int): The number of component-wise trial move cycles. The
            total number of trial moves will be iterations*ndim. Default: 100
        burn_in (int): The number of additional burn in iterations to
            to run in order to "equilibrate" the Markov chain. Default: 0
        tuning_cycles (int): The number of tuning cycles to run before the
            Markov chain in order to tune the step sizes of the trial moves.
            Each tuning cycle is 20 iterations, after which the trial move step
            sizes are updated to try and make the acceptance ratio between
            0.2 and 0.6. Default: 0
        proposal (str): The shape of the symmetric proposal distrbution to
            use during the trial moves: either "uniform"
            or "normal." Default: "uniform"
    References:
        None
    """

    def __init__(self, iterations=100, burn_in=0, tuning_cycles=0,
                 proposal='uniform'):
        """Initialize the sampler."""
        # Set the public attributes.
        self.iterations = iterations
        self.burn_in = burn_in
        self.tuning_cycles = tuning_cycles
        self.proposal = proposal
        # Private attributes.
        self._steps_per_tuning_cycle = 20
        # _first is used as switch for whether or not the sampler has been
        # called yet.
        self._first = True
        # Container for the trial move sizes (width of the proposal distributions).
        self._widths = list()
        # The number of dimensions being sampled (i.e., the number of
        # being sampled parameters)
        self._ndim = None
        return

    def __call__(self, nested_sampler, starting_point, starting_loglikelihood):
        """Run the sampler.

        Args:
            nested_sampler (:obj:`gleipnir.nestedsampling.NestedSampling`):
                Reference to the NestedSampling instance which is using
                this sampler.
                parameters that are being sampled.
            starting_point (obj:`numpy.ndarray`): The starting position of
                parameter vector for the parameters being sampled.
        """
        if self._first:
            self._ndim = len(nested_sampler.sampled_parameters)
            # Make a rough estimate of the widths of each parameters' prior.
            for sampled_parameter in nested_sampler.sampled_parameters:
                random_sample = sampled_parameter.rvs(100)
                minimum_sample_value = np.min(random_sample)
                maximum_sample_value = np.max(random_sample)
                width = maximum_sample_value - minimum_sample_value
                self._widths.append(0.5*width)
            self._widths = np.array(self._widths)
            self._first = False

        #starting_loglikelihood = nested_sampler.loglikelihood(starting_point)

        # Tuning cycles
        steps = self._widths.copy()
        acceptance = np.zeros(self._ndim)
        current_point = starting_point.copy()
        current_loglikelihood = starting_loglikelihood
        for i in range(self.tuning_cycles):
            for k in range(self._steps_per_tuning_cycle):
                rsteps = np.random.random(self._ndim)
                u = np.random.random(self._ndim)
                for j in range(self._ndim):
                    new_point = current_point.copy()
                    cur_pointj = current_point[j]
                    widthj = self._widths[j]
                    # Generate the appropriate proposal distribution
                    if self.proposal == 'normal':
                        new_pointj = norm.ppf(rsteps[j],loc=cur_pointj, scale=widthj)
                    else:
                        new_pointj = uniform.ppf(rsteps[j],loc=cur_pointj-(widthj/2.0), scale=widthj)

                    new_point[j] = new_pointj
                    cur_priorj = nested_sampler.sampled_parameters[j].prior(cur_pointj)
                    new_priorj = nested_sampler.sampled_parameters[j].prior(new_point[j])
                    ratio = new_priorj/cur_priorj
                    new_loglikelihood = nested_sampler.loglikelihood(new_point)
                    # Metropolis criterion with NS boundary
                    if (u[j] < ratio) and (new_loglikelihood > nested_sampler._current_loglikelihood_level):
                        # accept the new point and update
                        current_point[j] = new_pointj
                        current_loglikelihood = new_loglikelihood
                        acceptance[j] += 1.0
                # Adjust the step sizes
                acceptance_ratio = acceptance/self._steps_per_tuning_cycle
                less_than_mask = acceptance_ratio < 0.2
                gt_mask = acceptance_ratio > 0.6
                steps[less_than_mask] *= 0.66
                steps[gt_mask] *= 1.33
                acceptance[:] = 0.0

        # Start the sampling chain
        self._widths = steps.copy()
        current_point = starting_point.copy()

        for i in range(self.iterations+self.burn_in):
                rsteps = np.random.random(self._ndim)
                u = np.random.random(self._ndim)
                for j in range(self._ndim):
                    new_point = current_point.copy()
                    cur_pointj = current_point[j]
                    widthj = self._widths[j]
                    # Generate the appropriate proposal distribution
                    if self.proposal == 'normal':
                        new_pointj = norm.ppf(rsteps[j],loc=cur_pointj, scale=widthj)
                    else:
                        new_pointj = uniform.ppf(rsteps[j],loc=cur_pointj-(widthj/2.0), scale=widthj)

                    cur_priorj = nested_sampler.sampled_parameters[j].prior(cur_pointj)
                    new_priorj = nested_sampler.sampled_parameters[j].prior(new_point[j])
                    ratio = new_priorj/cur_priorj
                    #print("ratio",ratio, "cur_priorj", cur_priorj, "new_priorj", new_priorj, "cur_pointj", cur_pointj, "new_pointj", new_pointj, "rstepj", rsteps[j])
                    new_loglikelihood = nested_sampler.loglikelihood(new_point)
                    # Metropolis criterion with NS boundary
                    if (u[j] < ratio) and (new_loglikelihood > nested_sampler._current_loglikelihood_level):
                        # accept the new point and update
                        current_point[j] = new_pointj
                        current_loglikelihood = new_loglikelihood

        return current_point, current_loglikelihood

class UniformSampler(object):
    """Uniform sampling from the prior.
    This sampler uses uniform sampling from the prior distribution with
    rejection based on the current likelihood level; i.e., a new random sample
    is drawn from the prior and always accepted if it's likelihood is higher
    than the current nested likelihood-level. Note that as the Nested Sampling
    iterations progress the rejection rate will increase commensurately with
    the decrease in remaining prior-mass.

    Attributes:
        max_tries (int): The maximum number sample rejections allowed. Once
            reached, the sampler will just return the random selected survivor
            point that was passed in to the call function - prevents
            stalling out of the sampling when rejection rate is high.
            Default: 1000

    References:
        None
    """

    def __init__(self, max_tries=1000):
        """Initialize the sampler."""
        # Set the public attributes.
        self.max_tries = max_tries
        return

    def __call__(self, nested_sampler, starting_point, starting_loglikelihood):
        """Run the sampler.

        Args:
            nested_sampler (:obj:`gleipnir.nestedsampling.NestedSampling`):
                Reference to the NestedSampling instance which is using
                this sampler.
                parameters that are being sampled.
            starting_point (obj:`numpy.ndarray`): The starting position of
                parameter vector for the parameters being sampled.
        """
        for i in range(self.max_tries):
            new_point = np.array([sampled_parameter.rvs(1) for sampled_parameter in nested_sampler.sampled_parameters])
            new_loglikelihood = nested_sampler.loglikelihood(new_point)
            if new_loglikelihood < nested_sampler._current_loglikelihood_level:
                return new_point, new_loglikelihood
        return starting_point, starting_loglikelihood
