"""Samplers used to update points from Nested Sampling.

This module defines the classes for the samplers used to update a chosen
survivor point in order to replace the dead point during the
Nested Sampling runs via the gleipnir.nested_sampling.NestedSampling class.


"""

import numpy as np
from scipy.stats import uniform, norm


class MetropolisComponentWiseHardNSRejection(object):
    """Markov Chain Monte Carlo sampler using augmented Metropolis criterion and component-wise trial moves.
    This sampler uses a Markov Chain Monte Carlo method to augment a position
    as sampled from the prior density using the Metropolis criterion and an extra
    hard rejection for the Nested Sampling likelihood level. Trial moves
    are carried out in a component-wise fashion (i.e., parameters are looped over
    in order).

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
            use during the trial moves. The proposal can either be "uniform"
            or "normal." Default: "uniform"
    References:
        None
    """

    def __init__(self, iterations=100, burn_in=0, tuning_cycles=0, proposal='uniform'):
        """Initialize the sampler."""
        # Set the public attributes.
        self.iterations = iterations
        self.burn_in = burn_in
        self.tuning_cycles = tuning_cycles
        self.proposal = proposal
        # Private attributes.
        # _first is used as switch for whether or not the sampler has been
        # called yet.
        self._first = True
        # Container for the trial move sizes (width of the proposal distributions).
        self._widths = list()
        # The number of dimensions being sampled (i.e., the number of
        # being sampled parameters)
        self._ndim = None
        return

    def __call__(self, sampled_parameters, loglikelihood, start_param_vec, ns_boundary, **kwargs):
        """Run the sampler.

        Args:
            sampled_parameters (:obj:`list` of
                :obj:`gleipnir.sampled_parameter.SampledParameter`): The
                parameters that are being sampled.
            loglikelihood (function): The log likelihood function.
            start_param_vec (obj:`numpy.ndarray`): The starting position of
                parameter vector for the parameters being sampled.
            ns_boundary (float): The current lower likelihood bound from the
            Nested Sampling routine.
            kwargs (dict): Pass in any other method specific keyword arguments.
        """
        if self._first:
            self._ndim = len(sampled_parameters)
            for sampled_parameter in sampled_parameters:
                rs = sampled_parameter.rvs(100)
                mirs = np.min(rs)
                mars = np.max(rs)
                width = mars - mirs
                #print(width)
                self._widths.append(0.5*width)
            #steps.append(0.5*width)
            self._widths = np.array(self._widths)
            self._first = False

        start_likelihood = loglikelihood(start_param_vec)

        # Tuning cycles
        steps = self._widths.copy()
        acceptance = np.zeros(self._ndim)
        cur_point = start_param_vec.copy()
        cur_likelihood = start_likelihood
        for i in range(self.tuning_cycles):
            for k in range(20):
                rsteps = np.random.random(self._ndim)
                u = np.random.random(self._ndim)
                for j in range(self._ndim):
                    new_point = cur_point.copy()
                    cur_pointj = cur_point[j]
                    widthj = self._widths[j]
                    # Generate the appropriate proposal distribution
                    if self.proposal == 'normal':
                        new_pointj = norm.ppf(rsteps[j],loc=cur_pointj, scale=widthj)
                    else:
                        new_pointj = uniform.ppf(rsteps[j],loc=cur_pointj-(widthj/2.0), scale=widthj)
                    
                    new_point[j] = new_pointj
                    cur_priorj = sampled_parameters[j].prior(cur_pointj)
                    new_priorj = sampled_parameters[j].prior(new_point[j])
                    ratio = new_priorj/cur_priorj
                    new_likelihood = loglikelihood(new_point)
                    # Metropolis criterion with NS boundary
                    if (u[j] < ratio) and (new_likelihood > ns_boundary):
                        # accept the new point and update
                        cur_point[j] = new_pointj
                        cur_likelihood = new_likelihood
                        acceptance[j] += 1.0
                # Adjust the step sizes
                acceptance_ratio = acceptance/20.0
                less_than_mask = acceptance_ratio < 0.2
                gt_mask = acceptance_ratio > 0.6
                steps[less_than_mask] *= 0.66
                steps[gt_mask] *= 1.33
                acceptance[:] = 0.0

        # Start the sampling chain
        self._widths = steps.copy()
        cur_point = start_param_vec.copy()
        curr_likelihood = start_likelihood
        for i in range(self.iterations+self.burn_in):
                rsteps = np.random.random(self._ndim)
                u = np.random.random(self._ndim)
                for j in range(self._ndim):
                    new_point = cur_point.copy()
                    cur_pointj = cur_point[j]
                    widthj = self._widths[j]
                    # Generate the appropriate proposal distribution
                    if self.proposal == 'normal':
                        new_pointj = norm.ppf(rsteps[j],loc=cur_pointj, scale=widthj)
                    else:
                        new_pointj = uniform.ppf(rsteps[j],loc=cur_pointj-(widthj/2.0), scale=widthj)

                    cur_priorj = sampled_parameters[j].prior(cur_pointj)
                    new_priorj = sampled_parameters[j].prior(new_point[j])
                    ratio = new_priorj/cur_priorj
                    #print("ratio",ratio, "cur_priorj", cur_priorj, "new_priorj", new_priorj, "cur_pointj", cur_pointj, "new_pointj", new_pointj, "rstepj", rsteps[j])
                    new_likelihood = loglikelihood(new_point)
                    # Metropolis criterion with NS boundary
                    if (u[j] < ratio) and (new_likelihood > ns_boundary):
                        # accept the new point and update
                        cur_point[j] = new_pointj
                        cur_likelihood = new_likelihood


        return cur_point, cur_likelihood
