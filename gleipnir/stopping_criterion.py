"""Classes for defining the stopping criterion for a Nested Sampling run.

This module defines the classes used by gleinir.nested_sampling.NestedSampling
instances to define the stopping criterion for the Nested Sampling run.

"""

import numpy as np

class NumberOfIterations(object):
    """Stop after a fixed number of iteration.
    Attributes:
        n_iterations (int): The number of Nested Sampling iterations after
            which to terminate the run.
    """
    def __init__(self, iterations):
        """Initialize the NumberOfIterations stopping criterion.
        Args:
            iterations (int): Sets the n_iterations Attribute.
        """
        self.n_iterations = iterations
        return

    def __call__(self, nested_sampler):
        """Evaluate the criterion.
        Args:
            nested_sampler (:obj:gleipnir.nested_sampler.NestedSampling): The
                instance of the NestedSampling object to be tested.
        Return:
            bool : Should stop the Nested Sampling run if True, should
                continue running if False.
        """
        return nested_sampler._n_iterations >= self.n_iterations

class RemainingPriorMass(object):
    """Stop after the remaining amount of prior mass reaches a preset threshold.
    Attributes:
        cutoff (float): The remaining prior mass threshold that Nested Sampling
            iterations should reach after which to terminate the run.
    """
    def __init__(self, cutoff):
        """Initialize the RemainingPriorMass stopping criterion.
        Args:
            cutoff (float): Sets the cutoff Attribute.
        """
        self.cutoff = cutoff
        return

    def __call__(self, nested_sampler):
        """Evaluate the criterion.
        Args:
            nested_sampler (:obj:gleipnir.nested_sampler.NestedSampling): The
                instance of the NestedSampling object to be tested.
        Return:
            bool : Should stop the Nested Sampling run if True, should
                continue running if False.
        """
        return nested_sampler._alpha**nested_sampler._n_iterations <= self.cutoff
