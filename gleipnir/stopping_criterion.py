import numpy as np

class NumberOfIterations(object):

    def __init__(self, iterations):
        self.n_iterations = iterations
        return

    def __call__(self, nested_sampler):
        return nested_sampler._n_iterations >= self.n_iterations

class RemainingPriorMass(object):

    def __init__(self, cutoff):
        self.cutoff = cutoff
        return

    def __call__(self, nested_sampler):
        return nested_sampler._alpha**nested_sampler._n_iterations <= self.cutoff
