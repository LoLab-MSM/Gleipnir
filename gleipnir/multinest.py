import numpy as np
try:
    import pymultinest
    from pymultinest.solve import solve
except ImportError as err:
    #print(err)
    raise err

class MultiNestNestedSampling(object):

    def __init__(self, sampled_parameters, loglikelihood, population_size=None):

        self._sampled_parameters = sampled_parameters
        self._loglikelihood = loglikelihood
        self.nDims = len(sampled_parameters)
        self.nDerived = 0
        self.population_size = population_size
        if self.population_size is None:
            self.population_size = 25*self.nDims
        # make the likelihood function for polychord
        self.likelihood = loglikelihood
        # make the prior for polychord
        def prior(hypercube):
            return np.array([self._sampled_parameters[i].invcdf(value) for i,value in enumerate(hypercube)])

        self.prior = prior
        # multinest settings

        self.file_root = 'multinest_run' #string

        return


    def run(self, verbose=False):
        output = solve(LogLikelihood = self.likelihood, Prior=self.prior,
                       n_dims = self.nDims,
                       outputfiles_basename=self.file_root,
                       verbose=verbose)
        self.output = output
