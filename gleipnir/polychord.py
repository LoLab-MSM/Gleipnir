import numpy as np
import pypolychord
from pypolychord.settings import PolyChordSettings


class PolyChordNestedSampling(object):

    def __init__(self, sampled_parameters, loglikelihood, population_size=None):

        self._sampled_parameters = sampled_parameters
        self._loglikelihood = loglikelihood
        self.nDims = len(sampled_parameters)
        self.nDerived = 0
        self.population_size = population_size
        if self.population_size is None:
            self.population_size = 25*self.nDims
        # make the likelihood function for polychord
        def likelihood(theta):
            r2 = 0
            return loglikelihood(theta), [r2]
        self.likelihood = likelihood
        # make the prior for polychord
        def prior(hypercube):
            return np.array([self._sampled_parameters[i].invcdf(value) for i,value in enumerate(hypercube)])

        self.prior = prior
        # polychord settings
        self.settings = PolyChordSettings(self.nDims, self.nDerived, nlive=self.population_size) #settings is an object
        self.settings.file_root = 'polychord_run' #string
        self.settings.do_clustering = True
        self.settings.read_resume = False
        # make the polychord dumper function
        # param : array, array, array, float, float
        def dumper(live, dead, logweights, logZ, logZerr):
            print("Last dead point:", dead[-1]) # prints last element of dead (wich is an array)

        self.dumper = dumper
        return


    def run(self):
        output = pypolychord.run_polychord(self.likelihood, self.nDims, self.nDerived, self.settings, self.prior, self.dumper)
        self.output = output
