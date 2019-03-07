import numpy as np
import warnings
import pypolychord
from pypolychord.settings import PolyChordSettings


class PolyChordNestedSampling(object):

    def __init__(self, sampled_parameters, loglikelihood, population_size=None):

        self._sampled_parameters = sampled_parameters
        self._loglikelihood = loglikelihood
        self.nDims = len(sampled_parameters)
        self.nDerived = 0
        self.population_size = population_size
        self._post_eval = False
        self._posteriors = None
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
        self._output = output
        return output.logZ, output.logZerr

    @property
    def evidence(self):
        return np.exp(self._output.logZ)
    @evidence.setter
    def evidence(self, value):
        warnings.warn("evidence is not settable")

    @property
    def evidence_error(self):
        return np.exp(self._output.logZerr)
    @evidence_error.setter
    def evidence_error(self, value):
        warnings.warn("evidence_error is not settable")

    @property
    def log_evidence(self):
        return self._output.logZ
    @log_evidence.setter
    def log_evidence(self, value):
        warnings.warn("log_evidence is not settable")

    @property
    def log_evidence_error(self):
        return self._output.logZerr
    @log_evidence_error.setter
    def log_evidence_error(self, value):
        warnings.warn("log_evidence_error is not settable")

    @property
    def information(self):
        return None
    @information.setter
    def information(self, value):
        warnings.warn("information is not settable")


    def posteriors(self):
        # lazy evaluation of posteriors on first call to function.
        if not self._post_eval:
            samples = self._output.samples
            log_likelihoods = samples['loglike'].to_numpy()
            #weights = samples['weight'].to_numpy()
            #likelihoods = np.exp(log_likelihoods)
            #weight_times_likelihood = weights*likelihoods
            #norm_weights = weight_times_likelihood/weight_times_likelihood.sum()

            parms = samples.columns[2:]
            # print(len(self._dead_points[0]))
            # print(norm_weights)
            # import matplotlib.pyplot as plt
            # plt.hist(self._dead_points[0], weights=norm_weights)
            # plt.show()
            # print(parms)
            # nbins = int(np.sqrt(len(log_likelihoods)))
            # Rice bin count selection
            nbins = 2 * int(np.cbrt(len(log_likelihoods)))
            JP, edges = np.histogramdd(samples.values[:,2:], density=True, bins=nbins)
            nd = len(JP.shape)
            self._posteriors = dict()
            for ii in range(nd):
                others = tuple(jj for jj in range(nd) if jj!=ii)
                marginal = np.sum(JP, axis=others)
                edge = edges[ii]
                center = (edge[:-1] + edge[1:])/2.
                self._posteriors[parms[ii]] = (marginal, center)
            #self._posteriors = {parm:(self._dead_points[parm], norm_weights) for parm in parms}
            self._post_eval = True
        #print(post[0])
        return self._posteriors
