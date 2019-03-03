import numpy as np
import scipy

class SampledParameter(object):

    def __init__(self, name, prior):
        self.name = name
        self._prior = prior
        # map the pdf/pmf functions to a common label
        try:
            self.pf = prior.pdf
            self.logpf = prior.logpdf
        except:
            self.pf = prior.pmf
            self.logpf = prior.logpmf
        # compute the normalization factor for this prior
        self.norm = prior.cdf(np.inf)
        return

    def rvs(self, sample_shape):
        return self._prior.rvs(sample_shape)

    def logprior(self, value):
        return np.log(self.pf(value)/self.norm)

    def prior(self, value):
        return self.pf(value)/self.norm

    def invcdf(self, value):
        return self._prior.ppf(value)    
