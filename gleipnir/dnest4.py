import numpy as np
import warnings
try:
    import dnest4
except ImportError as err:
    #print(err)
    raise err


class DNest4Model(object):

    def __init__(self, log_likelihood_func, from_prior_func, widths):
        self._log_likelihood = log_likelihood_func
        self._from_prior = from_prior_func
        self._widths = widths
        self._n_dim = len(widths)
        return

    def log_likelihood(self, coords):
        return self._log_likelihood(coords)

    def from_prior(self):
        return self._from_prior()

    def perturb(self, coords):
        idx = np.random.randint(self._n_dim)
        # print(idx)
        coords[idx] += (self._widths[idx]*(np.random.uniform(size=1)-0.5))*0.5
        # Note: use the return value of wrap, unlike in C++
        #coords[i] = dnest4.wrap(coords[i], -0.5*self.width, 0.5*self.width)
        return 0.0


class DNest4NestedSampling(object):

    def __init__(self, sampled_parameters, loglikelihood, population_size=None,
                 n_diffusive_levels=20, dnest4_backend="memory",
                 **dnest4_kwargs):

        self._log_evidence = None
        self._information = None
        self._sampled_parameters = sampled_parameters
        self._loglikelihood = loglikelihood
        self._n_dims = len(sampled_parameters)
        self.population_size = population_size
        self.dnest4_backend = dnest4_backend
        self._n_levels = n_diffusive_levels
        self._dnest4_kwargs = dnest4_kwargs
        self._output = None
        self._post_eval = False
        if self.population_size is None:
            self.population_size = 25*self._n_dims
        # make the likelihood function for polychord
        self.likelihood = loglikelihood
        # make the prior for polychord
        def from_prior():
            return np.array([sampled_parameter.rvs(1)[0] for sampled_parameter in self._sampled_parameters])

        self._from_prior = from_prior
        widths = []
        for sampled_parameter in sampled_parameters:
            rv = sampled_parameter.rvs(100)
            width = rv.max() - rv.min()
            widths.append(width)
        widths = np.array(widths)
        self._widths = widths
        self._dnest4_model = DNest4Model(loglikelihood, self._from_prior, widths)

        return


    def run(self, verbose=False):

        if self.dnest4_backend == 'csv':
            # for CSVBackend, which is output data to disk
            backend = dnest4.backends.CSVBackend(".", sep=" ")
        else:
            # for the MemoryBackend, which is output data to memory
            backend = dnest4.backends.MemoryBackend()
        sampler = dnest4.DNest4Sampler(self._dnest4_model,
                                       backend=backend)
        output = sampler.sample(self._n_levels,
                                num_particles=self.population_size,
                                **self._dnest4_kwargs)
        self._output = output
        for i, sample in enumerate(output):
            if verbose and ((i + 1) % 100 == 0):
                stats = sampler.postprocess()
                print("Iteration: {0} log(Z): {1}".format(i,stats['log_Z']))
        stats = sampler.postprocess(resample=1)
        self._log_evidence = stats['log_Z']
        self._information = stats['H']
        logZ_err = np.sqrt(self._information/self.population_size)
        self._logZ_err = logZ_err
        ev_err = np.exp(logZ_err)
        self._evidence_error = ev_err
        self._evidence = np.exp(self._log_evidence)
        self._samples = np.array(sampler.backend.posterior_samples)

        return self.log_evidence, self.log_evidence_error

    @property
    def evidence(self):
        return self._evidence
    @evidence.setter
    def evidence(self, value):
        warnings.warn("evidence is not settable")

    @property
    def evidence_error(self):
        return self._evidence_error
    @evidence_error.setter
    def evidence_error(self, value):
        warnings.warn("evidence_error is not settable")

    @property
    def log_evidence(self):
        return self._log_evidence
    @log_evidence.setter
    def log_evidence(self, value):
        warnings.warn("log_evidence is not settable")

    @property
    def log_evidence_error(self):
        return self._logZ_err
    @log_evidence_error.setter
    def log_evidence_error(self, value):
        warnings.warn("log_evidence_error is not settable")

    @property
    def information(self):
        return self._information
    @information.setter
    def information(self, value):
        warnings.warn("information is not settable")

    def posteriors(self):
        #warnings.warn("Posteriors not yet implemented for DNest4 NS.")
        #pass
        # lazy evaluation of posteriors on first call to function.
        if not self._post_eval:
            # Here the samples are samples directly from the posterior
            # (i.e. equal weights)
            samples = self._samples
            #log_likelihoods = samples['loglike'].to_numpy()
            #weights = samples['weight'].to_numpy()
            #likelihoods = np.exp(log_likelihoods)
            #weight_times_likelihood = weights*likelihoods
            #norm_weights = weight_times_likelihood/weight_times_likelihood.sum()

            #parms = samples.columns[2:]
            # print(len(self._dead_points[0]))
            # print(norm_weights)
            # import matplotlib.pyplot as plt
            # plt.hist(self._dead_points[0], weights=norm_weights)
            # plt.show()
            # print(parms)
            # nbins = int(np.sqrt(len(samples)))
            # Rice bin count selection
            nbins = 2 * int(np.cbrt(len(samples)))
            nd = samples.shape[1]
            self._posteriors = dict()
            for ii in range(nd):
                marginal, edge = np.histogram(samples[:,ii], density=True, bins=nbins)
                center = (edge[:-1] + edge[1:])/2.
                self._posteriors[self._sampled_parameters[ii].name] = (marginal, center)
            #self._posteriors = {parm:(self._dead_points[parm], norm_weights) for parm in parms}
            self._post_eval = True
        #print(post[0])
        return self._posteriors
