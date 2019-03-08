import numpy as np
import pandas as pd
import warnings

class NestedSampling(object):

    def __init__(self, sampled_parameters, loglikelihood, sampler, population_size, stopping_criterion):
        # stor inputs
        self.sampled_parameters = sampled_parameters
        self.sampled_parameters_dict = {sp.name:sp for sp in sampled_parameters}
        self.loglikelihood = loglikelihood
        self.sampler = sampler
        self.population_size = population_size
        self._stopping_criterion = stopping_criterion

        # estimate of NS constriction factor
        self._alpha = population_size/(population_size+1)

        # NS accumulators

        self._evidence = 0.0
        self._evidence_error = 0.0
        self._logZ_err = 0.0
        self._log_evidence = 0.0
        self._information = 0.0
        self._H = 0.0
        self._previous_evidence = 0.0
        self.current_weight = 1.0
        self._previous_weight = 1.0

        self._n_iterations = 0
        self._dead_points = list()
        self._live_points = None
        self._post_eval = False
        self._posteriors = None

        # public vars
        # self.evidence = None
        # self.evidence_error = None
        # self.log_evidence = None
        # self.log_evidence_error = None
        # self.information = None
        # print(self._alpha)
        #quit()
        return

    def run(self, verbose=False):

        # zeroth iteration -- generate all the random samples
        if verbose:
            print("Generating the initial set of live points with population size {}...".format(self.population_size))
        live_points = dict()
        for i in range(self.population_size):
            for sampled_parameter_name in self.sampled_parameters_dict:
                name = sampled_parameter_name
                rs = self.sampled_parameters_dict[sampled_parameter_name].rvs(1)[0]
                if name not in live_points.keys():
                    live_points[name] = list([rs])
                else:
                    live_points[name].append(rs)

        self.live_points = pd.DataFrame(live_points)
        # print(self.live_points)
        # evaulate the log likelihood function for each live point
        if verbose:
            print("Evaluating the loglikelihood function for each live point...")
        log_likelihoods = np.array([self.loglikelihood(sampled_parameter_vector) for sampled_parameter_vector in self.live_points.values])

        # first iteration
        self._n_iterations += 1
        self.current_weight = 1.0 - self._alpha**self._n_iterations
        #print(self.current_weight)
        #quit()
        # get the lowest likelihood live point
        ndx = np.argmin(log_likelihoods)
        log_l = log_likelihoods[ndx]
        param_vec = self.live_points.values[ndx]
        # evaluate the priors
        #priors = [self.sampled_parameters_dict[name].prior(param) for name, param in zip(self.live_points.columns, param_vec)]
        #prior = np.array(priors)
        #joint_prior = prior.prod()
        # accumulate the evidence
        #dZ = self.current_weight*joint_prior*np.exp(log_l)
        joint_prior = 1.0
        dZ = self.current_weight*np.exp(log_l)
        # print(dZ, log_l)
        #quit()
        self._evidence += dZ
        # accumulate the information
        dH = dZ*log_l
        if np.isnan(dH): dH = 0.0
        self._H += dH
        if self._evidence > 0.0:
            self._information = -np.log(self._evidence)+self._H/self._evidence

        self._previous_weight = self.current_weight
        # add the lowest likelihood live point to dead points
        dpd = dict({'log_l': log_l, 'weight':self.current_weight})
        for k,val in enumerate(param_vec):
            dpd[self.sampled_parameters[k].name] = val
        #self._dead_points.append(dict({'log_l': log_l,
        #                               'weight': self.current_weight,
        #                               'param_vec': param_vec}))
        self._dead_points.append(dpd)

        if verbose:
            print("Iteration: {} Evidence estimate: {} Remaining prior mass: {}".format(self._n_iterations, self._evidence, self._alpha**self._n_iterations))
            print("Dead Point:")
            print(self._dead_points[-1])
        # subseqent iterations
        while not self.stopping_criterion():
            self._n_iterations += 1
            self.current_weight = self._alpha**(self._n_iterations-1.0) - self._alpha**self._n_iterations

            # replace the dead point with a modified survivor
            # choose at random
            r_p_ndx = int(np.random.random(1)*self.population_size)
            while r_p_ndx == ndx:
                r_p_ndx = int(np.random.random(1)*self.population_size)
            #print(r_p_ndx)
            # now make a new point from the survivor via the sampler
            r_p_param_vec = self.live_points.values[r_p_ndx]
            #if verbose and (self._n_iterations%10==0):
            #    print("Replacing the new dead point with a survivor modifed via MCMC...")
            updated_point_param_vec, u_log_l = self.sampler(self.sampled_parameters, self.loglikelihood, r_p_param_vec, log_l, self._alpha**self._n_iterations)
            log_likelihoods[ndx] = u_log_l
            self.live_points.values[ndx] = updated_point_param_vec
            # get the lowest likelihood live point
            ndx = np.argmin(log_likelihoods)
            log_l = log_likelihoods[ndx]
            param_vec = self.live_points.values[ndx]
            # evaluate the priors
            #priors = [self.sampled_parameters_dict[name].prior(param) for name, param in zip(self.live_points.columns, param_vec)]
            #prior = np.array(priors)
            #joint_prior = prior.prod()
            # accumulate the evidence
            #dZ = self.current_weight*joint_prior*np.exp(log_l)
            dZ = self.current_weight*np.exp(log_l)
            #print(dZ, log_l)
            self._evidence += dZ
            # accumulate the information
            # print(dZ, log_l)
            dH = dZ*log_l
            if np.isnan(dH): dH = 0.0
            self._H += dH

            if self._evidence > 0.0:
                self._information = -np.log(self._evidence)+self._H/self._evidence
            # print(self._n_iterations,self._alpha**self._n_iterations, self.current_weight, self._previous_weight, self._evidence)
            #if self.current_weight < 0.0:
            #    quit()

            # add the lowest likelihood live point to dead points
            dpd = dict({'log_l': log_l, 'weight':self.current_weight})
            for k,val in enumerate(param_vec):
                dpd[self.sampled_parameters[k].name] = val
            #self._dead_points.append(dict({'log_l': log_l,
            #                               'weight': self.current_weight,
            #                               'param_vec': param_vec}))
            self._dead_points.append(dpd)
            self._previous_weight = self.current_weight
            if verbose and (self._n_iterations%10==0):
                logZ_err = np.sqrt(self._information/self.population_size)
                ev_err = np.exp(logZ_err)
                print("Iteration: {} Evidence estimate: {} +- {} Remaining prior mass: {}".format(self._n_iterations, self._evidence, ev_err, self._alpha**self._n_iterations))
                print("Dead Point:")
                print(self._dead_points[-1])
        # accumulate the final bit for remaining surviving points
        weight = self._alpha**(self._n_iterations)
        likelihoods = np.exp(log_likelihoods)
        likelihoods_surv = np.array([likelihood for i,likelihood in enumerate(likelihoods) if i != ndx])
        l_m = likelihoods_surv.mean()
        self._evidence += weight*l_m
        # accumulate the information
        dH = weight*l_m*np.log(l_m)
        if np.isnan(dH): dH = 0.0
        self._H += dH
        if self._evidence > 0.0:
            self._information = -np.log(self._evidence)+self._H/self._evidence
        n_left = len(likelihoods_surv)
        a_weight = weight/n_left
        for i,l_likelihood in enumerate(log_likelihoods):
            if i != ndx:
                dpd = dict({'log_l': l_likelihood, 'weight':a_weight})
                for k,val in enumerate(self.live_points.values[i]):
                    dpd[self.sampled_parameters[k].name] = val
                #self._dead_points.append(dict({'log_l': log_l,
                #                               'weight': self.current_weight,
                #                               'param_vec': param_vec}))
                self._dead_points.append(dpd)
                #self._dead_points.append(dict({'log_l':l_likelihood,
                #                               'weight':a_weight,
                #                               'param_vec':self.live_points.values[i]}))
        logZ_err = np.sqrt(self._information/self.population_size)
        self._logZ_err = logZ_err
        ev_err = np.exp(logZ_err)
        self._evidence_error = ev_err
        self._log_evidence = np.log(self._evidence)
        self._dead_points = pd.DataFrame(self._dead_points)
        #print(pd.DataFrame(self._dead_points))
        #self.posteriors()
        return self._log_evidence, logZ_err

    def stopping_criterion(self):
        return self._stopping_criterion(self)

    @property
    def evidence(self):
        """Returns the estimate of the Bayesian evidence, or Z.
        """
        return self._evidence
    @evidence.setter
    def evidence(self, value):
        warnings.warn("evidence is not settable")

    @property
    def evidence_error(self):
        """Returns the (rough) estimate of the error in the evidence, or Z.

        The error in the evidence is computed as the approximation:
            exp(sqrt(information/population_size))
        """
        return self._evidence_error
    @evidence_error.setter
    def evidence_error(self, value):
        warnings.warn("evidence_error is not settable")

    @property
    def log_evidence(self):
        """Returns the estimate of the natural logarithm of the Bayesian evidence, or ln(Z).
        """
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
    #def information(self):

    def posteriors(self):
        if not self._post_eval:
            log_likelihoods = self._dead_points['log_l'].to_numpy()
            weights = self._dead_points['weight'].to_numpy()
            likelihoods = np.exp(log_likelihoods)
            norm_weights = (weights*likelihoods)/self.evidence
            gt_mask = norm_weights > 0.0
            parms = self._dead_points.columns[2:]
            # print(len(self._dead_points[0]))
            # print(norm_weights)
            # import matplotlib.pyplot as plt
            # plt.hist(self._dead_points[0], weights=norm_weights)
            # plt.show()
            # print(parms)
            #nbins = int(np.sqrt(len(norm_weights)))
            # Rice bin count selection
            nbins = 2 * int(np.cbrt(len(norm_weights[gt_mask])))
            # print(nbins)
            self._posteriors = dict()
            for parm in parms:
                marginal, edge = np.histogram(self._dead_points[parm][gt_mask], weights=norm_weights[gt_mask], density=True, bins=nbins)
                center = (edge[:-1] + edge[1:])/2.
                self._posteriors[parm] = (marginal, center)
            #self._posteriors = {parm:(self._dead_points[parm], norm_weights) for parm in parms}
            self._post_eval = True
        #print(post[0])
        return self._posteriors

    @property
    def dead_points(self):
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
