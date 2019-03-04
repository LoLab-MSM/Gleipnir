import numpy as np
import pandas as pd

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
        self._previous_evidence = 0.0
        self.current_weight = 1.0
        self._previous_weight = 1.0

        self._n_iterations = 0
        self._dead_points = list()
        self._live_points = None
        # print(self._alpha)
        #quit()
        return

    def run(self, verbose=False):

        # zeroth iteration -- generate all the random samples
        live_points = dict()
        for i in range(self.population_size):
            for sampled_parameter_name in self.sampled_parameters_dict:
                name = sampled_parameter_name
                rs = self.sampled_parameters_dict[sampled_parameter_name].rvs(1)
                if name not in live_points.keys():
                    live_points[name] = list([rs])
                else:
                    live_points[name].append(rs)

        self.live_points = pd.DataFrame(live_points)
        # evaulate the log likelihood function for each live point
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
        #print(dZ, log_l)
        #quit()
        self._evidence += dZ
        self._previous_weight = self.current_weight
        # add the lowest likelihood live point to dead points
        self._dead_points.append(dict({'log_l': log_l,
                                       'weight': self.current_weight,
                                       'param_vec': param_vec}))


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

            # print(self._n_iterations,self._alpha**self._n_iterations, self.current_weight, self._previous_weight, self._evidence)
            #if self.current_weight < 0.0:
            #    quit()

            # add the lowest likelihood live point to dead points
            self._dead_points.append(dict({'log_l': log_l,
                                           'weight': self.current_weight,
                                           'param_vec': param_vec}))
            self._previous_weight = self.current_weight
            if verbose:
                print("Iteration: {} Evidence estimate: {} Remaining prior mass: {}".format(self._n_iterations, self._evidence, self._alpha**self._n_iterations))
                print("Dead Point:")
                print(self._dead_points[-1])
        # accumulate the final bit for remaining surviving points
        weight = self._alpha**(self._n_iterations)
        likelihoods = np.exp(log_likelihoods)
        likelihoods_surv = np.array([likelihood for i,likelihood in enumerate(likelihoods) if i != ndx])
        l_m = likelihoods_surv.mean()
        self._evidence += weight*l_m
        n_left = len(likelihoods_surv)
        a_weight = weight/n_left
        for i,l_likelihood in enumerate(log_likelihoods):
            if i != ndx:
                self._dead_points.append(dict({'log_l':l_likelihood,
                                               'weight':a_weight,
                                               'param_vec':self.live_points.value[i]}))
        return

    def stopping_criterion(self):
        return self._stopping_criterion(self)

    def evidence(self):
        return self._evidence

    #def information(self):
