import numpy as np

class MetropolisComponentWiseHardNSRejection(object):

    def __init__(self, iterations=100, burn_in=0):
        self.iterations = iterations
        self.burn_in = burn_in
        self._first = True
        self.widths = []
        return

    def __call__(self, sampled_parameters, loglikelihood, start_param_vec, ns_boundary, compression_factor):

        # priors = [sampled_parameters[i].logprior(param) for i,param in enumerate(start_param_vec)]
        # priors = np.array(priors)
        # joint_prior = priors.prod()
        # thinning = 50
        ndim = len(sampled_parameters)
        #steps = list([])
        if self._first:
            for sampled_parameter in sampled_parameters:
                rs = sampled_parameter.rvs(100)
                mirs = np.min(rs)
                mars = np.max(rs)
                width = mars - mirs
                #print(width)
                self.widths.append(0.5*width)
            #steps.append(0.5*width)
            self.widths = np.array(self.widths)
            self._first = False
        steps = self.widths
        #steps = np.array(steps)
        acceptance = np.zeros(ndim)
        cur_point = start_param_vec.copy()
        #cur_jprior = joint_prior
        cur_likelihood = loglikelihood(cur_point)
        #accepted_points = list([])

        for i in range(self.burn_in):
            rsteps = steps*(np.random.random(ndim)-0.5)
            u = np.random.random(ndim)
            for j in range(ndim):
                new_point = cur_point.copy()
                cur_pointj = cur_point[j]
                new_pointj = cur_pointj + rsteps[j]
                new_point[j] = new_pointj
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

                    acceptance[j] += 1.0
                        #if i%thinning == 0:
                            #accepted_points.append(cur_point.copy())
            if (((i+1) % 20) == 0):
                acceptance_ratio = acceptance/20.0
                less_than_mask = acceptance_ratio < 0.2
                gt_mask = acceptance_ratio > 0.6
                steps[less_than_mask] *= 0.66
                steps[gt_mask] *= 1.33
                acceptance[:] = 0.0
                # print('i',i,'acceptance ratios: ', acceptance_ratio)

        self.width = steps
        for i in range(self.iterations):
            rsteps = steps*(np.random.random(ndim)-0.5)
            u = np.random.random(ndim)
            for j in range(ndim):
                new_point = cur_point.copy()
                cur_pointj = cur_point[j]
                new_pointj = cur_pointj + rsteps[j]
                new_point[j] = new_pointj
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
