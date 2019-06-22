"""Implementation of Nested Sampling.

This module defines the class used for Nested Sampling.

References:
    1. Skilling, John. "Nested sampling." AIP Conference Proceedings. Vol.
        735. No. 1. AIP, 2004.
    2. Skilling, John. "Nested sampling for general Bayesian computation."
        Bayesian analysis 1.4 (2006): 833-859.
    3. Skilling, John. "Nested sampling's convergence." AIP Conference
        Proceedings. Vol. 1193. No. 1. AIP, 2009.
"""

import numpy as np
import pandas as pd
import warnings
import itertools
import networkx as nx
# Testing only!
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist
#from gleipnir.utilities import k_nearest_neighbors as knn

class NestedSampling(object):
    """A Nested Sampler.
    This class is an implementation of the outer layer of the classic Nested
    Sampling algorithm.

    Attributes:
        sampled_parameters (list of :obj:gleipnir.sampled_parameter.SampledParameter):
            The parameters that are being sampled during the Nested Sampling
            run.
        loglikelihood (function): The log-likelihood function to use for
            assigning a likelihood to parameter vectors during the sampling.
        sampler (obj from gleipnir.samplers): The sampling scheme to be used
            when updating sample points.
        population_size (int): The number of points to use in the Nested
            Sampling active population.
        stopping_criterion (obj from gleipnir.stopping_criterion): The criterion
            that should be used to determine when to stop the Nested Sampling
            run.
    References:
        1. Skilling, John. "Nested sampling." AIP Conference Proceedings. Vol.
            735. No. 1. AIP, 2004.
        2. Skilling, John. "Nested sampling for general Bayesian computation."
            Bayesian analysis 1.4 (2006): 833-859.
        3. Skilling, John. "Nested sampling's convergence." AIP Conference
            Proceedings. Vol. 1193. No. 1. AIP, 2009.
    """

    def __init__(self, sampled_parameters, loglikelihood, sampler,
                 population_size, stopping_criterion):
        """Initialize the Nested Sampler."""
        # stor inputs
        self.sampled_parameters = sampled_parameters
        # Make a dictionary version of the sampled parameters
        self._sampled_parameters_dict = {sp.name:sp for sp in sampled_parameters}
        self.loglikelihood = loglikelihood
        self.sampler = sampler
        self.population_size = population_size
        self.stopping_criterion = stopping_criterion

        # estimate of NS constriction factor
        self._alpha = population_size/(population_size+1)

        # NS accumulators and other private attributes
        self._evidence = 0.0
        self._evidence_error = 0.0
        self._logZ_err = 0.0
        self._log_evidence = 0.0
        self._information = 0.0
        self._H = 0.0
        self._previous_evidence = 0.0
        self._current_weights = 1.0
        self._previous_weight = 1.0
        self._n_iterations = 0
        self._dead_points = list()
        self._live_points = None
        self._post_eval = False
        self._posteriors = None
        return

    def run(self, verbose=False):
        """Initiate the Nested Sampling run.
        Returns:
            tuple of (float, float): Tuple containing the natural logarithm
            of the evidence and its error estimate as computed from the
            Nested Sampling run: (log_evidence, log_evidence_error)
        """
        # Zeroth iteration -- generate all the random samples
        if verbose:
            print("Generating the initial set of live points with population size {}...".format(self.population_size))
        live_points = dict()
        for i in range(self.population_size):
            for sampled_parameter_name in self._sampled_parameters_dict:
                name = sampled_parameter_name
                rs = self._sampled_parameters_dict[sampled_parameter_name].rvs(1)[0]
                if name not in live_points.keys():
                    live_points[name] = list([rs])
                else:
                    live_points[name].append(rs)

        self._live_points = pd.DataFrame(live_points)

        # Evaulate the log likelihood function for each live point
        if verbose:
            print("Evaluating the loglikelihood function for each live point...")
        log_likelihoods = np.array([self.loglikelihood(sampled_parameter_vector) for sampled_parameter_vector in self._live_points.values])

        # first iteration
        self._n_iterations += 1
        self._current_weights = 1.0 - self._alpha**self._n_iterations

        # Get the lowest likelihood live point
        ndx = np.argmin(log_likelihoods)
        log_l = log_likelihoods[ndx]
        param_vec = self._live_points.values[ndx]
        dZ = self._current_weights*np.exp(log_l)
        self._evidence += dZ
        # Accumulate the information
        dH = dZ*log_l
        if np.isnan(dH): dH = 0.0
        self._H += dH
        if self._evidence > 0.0:
            self._information = -np.log(self._evidence)+self._H/self._evidence

        self._previous_weight = self._current_weights
        # Add the lowest likelihood live point to dead points -- use dict that
        # that can be easily converted to pandas DataFrame.
        dpd = dict({'log_l': log_l, 'weight':self._current_weights})
        for k,val in enumerate(param_vec):
            dpd[self.sampled_parameters[k].name] = val
        self._dead_points.append(dpd)

        if verbose:
            print("Iteration: {} Evidence estimate: {} Remaining prior mass: {}".format(self._n_iterations, self._evidence, self._alpha**self._n_iterations))
            print("Dead Point:")
            print(self._dead_points[-1])

        # subseqent iterations
        while not self._stopping_criterion():
            self._n_iterations += 1
            self._current_weights = self._alpha**(self._n_iterations-1.0) - self._alpha**self._n_iterations

            # Replace the dead point with a modified survivor.
            # Choose at random from the survivors.
            r_p_ndx = int(np.random.random(1)*self.population_size)
            while r_p_ndx == ndx:
                r_p_ndx = int(np.random.random(1)*self.population_size)
            # Now make a new point from the survivor via the sampler.
            r_p_param_vec = self._live_points.values[r_p_ndx]
            updated_point_param_vec, u_log_l = self.sampler(self.sampled_parameters, self.loglikelihood, r_p_param_vec, log_l)
            log_likelihoods[ndx] = u_log_l
            self._live_points.values[ndx] = updated_point_param_vec
            # Get the lowest likelihood live point.
            ndx = np.argmin(log_likelihoods)
            log_l = log_likelihoods[ndx]
            param_vec = self._live_points.values[ndx]
            # Accumulate the evidence.
            dZ = self._current_weights*np.exp(log_l)
            self._evidence += dZ
            # Accumulate the information.
            dH = dZ*log_l
            if np.isnan(dH): dH = 0.0
            self._H += dH
            if self._evidence > 0.0:
                self._information = -np.log(self._evidence)+self._H/self._evidence

            # Add the lowest likelihood live point to dead points
            dpd = dict({'log_l': log_l, 'weight':self._current_weights})
            for k,val in enumerate(param_vec):
                dpd[self.sampled_parameters[k].name] = val
            self._dead_points.append(dpd)

            self._previous_weight = self._current_weights
            if verbose and (self._n_iterations%10==0):
                logZ_err = np.sqrt(self._information/self.population_size)
                ev_err = np.exp(logZ_err)
                print("Iteration: {} Evidence estimate: {} +- {} Remaining prior mass: {}".format(self._n_iterations, self._evidence, ev_err, self._alpha**self._n_iterations))
                print("Dead Point:")
                print(self._dead_points[-1])

        # Accumulate the final bit for remaining surviving points.
        weight = self._alpha**(self._n_iterations)
        likelihoods = np.exp(log_likelihoods)
        likelihoods_surv = np.array([likelihood for i,likelihood in enumerate(likelihoods) if i != ndx])
        l_m = likelihoods_surv.mean()
        self._evidence += weight*l_m
        # Accumulate the information.
        dH = weight*l_m*np.log(l_m)
        if np.isnan(dH): dH = 0.0
        self._H += dH
        if self._evidence > 0.0:
            self._information = -np.log(self._evidence)+self._H/self._evidence
        n_left = len(likelihoods_surv)
        a_weight = weight/n_left
        # Add the final survivors to the dead points.
        for i,l_likelihood in enumerate(log_likelihoods):
            if i != ndx:
                dpd = dict({'log_l': l_likelihood, 'weight':a_weight})
                for k,val in enumerate(self._live_points.values[i]):
                    dpd[self.sampled_parameters[k].name] = val
                self._dead_points.append(dpd)

        logZ_err = np.sqrt(self._information/self.population_size)
        self._logZ_err = logZ_err
        ev_err = np.exp(logZ_err)
        self._evidence_error = ev_err
        self._log_evidence = np.log(self._evidence)
        # Convert the dead points dict to a pandas DataFrame.
        self._dead_points = pd.DataFrame(self._dead_points)

        return self._log_evidence, logZ_err

    def _stopping_criterion(self):
        """Wrapper function for the stopping criterion."""
        return self.stopping_criterion(self)

    @property
    def evidence(self):
        """float: Estimate of the Bayesian evidence, or Z.
        """
        return self._evidence
    @evidence.setter
    def evidence(self, value):
        warnings.warn("evidence is not settable")

    @property
    def evidence_error(self):
        """float: Estimate (rough) of the error in the evidence, or Z.

        The error in the evidence is computed as the approximation:
            exp(sqrt(information/population_size))
        """
        return self._evidence_error
    @evidence_error.setter
    def evidence_error(self, value):
        warnings.warn("evidence_error is not settable")

    @property
    def log_evidence(self):
        """float: Estimate of the natural logarithm of the Bayesian evidence, or ln(Z).
        """
        return self._log_evidence
    @log_evidence.setter
    def log_evidence(self, value):
        warnings.warn("log_evidence is not settable")

    @property
    def log_evidence_error(self):
        """float: Estimate (rough) of the error in the natural logarithm of the evidence.
        """
        return self._logZ_err
    @log_evidence_error.setter
    def log_evidence_error(self, value):
        warnings.warn("log_evidence_error is not settable")

    @property
    def information(self):
        """float: Estimate of the Bayesian information, or H."""
        return self._information
    @information.setter
    def information(self, value):
        warnings.warn("information is not settable")

    def posteriors(self):
        """Estimates of the posterior marginal probability distributions of each parameter.
        Returns:
            dict of tuple of (numpy.ndarray, numpy.ndarray): The histogram
                estimates of the posterior marginal probability distributions.
                The returned dict is keyed by the sampled parameter names and
                each element is a tuple with (marginal_weights, bin_centers).
        """
        # Lazy evaluation at first call of the function and store results
        # so that subsequent calls don't have to recompute.
        if not self._post_eval:
            log_likelihoods = self._dead_points['log_l'].to_numpy()
            weights = self._dead_points['weight'].to_numpy()
            likelihoods = np.exp(log_likelihoods)
            norm_weights = (weights*likelihoods)/self.evidence
            gt_mask = norm_weights > 0.0
            parms = self._dead_points.columns[2:]
            # Rice bin count selection
            nbins = 2 * int(np.cbrt(len(norm_weights[gt_mask])))
            self._posteriors = dict()
            for parm in parms:
                marginal, edge = np.histogram(self._dead_points[parm][gt_mask], weights=norm_weights[gt_mask], density=True, bins=nbins)
                center = (edge[:-1] + edge[1:])/2.
                self._posteriors[parm] = (marginal, center)
            self._post_eval = True
        return self._posteriors

    def akaike_ic(self):
        """Estimate Akaike Information Criterion.
        This function estimates the Akaike Information Criterion (AIC) for the
        model simulated with Nested Sampling (NS). It does so by using the
        largest likelihood value found during the NS run and using that as
        the maximum likelihood estimate. The AIC formula is given by:
            AIC = 2k - 2ML,
        where k is number of sampled parameters and ML is maximum likelihood
        estimate.

        Returns:
            float: The AIC estimate.
        """
        mx = self._dead_points.max()
        ml = mx['log_l']
        k = len(self.sampled_parameters)
        return  2.*k - 2.*ml

    def bayesian_ic(self, n_data):
        """Estimate Bayesian Information Criterion.
        This function estimates the Bayesian Information Criterion (BIC) for the
        model simulated with Nested Sampling (NS). It does so by using the
        largest likelihood value found during the NS run and taking that as
        the maximum likelihood estimate. The BIC formula is given by:
            BIC = ln(n_data)k - 2ML,
        where n_data is the number of data points used in computing the likelihood
        function fitting, k is number of sampled parameters, and ML is maximum
        likelihood estimate.

        Args:
            n_data (int): The number of data points used when comparing to data
                in the likelihood function.

        Returns:
            float: The BIC estimate.
        """
        mx = self._dead_points.max()
        ml = mx['log_l']
        k = len(self.sampled_parameters)
        return  np.log(n_data)*k - 2.*ml

    def deviance_ic(self):
        """Estimate Deviance Information Criterion.
        This function estimates the Deviance Information Criterion (DIC) for the
        model simulated with Nested Sampling (NS). It does so by using the
        posterior distribution estimates computed from the NS outputs.
        The DIC formula is given by:
            DIC = p_D + D_bar,
        where p_D = D_bar - D(theta_bar), D_bar is the posterior average of
        the deviance D(theta)= -2*ln(L(theta)) with L(theta) the likelihood
        of parameter set theta, and theta_bar is posterior average parameter set.

        Returns:
            float: The DIC estimate.
        """
        log_likelihoods = self._dead_points['log_l'].to_numpy()
        weights = self._dead_points['weight'].to_numpy()
        likelihoods = np.exp(log_likelihoods)
        norm_weights = (weights*likelihoods)/self.evidence
        gt_mask = norm_weights > 0.0
        parms = self._dead_points.columns[2:]
        params = self._dead_points[parms]
        D_of_theta = -2.*log_likelihoods[gt_mask]
        D_bar = np.average(D_of_theta, weights=norm_weights[gt_mask])
        theta_bar = np.average(params[gt_mask], weights=norm_weights[gt_mask], axis=0)
        D_of_theta_bar = -2. * self.loglikelihood(theta_bar)
        p_D = D_bar - D_of_theta_bar
        return p_D + D_bar

    @property
    def dead_points(self):
        """The set of dead points collected during the Nested Sampling run."""
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



    def landscape(self):
        # Get the samples.
        ns_samples = self.dead_points
        # Pull out parameters.
        pnames = ns_samples.columns[2:]
        params = ns_samples[pnames].to_numpy()
        # loglikelihoods.
        loglikelihoods = ns_samples['log_l'].to_numpy()
        # Weights.
        weights = ns_samples['weight'].to_numpy()
        # Compute the k-nearest neighbors--uses Euclidean distance between
        # parameter vectors.
        #print(params)
        param_knn = _knn(params, loglikelihoods, k=6)
        #quit()
        # Build the knn-network from the knn lists.
        graph = nx.Graph(name='base')
        # First add the nodes with loglikelihoods and weights
        for key in param_knn.keys():
            graph.add_node(key, loglikelihood=loglikelihoods[key],
                           weight=weights[key])
        # Now add the edges with distance.
        for key in param_knn.keys():
            my_knn = param_knn[key]
            #print("key:", key, len(my_knn))
            for item in my_knn:
            #    print("add edge: ", key, item[0])
                graph.add_edge(key, item[0], distance=item[1])
        # Prune out any disjoint nodes from the initial graph.
        subs = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]
        for sub in subs:
            if len(sub) == 1:
                for node in sub.nodes:
                    graph.remove_node(node)
        #print(graph.edges)
        #quit()
        _basin_label = itertools.cycle(('base', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S'))
        basin_graph = nx.DiGraph()
        basin_points = dict()
        current_basin = next(_basin_label)
        basin_graph.add_node(current_basin)
        basin_points[current_basin] = list()
        subgraphs = list([graph])
        basins = list()
        basins.append(current_basin)
        # Now we start pruning and separating basins
        for i in range(len(loglikelihoods)):
            # Find the subgraph with that node.
            for j, sg in enumerate(subgraphs):
                if i in sg.nodes:
                    print(graph.name)
                    #Add to the basins list
                    basin_points[sg.name].append(i)
                    # Remove the node
                    sg.remove_node(i)
                    # Check for splitting.
                    subs = [sg.subgraph(c).copy() for c in nx.connected_components(sg)]
                    #print("number of subs: ", len(subs))
                    #for s, sub in enumerate(subs):
                    #    print("s ",s," size ",len(sub))
                    #quit()

                    if len(subs) > 1:
                        # Remove dangling nodes
                        for sub in subs:
                            if len(sub) < 2:
                                for node in sub.nodes:
                                #node = list(sub.nodes)[0]
                                    sg.remove_node(node)
                                    #subgraphs[j].remove_node(node)
                    subs = [sg.subgraph(c).copy() for c in nx.connected_components(sg)]
                    print("number of subgraphs", len(subs))
                    if len(subs) > 0:
                        print("len(subs[0])", len(subs[0]))
                    if (len(subs) > 1) and (len(subs) < 15):
                        # Caused a split.
                        top_basin = sg.name
                        # Remove the original subgraph from the master list.
                        del subgraphs[j]
                        # Now add the its subgraphs to the master list.
                        for sub in subs:
                            if len(sub) > 1:
                                # Create new copies with appropriate names.
                                subg = nx.Graph(sub,name=next(_basin_label))
                                print("created new split: ",subg.name)
                                subgraphs.append(subg)
                                basin_graph.add_edge(top_basin, subg.name)
                                basin_points[subg.name] = list()
                                basins.append(subg.name)
                    elif len(subs) == 0:
                        del subgraphs[j]

                    # Now break the enumerate since we already found the
                    # right subgraph.
                    break
        landscape_points = list()
        #quit()
        basin = 'base'

        nbasins = len(basins)

        # Compute the widths for each point
        widths = dict()
        for b, basin in enumerate(basins):
            bps = basin_points[basin]
            descend = nx.descendants(basin_graph, basin)
            descend_weight = 0.0
            for d in descend:
                descend_weight += weights[basin_points[d]].sum()
            for idx, i in enumerate(bps):
                my_weight = weights[i]
                follow_weight = weights[bps[idx+1:]].sum()
                width = descend_weight + my_weight + follow_weight
                #dbps = dict({'center':b,'loglikelihood':loglikelihoods[i], 'weight':my_weight, 'width':width})
                #landscape_points.append(dbps)
                widths[i] = width
        big_sibling = dict()
        for i in range(nbasins):
            big_sibling[basins[i]] = True
        has_center = dict()
        for i in range(nbasins):
            has_center[basins[i]] = False

        basin_centers = dict()
        basin_centers['base'] = 0.0
        has_center['base'] = True
        for b, basin in enumerate(basins[1:]):
            if not has_center[basin]:
                my_width = widths[basin_points[basin][0]]
                parents = list(basin_graph.predecessors(basin))
                parent = parents[0]
                p_low = basin_points[parent][-1]
                width_p_low = widths[p_low]
                if len(parents) > 0:
                    successors = list(basin_graph.successors(parent))
                    siblings = [successor for successor in successors if successor != basin]

                my_center =  (basin_centers[parent] - width_p_low/2.0) + my_width/2.0

                basin_centers[basin] = my_center

                has_center[basin] = True
                #has_center[sibling] = True
                for s, sibling in enumerate(siblings):
                    sib_width = widths[basin_points[sibling][0]]
                    prev_sibs = siblings[:s]
                    prev_width = 0.0
                    for prev_sib in prev_sibs:
                        prev_width += widths[basin_points[prev_sib][0]]
                    sib_center = (basin_centers[parent] - width_p_low/2.0) + my_width + prev_width + sib_width/2.0
                    basin_centers[sibling] = sib_center
                    has_center[sibling] = True
                    #sib_center = (basin_centers[parent] - width_p_low/2.0) + my_width + sib_width/2.0
        print(basin_centers)

        # First do the base left
        bps = basin_points['base']
        for i in reversed(bps):
            pos = basin_centers['base'] - widths[i]/2.0
            dbps = dict({'x':pos,'loglikelihood':loglikelihoods[i], 'weight':weights[i], 'width':widths[i]})
            landscape_points.append(dbps)
        # Now do the children
        for basin in basins[1:]:
            bps = basin_points[basin]
            # Forward
            for idx, i in enumerate(bps):
                pos = basin_centers[basin] - widths[i]/2.0
                dbps = dict({'x':pos,'loglikelihood':loglikelihoods[i], 'weight':weights[i], 'width':widths[i]})
                landscape_points.append(dbps)
            # Reverse
            for i in reversed(bps):
                pos = basin_centers[basin] + widths[i]/2.0
                dbps = dict({'x':pos,'loglikelihood':loglikelihoods[i], 'weight':weights[i], 'width':widths[i]})
                landscape_points.append(dbps)
        # Now do the base right
        bps = basin_points['base']
        for idx, i in enumerate(bps):
            pos = basin_centers['base'] + widths[i]/2.0
            dbps = dict({'x':pos,'loglikelihood':loglikelihoods[i], 'weight':weights[i], 'width':widths[i]})
            landscape_points.append(dbps)
        #
        # for b, basin in enumerate(basins):
        #     bps = basin_points[basin]
        #     descend = nx.descendants(basin_graph, basin)
        #     descend_weight = 0.0
        #     for d in descend:
        #         descend_weight += weights[basin_points[d]].sum()
        #     for idx, i in enumerate(bps):
        #         my_weight = weights[i]
        #         follow_weight = weights[bps[idx+1:]].sum()
        #         width = descend_weight + my_weight + follow_weight
        #         dbps = dict({'center':b,'loglikelihood':loglikelihoods[i], 'weight':my_weight, 'width':width})
        #         landscape_points.append(dbps)

        landscape_points = pd.DataFrame(landscape_points)
        #nx.draw_spring(basin_graph, with_labels=True)
        #plt.show()
        plt.plot(landscape_points['x'], -1.0*landscape_points['loglikelihood'])
        plt.show()


    def landscape_2(self):
        # Get the samples.
        ns_samples = self.dead_points
        # Pull out parameters.
        pnames = ns_samples.columns[2:]
        params = ns_samples[pnames].to_numpy()
        # loglikelihoods.
        loglikelihoods = ns_samples['log_l'].to_numpy()
        # Weights.
        weights = ns_samples['weight'].to_numpy()
        # Compute the k-nearest neighbors--uses Euclidean distance between
        # parameter vectors with lower likelihood.
        param_knn = _knn(params, loglikelihoods, k=6)
        #quit()
        # Build the knn-network from the knn lists.
        graph = nx.Graph(name='base')
        # First add the nodes with loglikelihoods and weights
        for key in param_knn.keys():
            graph.add_node(key, loglikelihood=loglikelihoods[key],
                           weight=weights[key])
        # Now add the edges with distance.
        for key in param_knn.keys():
            my_knn = param_knn[key]
            #print("key:", key, len(my_knn))
            for item in my_knn:
            #    print("add edge: ", key, item[0])
                graph.add_edge(key, item[0], distance=item[1])
        # Prune out any disjoint nodes from the initial graph.
        subs = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]
        for sub in subs:
            if len(sub) == 1:
                for node in sub.nodes:
                    graph.remove_node(node)

        basin_graph = nx.DiGraph()
        # Now we start pruning and separating basins
        for i in range(len(loglikelihoods)):
            if i in graph.nodes:
                graph.remove_node(i)
                #basin_graph.add_node(i)
                my_knn = param_knn[i]
                #gnodes = list(graph.nodes())
                # Check for splitting.
                subs = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]
                for sub in subs:
                    snodes = list(sub.nodes)
                    connect = list()
                    connectl = list()
                    for snode in sub.nodes:
                        snknn = param_knn[snode]
                        snknnidx = [snknni[0] for snknni in snknn]

                        if i in snknnidx:
                            connect.append(snode)
                            connectl.append(loglikelihoods[snode])


                    if len(connect)>0:
                        connectl = np.array(connectl)
                        mlidx = np.argmin(connectl)
                        cmlidx = connect[mlidx]
                        basin_graph.add_edge(i, cmlidx)


        nx.draw_spring(basin_graph, with_labels=True)
        plt.show()

def _knn(X, likelihoods, k=1):
    """Determine the k-nearest neighbors of each point within a random variate sample.
    This function uses Euclidean distance as the distance metric for
    determining the nearest neighbors.
    Args:
        X (numpy.array): A random variate sample.
        k (int): The number of nearest neighbors to find. Defaults to 1.

    Returns:
        dict: A dictionary keyed to the sample indices of points from the input
            random variate sample. Each element is a sorted list of the
            k-nearest neighbors of the form
            [[index, distance], [index, distance]...]

    """
    #length
    nX = len(X)
    #initialize knn dict
    knn = {key: [] for key in range(nX)}
    #make sure X has the right shape for the cdist function
    X = np.reshape(X, (nX,-1))
    dists_arr = cdist(X, X)
    distances = [[i,j,dists_arr[i,j]] for i in range(nX-1) for j in range(i+1,nX)]
    #sort distances
    distances.sort(key=lambda x: x[2])
    #pick up the k nearest
    for d in distances:
        i = d[0]
        j = d[1]
        dist = d[2]
        li = likelihoods[i]
        lj = likelihoods[j]
        #print("dist", dist)
        if (len(knn[i]) < k) and (li > lj):
            #print("dist", dist)
            knn[i].append([j, dist])
        elif (len(knn[j]) < k) and (lj > li):
            knn[j].append([i, dist])
    return knn
