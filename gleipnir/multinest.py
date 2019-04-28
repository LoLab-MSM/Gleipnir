"""Implementation on top of MultiNest via PyMultiNest.

This module defines the class for Nested Sampling using the MultiNest
program via its Python wrapper PyMultiNest. Note that PyMultiNest and MultiNest
have to be built and installed separately (from gleipnir) before this module
can be used.

PyMultiNest: https://github.com/JohannesBuchner/PyMultiNest
MultiNest: https://github.com/JohannesBuchner/MultiNest

References:
    MultiNest:
    1. Feroz, Farhan, and M. P. Hobson. "Multimodal nested sampling: an
        efficient and robust alternative to Markov Chain Monte Carlo
        methods for astronomical data analyses." Monthly Notices of the
        Royal Astronomical Society 384.2 (2008): 449-463.
    2. Feroz, F., M. P. Hobson, and M. Bridges. "MultiNest: an efficient
        and robust Bayesian inference tool for cosmology and particle
        physics." Monthly Notices of the Royal Astronomical Society 398.4
        (2009): 1601-1614.
    3. Feroz, F., et al. "Importance nested sampling and the MultiNest
        algorithm." arXiv preprint arXiv:1306.2144 (2013).
    PyMultiNest:
    4. Buchner, J., et al. "X-ray spectral modelling of the AGN obscuring
        region in the CDFS: Bayesian model selection and catalogue."
        Astronomy & Astrophysics 564 (2014): A125.

"""

import numpy as np
import pandas as pd
import warnings
import itertools
import networkx as nx
# Testing only!
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist
try:
    import pymultinest
    from pymultinest.solve import solve
    from pymultinest.analyse import Analyzer
except ImportError as err:
    #print(err)
    raise err

class MultiNestNestedSampling(object):
    """Nested Sampling using MultiNest.
    PyMultiNest: https://github.com/JohannesBuchner/PyMultiNest
    MultiNest: https://github.com/JohannesBuchner/MultiNest

    Attributes:
        sampled_parameters (list of :obj:gleipnir.sampled_parameter.SampledParameter):
            The parameters that are being sampled during the Nested Sampling
            run.
        loglikelihood (function): The log-likelihood function to use for
            assigning a likelihood to parameter vectors during the sampling.
        population_size (int): The number of points to use in the Nested
            Sampling active population. Default: None -> gets set to
            25*(number of sampled parameters) if left at default.
        multinest_kwargs (dict): Additional keyword arguments that should be
            passed to the PyMultiNest MultiNest solver.
    References:
        1. Feroz, Farhan, and M. P. Hobson. "Multimodal nested sampling: an
            efficient and robust alternative to Markov Chain Monte Carlo
            methods for astronomical data analyses." Monthly Notices of the
            Royal Astronomical Society 384.2 (2008): 449-463.
        2. Feroz, F., M. P. Hobson, and M. Bridges. "MultiNest: an efficient
            and robust Bayesian inference tool for cosmology and particle
            physics." Monthly Notices of the Royal Astronomical Society 398.4
            (2009): 1601-1614.
        3. Feroz, F., et al. "Importance nested sampling and the MultiNest
            algorithm." arXiv preprint arXiv:1306.2144 (2013).
        4. Buchner, J., et al. "X-ray spectral modelling of the AGN obscuring
            region in the CDFS: Bayesian model selection and catalogue."
            Astronomy & Astrophysics 564 (2014): A125.
    """

    def __init__(self, sampled_parameters, loglikelihood, population_size=None,
                 **multinest_kwargs):
        """Initialize the MultiNest Nested Sampler."""
        self.sampled_parameters = sampled_parameters
        self.loglikelihood = loglikelihood
        self.population_size = population_size
        self.multinest_kwargs = multinest_kwargs

        self._nDims = len(sampled_parameters)
        self._nDerived = 0
        self._output = None
        self._post_eval = False
        if self.population_size is None:
            self.population_size = 25*self._nDims

        # Make the prior function for PyMultiNest.
        def prior(hypercube):
            return np.array([self.sampled_parameters[i].invcdf(value) for i,value in enumerate(hypercube)])

        self._prior = prior
        # multinest settings
        self._file_root = 'multinest_run' #string

        return


    def run(self, verbose=False):
        """Initiate the MultiNest Nested Sampling run."""
        output = solve(LogLikelihood=self.loglikelihood, Prior=self._prior,
                       n_dims = self._nDims,
                       n_live_points=self.population_size,
                       outputfiles_basename=self._file_root,
                       verbose=verbose,
                       **self.multinest_kwargs)
        self._output = output
        return self.log_evidence, self.log_evidence_error

    @property
    def evidence(self):
        """float: Estimate of the Bayesian evidence, or Z."""
        return np.exp(self._output['logZ'])
    @evidence.setter
    def evidence(self, value):
        warnings.warn("evidence is not settable")

    @property
    def evidence_error(self):
        """float: Estimate (rough) of the error in the evidence, or Z."""
        return np.exp(self._output['logZerr'])
    @evidence_error.setter
    def evidence_error(self, value):
        warnings.warn("evidence_error is not settable")

    @property
    def log_evidence(self):
        """float: Estimate of the natural logarithm of the Bayesian evidence, or ln(Z).
        """
        return self._output['logZ']
    @log_evidence.setter
    def log_evidence(self, value):
        warnings.warn("log_evidence is not settable")

    @property
    def log_evidence_error(self):
        """float: Estimate of the error in the natural logarithm of the evidence.
        """
        return self._output['logZerr']
    @log_evidence_error.setter
    def log_evidence_error(self, value):
        warnings.warn("log_evidence_error is not settable")

    @property
    def information(self):
        """None: Not implemented yet->Estimate of the Bayesian information, or H."""
        return None
    @information.setter
    def information(self, value):
        warnings.warn("information is not settable")

    def posteriors(self, nbins=None):
        """Estimates of the posterior marginal probability distributions of each parameter.
        Returns:
            dict of tuple of (numpy.ndarray, numpy.ndarray): The histogram
                estimates of the posterior marginal probability distributions.
                The returned dict is keyed by the sampled parameter names and
                each element is a tuple with (marginal_weights, bin_centers).
        """
        # Lazy evaluation at first call of the function and store results
        # so that subsequent calls don't have to recompute.
        print('nbins', nbins)
        if not self._post_eval:
            # Here the samples are samples directly from the posterior
            # (i.e. equal weights).
            samples = self._output['samples']
            # Rice bin count selection
            if nbins is None:
                nbins = 2 * int(np.cbrt(len(samples)))
            print('nbins', nbins)
            nd = samples.shape[1]
            self._posteriors = dict()
            for ii in range(nd):
                marginal, edge = np.histogram(samples[:,ii], density=True, bins=nbins)
                center = (edge[:-1] + edge[1:])/2.
                self._posteriors[self.sampled_parameters[ii].name] = (marginal, center)
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
        mn_data = Analyzer(len(self.sampled_parameters), self._file_root, verbose=False).get_data()
        log_ls = -0.5*mn_data[:,1]
        ml = log_ls.max()
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
        mn_data = Analyzer(len(self.sampled_parameters), self._file_root, verbose=False).get_data()
        log_ls = -0.5*mn_data[:,1]
        ml = log_ls.max()
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
        mn_data = Analyzer(len(self.sampled_parameters), self._file_root, verbose=False).get_data()
        params = mn_data[:,2:]
        log_likelihoods = -0.5*mn_data[:,1]
        prior_mass = mn_data[:,0]
        norm_weights = (prior_mass*np.exp(log_likelihoods))/self.evidence
        nw_mask = np.isnan(norm_weights)
        if  np.any(nw_mask):
            return np.inf
        D_of_theta = -2.*log_likelihoods
        D_bar = np.average(D_of_theta, weights=norm_weights)
        theta_bar = np.average(params, axis=0, weights=norm_weights)
        D_of_theta_bar = -2. * self.loglikelihood(theta_bar)
        p_D = D_bar - D_of_theta_bar
        return p_D + D_bar

    def landscape(self):

        # Get the samples.
        mn_data = Analyzer(len(self.sampled_parameters), self._file_root, verbose=False).get_data()
        print(mn_data)
        # Pull out parameters.
        params = mn_data[-2000:,2:]
        # loglikelihoods.
        loglikelihoods = -0.5*mn_data[-2000:,1]
        print(min(loglikelihoods))
        quit()
        # Weights.
        weights = mn_data[-2000:,0]
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
        _basin_label = itertools.cycle(('base', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'))
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
                            if len(sub) < 12:
                                for node in sub.nodes:
                                #node = list(sub.nodes)[0]
                                    sg.remove_node(node)
                                    #subgraphs[j].remove_node(node)
                    subs = [sg.subgraph(c).copy() for c in nx.connected_components(sg)]
                    print("number of subgraphs", len(subs))
                    if len(subs) > 0:
                        print("len(subs[0])", len(subs[0]))
                    if len(subs) > 1:
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
        if (len(knn[i]) < k) and (li < lj):
            #print("dist", dist)
            knn[i].append([j, dist])
        if (len(knn[j]) < k) and (lj < li):
            knn[j].append([i, dist])
    return knn
