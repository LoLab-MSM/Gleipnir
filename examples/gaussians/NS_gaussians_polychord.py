"""
Implementation of a 5-dimensional Gaussian problem and its Nested Sampling
using MultiNest via Gleipnir.

Adapted from the DNest4 python gaussian example:
https://github.com/eggplantbren/DNest4/blob/master/python/examples/gaussian/gaussian.py
"""

import numpy as np
from scipy.stats import uniform
from scipy.special import erf
from gleipnir.sampled_parameter import SampledParameter
from gleipnir.polychord import PolyChordNestedSampling



# Number of paramters to sample is 5
ndim = 5

# Define the loglikelihood function
def loglikelihood(sampled_parameter_vector):
    const = -0.5*np.log(2*np.pi)
    return -0.5*np.sum(sampled_parameter_vector**2) + ndim * const
width = 10.0
def analytic_log_evidence(ndim, width):
      lZ = (ndim * np.log(erf(0.5*width/np.sqrt(2)))) - (ndim * np.log(width))
      return lZ

if __name__ == '__main__':

    # Set up the list of sampled parameters: the prior is Uniform(-5:5) --
    # we are using a fixed uniform prior from scipy.stats
    sampled_parameters = [SampledParameter(name=i, prior=uniform(loc=-5.0,scale=10.0)) for i in range(ndim)]

    # Set the active point population size
    population_size = 500
    # Setup the Nested Sampling run
    n_params = len(sampled_parameters)
    print("Sampling a total of {} parameters".format(n_params))
    #population_size = 10
    print("Will use NS population size of {}".format(population_size))
    # Construct the Nested Sampler
    PCNS = PolyChordNestedSampling(sampled_parameters=sampled_parameters,
                                   loglikelihood=loglikelihood,
                                   population_size=population_size)
    #print(PCNS.likelihood(np.array([1.0])))
    #quit()
    # run it
    log_evidence, log_evidence_error = PCNS.run(verbose=True)
    # Print the output
    print("log_evidence: {} +- {} ".format(log_evidence, log_evidence_error))
    print("analytic log_evidence: {}".format(analytic_log_evidence(ndim, width)))
    best_fit_l = PCNS.best_fit_likelihood()
    print("Max likelihood parms: ", best_fit_l)
    best_fit_p, fit_error = PCNS.best_fit_posterior()
    print("Max posterior weight parms ", best_fit_p)
    print("Max posterior weight parms error ", fit_error)
    #try plotting a marginal distribution
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
        # Get the posterior distributions -- the posteriors are return as dictionary
        # keyed to the names of the sampled paramters. Each element is a histogram
        # estimate of the marginal distribution, including the heights and centers.
        posteriors = PCNS.posteriors()
        # Lets look at the first paramter
        marginal, edges, centers = posteriors[list(posteriors.keys())[0]]
        # Plot with seaborn
        sns.distplot(centers, bins=edges, hist_kws={'weights':marginal})
        # Uncomment next line to plot with plt.hist:
        # plt.hist(centers, bins=edges, weights=marginal)
        plt.show()
    except ImportError:
        pass
