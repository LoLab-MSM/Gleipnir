"""
Implementation of the 2-dimensional 'Egg Carton' problem and its sampling
using an implementation of classic Nested Sampling via Gleipnir.

Adapted from the pymultinest_demo.py at:
https://github.com/JohannesBuchner/PyMultiNest/blob/master/pymultinest_demo.py

The likelihood landscape has an egg carton-like shape; see slide 15 from:
http://www.nbi.dk/~koskinen/Teaching/AdvancedMethodsInAppliedStatistics2016/Lecture14_MultiNest.pdf

"""

import numpy as np
from scipy.stats import uniform
from gleipnir.sampled_parameter import SampledParameter
from gleipnir.nestedsampling import NestedSampling
from gleipnir.nestedsampling.samplers import MetropolisSampler
from gleipnir.nestedsampling.stopping_criterion import NumberOfIterations


# Number of paramters to sample is 2
ndim = 2

# Define the loglikelihood function
def loglikelihood(sampled_parameter_vector):
    chi = (np.cos(sampled_parameter_vector)).prod()
    return (2. + chi)**5

if __name__ == '__main__':

    # Set up the list of sampled parameters: the prior is Uniform(0:10*pi) --
    # we are using a fixed uniform prior from scipy.stats
    sampled_parameters = [SampledParameter(name=i, prior=uniform(loc=0.0,scale=10.0*np.pi)) for i in range(ndim)]

    # Set the active point population size
    population_size = 500
    # Setup the sampler to use when updated points during the NS run --
    # Here we are using an implementation of the Metropolis Monte Carlo algorithm
    # with component-wise trial moves and augmented acceptance criteria that adds a
    # hard rejection constraint for the NS likelihood boundary.
    sampler = MetropolisSampler(iterations=50, tuning_cycles=2)
    # Setup the stopping criterion for the NS run -- We'll use a fixed number of
    # iterations: 10*population_size
    stopping_criterion = NumberOfIterations(10*population_size)
    # Construct the Nested Sampler -- Using the MCMC sampler with hard rejection
    # of likelihood levels is an implementation of the classic NS algorithm.
    NS = NestedSampling(sampled_parameters=sampled_parameters,
                        loglikelihood=loglikelihood,
                        population_size=population_size,
                        sampler=sampler,
                        stopping_criterion=stopping_criterion)
    # run it
    log_evidence, log_evidence_error = NS.run(verbose=True)

    # log Evidence (lnZ) should be approximately 236
    print("log_evidence: {} +- {} ".format(log_evidence, log_evidence_error))

    # Retrieve the evidence and information
    evidence = NS.evidence
    error = NS.evidence_error
    information = NS.information
    print("evidence: {} +- {}".format(evidence, error))
    # exp(-information) is an estimate of the compression factor from prior to posterior
    print("Information: {} exp(-Information): {}".format(information, np.exp(-information)))
    # We can also pull out an estimate of the Akaike Information Criterion (AIC)
    aic = NS.akaike_ic()
    print("AIC estimate: {}".format(aic))
    # Bayesian Information Criterion (BIC)
    bic = NS.bayesian_ic(2)
    print("BIC estimate: {}".format(bic))
    # Deviance Information Criterion (DIC)
    dic = NS.deviance_ic()
    print("DIC estimate: {}".format(dic))

    #try plotting a marginal distribution
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
        # Get the posterior distributions -- the posteriors are return as dictionary
        # keyed to the names of the sampled paramters. Each element is a histogram
        # estimate of the marginal distribution, including the heights and centers.
        posteriors = NS.posteriors()
        # Lets look at the first paramter
        marginal, edges, centers = posteriors[list(posteriors.keys())[0]]
        # Plot with seaborn
        sns.distplot(centers, bins=edges, hist_kws={'weights':marginal})
        # Uncomment next line to plot with plt.hist:
        # plt.hist(centers, bins=edges, weights=marginal)
        plt.show()
    except ImportError:
        pass
