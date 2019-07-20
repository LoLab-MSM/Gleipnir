"""
Implementation of finding the parameters of line with three data
points that have uncertainty using DNest4 via Gleipnir.
This is a 2 parameter problem.

Adapted from the Nestle 'Getting started' example at:
http://kylebarbary.com/nestle/
"""

import numpy as np
from scipy.stats import uniform
from gleipnir.sampled_parameter import SampledParameter
from gleipnir.dnest4 import DNest4NestedSampling



# Setupp the data points that are being fitted.
data_x = np.array([1., 2., 3.])
data_y = np.array([1.4, 1.7, 4.1])
data_yerr = np.array([0.2, 0.15, 0.2])

# Define the loglikelihood function
def loglikelihood(theta):
    y = theta[1] * data_x + theta[0]
    chisq = np.sum(((data_y - y) / data_yerr)**2)
    return -chisq / 2.

if __name__ == '__main__':

    # Set up the list of sampled parameters: the prior is Uniform(-5:5) --
    # we are using a fixed uniform prior from scipy.stats
    parm_names = list(['m', 'b'])
    sampled_parameters = [SampledParameter(name=p, prior=uniform(loc=-5.0,scale=10.0)) for p in parm_names]

    # Set the active point population size
    population_size = 100
    # Setup the Nested Sampling run
    n_params = len(sampled_parameters)
    print("Sampling a total of {} parameters".format(n_params))
    #population_size = 10
    print("Will use NS population size of {}".format(population_size))
    # Construct the Nested Sampler
    DNS = DNest4NestedSampling(sampled_parameters,
                               loglikelihood,
                               population_size,
                               num_steps=1000)
    #print(PCNS.likelihood(np.array([1.0])))
    #quit()
    # run it
    log_evidence, log_evidence_error = DNS.run(verbose=True)
    # Print the output
    print("log_evidence: {} +- {} ".format(log_evidence, log_evidence_error))
    best_fit_l = DNS.best_fit_likelihood()
    print("Max likelihood parms: ", best_fit_l)
    best_fit_p, fit_error = DNS.best_fit_posterior()
    print("Max posterior weight parms ", best_fit_p)
    print("Max posterior weight parms error ", fit_error)
    # Information criteria
    # Akaike
    aic = DNS.akaike_ic()
    # Bayesian
    bic = DNS.bayesian_ic(3)
    # Deviance
    dic = DNS.deviance_ic()
    print("AIC ",aic, " BIC ", bic, " DIC ",dic)
    #try plotting a marginal distribution
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
        # Get the posterior distributions -- the posteriors are return as dictionary
        # keyed to the names of the sampled paramters. Each element is a histogram
        # estimate of the marginal distribution, including the heights and centers.
        posteriors = DNS.posteriors()
        # Lets look at the first paramter
        marginal, edges, centers = posteriors[list(posteriors.keys())[0]]
        # Plot with seaborn
        sns.distplot(centers, bins=edges, hist_kws={'weights':marginal})
        # Uncomment next line to plot with plt.hist:
        # plt.hist(centers, bins=edges, weights=marginal)
        plt.show()
    except ImportError:
        pass
