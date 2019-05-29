"""
Implementation of a 1-dimensional multi-modal likelihood problem and its
sampling using PolyChord via Gleipnir.

Adapted from Example1 of the PyMultiNest tutorial:
http://johannesbuchner.github.io/pymultinest-tutorial/example1.html
"""

import numpy as np
from scipy.stats import uniform
import matplotlib.pyplot as plt
from gleipnir.sampled_parameter import SampledParameter
from gleipnir.polychord import PolyChordNestedSampling

# Number of paramters to sample is 1
ndim = 1
# Set up the list of sampled parameters: the prior is Uniform(0:2) --
# we are using a fixed uniform prior from scipy.stats
sampled_parameters = [SampledParameter(name=i, prior=uniform(loc=0.0,scale=2.0)) for i in range(ndim)]
# Mode positions for the multi-modal likelihood
positions = np.array([0.1, 0.2, 0.5, 0.55, 0.9, 1.1])
# its width
width = 0.01

# Define the loglikelihood function
def loglikelihood(sampled_parameter_vector):
    diff = sampled_parameter_vector[0] - positions
    diff_scale = diff / width
    l = np.exp(-0.5 * diff_scale**2) / (2.0*np.pi*width**2)**0.5
    logl = np.log(l.mean())
    if logl < -1000.0:
        logl = -1000.0
    return logl


# Construct the Nested Sampler
PCNS = PolyChordNestedSampling(sampled_parameters=sampled_parameters,
                    loglikelihood=loglikelihood, population_size=500)
#print(PCNS.likelihood(np.array([1.0])))
#quit()
# run it
log_evidence, log_evidence_error = PCNS.run()
# Print the output
#print(PCNS.output)
# Evidence should be 1/2
print("log_evidence: ", log_evidence)
print("evidence: ", PCNS.evidence)

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
