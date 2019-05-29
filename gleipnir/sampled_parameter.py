"""Class for defining the parameters to be sampled during a Nested Sampling run.

This module defines the class for defining the parmeters and their
prior distributions that are to sampled and used during the Nested Sampling run.


"""

import numpy as np

class SampledParameter(object):
    """A parameter that will be sampled during a Nested Sampling run.
    Attributes:
        name (str,int): The name of this parameter.
        prior_dist (:obj:): The prior distribution object. This can be a fixed
            distribution from scipy.stats (e.g., scipy.stats.uniform) or
            a user-defined distribution class with the appropriate
            functions.
    """

    def __init__(self, name, prior):
        """Initialize the sampled parameter.
        Args:
            name (str,int): set the name Attribute.
            prior (:obj:): set the prior_dist Attribute.
        """
        self.name = name
        self.prior_dist = prior
        # Map the pdf/pmf functions to a common label.
        try:
            self.pf = prior.pdf
            self.logpf = prior.logpdf
        except:
            self.pf = prior.pmf
            self.logpf = prior.logpmf
        # Compute the normalization factor for this prior.
        self._norm = prior.cdf(np.inf)
        return

    def rvs(self, sample_shape):
        """Random variate sample.
        Args:
            sample_shape (int, tuple): The array size/shape for the random
            variate.
        Returns:
            (numpy.array): The set of random variate samples with length/shape
                sample_shape drawn form the prior distrbution.
        """
        return self.prior_dist.rvs(sample_shape)


    def logprior(self, value):
        """Natural logarithm of the normalized probaility function.
        Args:
            value (float, numpy.array): A value from the prior space.
        Returns:
            (float, numpy.array): The natural logarithm of the normalized
                probability function at the input value.
        """
        return np.log(self.pf(value)/self._norm)

    def prior(self, value):
        """Normalized prior density/mass.
        Args:
            value (float, numpy.array): A value from the prior space.
        Returns:
            float, numpy.array: The normalized prior density/mass at the
                specicied value of the prior space.
        """
        return self.pf(value)/self._norm

    def invcdf(self, value):
        """The inverted cumulative density/mass function.
        Args:
            value (float, numpy.array): A value from [0:1].
        Returns:
            float, numpy.array: The inverted cumulative distrbution/mass
                values at the specified value.
        """
        return self.prior_dist.ppf(value)
