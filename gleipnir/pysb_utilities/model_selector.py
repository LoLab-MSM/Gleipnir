"""Defines the ModelSelector class, which is a subclass of HypSelector.
"""

from .hyp_selector import HypSelector


class ModelSelector(HypSelector):
    """A model selector using Nested Sampling-based model selection.
    ModelSelector is sub-classed from HypSelector.

    Args:
        models (list of :obj:pysb.Model): A list PySB models to perform model
        selection on. Filename of the input HypBuilder model csv file.

    Attributes:
        nested_samplers (list of :obj:): A list containing the Nested Sampler
            objects. Must call the gen_nested_samplers function build the
            Sampler instances.
        selection (pandas.DataFrame): The DataFrame containing the sorted
            set of models, including their name, log_evidence, and
            log_evidence_error values. The values are sorted in descending
            order by the log_evidence values. Only generated after calling
            the run_nested_sampling function.
        models

    """

    def __init__(self, models):
        """Inits ModelSelector."""
        self.nested_samplers = None
        self._nested_sample_its = None
        self.selection = None
        self.models = models
        return

    def load_models(self):
        """Does nothing.
        """
        pass

    def append_to_models(self, line):
        """Does nothing.
        """
        pass

    def number_of_models(self):
        """Number of models being tested.

        Returns:
            int: The number of models.

        """
        return len(self.models)
