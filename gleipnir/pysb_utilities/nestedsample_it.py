import importlib
import os.path
try:
    import pysb
    from pysb.simulator import ScipyOdeSimulator
except ImportError:
    pass
from scipy.stats import norm, uniform
import numpy as np
from gleipnir.sampled_parameter import SampledParameter


def is_numbers(inputString):
    return all(char.isdigit() for char in inputString)

def parse_directive(directive, priors, no_sample, Keq_sample):
    words = directive.split()
    if words[1] == 'prior':
        if is_numbers(words[2]):
            par_idx = int(words[2])
            par = model.parameters[par_idx].name
        else:
            par = words[2]
        priors[par] = words[3]
    elif words[1] == 'no-sample':
        if is_numbers(words[2]):
            par_idx = int(words[2])
            par = model.parameters[par_idx].name
        else:
            par = words[2]
        no_sample.append(par)
    elif words[1] == 'sample_Keq':
        if is_numbers(words[2]):
            par_idx = int(words[2])
            par_f = model.parameters[par_idx].name
        if is_numbers(words[3]):
            par_idx = int(words[3])
            par_r = model.parameters[par_idx].name
        if is_numbers(words[4]):
            par_idx = int(words[4])
            par_rep = model.parameters[par_idx].name
        no_sample.append(par_rep)
        Keq_sample.append((par_f, par_r, par_rep))
    return

def prune_no_samples(parameters, no_sample):
    pruned_pars = [parameter for parameter in parameters if parameter[0].name not in no_sample]

    return pruned_pars

def update_with_Keq_samples(parameters, Keq_sample):
    k_reps = [sample[2] for sample in Keq_sample]
    pruned_pars = [parameter for parameter in parameters if parameter[0].name not in k_reps]
    return pruned_pars

def write_norm_param(p_name, p_val):
    line = "sp_{} = SampledParameter(\'{}\', norm(loc=np.log10({}), scale=2.0))\n".format(p_name, p_name, p_val)
    return line

def write_uniform_param(p_name, p_val):
    line = "sp_{} = SampledParameter(\'{}\', uniform(loc=np.log10({})-1.0, scale=2.0))\n".format(p_name, p_name, p_val)
    return line

class NestIt(object):
    """Container to store parameters and priors for sampling.
    This object is geared towards flagging PySB model parameters at the level of
    model definition for Nested Sampling. However, it could be used outside
    model definition.

    Attributes:
        parms (dict of :obj:): A dictionary keyed to the parameter names. The
            values are the parameter priors as given in the call function.

    """

    def __init__(self):
        self.parms = dict()
        return

    def __call__(self, parameter, prior=None):
        """Add a parameter to the list.

        Args:
            parameter (:obj:pysb.Parameter): The parameter to be registered for
                Nested Sampling.
            prior (:obj:scipy.stats.RVS): The prior for the parameter. Should
                Should be defined using log10 scale. Default: None
                If None, a uniform prior centered on the parameter.value
                is used with a scale 4 orders of magnitude.
        """
        if prior is None:
            prior = uniform(loc=np.log10(parameter.value)-2.0, scale=4.0)
        self.parms[parameter.name] = prior
        return parameter

    def __getitem__(self, key):
        return self.parms[key]

    def __setitem__(self, key, prior):
        self.parms[key] = prior

    def __delitem__(self, key):
        del self.parm[key]

    def __contains__(self, key):
        return (key in self.names)

    def __iadd__(self, parm):
        self.__call__(parm)
        return self

    def __isub__(self, parm):
        try:
            name = parm.name
            self.__delitem__(name)
        except:
            self.__delitem__(parm)
        return self

    def names(self):
        return list(self.parms.keys())

    def keys(self):
        return self.parms.keys()

    def mask(self, model_parameters):
        names = self.names()
        return [(parm.name in names) for parm in model_parameters]

    def priors(self):
        return [self.parm[name] for name in self.parm.keys()]

    def sampled_parameters(self):
        return [SampledParameter(name, self.parm[name]) for name in self.keys()]


class NestedSampleIt(object):
    """Create instances of Nested Samling objects for PySB models.

    Args:
        model (pysb.Model): The instance of the PySB model that you want to run
            Nested Sampling on.
        observable_data (dict of tuple): Defines the observable data to
            use when computing the loglikelihood function. It is a dictionary
            keyed to the model Observables (or species names) that the
            data corresponds to. Each element is a 3 item tuple of format:
            (:numpy.array:data, None or :numpy.array:data_standard_deviations,
            None or :list like:time_idxs or :list like:time_mask).
        timespan (numpy.array): The timespan for model simulations.
        solver (:obj:): The ODE solver to use when running model simulations.
            Defaults to pysb.simulator.ScipyOdeSimulator.
        solver_kwargs (dict): Dictionary of optional keyword arguments to
            pass to the solver when it is initialized. Defaults to dict().
        nest_it (:obj:gleipnir.pysb_utilities.nestedsample_it.NestIt): An
            instance of the NestIt class with the data about the parameters to
            be sampled. If None (and builder is None), the default parameters to be sampled
            are the kinetic rate parameters with uniform priors of four orders of
            magnitude. Default: None
        builder (:obj:pysb.builder.Builder): An instance of the Builder class
            with the data about the parameters to be sampled. If None
            (and nest_it is None), the default parameters to be sampled
            are the kinetic rate parameters with uniform priors of four orders
            of magnitude. Default: None

    Attributes:
        model
        observable_data
        timespan
        solver
        solver_kwargs

    """
    def __init__(self, model, observable_data, timespan,
                 solver=pysb.simulator.ScipyOdeSimulator,
                 solver_kwargs=None, nest_it=None, builder=None):
        """Inits the NestedSampleIt."""
        if solver_kwargs is None:
            solver_kwargs = dict()
        self.model = model
        self.observable_data = observable_data
        self.timespan = timespan
        self.solver = solver
        self.solver_kwargs = solver_kwargs
        # self.ns_version = None
        self._ns_kwargs = None
        self._like_data = dict()
        self._data = dict()
        self._data_mask = dict()
        for observable_key in observable_data.keys():
            self._like_data[observable_key] = norm(loc=observable_data[observable_key][0],
                                               scale=observable_data[observable_key][1])
            self._data[observable_key] = observable_data[observable_key][0]
            self._data_mask[observable_key] = observable_data[observable_key][2]
            # print(observable_data[observable_key][2])
            if observable_data[observable_key][2] is None:
                self._data_mask[observable_key] = range(len(self.timespan))
        self._model_solver = solver(self.model, tspan=self.timespan, **solver_kwargs)
        if nest_it is not None:
            parm_mask = nest_it.mask(model.parameters)
            self._sampled_parameters = [SampledParameter(parm.name, nest_it[parm.name]) for i,parm in enumerate(model.parameters) if parm_mask[i]]
            self._rate_mask = parm_mask
        elif builder is not None:
            pnames = [parm.name for parm in builder.estimate_params]
            self._rate_mask = [(parm.name in pnames) for parm in model.parameters]
            self._sampled_parameters = [SampledParameter(parm.name, builder.priors[pnames.index(parm.name)]) for i,parm in enumerate(model.parameters) if self._rate_mask[i]]
        else:
            params = list()
            for rule in model.rules:
                if rule.rate_forward:
                     params.append(rule.rate_forward)
                if rule.rate_reverse:
                     params.append(rule.rate_reverse)
            rate_mask = [model.parameters.index(param) for param in params]
            self._sampled_parameters = [SampledParameter(param.name, uniform(loc=np.log10(param.value)-2.0, scale=4.0)) for i,param in enumerate(model.parameters) if i in rate_mask]
            self._rate_mask = rate_mask

        self._param_values = np.array([param.value for param in model.parameters])
        return


    def logpdf_loglikelihood(self, position):
        """Compute the loglikelihood using the normal distribution estimator.

        Args:
            position (numpy.array): The parameter vector the compute loglikelihood
                of.

        Returns:
            float: The natural logarithm of the likelihood estimate.

        """
        Y = np.copy(position)
        params = self._param_values.copy()
        params[self._rate_mask] = 10**Y
        sim = self._model_solver.run(param_values=[params]).all
        logl = 0.
        for observable in self._like_data.keys():
            sim_vals = sim[observable][self._data_mask[observable]]
            logl += np.sum(self._like_data[observable].logpdf(sim_vals))
        if np.isnan(logl):
            return -np.inf
        return logl

    def mse_loglikelihood(self, position):
        """Compute the loglikelihood using the negative mean squared error estimator.

        Args:
            position (numpy.array): The parameter vector the compute loglikelihood of.

        Returns:
            float: The natural logarithm of the likelihood estimate.

        """
        Y = np.copy(position)
        params = self._param_values.copy()
        params[self._rate_mask] = 10**Y
        sim = self._model_solver.run(param_values=[params]).all
        logl = 0.0
        for observable in self._like_data.keys():
            sim_vals = sim[observable][self._data_mask[observable]]
            logl -= np.mean((self._data[observable]-sim_vals)**2)
        if np.isnan(logl):
            return -np.inf
        return logl

    def sse_loglikelihood(self, position):
        """Compute the loglikelihood using the negative sum of squared errors estimator.

        Args:
            position (numpy.array): The parameter vector the compute loglikelihood
                of.

        Returns:
            float: The natural logarithm of the likelihood estimate.

        """
        Y = np.copy(position)
        params = self._param_values.copy()
        params[self._rate_mask] = 10**Y
        sim = self._model_solver.run(param_values=[params]).all
        logl = 0.0
        for observable in self._like_data.keys():
            sim_vals = sim[observable][self._data_mask[observable]]
            logl -= np.sum((self._data[observable]-sim_vals)**2)
        if np.isnan(logl):
            return -np.inf
        return logl

    def __call__(self, ns_version='gleipnir-classic',
                 ns_population_size=1000, ns_kwargs=None,
                 log_likelihood_type='logpdf'):
        """Call the NestedSampleIt instance to construct to instance of the NestedSampling object.

        Args:
                ns_version (str): Defines which version of Nested Sampling to use.
                    Options are 'gleipnir-classic'=>Gleipnir's built-in implementation
                    of the classic Nested Sampling algorithm, 'multinest'=>Use the
                    MultiNest code via Gleipnir, 'polychord'=>Use the PolyChord code
                    via Gleipnir, or 'dnest4'=>Use the DNest4 program via Gleipnir.
                    Defaults to 'gleipnir-classic'.
                ns_population_size (int): Set the size of the active population
                    of sample points to use during Nested Sampling runs.
                    Defaults to 1000.
                ns_kwargs (dict): Dictionary of any additional optional keyword
                    arguments to pass to NestedSampling object constructor.
                    Defaults to dict().
                log_likelihood_type (str): Define the type of loglikelihood estimator
                    to use. Options are 'logpdf'=>Compute the loglikelihood using
                    the normal distribution estimator, 'mse'=>Compute the
                    loglikelihood using the negative mean squared error estimator,
                    'sse'=>Compute the loglikelihood using the negative sum of
                     squared errors estimator. Defaults to 'logpdf'.

        Returns:
            type: Description of returned object.

        """
        if ns_kwargs is None:
            ns_kwargs = dict()
        # self.ns_version = ns_version
        self._ns_kwargs = ns_kwargs
        population_size = ns_population_size
        if log_likelihood_type == 'mse':
            loglikelihood = self.mse_loglikelihood
        elif log_likelihood_type == 'sse':
            loglikelihood = self.sse_loglikelihood
        else:
            loglikelihood = self.logpdf_loglikelihood
        if ns_version == 'gleipnir-classic':
            from gleipnir.nestedsampling import NestedSampling
            from gleipnir.nestedsampling.samplers import MetropolisComponentWiseHardNSRejection
            # from gleipnir.sampled_parameter import SampledParameter
            from gleipnir.nestedsampling.stopping_criterion import NumberOfIterations
            # population_size = 100*len(self._sampled_parameters)
            sampler = MetropolisComponentWiseHardNSRejection(iterations=10,
                                                             burn_in=10,
                                                             tuning_cycles=1)
            # Setup the stopping criterion for the NS run -- We'll use a fixed number of
            # iterations: 10*population_size
            stopping_criterion = NumberOfIterations(10*population_size)
            # Construct the Nested Sampler
            nested_sampler = NestedSampling(sampled_parameters=self._sampled_parameters,
                                loglikelihood=loglikelihood,
                                sampler=sampler,
                                population_size=population_size,
                                stopping_criterion=stopping_criterion)
            # self._nested_sampler = NS
        elif ns_version == 'multinest':
            from gleipnir.multinest import MultiNestNestedSampling
            # population_size = 100*len(self._sampled_parameters)
            nested_sampler = MultiNestNestedSampling(sampled_parameters=self._sampled_parameters,
                                           loglikelihood=loglikelihood,
                                           population_size=population_size,
                                           **self._ns_kwargs)
            #self._nested_sampler = MNNS
        elif ns_version == 'polychord':
            from gleipnir.polychord import PolyChordNestedSampling
            nested_sampler = PolyChordNestedSampling(sampled_parameters=self._sampled_parameters,
                                           loglikelihood=loglikelihood,
                                           population_size=population_size)
        elif ns_version == 'dnest4':
            from gleipnir.dnest4 import DNest4NestedSampling
            if not ('num_steps' in list(self._ns_kwargs.keys())):
                self._ns_kwargs['num_steps'] = 100*population_size
                # num_steps = 100*population_size
            nested_sampler = DNest4NestedSampling(sampled_parameters=sampled_parameters,
                                           loglikelihood=loglikelihood,
                                           population_size=population_size,
                                           **self._ns_kwargs)

        return nested_sampler


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', metavar='model_file', type=str, help='The model.py file that you want run NS on.')
    parser.add_argument('output_path', metavar='output_path', type=str, help='The file location where you want to output the NS run script.')
    args = parser.parse_args()
    # get the input script from command line imputs
    model_file = os.path.abspath(args.model_file)
    # get the location to dump the output
    output_path = os.path.abspath(args.output_path)

    print("Using model from file: {}".format(model_file))
    base=os.path.basename(model_file)
    model_module_name = os.path.splitext(base)[0]
    print("With name: {}".format(model_module_name))
    default_prior_shape = 'norm'
    print("The default prior shape is: {}".format(default_prior_shape))
    #print(model_file)

    #print(model_module_name)
    model_module = importlib.import_module(model_module_name)
    model = getattr(model_module, 'model')
    #print(model)
    priors = dict()
    no_sample = list()
    Keq_sample = list()
    #Read the file and parse any #NESTEDSAMPLE_IT directives
    print("Parsing the model for any #NESTEDSAMPLE_IT directives...")
    with open(model_file, 'r') as file_obj:
        for line in file_obj:
            words = line.split()
            if len(words) > 1:
                if words[0] == '#NESTEDSAMPLE_IT':
                    parse_directive(line, priors, no_sample, Keq_sample)

    #now we need to extract a list of kinetic parameters
    parameters = list()
    print("Inspecting the model and pulling out kinetic parameters...")
    for rule in model.rules:
        print(rule.rate_forward, rule.rate_reverse)
        #print(rule_keys)
        if rule.rate_forward:
            param = rule.rate_forward
            #print(param)
            parameters.append([param,'f'])
        if rule.rate_reverse:
            param = rule.rate_reverse
            #print(param)
            parameters.append([param, 'r'])
    #print(no_sample)
    parameters = prune_no_samples(parameters, no_sample)
    parameters = update_with_Keq_samples(parameters, Keq_sample)
    #print(parameters)
    print("Found the following kinetic parameters:")
    print("{}".format(parameters))
    #default the priors to norm - i.e. normal distributions
    for parameter in parameters:
        name = parameter[0].name
        if name not in priors.keys():
            priors[name] = default_prior_shape

    # Obtain mask of sampled parameters to run simulation in the likelihood function
    parameters_idxs = [model.parameters.index(parameter[0]) for parameter in parameters]
    rates_mask = [i in parameters_idxs for i in range(len(model.parameters))]
    param_values = [p.value for p in model.parameters]

    out_file = open("run_NS_"+base, 'w')

    print("Writing to Gleipnir NS run script: run_NS_{}".format(base))
    out_file.write("\'\'\'\nGenerated by nestedsample_it\n")
    out_file.write("Gleipnir NS run script for {} \n".format(base))
    out_file.write("\'\'\'")
    out_file.write("\n")
    out_file.write("from pysb.simulator import ScipyOdeSimulator\n")
    out_file.write("import numpy as np\n")
    out_file.write("from scipy.stats import norm,uniform\n")
    out_file.write("from gleipnir.nested_sampling import NestedSampling\n")
    out_file.write("from gleipnir.samplers import MetropolisComponentWiseHardNSRejection\n")
    out_file.write("from gleipnir.sampled_parameter import SampledParameter\n")
    out_file.write("from gleipnir.stopping_criterion import NumberOfIterations\n")

    #out_file.write("import inspect\n")
    #out_file.write("import os.path\n")
    out_file.write("from "+model_module_name+" import model\n")
    out_file.write("\n")

    out_file.write("# Initialize PySB solver object for running simulations.\n")
    out_file.write("# Simulation timespan should match experimental data.\n")
    out_file.write("tspan = np.linspace(0,10, num=100)\n")
    out_file.write("solver = ScipyOdeSimulator(model, tspan=tspan)\n")
    out_file.write("parameters_idxs = " + str(parameters_idxs)+"\n")
    out_file.write("rates_mask = " + str(rates_mask)+"\n" )
    out_file.write("param_values = np.array([p.value for p in model.parameters])\n" )
    out_file.write("\n")
    out_file.write("# USER must add commands to import/load any experimental\n")
    out_file.write("# data for use in the likelihood function!\n")
    out_file.write("experiments_avg = np.load()\n")
    out_file.write("experiments_sd = np.load()\n")
    out_file.write("like_data = norm(loc=experiments_avg, scale=experiments_sd)\n")
    out_file.write("# USER must appropriately update loglikelihood function!\n")
    out_file.write("def loglikelihood(position):\n")
    out_file.write("    Y=np.copy(position)\n")
    out_file.write("    param_values[rates_mask] = 10 ** Y\n")
    out_file.write("    sim = solver.run(param_values=param_values).all\n")
    out_file.write("    logp_data = np.sum(like_data.logpdf(sim['observable']))\n")
    out_file.write("    if np.isnan(logp_data):\n")
    out_file.write("        logp_data = -np.inf\n")
    out_file.write("    return logp_data\n")
    out_file.write("\n")
    #write the sampled params lines
    out_file.write("sampled_parameters = list()\n")
    for parameter in parameters:
        name = parameter[0].name
        value = parameter[0].value
        prior_shape = priors[name]
        print("Will sample parameter {} with {} prior around {}".format(name, prior_shape, value))
        if prior_shape == 'uniform':
            line = write_uniform_param(name, value)
            ps_name = line.split()[0]
            out_file.write(line)
            out_file.write("sampled_parameters.append({})\n".format(ps_name))
        else:
            line = write_norm_param(name, value)
            ps_name = line.split()[0]
            out_file.write(line)
            out_file.write("sampled_parameters.append({})\n".format(ps_name))

    out_file.write("# Setup the Nested Sampling run\n")
    out_file.write("n_params = len(sampled_parameters)\n")
    out_file.write("population_size = 10*n_params\n")
    out_file.write("# Setup the sampler to use when updating points during the NS run --\n")
    out_file.write("# Here we are using an implementation of the Metropolis Monte Carlo algorithm\n")
    out_file.write("# with component-wise trial moves and augmented acceptance criteria that adds a\n")
    out_file.write("# hard rejection constraint for the NS likelihood boundary.\n")
    out_file.write("sampler = MetropolisComponentWiseHardNSRejection(iterations=500, burn_in=100)\n")
    out_file.write("# Setup the stopping criterion for the NS run -- We'll use a fixed number of\n")
    out_file.write("# iterations: 10*population_size\n")
    out_file.write("stopping_criterion = NumberOfIterations(10*population_size)\n")
    out_file.write("# Construct the Nested Sampler\n")
    out_file.write("NS = NestedSampling(sampled_parameters=sampled_parameters,\n")
    out_file.write("                    loglikelihood=loglikelihood, sampler=sampler,\n")
    out_file.write("                    population_size=population_size,\n")
    out_file.write("                    stopping_criterion=stopping_criterion)\n")
    out_file.write("# run it\n")
    out_file.write("NS.run()\n")
    out_file.write("# Retrieve the evidence\n")
    out_file.write("evidence = NS.evidence()\n")
    out_file.write("print(\"evidence: \",evidence)\n")
    out_file.write("print(\"log_evidence: \", np.log(evidence))\n")


    out_file.close()
    print("nestedsample_it is complete!")
    print("END OF LINE.")
