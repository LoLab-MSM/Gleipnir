import numpy as np
try:
    import pysb
    from pysb.simulator import ScipyOdeSimulator
except ImportError as err:
    raise err
from gleipnir.pysb_utilities import HypSelector


if __name__ == '__main__':
    # The HypBuilder format model csv file.
    model_csv = 'grouped_reactions.csv'
    # The timespan of the simulations.
    tspan = np.linspace(0, 5, 20)
    # Define what ODE solver to use.
    solver = ScipyOdeSimulator
    # Load the data.
    data = np.load("model_0_AB_complex_data.npy")
    # Define the fancy indexer or mask for the time points that the data
    # corresponds to. -- In this case it is the last ten (out of 20) time points.
    data_time_idxs = np.array(list(range(len(tspan))))[10:]
    # Generate the observable data tuple for this observable: (data, data_sd, data_time_idxs)
    obs_data_t = tuple((data,None,data_time_idxs))
    # Generate the dictionary of observable data that is to be used in
    # computing the likelihood. -- Here we are just using the AB_complex
    # observable, which is the amount of A(B=1)%B(A=1).
    observable_data = dict()
    observable_data['AB_complex'] = obs_data_t

    # Build the HypSelector.
    selector = HypSelector(model_csv)
    # Check the number of models that were generated.
    n_models = selector.number_of_models()
    print("Generated {} models from input csv".format(n_models))
    # Append the needed observable to the model files
    obs_line = "Observable(\'AB_complex\',A(B=1)%B(A=1))"
    selector.append_to_models(obs_line)
    # quit()
    # Now let's construct the Nested Samplers for the models.
    # ns_version='gleipnir-classic' will use Gleipnir's built-in implementation
    # of the classic Nested Sampling algorithm.
    # ns_population_size=100 will set the active population size for the Nested
    # Sampling runs to 100.
    # log_likelihood_type='mse' will use the minus of the Mean Squared Error (mse)
    # as the log_likelihood estimator.
    selector.gen_nested_samplers(tspan, observable_data, solver=solver,
                                 ns_version='gleipnir-classic',
                                 ns_population_size=1000,
                                 log_likelihood_type='mse')
    #print(selector.nested_samplers[0])
    #print(selector.nested_sample_its[0]._data_mask)
    #quit()
    #selector.nested_samplers[0].run(verbose=True)
    #quit()
    # Do the Nested Sampling runs. -- The output is a pandas DataFrame.
    selections = selector.run_nested_sampling(nprocs=1)
    print(selections)
