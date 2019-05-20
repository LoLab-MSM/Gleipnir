"""
Nesting Sampling of the JARM (https://github.com/LoLab-VU/JARM) PySB model
parameters using PolyChord via Gleipnir.

Adapted from the calibrate_pydream.py script at:
https://github.com/LoLab-VU/JARM/blob/master/model_analysis/calibrate_pydream.py

"""
import numpy as np
from pysb.simulator import ScipyOdeSimulator
from scipy.stats import norm, halfnorm, uniform
from jnk3_no_ask1 import model
import pandas as pd
from equilibration_function import pre_equilibration

#NS
from gleipnir.sampled_parameter import SampledParameter
from gleipnir.dnest4 import DNest4NestedSampling



# Initialize PySB solver

exp_data = pd.read_csv('./data/exp_data_3min.csv')

tspan = np.linspace(0, exp_data['Time (secs)'].values[-1], 181)
t_exp_mask = [idx in exp_data['Time (secs)'].values[:] for idx in tspan]

solver = ScipyOdeSimulator(model, tspan=tspan)

like_mkk4_arrestin_pjnk3 = norm(loc=exp_data['pTyr_arrestin_avg'].values,
                                scale=exp_data['pTyr_arrestin_std'].values)

like_mkk7_arrestin_pjnk3 = norm(loc=exp_data['pThr_arrestin_avg'].values,
                                scale=exp_data['pThr_arrestin_std'].values)

like_mkk4_noarrestin_pjnk3 = norm(loc=exp_data['pTyr_noarrestin_avg'].values[4:],
                                scale=exp_data['pTyr_noarrestin_std'].values[4:])
like_mkk4_noarrestin_pjnk3_04 = halfnorm(loc=exp_data['pTyr_noarrestin_avg'].values[:4],
                                       scale=exp_data['pTyr_noarrestin_std'].values[:4])
like_mkk7_noarrestin_pjnk3 = norm(loc=exp_data['pThr_noarrestin_avg'].values,
                                scale=exp_data['pThr_noarrestin_std'].values)

like_thermobox = norm(loc=1, scale=1e-0)

# Add PySB rate parameters to be sampled as unobserved random variables to
# NS with normal priors

## New kds in jnk3 mkk4/7

idx_pars_calibrate = [1, 5, 9, 11, 15, 17, 19, 23, 25, 27, 31, 35, 36, 37, 38, 39, 41, 43] #pydream3

rates_of_interest_mask = [i in idx_pars_calibrate for i, par in enumerate(model.parameters)]

# Index of Initial conditions of Arrestin
arrestin_idx = [44]
jnk3_initial_value = 0.6  # total jnk3
jnk3_initial_idxs = [47, 48, 49]
kcat_idx = [36, 37]

param_values = np.array([p.value for p in model.parameters])

sampled_parameters = [SampledParameter(i, uniform(loc=np.log10(5E-8), scale=np.log10(1.9E3)-np.log10(5E-8)))
                           for i,pa in enumerate(param_values[rates_of_interest_mask])]

# # We calibrate the pMKK4 - Arrestin-3 reverse reaction rate. We have experimental data
# # for this interaction and know that the k_r varies from 160 to 1068 (standard deviation)
sampled_parameters[0] = SampledParameter(0, uniform(loc=np.log10(120), scale=np.log10(1200)-np.log10(120)))
sampled_parameters[6] = SampledParameter(6, uniform(loc=np.log10(28), scale=np.log10(280)-np.log10(28)))


def evaluate_cycles(pars1):
    boxes = np.zeros(4)
    box1 = (pars1[21]/pars1[20]) * (pars1[23]/pars1[22]) * (1 / (pars1[1] / pars1[0])) * \
           (1 / (pars1[5]/pars1[4]))

    box2 = (pars1[21] / pars1[20]) * (pars1[25] / pars1[24]) * (1 / (pars1[3] / pars1[2])) * \
           (1 / (pars1[27] / pars1[26]))

    box3 = (pars1[13] / pars1[12]) * (pars1[23] / pars1[22]) * (1 / (pars1[1] / pars1[0])) * \
           (1 / (pars1[15] / pars1[14]))

    box4 = (pars1[7] / pars1[6]) * (pars1[25] / pars1[24]) * (1 / (pars1[3] / pars1[2])) * \
           (1 / (pars1[11] / pars1[10]))

    boxes[0] = box1
    boxes[1] = box2
    boxes[2] = box3
    boxes[3] = box4
    return boxes

def loglikelihood(position):
    Y = np.copy(position)
    param_values[rates_of_interest_mask] = 10 ** Y

    pars1 = np.copy(param_values)
    pars2 = np.copy(param_values)
    # thermoboxes
    boxes = evaluate_cycles(pars1)

    logp_boxes = like_thermobox.logpdf(boxes).sum()
    if np.isnan(logp_boxes):
        logp_boxes = -np.inf

    # Pre-equilibration
    time_eq = np.linspace(0, 100, 100)
    pars_eq1 = np.copy(param_values)
    pars_eq2 = np.copy(param_values)

    pars_eq2[arrestin_idx] = 0
    pars_eq2[jnk3_initial_idxs] = [0.592841488, 0, 0.007158512]

    all_pars = np.stack((pars_eq1, pars_eq2))
    all_pars[:, kcat_idx] = 0  # Setting catalytic reactions to zero for pre-equilibration
    try:
        eq_conc = pre_equilibration(model, time_eq, all_pars)[1]
    except:
        logp_total = -np.inf
        return logp_total


    # Simulating models with initials from pre-equilibration and parameters for condition with/without arrestin
    pars2[arrestin_idx] = 0
    pars2[jnk3_initial_idxs] = [0.592841488, 0, 0.007158512]
    sim = solver.run(param_values=[pars1, pars2], initials=eq_conc).all
    logp_mkk4_arrestin = np.sum(like_mkk4_arrestin_pjnk3.logpdf(sim[0]['pTyr_jnk3'][t_exp_mask] / jnk3_initial_value))
    logp_mkk7_arrestin = np.sum(like_mkk7_arrestin_pjnk3.logpdf(sim[0]['pThr_jnk3'][t_exp_mask] / jnk3_initial_value))

    # No arrestin simulations/experiments

    logp_mkk4_noarrestin = np.sum(like_mkk4_noarrestin_pjnk3.logpdf(sim[1]['pTyr_jnk3'][t_exp_mask][4:] / jnk3_initial_value))
    logp_mkk4_noarrestin_04 = np.sum(like_mkk4_noarrestin_pjnk3_04.logpdf(sim[1]['pTyr_jnk3'][t_exp_mask][:4] / jnk3_initial_value))
    logp_mkk7_noarrestin = np.sum(like_mkk7_noarrestin_pjnk3.logpdf(sim[1]['pThr_jnk3'][t_exp_mask] / jnk3_initial_value))
    logp_mkk4_noarrestin_total = logp_mkk4_noarrestin + logp_mkk4_noarrestin_04


    # If model simulation failed due to integrator errors, return a log probability of -inf.
    logp_total = logp_mkk4_arrestin + logp_mkk7_arrestin + logp_mkk4_noarrestin_total + \
                 logp_mkk7_noarrestin + logp_boxes
    if np.isnan(logp_total):
        logp_total = -np.inf

    return logp_total


if __name__ == '__main__':

    # Setup the Nested Sampling run
    n_params = len(sampled_parameters)
    print("Sampling a total of {} parameters".format(n_params))
    population_size = 100
    print("Will use NS population size of {}".format(population_size))
    # Construct the Nested Sampler
    DNS = DNest4NestedSampling(sampled_parameters=sampled_parameters,
                               loglikelihood=loglikelihood,
                               population_size=population_size,
                               n_diffusive_levels=10,
                               num_steps=1000,
                               num_per_step=100)

    # Launch the Nested Sampling run.
    log_evidence, log_evidence_error = DNS.run()
    # Print the output
    print("log_evidence: {} +- {} ".format(log_evidence, log_evidence_error))
    # Get the estimates of the posterior probability distrbutions of the
    # parameters.
    posteriors = DNS.posteriors()
    # Save the posterior estimates.
    for parm in posteriors.keys():
        marginal, centers = posteriors[parm]
        np.save("post_multinest_marginal_weights_parm_{}.npy".format(parm), marginal, allow_pickle=False)
        np.save("post_multinest_marginal_centers_parm_{}.npy".format(parm), centers, allow_pickle=False)

    # Try plotting a marginal distribution
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
        # Get the posterior distributions -- the posteriors are return as dictionary
        # keyed to the names of the sampled paramters. Each element is a histogram
        # estimate of the marginal distribution, including the heights and centers.

        # Lets look at the first paramter
        marginal, centers = posteriors[list(posteriors.keys())[0]]
        # Plot with seaborn
        sns.distplot(centers, bins=centers, hist_kws={'weights':marginal})
        # Uncomment next line to plot with plt.hist:
        # plt.hist(centers, bins=centers, weights=marginal)
        plt.show()
    except ImportError:
        pass
