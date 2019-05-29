"""
Plot the posterior estimates generated for JARM parameters from Nested Sampling.

Adapted from the pars_dists_plot.py script at:
https://github.com/LoLab-VU/JARM/blob/master/model_analysis/pars_dists_plot.py
"""

import numpy as np
from jnk3_no_ask1 import model

idx_pars_calibrate = [1, 5, 9, 11, 15, 17, 19, 23, 25, 27, 31, 35, 36, 37, 38, 39, 41, 43] #pydream3

ndims = len(idx_pars_calibrate)
colors = sns.color_palette(n_colors=ndims)
rows = 6
columns = 3
counter = 0


import seaborn as sns
import matplotlib.pyplot as plt
f, axes = plt.subplots(rows, columns, figsize=(7, 7), sharex=True)
for r in range(rows):
    for c in range(columns):
        weights = np.load("post_multinest_marginal_weights_parm_"+str(counter)+".npy")
        edges = np.load("post_multinest_marginal_edges_parm_"+str(counter)+".npy")
        centers = np.load("post_multinest_marginal_centers_parm_"+str(counter)+".npy")
        axes[r, c].hist(centers, bins=edges, color=colors[counter], weights=weights)
        #axes[r, c].hist(centers, bins=50, color=colors[counter], weights=weights, density=True)
        axes[r, c].set_title(model.parameters[idx_pars_calibrate[counter]].name, fontdict={'fontsize':8})
        # axes[r, c].set_xlim(-6, 6)
        counter += 1

        if counter > len(idx_pars_calibrate):
            break
f.add_subplot(111, frameon=False)
f.subplots_adjust(wspace=0.4)
f.subplots_adjust(hspace=0.5)
# hide tick and tick label of the big axes
plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
plt.grid(False)
plt.xlabel("Log(Parameter value)", fontsize=14)
plt.ylabel("Probability", fontsize=14, labelpad=15)

# plt.show()
plt.savefig('pars_post_dist_plot.pdf', format='pdf', bbox_inches="tight")
