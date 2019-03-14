# Gleipnir  

Gleipnir is a python toolkit that provides an easy to use interface for Nested Sampling that is similar to calibration tools such as [PyDREAM](https://github.com/LoLab-VU/PyDREAM) and [SimplePSO](https://github.com/LoLab-VU/ParticleSwarmOptimization).
In addition to its built-in implementation of the classic Nested Sampling algorithm, Gleipnir provides a common interface to the major Nested Sampling implementations such as MultiNest, PolyChord, and DNest4.

Gleipnir can be used to compute the Bayesian evidence of models with respect to the parameter priors and the likelilood of parameter vectors dictated by the (calibration) data, allowing users to perform model selection. As a side-effect of the evidence calculation, estimates of the posterior distributions of the parameters can also be generated; therefore, Gleipnir can also be used for Bayesian model calibration.

### What is Nested Sampling?

Nested Sampling is a numerical integration scheme to estimate the Bayesian evidence (i.e., the normalization factor or denominator in Bayes formula for probability distributions).

In particular, Nested Sampling was
designed to handle cases where the evidence integral is high-dimensional and the likelihood is exponentially localized in the probability mass of the prior distribution of the sampled dimensions. In the Nested Sampling approach, the evidence is first converted from a (possibly) multi-dimensional integral into a one-dimensional integral taken over a mapping of the likelihood function to elements of the unit prior probability mass (X). In principle, this is achieved by using a top-down
approach in which sample points are drawn according to the prior distribution, and the unit prior probability is subdivided
into equal fractional elements from X = 1 down to X = 0 and
mapped to the likelihood function, L(X), via a likelihood sorting routine.

The Nested Sampling method was originally developed by John Skilling; see the following references:
  1. Skilling, John. "Nested sampling." AIP Conference Proceedings. Vol.
    735. No. 1. AIP, 2004.
  2. Skilling, John. "Nested sampling for general Bayesian computation."
    Bayesian analysis 1.4 (2006): 833-859.
  3. Skilling, John. "Nested sampling’s convergence." AIP Conference
    Proceedings. Vol. 1193. No. 1. AIP, 2009.

------

# Install

| **! Warning** |
| :--- |
|  Gleipnir is still under heavy development and may rapidly change. |

## Gleipnir run dependencies
Gleipnir has the following core dependencies:
   * NumPy - http://www.numpy.org/
   * SciPy - https://www.scipy.org/
   * pandas - https://pandas.pydata.org/

### To use PolyChord
   * pypolychord - https://github.com/PolyChord/PolyChordLite

### To use MultiNest
   * PyMultiNest - https://github.com/JohannesBuchner/PyMultiNest
   * MultiNest -  https://github.com/JohannesBuchner/MultiNest

### To use DNest4
   * DNest4 - https://github.com/eggplantbren/DNest4

Gleipnir is compatible with Python 3.6

The following section describes the process for setting up the dependencies using a conda environment.

## Setup and install using Anaconda's conda tool

First, clone or download the GitHub repo
```
git clone https://github.com/LoLab-VU/Gleipnir.git
```
Then create a new conda environment for Gleipnir and activate it:
```
conda create --name gleipnir python=3.6
conda activate gleipnir
```

Then install all the run dependencies:
```
conda install numpy scipy pandas
```
There is currently no installer for Gleipnir, so just add the repo to your PYTHONPATH.

If you want to run pysb models:
```
conda install -c alubbock pysb
```

### PolyChord
If you want to use the PolyChordNestedSampling object for the Nested Sampling runs then download, build, and install pypolychord following instructions in the README at:

https://github.com/PolyChord/PolyChordLite

Notes:
 * Installs into your .local/lib python site-packages.
 * Requires gfortran (f77 compiler) and lipopenmpi-dev (development libraries for MPI) to build the code.

### MultiNest
If you want to use the MultiNestNestedSampling object for Nested Sampling download, build, and install PyMultiNest and MultiNest following the instructions at:
http://johannesbuchner.github.io/PyMultiNest/install.html

### DNest4
If you want use the DNest4NestedSampling object for Nested Sampling then download,
build, and install DNest4 and its Python bindings following the instructions in the README at:
https://github.com/eggplantbren/DNest4

Notes:
 * Requires a c++ compiler with c++11 standard libraries.
 * Requires Cython and numba for python bindings to compile and install

------

# Usage

Checkout the [examples](Gleipnir/examples) to see how to setup Nested Sampling runs using Gleipnir.

------

# Utilities
## nestedsample_it

nestedsample_it is a utility that helps generate a template Nested Sampling run script for a PySB model. nestedsample_it reads the model file, imports and pulls out all the kinetic parameters, and then writes out a run_NS script for that model. nestedsample_it currently writes out a run script for classic Nested Sampling via Gleipnir, so you'll need to modify it to use one of the other Nested Samplers (MultiNest, PolyChord, or DNest4). And you will need to edit the run script to load any data and modify the loglikelihood function, but nestedsample_it should give you a good starting point.

Run nestedsample_it from the command line:
```
python -m glepnir.pysb_utilities.nestedsample_it model.py
```      
