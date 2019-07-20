# Gleipnir

![Python version badge](https://img.shields.io/badge/python-3.6-blue.svg)
[![license](https://img.shields.io/github/license/LoLab-VU/Gleipnir.svg)](LICENSE)
![version](https://img.shields.io/badge/version-0.25.0-orange.svg)
[![release](https://img.shields.io/github/release-pre/LoLab-VU/Gleipnir.svg)](https://github.com/LoLab-VU/Gleipnir/releases/tag/v0.18.0)
[![anaconda cloud](https://anaconda.org/blakeaw/gleipnir/badges/version.svg)](https://anaconda.org/blakeaw/gleipnir)
[![DOI](https://zenodo.org/badge/173688080.svg)](https://zenodo.org/badge/latestdoi/173688080)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/e117a46ae8b241539742ab00f8cd1b38)](https://www.codacy.com/app/blakeaw1102/Gleipnir?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=LoLab-VU/Gleipnir&amp;utm_campaign=Badge_Grade)

<p align="center">
  <img width="100" height="100" src="./images/gleipnir_logo_2.png">
</p>

Gleipnir is a python toolkit that provides an easy to use interface for Bayesian parameter inference and model selection using Nested Sampling. It has a built-in implementation of the Nested Sampling algorithm but also provides a common interface to the Nested Sampling implementations MultiNest, PolyChord, dyPolyChord, DNest4, and Nestle.
Although Gleipnir provides a general framework for running Nested Sampling simulations, it was created with biological models in mind. It therefore supplies additional tools for working with biological models in the PySB format (see the PySB Utilities section). Likewise, Gleipnir's API was designed to be familiar to users of [PyDREAM](https://github.com/LoLab-VU/PyDREAM) and [SimplePSO](https://github.com/LoLab-VU/ParticleSwarmOptimization), which are primarily used for biological model calibration.

### What is Nested Sampling?

Nested Sampling is a numerical integration scheme for estimating the marginal likelihood, or in Nested Sampling parlance, the 'evidence' of high-dimensional models.
As a side-effect of the evidence calculation, estimates of the posterior probability distributions of model parameters can also be generated.   

In particular, Nested Sampling was
designed to handle evaluate the evidence of high-dimensional models where the likelihood is exponentially localized in the prior probability mass. In the Nested Sampling approach, the evidence is first converted from a (possibly) multi-dimensional integral into a one-dimensional integral taken over a mapping of the likelihood function to elements of the unit prior probability mass (X). In principle, this is achieved by using a top-down
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

Gleipnir installs as the `gleipnir` package. It is compatible with Python 3.6.

#### conda install
Although not absolutely required, we recommend using the [Anaconda](https://www.anaconda.com/) Python distribution and the [conda](https://conda.io/en/latest/) package manager.

`gleipnir` can be installed from the terminal using `conda`:
```
conda intall -c blakeaw gleipnir
```
Note that `gleipnir` has the following core dependencies which will also be installed:
   * [NumPy](http://www.numpy.org/)
   * [SciPy](https://www.scipy.org/)
   * [pandas](https://pandas.pydata.org/)

Alternatively, for convenience, a `gleipnir` environment can be downloaded/created that has `gleipnir`, its core dependencies, as well as several optional/recommended packages; the optional/recommended packages include `pysb`, `hypbuilder`, `matplotlib`, `seaborn`, and `jupyter`.
From the terminal:
```
conda env create blakeaw/gleipnir
```
and then activate it with:
```
conda activate gleipnir
```

Additionally, there is another `gleipnir` environment for linux-64 that can be downloaded/created that has `gleipnir`, its core dependencies, as well as most of the recommended additional software packages; note that the versions of packages are pinned to exact version numbers in this environment file.
From the terminal:
```
conda env create blakeaw/gleipnir-all-linux64
```
and then activate it with:
```
conda activate gleipnir
```

#### pip install
You can install the `gleipnir` package using `pip` sourced from the GitHub repo:
```
pip install -e git+https://github.com/LoLab-VU/Gleipnir@v0.18.0#egg=gleipnir
```
However, this will not automatically install the core dependencies. You will have to do that separately:
```
pip install numpy scipy pandas
```

### Recommended additional software

The following software is not required for the basic operation of Gleipnir, but provides extra capabilities and features when installed.

#### PySB
[PySB](http://pysb.org/) is needed to run PySB models and it is needed if you want to use the gleipnir.pysb_utilities module:
```
conda install -c alubbock pysb
```

#### HypBuilder
If you want use the HypSelector class from gleipnir.pysb_utilities then you
need to have [HypBuilder](https://github.com/LoLab-VU/HypBuilder):
```
conda install -c blakeaw hypbuilder
```

#### Jupyter
If you want to run the Jupyter IPython notebooks that come with Gleipnir then you need to install [Jupyter](https://jupyter.org/):
```
conda install jupyter
```

#### Plotting packages:
We recommend installing [Matplotlib](https://matplotlib.org/) and [seaborn](https://seaborn.pydata.org/) to generate plots. Note that some of the Gleipnir examples will use these packages if they are installed to generate sample plots. Matplotlib is also needed for one of the Jupyter notebooks.
```
conda install matplotlib seaborn
```

#### MultiNest
If you want to run Nested Sampling simulations using Gleipnir's MultiNest interface class object, MultiNestNestedSampling (from the gleipnir.multinest module), then you will need to install [PyMultiNest](https://github.com/JohannesBuchner/PyMultiNest) and [MultiNest](https://github.com/JohannesBuchner/MultiNest). Build and install instructions for getting PyMultiNest and MultiNest from source can be found at:
http://johannesbuchner.github.io/PyMultiNest/install.html

PyMultiNest is available on PyPI:
```
pip install pymultinest
```
Note that in addition to MultiNest, `pymultinest` requires `numpy`, `scipy`, and `matplotlib` to run. It also optionally requires `mpi4py` to run MultiNest with MPI parallelization.

You can get a linux-64 conda build of MultiNest from the [blakeaw conda channel](https://anaconda.org/blakeaw/multinest):
```
conda install -c blakeaw multinest
```
Note that this conda build of MultiNest requires packages from the `anaconda` and `conda-forge` channels, so you'll need to add them to the channel list in your conda config (.condarc) file. You can also install a build of `mpi4py` that is compatible with this build of `multinest` from the [blakeaw conda channel](https://anaconda.org/blakeaw/mpi4py):
```
conda install -c blakeaw mpi4py
```

Additionally, a separate set of third party instructions for building and installing on Mac OS can be found at:
http://astrobetter.com/wiki/MultiNest+Installation+Notes

Also, this PyMultiNest GitHub issue may be helpful if you run into library path problems on Mac OS:
https://github.com/JohannesBuchner/PyMultiNest/issues/89

#### PolyChord
If you want run Nested Sampling simulations using [PolyChord](https://github.com/PolyChord/PolyChordLite) via the
PolyChordNestedSampling class from the gleipnir.polychord, then you will need to install pypolychord (for PolyChordLite version >= 1.16). Build and install instructions are in the README at:
https://github.com/PolyChord/PolyChordLite

However, as per [PolyChordLite GitHub Issue 11](https://github.com/PolyChord/PolyChordLite/issues/11) there is a version of pypolychord on PyPI which should work for linux-64:
```
pip install pypolychord
```
But note that the current version of pypolychord on PyPI (as of 07-01-2019) is not the most recent version, and some of the extra functionality provided by Gleipnir will not work with it.

Special Notes for builds from source on linux-64:
 * Installs into your .local/lib python site-packages.
 * Requires gfortran (f77 compiler) and lipopenmpi-dev (development libraries for MPI) to build the code.

#### dyPolyChord
If you want to run Nested Sampling simulations using
[dyPolyChord](https://github.com/ejhigson/dyPolyChord) using Gleipnir's interface object, dyPolyChordNestedSampling (from the gleipnir.dypolychord module), then you will need to install dyPolyChord (available on PyPI):
```
pip install dyPolyChord
```
Note that dyPolyChord requires PolyChord to run, so its use via Gleipnir requires the pypolychord package; see the the previous section. Also note that in addition to PolyChord, `dyPolyChord` requires `numpy`, `scipy`, and `nestcheck` to run. It also optionally requires `mpi4py` to run with MPI parallelization.
For additional information check out the [dyPolyChord documentation](https://dypolychord.readthedocs.io/en/latest/index.html).

#### DNest4
If you want run Nested Sampling simulations using [DNest4](https://github.com/eggplantbren/DNest4) via the DNest4NestedSampling class from the gleipnir.dnest4 module, then you will need to get DNest4 and its Python bindings. Instructions for building and installing from source can be found in the README at:
https://github.com/eggplantbren/DNest4

Additionally, a linux-64 conda build of dnest4 can be installed from
the [blakeaw conda channel](https://anaconda.org/blakeaw/dnest4):
```
conda install -c blakeaw dnest4
```

Special Notes for building and installing from source:
 * Requires a c++ compiler with c++11 standard libraries.
 * Requires Cython and numba for python bindings to compile and install

#### Nestle
If you want to run Nested Sampling simulations using
[Nestle](https://github.com/kbarbary/nestle) via Gleipnir's interface object, NestleNestedSampling (from the gleipnir.nestle module), then you will need to install Nestle (available on PyPI):
```
pip install nestle
```
Note that Nestle requires `numpy` to run (also required for gleipnir), and it also optionally requires `scipy`.

For additional information check out the [Nestle documentation](http://kylebarbary.com/nestle/).

------

# License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

------

# Documentation and Usage

Checkout the Jupyter Notebooks (more in the pipeline):
 1. [Intro to Nested Sampling with Gleipnir](./jupyter_notebooks/Intro_to_Nested_Sampling_with_Gleipnir.ipynb)
 2. [Nested Sampling Classes](./jupyter_notebooks/Nested_Sampling_classes.ipynb)
 3. [HypSelector Example](./jupyter_notebooks/HypSelector_example.ipynb)
 4. [ModelSelector Example](./jupyter_notebooks/ModelSelector_example.ipynb)

Also checkout the [examples](./examples) to see example scripts that show how to setup Nested Sampling runs using Gleipnir.

------

# Contact

To report problems or bugs please open a
[GitHub Issue](https://github.com/LoLab-VU/Gleipnir/issues). Additionally, any
comments, suggestions, or feature requests for Gleipnir can also be submitted as a
[GitHub Issue](https://github.com/LoLab-VU/Gleipnir/issues).

------

# PySB Utilities

## nestedsample_it
nestedsample_it is a utility that helps generate a Nested Sampling run script or NestedSampling objects for a PySB model.

### Commmand line use
nestedsample_it can be used as a command line utility to generate a template Nested Sampling run script for a PySB model. nestedsample_it reads the model file, imports and pulls out all the kinetic parameters, and then writes out a run_NS script for that model. nestedsample_it currently writes out a run script for classic Nested Sampling via Gleipnir, so you'll need to modify it to use one of the other Nested Samplers (MultiNest, PolyChord, or DNest4). And you will need to edit the run script to load any data and modify the loglikelihood function, but nestedsample_it should give you a good starting point.

Run nestedsample_it from the command line with following format:
```
python -m glepnir.pysb_utilities.nestedsample_it model.py output_path
```      
where output_path is the directory/folder location where you want the generated script
to be saved.

The command line version of nestedsample_it also has support for a limited set of #NESTEDSAMPLE_IT directives which can be added to model files. The current directives are:
  * #NESTEDSAMPLE_IT prior [param_name, param_index] [norm, uniform]  
    * Specify the type of prior to assign to a parameter. The parameter can either be specified by its name or its index (in model.parameters). The priors that can be assigned are either norm or uniform; note that uniform is the default for all parameters.  
  * #NESTEDSAMPLE_IT no-sample [param_name, param_index]
     * Specify a fixed parameter (i.e., not to included in sampling). The parameter can either be specified by its name or its index (in model.parameters).          


### Progammatic use via the NestedSampleIt class
The nestedsample_it utility can be used progammatically via the NestedSampleIt
class. It's importable from the pysb_utilities module:
```python
from gleipnir.pysb_utilities import NestedSampleIt
```
The NestedSampleIt class can build an instance of a NestedSampling object.  
 Here's a faux minimal example:
```python
from my_pysb_model import model as my_model
from gleipnir.pysb_utilities import NestedSampleIt
import numpy as np

timespan = np.linspace(0., 10., 10)
data = np.load('my_data.npy')
data_sd = np.load('my_data_sd.npy')
observable_data = dict()
time_idxs = list(range(len(timespan)))
observable_data['my_observable'] = (data, data_sd, time_idxs)
# Initialize the NestedSampleIt instance with the model details.
sample_it = NestedSampleIt(my_model, observable_data, timespan)
# Now build the NestedSampling object. -- All inputs are
# optional keyword arguments.
nested_sampler = sample_it(ns_version='gleipnir-classic',
                           ns_population_size=100,
                           ns_kwargs=dict(),
                           log_likelihood_type='logpdf')
# Then you can run the nested sampler.
log_evidence, log_evidence_error = nested_sampler.run()
```

NestedSampleIt constructs the NestedSampling object to sample all of a model's kinetic rate parameters. It assumes that the priors are uniform with size 4 orders of magnitude and centered on the values defined in the model.

In addition, NestedSampleIt crrently has three pre-defined loglikelihood functions with different estimators. They can be specified with the keyword parameter log_likelihood_type:
```python
# Now build the NestedSampling object.
nested_sampler = sample_it(log_likelihood_type='logpdf')
```
The options are
  * 'logpdf'=>Compute the loglikelihood using the
normal distribution estimator
  * 'mse'=>Compute the loglikelihood using the
negative mean squared error estimator
  * 'sse'=>Compute the loglikelihood using
the negative sum of squared errors estimator.
The default is 'logpdf'.
Each of these functions computes the loglikelihood estimate using the timecourse output of a model simulation for each observable defined in the `observable_data` dictionary.
If you want to use a different or more complicated likelihood function with NestedSampleIt then you'll need to subclass it and override one of the existing loglikelihood functions.  

#### NestIt
The nestedsample_it module has a built-in helper class, NestIt, which can be used in conjunction of with NestedSampleIt class. NestIt can be used at the level of PySB model definition to log which parameters to include in
a Nested Sampling run. It can be imported from the pysb_utilities module:
```python
from gleipnir.pysb_utilities import NestIt
```
It is passed at instantiation to the NestedSampleIt class, which uses it
to build the sampled parameters list and parameter mask for the likelihood
function.
See the following example files:

   * [dimerization_model_nestit](./examples/pysb_dimerization_model/dimerization_model_nestit.py) - example model definition using NestIt to flag parameters.
   * [run_NS_NestedSampleIt_NestIt_dimerization_model](./examples/pysb_dimerization_model/run_NS_NestedSampleIt_NestIt_dimerization_model.py) - example use of NestIt with NestedSampleIt.

Note that if you flag a parameter for sampling without setting a prior, NestIt will by default assign the parameter a uniform prior centered on the parameter's value with a width of 4 orders of magnitude.  

#### Builder class from pysb.builder

The Builder class from pysb.builder can also be used in conjunction with the NestedSampleIt class. The Builder class itself is a wrapper class that can be used to construct a PySB model and set parameter priors, logging them for sampling. Although
this feature was originally intended for use with the BayesSB package, the NestedSampleIt class supports it as a logger for sampled parameters.
The instance of the Builder is passed at instantiation to the NestedSampleIt class, which uses it to build the sampled parameters list and parameter mask for the likelihood function.
See the following example files:

   * [dimerization_model_builder](./examples/pysb_dimerization_model/dimerization_model_builder.py) - example model definition using Builder to construct a PySB model and flag parameters for sampling.
   * [run_NS_NestedSampleIt_Builder_dimerization_model](./examples/pysb_dimerization_model/run_NS_NestedSampleIt_Builder_dimerization_model.py) - example use of Builder with NestedSampleIt.

Note that you have to explicitly set a prior for each parameter that you want to sample when you add it your model with the builder.parameter function. If no
prior is given the parameter won't be included as a sampled parameter in the Nested Sampling run.

## HypSelector

HypSelector is a tool for hypothesis selection using [HypBuilder](https://github.com/LoLab-VU/HypBuilder) and Nested Sampling-based model selection. Models embodying different hypotheses (e.g., optional reactions) can be defined using the HypBuilder csv syntax. HypSelector then allows users to easily compare all the hypothetical model variants generated by HypBuilder by performing Nested Sampling to compute their evidences and thereby do model selection; HypSelector also provides functionality to estimate Bayes factors from the evidence estimates, as well as estimators for the Akaike, Bayesian, and Deviance information criteria computed from the Nested Sampling outputs. See the [grouped reactions example](./examples/HypSelector/grouped_reactions) or the [HypSelector Example Jupyter Notebook](./jupyter_notebooks/HypSelector_example.ipynb) to see example usage of HypSelector.

## ModelSelector

Similar to HypSelector, ModelSelector is a tool for PySB model selection using Nested Sampling-based model selection. ModelSelector allows users to easily compare model variants written in PySB and see which one may best explain a dataset by performing Nested Sampling to compute their evidences and thereby do model selection; ModelSelector also provides functionality to estimate Bayes factors from the evidence estimates, as well as estimators for the Akaike, Bayesian, and Deviance information criteria computed from the Nested Sampling outputs. See the [ModelSelector Example Jupyter Notebook](./jupyter_notebooks/ModelSelector_example.ipynb) to see example usage of ModelSelector.

------

# Citing

If you use the Gleipnir software in your research, please cite it. You can export the  Gleipnir citation in your preferred format from its [Zenodo DOI](http://doi.org/10.5281/zenodo.3036345) entry.  

Also, please cite the following references as appropriate for software used with/via Gleipnir:

#### Packages from the SciPy ecosystem

These include NumPy, SciPy, Pandas, and Matplotlib for which references can be obtained from:
https://www.scipy.org/citing.html

#### PySB
  1. Lopez, C. F., Muhlich, J. L., Bachman, J. A. & Sorger, P. K. Programming biological models in Python using PySB. Mol Syst Biol 9, (2013). doi:[10.1038/msb.2013.1](dx.doi.org/10.1038/msb.2013.1)

#### MultiNest
  1. Feroz, Farhan, and M. P. Hobson. "Multimodal nested sampling: an
      efficient and robust alternative to Markov Chain Monte Carlo
      methods for astronomical data analyses." Monthly Notices of the
      Royal Astronomical Society 384.2 (2008): 449-463.
  2. Feroz, F., M. P. Hobson, and M. Bridges. "MultiNest: an efficient
      and robust Bayesian inference tool for cosmology and particle
      physics." Monthly Notices of the Royal Astronomical Society 398.4
      (2009): 1601-1614.
  3. Feroz, F., et al. "Importance nested sampling and the MultiNest
      algorithm." arXiv preprint arXiv:1306.2144 (2013).

#### PyMultiNest:
  1. Buchner, J., et al. "X-ray spectral modelling of the AGN obscuring region in the CDFS: Bayesian model selection and catalogue." Astronomy & Astrophysics 564 (2014): A125.

#### PolyChord
  1. Handley, W. J., M. P. Hobson, and A. N. Lasenby. "PolyChord: nested sampling for cosmology." Monthly Notices of the Royal Astronomical Society: Letters 450.1 (2015): L61-L65.
  2. Handley, W. J., M. P. Hobson, and A. N. Lasenby. "POLYCHORD:
    next-generation nested sampling." Monthly Notices of the Royal
    Astronomical Society 453.4 (2015): 4384-4398.

#### DNest4

  1. Brewer, B. J., Pártay, L. B., & Csányi, G. (2011). Diffusive nested
        sampling. Statistics and Computing, 21(4), 649-656        
  2. Brewer, B., & Foreman-Mackey, D. (2018). DNest4: Diffusive Nested Sampling in C++ and Python. Journal of Statistical Software, 86(7), 1 - 33. doi:[10.18637/jss.v086.i07](http://dx.doi.org/10.18637/jss.v086.i07)

#### Nestle
Cite the GitHub repo: https://github.com/kbarbary/nestle


#### seaborn
Reference can be exported from the [seaborn Zeondo DOI entry](https://doi.org/10.5281/zenodo.592845)
