# Install

| **! Warning** |
| :--- |
|  Gleipnir is still under heavy development and may rapidly change. |

#### Gleipnir run dependencies
Gleipnir has the following core dependencies:
   * NumPy - http://www.numpy.org/
   * SciPy - https://www.scipy.org/
   * pandas - https://pandas.pydata.org/

##### To use PolyChord
   * pypolychord - https://github.com/PolyChord/PolyChordLite

##### To use MultiNest
   * PyMultiNest - https://github.com/JohannesBuchner/PyMultiNest
   * MultiNest -  https://github.com/JohannesBuchner/MultiNest

##### To use DNest4
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

##### PolyChord
If you want to use the PolyChordNestedSampling object for the Nested Sampling runs then download, build, and install pypolychord following instructions in the README at:

https://github.com/PolyChord/PolyChordLite

Notes:
 * Installs into your .local/lib python site-packages.
 * Requires gfortran (f77 compiler) and lipopenmpi-dev (development libraries for MPI) to build the code.

##### MultiNest
If you want to use the MultiNestNestedSampling object for Nested Sampling download, build, and install PyMultiNest and MultiNest following the instructions at:
http://johannesbuchner.github.io/PyMultiNest/install.html

##### DNest4
If you want use the DNest4NestedSampling object for Nested Sampling then download,
build, and install DNest4 and its Python bindings following the instructions in the README at:
https://github.com/eggplantbren/DNest4

Notes:
 * Requires a c++ compiler with c++11 standard libraries.
 * Requires Cython and numba for python bindings to compile and install


#### Generate the run script for a PySB model using the nestedsample_it utility.
If you have new pysb model that you want to run Nested Sampling on you can use the nestedsample_it utility script to build a run file. nestedsample_it read the model file and imports and pulls out all the kinetic parameters and writes out a run_NS script for that model. You wil still need to edit the output script to load any data and modify the loglikelihood function, but nestedsample_it should give a good starting point. Run from the command line:
```
python -m glepnir.pysb_utilities.nestedsample_it model.py
```      
