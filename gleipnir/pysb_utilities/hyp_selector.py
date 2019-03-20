import numpy as np
import pandas as pd
import os
import shutil
import glob
import importlib
import warnings
import multiprocessing
from multiprocessing import Pool
try:
    import HypBuilder
    from HypBuilder import ModelAssembler
except ImportError as err:
    raise err
import pysb
from .nestedsample_it import NestedSampleIt

_hypb_dir = os.path.dirname(HypBuilder.__file__)
library_file = os.path.join(_hypb_dir, "HB_library.txt")

class HypSelector(object):
    """
    """
    def __init__(self, model_csv, hb_library=None):
        """
        """
        #self.model_csv = os.path.abspath(model_csv)
        self.model_csv = model_csv
        if hb_library is None:
            self.hb_library = library_file
        else:
            self.hb_library = hb_library
        self.nested_samplers = None
        self.nested_sample_its = None
        self.selection = None
        self._mod_basename = os.path.basename(self.model_csv).split('.')[0]
        self._hypb_outputdir = os.path.join('./output',self._mod_basename)

        # Assemble the models
        ModelAssembler(self.hb_library, self.model_csv)
        # get the output models
        self._model_files = glob.glob(os.path.join(self._hypb_outputdir,'model_*.py'))
        print(self._model_files)
        # Now lets make a new models dir with an __init__.py, so we can easily
        # import all the models
        try:
            os.makedirs('hb_models')
        except OSError:
            pass
        with open('hb_models/__init__.py','w') as init:
            pass

        for model_file in self._model_files:
            mbase = os.path.basename(model_file)
            new_path = os.path.join('./hb_models', mbase)
            os.rename(model_file, new_path)
        self._model_files = glob.glob(os.path.join('hb_models','model_*.py'))
        print(self._model_files)
        # Remove the old outputs dir
        try:
            shutil.rmtree('./output')
        except OSError:
            pass
        # Load the models
        self.models = None
        # self.load_models()
        # for i, model_file in enumerate(self._model_files):
        #         model_module = importlib.import_module("hb_models.model_{}".format(i))
        #         model = getattr(model_module, 'model')
        #         self.models.append(model)
        return

    def load_models(self):
        # Load the models
        self.models = list()
        for i, model_file in enumerate(self._model_files):
                model_module = importlib.import_module("hb_models.model_{}".format(i))
                model = getattr(model_module, 'model')
                self.models.append(model)
        return

    def number_of_models(self):
        return len(self._model_files)

    def append_to_models(self, line):
        for mf in self._model_files:
            with open(mf,'a') as mfo:
                mfo.write(line)
        return

    def gen_nested_samplers(self, timespan, observable_data,
                        solver=pysb.simulator.ScipyOdeSimulator,
                        solver_kwargs=dict(), ns_version='gleipnir-classic',
                        ns_population_size=1000, ns_kwargs=dict(),
                        log_likelihood_type='logpdf'):
        """
        """
        print(ns_version)
        if self.models is None:
            self.load_models()
        if ns_version == 'multinest':
            if 'sampling_efficiency' not in list(ns_kwargs.keys()):
                ns_kwargs['sampling_efficiency'] = 0.3    
        ns_sample_its = list()
        ns_samplers = list()
        for i,model in enumerate(self.models):
            sample_it = NestedSampleIt(model, observable_data, timespan,
                                       solver=solver,
                                       solver_kwargs=solver_kwargs)
            ns_sampler = sample_it(ns_version,
                                   ns_population_size=ns_population_size,
                                   ns_kwargs=ns_kwargs,
                                   log_likelihood_type=log_likelihood_type)
            # Guard patch for multinest and polychord file outputs, so
            # each model run has its own file names.
            if ns_version == 'multinest':
                ns_sampler._file_root="multinest_run_model_{}_".format(i)
                # print(ns_sampler._file_root)
            elif ns_version == 'polychord':
                ns_sampler._settings.file_root="polychord_run_model_{}_".format(i)
            #elif ns_sampler:
            # if ns_version == 'multinest':
            #     print(ns_sampler._file_root)
            # quit()
            ns_sample_its.append(sample_it)
            ns_samplers.append(ns_sampler)
        self.nested_sample_its = ns_sample_its
        self.nested_samplers = ns_samplers
        return

    def run_nested_sampling(self, nprocs=1):
        if self.nested_samplers is None:
            warnings.warn("Unable to run. Must call the 'gen_nested_samplers' function first!")
            return
        ns_samplers = self.nested_samplers
        if nprocs > 1:
            def run_ns(nested_sampler):
                nested_sampler.run()
                return nested_sampler

            p = Pool(nprocs)
            ns_runs = p.map(run_ns, ns_samplers)
            p.close()
            self.nested_samplers = ns_runs
        else:
            for i in range(len(self.nested_samplers)):
                self.nested_samplers[i].run()
        frame = list()
        for i,ns in enumerate(self.nested_samplers):
            data_d = dict()
            data_d['model'] = "model_{}".format(i)
            data_d['log_evidence'] = ns.log_evidence
            data_d['log_evidence_error'] = ns.log_evidence_error
            frame.append(data_d)
        selection = pd.DataFrame(frame)
        selection.sort_values(by=['log_evidence'], ascending=False, inplace=True)
        self.selection = selection
        return selection
