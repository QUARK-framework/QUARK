from copy import deepcopy
import json
from time import time
import logging

from modules.Core import Core

from modules.applications.Application import Application as Application_NEW
from quark2_adapter.legacy_classes.Application import Application as Application_OLD
from modules.applications.Mapping import Mapping as Mapping_NEW
from quark2_adapter.legacy_classes.Mapping import Mapping as Mapping_OLD
from modules.solvers.Solver import Solver as Solver_NEW
from quark2_adapter.legacy_classes.Solver import Solver as Solver_OLD
from modules.devices.Device import Device as Device_NEW
from quark2_adapter.legacy_classes.Device import Device as Device_OLD


WARNING_MSG = 'Class "%s" is inheriting from depricated base class. Please refactor your class.'


class ApplicationAdapter(Application_NEW, Application_OLD):
    def __init__(self, application_name, *args, **kwargs):
        logging.warning(WARNING_MSG,  self.__class__.__name__)
        Application_NEW.__init__(self, application_name)
        Application_OLD.__init__(self, application_name)
        self.args = args
        self.kwargs = kwargs

        self.problem_conf_hash = None
        self.problems = {}

    @property
    def submodule_options(self):
        return self.mapping_options

    @submodule_options.setter
    def submodule_options(self, options):
        self.mapping_options = options

    def get_default_submodule(self, option: str) -> Core:
        return self.get_mapping(option)

    def preprocess(self, input_data: any, config: dict, **kwargs) -> (any, float):
        start = time()
        logging.warning(WARNING_MSG, self.__class__.__name__)

        rep_count = kwargs["rep_count"]

        #create a hash value for identifying the problem configuration
        #compare https://stackoverflow.com/questions/5884066/hashing-a-dictionary
        problem_conf_hash = json.dumps(config, sort_keys=True)

        if self.problem_conf_hash != problem_conf_hash:
            self.problems = {}
            self.problem_conf_hash = problem_conf_hash

        problem_key = rep_count if self.regenerate_on_iteration(config) else "dummy"
        if problem_key in self.problems:
            self.problem, creation_time = self.problems[problem_key]
        else:
            start = time()
            logging.info("generate new problem instance")
            self.problem = self.generate_problem(config, rep_count)
            creation_time = (time() - start)*1000
            self.problems[problem_key] = (self.problem, creation_time)
        return self.problem, creation_time

    def postprocess(self, input_data: any, config: dict, **kwargs) -> (any, float):
        processed_solution, time_processing = self.process_solution(input_data)
        solution_validity, time_validate = self.validate(processed_solution)
        if solution_validity:
            solution_quality, time_evaluate = self.evaluate(processed_solution)
        else:
            solution_quality, time_evaluate = None, 0.0
            
       
        self.metrics.add_metric("time_to_validation", time_validate)
        self.metrics.add_metric("time_to_validation_unit", "ms")
        self.metrics.add_metric("solution_validity", solution_validity)
        self.metrics.add_metric("solution_quality", solution_quality)
        self.metrics.add_metric("solution_quality_unit", self.get_solution_quality_unit())
        return (solution_validity, solution_quality), time_validate+time_evaluate+time_processing


class MappingAdapter(Mapping_NEW, Mapping_OLD):
        
        
    def __init__(self,*args, **kwargs):
        
        Mapping_NEW.__init__(self)
        Mapping_OLD.__init__(self)

    @property
    def submodule_options(self):
        return self.solver_options

    @submodule_options.setter
    def submodule_options(self, options):
        self.solver_options = options

    def get_default_submodule(self, option: str) -> Core:
        return self.get_solver(option)

    def preprocess(self, input_data: any, config: dict, **kwargs) -> (any, float):
        logging.warning(WARNING_MSG, self.__class__.__name__)
        return self.map(input_data, config=config)

    def postprocess(self, input_data: any, config: dict, **kwargs) -> (any, float):
        logging.info("Calling %s.reverse_map", __class__.__name__)
        processed_solution, postprocessing_time = self.reverse_map(input_data)

        # self.metrics.add_metric("processed_solution", ["%s: %s" % (
        #     sol.__class__.__name__, sol) for sol in processed_solution])
        return processed_solution, postprocessing_time


def recursive_replace_dict_keys(obj: any)-> any:
    """replace values used as dict-keys by its str(), to make the object json compatible"""
    obj_new = None
    if isinstance(obj, dict):
        obj_new = {}
        for key in obj:
            obj_new[str(key)] = recursive_replace_dict_keys(obj[key])
    elif isinstance(obj, list):
        obj_new = []
        for i, element in enumerate(obj):
            obj_new.append(recursive_replace_dict_keys(element))
    elif isinstance(obj, tuple):
        obj_new = tuple(recursive_replace_dict_keys(element) for element in obj)
    else:
        obj_new = obj

    return obj_new


class SolverAdapter(Solver_NEW, Solver_OLD):

    def __init__(self,*args, **kwargs):
        
        Solver_NEW.__init__(self)
        Solver_OLD.__init__(self)

    @property
    def submodule_options(self):
        return self.device_options

    @submodule_options.setter
    def submodule_options(self, options):
        self.device_options = options

    def get_default_submodule(self, option: str) -> Core:
        return self.get_device(option)

    def preprocess(self, input_data: any, config: dict, **kwargs) -> (any, float):
        logging.warning(WARNING_MSG, self.__class__.__name__)
        return input_data, 0.0

    def postprocess(self, input_data: any, config: dict, **kwargs) -> (any, float):
        run_kwargs = {
            "store_dir": kwargs["store_dir"], "repetition": kwargs["rep_count"]}
        raw_solution, runtime, additional_solver_information = self.run(input_data["mapped_problem"],
                                                                        device_wrapper=input_data["device"], config=config, **run_kwargs)
        self.metrics.add_metric("additional_solver_information", dict(
            additional_solver_information))
        self.metrics.add_metric("solution_raw", self.raw_solution_to_json(raw_solution))
        # self.metrics.add_metric("raw_solution_energy", raw_solution[0])
        # self.metrics.add_metric("raw_solution_probabilities", {str(
        #     bitstring): probability for bitstring, probability in sorted(raw_solution[1], key=lambda x: -x[1])[:10]})
        return raw_solution, runtime

    def raw_solution_to_json(self, raw_solution):
        '''
        If raw_solution is not directly representable as json this method has to be overwritten.
        Note that using 'recursive_replace_dict_keys' might help.
        '''
        return raw_solution


class DeviceAdapter(Device_NEW, Device_OLD):

    def get_default_submodule(self, option: str) -> Core:
        return None

    def __init__(self, name):
        # name = self.__class__.__name__
        Device_NEW.__init__(self, name)
        Device_OLD.__init__(self, name)
        self.device_name = name

    def preprocess(self, input_data: any, config: dict, **kwargs) -> (any, float):
        logging.warning(WARNING_MSG, self.__class__.__name__)
        self.set_config(config)
        return {"mapped_problem": input_data, "device": self}, 0.




class LocalAdapter(DeviceAdapter):
    """
    Some Solvers (often classical) also can run on a normal local environment without any specific device or setting needed.
    """

    def __init__(self):
        """
        Constructor method
        """
        DeviceAdapter.__init__(self, name="local")
        self.device = None
