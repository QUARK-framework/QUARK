#  Copyright 2023 science + computing ag
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#

from abc import ABC
import json
import logging
from time import time

from quark.modules.Core import Core
from quark.modules.applications.Application import Application as Application_NEW
from quark.modules.applications.Mapping import Mapping as Mapping_NEW
from quark.modules.solvers.Solver import Solver as Solver_NEW
from quark.modules.devices.Device import Device as Device_NEW
from quark.quark2_adapter.legacy_classes.Application import Application as Application_OLD
from quark.quark2_adapter.legacy_classes.Mapping import Mapping as Mapping_OLD
from quark.quark2_adapter.legacy_classes.Solver import Solver as Solver_OLD
from quark.quark2_adapter.legacy_classes.Device import Device as Device_OLD


WARNING_MSG = 'Class "%s" is inheriting from deprecated base class. Please refactor your class.'


class ApplicationAdapter(Application_NEW, Application_OLD, ABC):
    """
    If you have a concrete Application written with an older QUARK version (before QUARK2) you can
    replace
    .. code-block:: python

        from applications.Application import Application

    by
    .. code-block:: python

        from quark2_adapter/adapters import ApplicationAdapter as Application

    to get your Application running with QUARK2.
    """

    def __init__(self, application_name: str, *args, **kwargs):
        """
        Constructor method.
        """
        logging.warning(WARNING_MSG, self.__class__.__name__)
        Application_NEW.__init__(self, application_name)
        Application_OLD.__init__(self, application_name)
        self.args = args
        self.kwargs = kwargs

        self.problem_conf_hash = None
        self.problems = {}

    @property
    def submodule_options(self) -> list[str]:
        """
        Maps the old attribute mapping_options to the new attribute submodule_options.
        """
        return self.mapping_options

    @submodule_options.setter
    def submodule_options(self, options: list[str]):
        """
        Maps the old attribute mapping_options to the new attribute submodule_options.

        :param options: List of submodule options
        """
        self.mapping_options = options

    def get_default_submodule(self, option: str) -> Core:
        """
        Maps the old method get_mapping to the new get_default_submodule.

        :param option: String with the chosen submodule
        :return: Module of type Core
        """
        return self.get_mapping(option)

    def preprocess(self, input_data: any, config: dict, **kwargs) -> tuple[any, float]:
        """
        Implements Application_NEW.preprocess using the Application_OLD interface.

        :param input_data: Data for the module, comes from the parent module if that exists
        :param config: Config for the module
        :param kwargs: Optional keyword arguments
        :return: The output of the preprocessing and the time it took to preprocess
        """
        start = time()
        logging.warning(WARNING_MSG, self.__class__.__name__)

        rep_count = kwargs["rep_count"]

        # create a hash value for identifying the problem configuration
        # compare https://stackoverflow.com/questions/5884066/hashing-a-dictionary
        problem_conf_hash = json.dumps(config, sort_keys=True)

        if self.problem_conf_hash != problem_conf_hash:
            self.problems = {}
            self.problem_conf_hash = problem_conf_hash

        problem_key = rep_count if self.regenerate_on_iteration(config) else "dummy"
        if problem_key in self.problems:
            self.problem, creation_time = self.problems[problem_key]
        else:
            start = time()
            logging.info("Generating new problem instance")
            self.problem = self.generate_problem(config, rep_count)
            creation_time = (time() - start) * 1000
            self.problems[problem_key] = (self.problem, creation_time)

        return self.problem, creation_time

    def postprocess(self, input_data: any, config: dict, **kwargs) -> tuple[any, float]:
        """
        Implements Application_NEW.postprocess using the Application_OLD interface.

        :param input_data: Input data comes from the submodule if that exists
        :param config: Config for the module
        :param kwargs: Optional keyword arguments
        :return: The output of the postprocessing and the time it took to postprocess
        """
        processed_solution, time_processing = self.process_solution(input_data)
        solution_validity, time_validate = self.validate(processed_solution)
        solution_quality, time_evaluate = (self.evaluate(processed_solution)
                                           if solution_validity else (None, 0.0))

        self.metrics.add_metric("time_to_validation", time_validate)
        self.metrics.add_metric("time_to_validation_unit", "ms")
        self.metrics.add_metric("solution_validity", solution_validity)
        self.metrics.add_metric("solution_quality", solution_quality)
        self.metrics.add_metric("solution_quality_unit", self.get_solution_quality_unit())

        return (solution_validity, solution_quality), time_validate + time_evaluate + time_processing


class MappingAdapter(Mapping_NEW, Mapping_OLD, ABC):
    """
    If you have a concrete Mapping written with an older QUARK version (before QUARK2) you can
    replace
    .. code-block:: python

        from applications.Mapping import Mapping

    by
    .. code-block:: python

        from quark2_adapter/adapters import MappingAdapter as Mapping

    to get your Mapping running with QUARK2.
    """

    def __init__(self, *args, **kwargs):
        """
        Constructor method.
        """
        Mapping_NEW.__init__(self)
        Mapping_OLD.__init__(self)

    @property
    def submodule_options(self) -> list[str]:
        """
        Maps the old attribute solver_options to the new attribute submodule_options.
        """
        return self.solver_options

    @submodule_options.setter
    def submodule_options(self, options: list[str]):
        """
        Maps the old attribute solver_options to the new attribute submodule_options.

        :param options: List of solver options
        """
        self.solver_options = options

    def get_default_submodule(self, option: str) -> Core:
        """
        Maps the old method get_solver to the new get_default_submodule.

        :param option: String with the chosen submodule
        :return: Module of type Core
        """
        return self.get_solver(option)

    def preprocess(self, input_data: any, config: dict, **kwargs) -> tuple[any, float]:
        """
        Implements Mapping_NEW.preprocess using the Mapping_OLD interface.
        """
        logging.warning(WARNING_MSG, self.__class__.__name__)
        return self.map(input_data, config=config)

    def postprocess(self, input_data: any, config: dict, **kwargs) -> tuple[any, float]:
        """
        Implements Mapping_NEW.postprocess using the Mapping_OLD interface.
        """
        logging.info("Calling %s.reverse_map", __class__.__name__)
        processed_solution, postprocessing_time = self.reverse_map(input_data)
        return processed_solution, postprocessing_time


def recursive_replace_dict_keys(obj: any) -> any:
    """
    Replace values used as dictionary keys by their string representation
    to make the object JSON-compatible.

    :param obj: The object to convert
    .return: The object with all dictionary keys converted to strings
    """
    obj_new = None
    if isinstance(obj, dict):
        obj_new = {str(key): recursive_replace_dict_keys(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        obj_new = [recursive_replace_dict_keys(element) for element in obj]
    elif isinstance(obj, tuple):
        obj_new = tuple(recursive_replace_dict_keys(element) for element in obj)
    else:
        obj_new = obj

    return obj_new


class SolverAdapter(Solver_NEW, Solver_OLD, ABC):
    """
    If you have a concrete Solver written with an older QUARK version (before QUARK2) you can
    replace
    .. code-block:: python

        from solvers.Solver import Solver

    by
    .. code-block:: python

        from quark2_adapter/adapters import SolverAdapter as Solver

    to get your Solver running with QUARK2.
    """

    def __init__(self, *args, **kwargs):
        """
        Constructor method.
        """
        Solver_NEW.__init__(self)
        Solver_OLD.__init__(self)

    @property
    def submodule_options(self) -> list[str]:
        """
        Maps the old attribute device_options to the new attribute submodule_options.
        """
        return self.device_options

    @submodule_options.setter
    def submodule_options(self, options: list[str]):
        """
        Maps the old attribute device_options to the new attribute submodule_options.

        :param options: List of device options
        """
        self.device_options = options

    def get_default_submodule(self, option: str) -> Core:
        """
        Maps the old method get_device to the new get_default_submodule.

        :param option: String with the chosen submodule
        :return: Module of type Core
        """
        return self.get_device(option)

    def preprocess(self, input_data: any, config: dict, **kwargs) -> tuple[any, float]:
        """
        Implements Solver_NEW.preprocess using the Solver_OLD interface.

        :param input_data: Data for the module, comes from the parent module if that exists
        :param config: Config for the module
        :param kwargs: Optional keyword arguments
        :return: The output of the preprocessing and the time it took to preprocess
        """
        logging.warning(WARNING_MSG, self.__class__.__name__)
        return input_data, 0.0

    def postprocess(self, input_data: any, config: dict, **kwargs) -> tuple[any, float]:
        """
        Implements Solver_NEW.postprocess using the Solver_OLD interface.

        :param input_data: Data passed to the run function of the solver
        :param config: Solver config
        :param kwargs: Optional keyword arguments
        :return: Output and time needed
        """
        run_kwargs = {
            "store_dir": kwargs["store_dir"],
            "repetition": kwargs["rep_count"]
        }
        raw_solution, runtime, additional_solver_information = self.run(
            input_data["mapped_problem"],
            device_wrapper=input_data["device"],
            config=config,
            **run_kwargs
        )

        self.metrics.add_metric(
            "additional_solver_information", dict(additional_solver_information)
        )
        self.metrics.add_metric(
            "solution_raw", self.raw_solution_to_json(raw_solution)
        )

        return raw_solution, runtime

    def raw_solution_to_json(self, raw_solution: any) -> any:
        """
        In case the raw_solution provided by your concrete solver is not directly representable as json
        you can overwrite SolverAdapter.raw_solution_to_json in your Solver to implement the conversion
        to json.
        Note that using 'recursive_replace_dict_keys' provided by this module might help.

        :param raw_solution: The raw solution
        :return: JSON-compatible representation of the raw solution
        """
        return raw_solution


class DeviceAdapter(Device_NEW, Device_OLD):
    """
    If you have a concrete Device written with an older QUARK version (before QUARK2) you can
    replace
    .. code-block:: python

        from devices.Device import Device

    by
    .. code-block:: python

        from quark2_adapter/adapters import DeviceAdapter as Device

    to get your Device running with QUARK2.
    """

    def __init__(self, name: str):
        """
        Constructor method.
        """
        Device_NEW.__init__(self, name)
        Device_OLD.__init__(self, name)
        self.device_name = name

    def get_default_submodule(self, option: str) -> Core:
        """
        Implements get_default_submodule by returning None as before QUARK2 a Device
        could not have submodules.

        :param option: String with the chosen submodule
        :return: None
        """
        return None

    def preprocess(self, input_data: any, config: dict, **kwargs) -> tuple[any, float]:
        """
        Implements Device_NEW.preprocess using the Device_OLD interface.

        :param input_data: Data for the module, comes from the parent module if that exists
        :param config: Config for the device
        :param kwargs: Optional keyword arguments
        :return: The output of the preprocessing and the time it took to preprocess
        """
        logging.warning(WARNING_MSG, self.__class__.__name__)
        self.set_config(config)
        return {"mapped_problem": input_data, "device": self}, 0.


class LocalAdapter(DeviceAdapter):
    """
    If you have been using the device devices.Local.Local from an older QUARK version (before QUARK2)
    you can replace
    .. code-block:: python

        from devices.Local import Local

    by
    .. code-block:: python

        from quark2_adapter/adapters import LocalAdapter as Local

    to get your code running with QUARK2.
    """

    def __init__(self):
        """
        Constructor method.
        """
        DeviceAdapter.__init__(self, name="local")
        self.device = None
