#  Copyright 2021 The QUARK Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from __future__ import annotations  # Needed if you want to type hint a method with the type of the enclosing class

import os
from abc import ABC, abstractmethod
import logging
from typing import final
import sys
from utils import _get_instance_with_sub_options

from Metrics import Metrics


class Core(ABC):
    """
    Core Module for QUARK used by all other Modules that are part of a benchmark process
    """

    def __init__(self, name: str = None):
        """
        Constructor method
        :param name: name used to identify this QUARK module. If not specified class name will be used as default.
        :type name: str
        """
        self.submodule_options = []
        self.sub_options = []
        self.preprocessed_input = None
        self.postprocessed_input = None
        if name is None:
            name = self.__class__.__name__
        self.name = name
        self.metrics = Metrics(name, os.path.relpath(sys.modules[self.__module__].__file__))

    @abstractmethod
    def get_parameter_options(self) -> dict:
        """
        Returns the parameters for a given module

        Should always be in this format:

        .. code-block:: json

            {
               "parameter_name":{
                  "values":[1, 2, 3],
                  "description":"How many nodes do you need?"
               },
                "parameter_name_2":{
                  "values":["x", "y"],
                  "description":"Which type of problem do you want?"
               }
            }

        :return: Available settings for this application
        :rtype: dict
        """

    @final
    def get_submodule(self, option: str) -> any:
        """
        Submodule is instantiated according to the information given in self.sub_options.
        If self.sub_options is None, get_default_submodule is called as a fallback.

        :param option: String with the options
        :type option: str
        :return: Instance of a module
        :rtype: any
        """
        if self.sub_options is None or not self.sub_options:
            return self.get_default_submodule(option)
        return _get_instance_with_sub_options(self.sub_options, option)

    # TODO Think if the naming of get_default_submodule can be improved to better reflect its function.
    @abstractmethod
    def get_default_submodule(self, option: str) -> Core:
        """
        Given an option string by the user, this returns a submodule

        :param option: String with the chosen submodule
        :type option: str
        :return: Module of type Core
        :rtype: Core
        """
        raise NotImplementedError("Please don't use the base version of this method. "
                                  "Implement your own override instead.")

    def preprocess(self, input_data: any, config: dict, **kwargs) -> (any, float):
        """
        Essential method for the benchmarking process. Is always executed before traversing down to the next module,
        passing the data returned by this function.

        :param input_data: Data for the module, comes from the parent module if that exists
        :type input_data: any
        :param config: Config for the module
        :type config: dict
        :param kwargs: Optional keyword arguments
        :type kwargs: dict
        :return: The output of the preprocessing and the time it took to preprocess
        :rtype: (any, float)
        """
        return input_data, 0.0

    def postprocess(self, input_data: any, config: dict, **kwargs) -> (any, float):
        """
        Essential Method for the benchmarking process. Is always executed after the submodule is finished. The data by
        this method is passed up to the parent module.

        :param input_data: Input data comes from the submodule if that exists
        :type input_data: any
        :param config: Config for the module
        :type config: dict
        :param kwargs: Optional keyword arguments
        :type kwargs: dict
        :return: The output of the postprocessing and the time it took to postprocess
        :rtype: (any, float)
        """
        return input_data, 0.0

    @final
    def get_available_submodule_options(self) -> list:
        """
        Gets list of available options

        :return: List of module options
        :rtype: list
        """
        if self.sub_options is None or not self.sub_options:
            return self.submodule_options
        else:
            return [o["name"] for o in self.sub_options]

    @staticmethod
    def get_requirements() -> list:
        """
        Returns the required pip packages for this module. Optionally, version requirements can be added.

        :return: List of dictionaries
        :rtype: list
        """
        return []
