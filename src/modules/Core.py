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
import os
from abc import ABC, abstractmethod
from time import time
import logging
from typing import final
import sys
from utils import _get_instance_with_sub_options


from Metrics import Metrics


class Core(ABC):
    """
    Core Module for QUARK, is used by all other Modules that are part of a benchmark process.
    """

    def __init__(self):
        """
        Constructor method
        """
        self.submodule_options = []
        self.sub_options = []
        self.preprocessed_input = None
        self.postprocessed_input = None
        self.metrics = Metrics(self.__class__.__name__, os.path.relpath(sys.modules[self.__module__].__file__))

    @abstractmethod
    def get_parameter_options(self) -> dict:
        """
        Method to return the parameters for a given module.

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

        :return: Available application settings for this application
        :rtype: dict
        """

    @final
    def get_submodule(self, option: str) -> any:
        """
        If self.sub_options is not None a submobule is instantiated according to the information given in
        self.sub_options.
        Otherwise, get_default_submodule is called as fall back.

        :param option: String with the option
        :type option: str
        :return: instance of a mapping class
        :rtype: any
        """
        if self.sub_options is None or not self.sub_options:
            return self.get_default_submodule(option)
        return _get_instance_with_sub_options(self.sub_options, option)

    def get_default_submodule(self, option: str) -> any:
        """

        :param option: String with the chosen submodule
        :type option: str
        :return: Module of type Core
        :rtype: Core
        """
        return None

    def preprocess(self, input_data, config, **kwargs) -> (any, float):
        """
        Essential Method for the benchmarking process. Is always executed before traversing down to the next module,
        passing the data returned by this function.

        :param input_data:
        :type input_data:
        :param config:
        :type config:
        :param kwargs:
        :type kwargs:
        :return: The output of the precprocessing and the time it took to preprocess
        :rtype: (any, float)
        """
        return input_data, 0.0

    def postprocess(self, input_data, config, **kwargs) -> (any, float):
        """
        Essential Method for the benchmarking process. Is always executed after the submodule is finished. The data by
        this method is passed up to the parent module.

        :param input_data:
        :type input_data:
        :param config:
        :type config:
        :param kwargs:
        :type kwargs:
        :return: The output of the postprocessing and the time it took to postprocess
        :rtype: (any, float)
        """
        return input_data, 0.0

    @final
    def get_available_submodule_options(self) -> list:
        """
        Get list of available  options.

        :return: list of mapping options
        :rtype: list
        """
        if self.sub_options is None or not self.sub_options:
            return self.submodule_options
        else:
            return [o["name"] for o in self.sub_options]
