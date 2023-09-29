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

from abc import ABC, abstractmethod
from time import time
from utils import _get_instance_with_sub_options



class Solver(ABC):
    """
    The solver is responsible for finding feasible and high-quality solutions of the formulated problem, i.e., of the
    defined objective function.
    """

    def __init__(self):
        """
        Constructor method
        """
        self.device_options = []
        self.sub_options = None
        super().__init__()

    @abstractmethod
    def run(self, mapped_problem, device, config, **kwargs) -> (any, float, dict):
        """
        This function runs the solving algorithm on a mapped problem instance and returns a solution.

        :param mapped_problem: a representation of the problem that the solver can solve
        :type mapped_problem: any
        :param device: a device the solver can leverage for the algorithm
        :type device: any
        :param config: settings for the solver such as hyperparameters
        :type config: any
        :param kwargs: optional additional settings
        :type kwargs: any
        :return: Solution, the time it took to compute it and some optional additional information
        :rtype: tuple(any, float, dict)
        """
        pass

    @abstractmethod
    def get_parameter_options(self) -> dict:
        """
        Method to return the parameters to fine tune the solver.

        Should always be in this format:

        .. code-block:: json

           {
               "parameter_name":{
                  "values":[1, 2, 3],
                  "description":"How many reads do you want?"
               }
           }

        :return: Available solver settings for this solver
        :rtype: dict
        """
        pass

    def get_submodule(self, device_option: str) -> any:
        """
        If self.sub_options is not None, a device is instantiated according to the information given in self.sub_options.
        Otherwise, get_device is called as fall back.

        :param device_option: String with the option
        :type device_option: str
        :return: instance of the device class
        :rtype: any
        """
        if self.sub_options is None:
            return self.get_device(device_option)
        else:
            return _get_instance_with_sub_options(self.sub_options, device_option, device_option)

    @abstractmethod
    def get_device(self, device_option: str) -> any:
        """
        Returns the default device based on string. This applies only if
        self.sub_options is None. See get_submodule.

        :param device_option:
        :type device_option: str
        :return: instance of the device class
        :rtype: any
        """
        pass

    def get_available_device_options(self) -> list:
        """
        Returns list of devices.

        :return: list of devices
        :rtype: list
        """
        if self.sub_options is None:
            return self.device_options
        else:
            return [o["name"] for o in self.sub_options]
