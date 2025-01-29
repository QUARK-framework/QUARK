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

from abc import ABC
from quark.modules.Core import Core
from quark.utils import start_time_measurement, end_time_measurement


class Device(Core, ABC):
    """
    The device class abstracts away details of the physical device, such as submitting a task to the quantum system.
    """

    def __init__(self, device_name: str):
        """
        Constructor method.

        :param device_name: Name of the device
        """
        super().__init__(device_name)
        self.device = None
        self.config = None
        self.device_name = self.name

    def get_parameter_options(self) -> dict:
        """
        Returns the parameters to fine-tune the device.

        Should always be in this format:
        .. code-block:: json

           {
               "parameter_name":{
                  "values":[1, 2, 3],
                  "description":"How many reads do you want?"
               }
           }

        :return: Available device settings for this device
        """
        return {}

    def set_config(self, config):
        """
        Sets the device configuration.

        :param config: Configuration settings for the device
        """
        self.config = config

    def preprocess(self, input_data: any, config: dict, **kwargs) -> tuple[any, float]:
        """
        Returns instance of device class (self) and time it takes to call config.

        :param input_data: Input data (not used)
        :param config: Config for the device
        :param kwargs: Optional keyword arguments
        :return: Output and time needed
        """
        start = start_time_measurement()
        self.config = config
        return self, end_time_measurement(start)

    def postprocess(self, input_data: any, config: dict, **kwargs) -> tuple[any, float]:
        """
        Returns input data and adds device name to the metrics class instance.

        :param input_data: Input data passed by the parent module
        :param config: Solver config
        :param kwargs: Optional keyword arguments
        :return: Output and time needed
        """
        start = start_time_measurement()
        self.metrics.add_metric("device", self.get_device_name())
        return input_data, end_time_measurement(start)

    def get_device(self) -> any:
        """
        Returns device.

        :return: Instance of the device class
        """
        return self.device

    def get_device_name(self) -> str:
        """
        Returns the device name.

        :return: Name of the device
        """
        return self.device_name
