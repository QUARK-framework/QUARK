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


class Device(ABC):
    """
    The device class abstracts away details of the physical device,
    such as submitting a task to the quantum system.
    """

    def __init__(self, device_name: str):
        """
        Constructor method.

        :param device_name: Name of the device
        """
        self.device = None
        self.device_name = device_name
        self.config = None

    def get_parameter_options(self) -> dict:
        """
        Method to return the parameters to fine tune the device.

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

    def set_config(self, config: dict) -> None:
        """
        Sets the device configuration.

        :param config: Configuration dictionary
        """
        self.config = config

    def get_device(self) -> any:
        """
        Returns the device instance.

        :return: Instance of the device
        """
        return self.device

    def get_device_name(self) -> str:
        """
        Returns the name of the Device.

        :return: Name of the device
        """
        return self.device_name
