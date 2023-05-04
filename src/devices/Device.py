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
    The device class abstracts away details of the physical device, such as submitting a task to the quantum system.
    """

    def __init__(self, device_name: str):
        """
        Constructor method
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
        :rtype: dict
        """
        return {
            
        }

    def set_config(self, config):
        self.config = config

    def get_device(self) -> any:
        """
        Returns Device.

        :return: Instance of the device class
        :rtype: any
        """
        return self.device

    def get_device_name(self) -> str:
        """
        Returns Device name.

        :return: Name of the device
        :rtype: str
        """
        return self.device_name
