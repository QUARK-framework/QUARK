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

from quark.modules.devices.Device import Device


class HelperClass(Device):
    """
    Some solvers like Pennylane, only needs strings for setting up the device and not a standalone class.

    TODO: Maybe refactor this once we think of a better structure for this
    """

    def __init__(self, device_name: str):
        """
        Constructor method.

        :param device_name: The name of the device
        """
        super().__init__(device_name=device_name)
        self.device = device_name
        self.submodule_options = []

    def get_parameter_options(self) -> dict:
        """
        Returns empty dictionary as this solver has no configurable settings.

        :return: Empty dict
        """
        return {}

    def get_default_submodule(self, option: str) -> None:
        """
        Raises ValueError as this module has no submodules.

        :param option: Option name
        :raises ValueError: If called, since this module has no submodules
        """
        raise ValueError("This module has no submodules.")
