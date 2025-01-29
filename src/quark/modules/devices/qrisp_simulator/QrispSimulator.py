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
from typing import TypedDict

from quark.modules.Core import Core
from quark.modules.devices.Device import Device


class QrispSimulator(Device, ABC):
    """
    Abstract class to use the Qrisp Simulator.
    """

    def __init__(self):
        """
        Constructor method.
        """
        super().__init__(device_name="qrisp_simulator")
        self.submodule_options = []
        self.backend = None

    def get_backend(self) -> any:
        """
        Returns backend.

        :return: Instance of the backend class
        """
        if self.backend is None:
            raise AttributeError("The 'backend' attribute has not been set.")
        return self.backend

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Return requirements of this module.

        :return: List of dict with requirements of this module
        """
        return [{"name": "qrisp", "version": "0.5.2"}]

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this application, not Applicable for Qrisp Simulator for now.

        Example: "doppler": {
                "values": [False, True],
                "description": "Simulate doppler noise? Has a large impact on performance!"
            }
        """
        # TODO once optional noisy simulation is done in qrisp
        return {}

    class Config(TypedDict):
        """
        Attributes of a valid config.
        """
        doppler: bool

    def get_backend_config(self):
        """
        Returns backend configurations.

        :return: Backend config for the emulator
        """
        if self.backend is None:
            raise AttributeError("The 'backend' attribute has not been set.")
        if not hasattr(self.backend, 'config'):
            raise AttributeError("The 'backend' object has no attribute 'config'.")
        return self.backend.config

    def get_default_submodule(self, option: str) -> Core:
        """
        Returns the default submodule based on the given option.

        :param option: The submodule option to select
        :return: Instance of the selected submodule
        :raises NotImplemented: If the provided option is not implemented
        """
        raise ValueError("This module has no submodules.")
