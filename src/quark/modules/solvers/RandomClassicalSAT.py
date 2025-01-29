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

from typing import TypedDict
import logging

import numpy as np

from pysat.formula import WCNF

from quark.modules.solvers.Solver import Solver
from quark.modules.Core import Core
from quark.utils import start_time_measurement, end_time_measurement


class RandomSAT(Solver):
    """
    Classic Random Solver for the SAT problem.
    """

    def __init__(self):
        """
        Constructor method.
        """
        super().__init__()
        self.submodule_options = ["Local"]

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Return requirements of this module.

        :return: List of dict with requirements of this module
        """
        return [
            {"name": "python-sat", "version": "1.8.dev13"},
            {"name": "numpy", "version": "1.26.4"}
        ]

    def get_default_submodule(self, option: str) -> Core:
        if option == "Local":
            from quark.modules.devices.Local import Local  # pylint: disable=C0415
            return Local()
        else:
            raise NotImplementedError(f"Device Option {option} not implemented")

    def get_parameter_options(self) -> dict:
        """
        Returns empty dict as this solver has no configurable settings.

        :return: Empty dict
        """
        return {}

    class Config(TypedDict):
        """
        Empty config as this solver has no configurable settings.
        """
        pass

    def run(self, mapped_problem: WCNF, device_wrapper: any, config: Config, **kwargs: dict) \
            -> tuple[list, float, dict]:
        """
        The given application is a problem instance from the pysat library.
        This generates a random solution to the problem.

        :param mapped_problem: The WCNF representation of the SAT problem
        :param device_wrapper: Local device
        :param config: Empty dict
        :param kwargs: No additionally settings needed
        :return: Solution, the time it took to compute it and optional additional information
        """
        logging.info(
            f"Got SAT problem with {mapped_problem.nv} variables, {len(mapped_problem.hard)} constraints and"
            f" {len(mapped_problem.soft)} tests."
        )

        start = start_time_measurement()
        sol = [(i + 1) * np.random.choice([-1, 1]) for i in range(mapped_problem.nv)]

        return sol, end_time_measurement(start), {}
