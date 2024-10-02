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

from pysat.examples.rc2 import RC2
from pysat.formula import WCNF

from modules.solvers.Solver import *
from utils import start_time_measurement, end_time_measurement


class ClassicalSAT(Solver):
    """
    Classical RC2 SAT solver.
    """

    def __init__(self):
        """
        Constructor method
        """
        super().__init__()
        self.submodule_options = ["Local"]

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Return requirements of this module

        :return: list of dict with requirements of this module
        :rtype: list[dict]
        """
        return [
            {
                "name": "python-sat",
                "version": "1.8.dev13"
            }
        ]

    def get_default_submodule(self, option: str) -> Core:
        if option == "Local":
            from modules.devices.Local import Local  # pylint: disable=C0415
            return Local()
        else:
            raise NotImplementedError(f"Device Option {option} not implemented")

    def get_parameter_options(self) -> dict:
        """
        Returns empty dict as this solver has no configurable settings

        :return: empty dict
        :rtype: dict
        """
        return {

        }

    class Config(TypedDict):
        """
        Empty config as this solver has no configurable settings
        """
        pass

    def run(self, mapped_problem: WCNF, device_wrapper: any, config: any, **kwargs: dict) -> (list, float):
        """
        The given application is a problem instance from the pysat library. This uses the rc2 maxsat solver
        given in that library to return a solution.

        :param mapped_problem:
        :type mapped_problem: WCNF
        :param device_wrapper: Local device
        :type device_wrapper: any
        :param config: empty dict
        :type config: Config
        :param kwargs: no additionally settings needed
        :type kwargs: any
        :return: Solution, the time it took to compute it and optional additional information
        :rtype: tuple(list, float, dict)
        """

        logging.info(
            f"Got problem with {mapped_problem.nv} variables, {len(mapped_problem.hard)} constraints and"
            f" {len(mapped_problem.soft)} tests."
        )

        start = start_time_measurement()
        # we use rc2 solver to compute the optimal solution
        with RC2(mapped_problem) as rc2:
            sol = rc2.compute()

        return sol, end_time_measurement(start), {}
