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

from devices.Local import Local
from solvers.Solver import *
import numpy as np
import logging
from pysat.formula import WCNF


class RandomSAT(Solver):
    """
    Classic Random Solver for the SAT problem.
    """

    def __init__(self):
        """
        Constructor method
        """
        super().__init__()
        self.device_options = ["Local"]

    def get_device(self, device_option: str) -> Local:
        if device_option == "Local":
            return Local()
        else:
            raise NotImplementedError(f"Device Option {device_option} not implemented")

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

    def run(self, mapped_problem: WCNF, device_wrapper: any, config: Config, **kwargs: dict) -> (list, float):
        """
        The given application is a problem instance from the pysat library. This generates a random solution to the
        problem.

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

        start = int(round(time() * 1000))
        sol = [(i + 1) * np.random.choice([-1, 1]) for i in range(mapped_problem.nv)]

        return sol, int(round(time() * 1000)) - start, {}
