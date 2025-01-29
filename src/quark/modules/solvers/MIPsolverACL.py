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
#
# copyright Pulp:
# Copyright (c) 2002-2005, Jean-Sebastien Roy
# Modifications Copyright (c) 2007- Stuart Anthony Mitchell
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.

from typing import TypedDict
import pulp

from quark.modules.solvers.Solver import Solver
from quark.modules.Core import Core
from quark.utils import start_time_measurement, end_time_measurement


class MIPaclp(Solver):
    """
    Classical mixed integer problem (MIP) solver for the auto-carrier loading problem (ACLP).
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
        return [{"name": "pulp", "version": "2.9.0"},]

    def get_default_submodule(self, option: str) -> Core:
        """
        Returns the default submodule based on the provided option.

        :param option: The name of the submodule
        :return: Instance of the default submodule
        """
        if option == "Local":
            from quark.modules.devices.Local import Local  # pylint: disable=C0415
            return Local()
        else:
            raise NotImplementedError(f"Device Option {option} not implemented")

    def get_parameter_options(self) -> dict:
        """
        Returns empty dictionary as this solver has no configurable settings.

        :return: Empty dict
        """
        return {}

    class Config(TypedDict):
        """
        Empty config as this solver has no configurable settings.
        """
        pass

    def run(self, mapped_problem: dict, device_wrapper: any, config: Config, **kwargs: dict) \
            -> tuple[dict, float, dict]:
        """
        Solve the ACL problem as a mixed integer problem (MIP).

        :param mapped_problem: Linear problem in form of a dictionary
        :param device_wrapper: Local device
        :param config: Empty dict
        :param kwargs: No additionally settings needed
        :return: Solution, the time it took to compute it and optional additional information
        """
        # Convert dict of problem instance to LP problem
        _, problem_instance = pulp.LpProblem.from_dict(mapped_problem)
        start = start_time_measurement()

        # Solve problem and store relevant data in dictionary
        status = problem_instance.solve()
        obj_value = pulp.value(problem_instance.objective)
        solution_data = {"status": pulp.LpStatus[status], "obj_value": obj_value}
        variables = {}
        for v in problem_instance.variables():
            variables[v.name] = v.varValue
            solution_data["variables"] = variables

        return solution_data, end_time_measurement(start), {}
