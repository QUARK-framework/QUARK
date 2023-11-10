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

import networkx as nx
import numpy as np
from pulp import *

from modules.applications.Application import *
from modules.applications.optimization.Optimization import Optimization
from utils import start_time_measurement, end_time_measurement


class ACL(Optimization):
    """
    \"The auto carrier loading problem - todo."
    (source: x)
    """

    def __init__(self):
        """
        Constructor method
        """
        super().__init__("ACL")
        self.submodule_options = ["MIPsolverACL", "QUBO"]

    @staticmethod
    def get_requirements() -> list:
        return [
            {
                "name": "pulp",
                "version": "2.7.0"
            },
        ]

    def get_default_submodule(self, option: str) -> Core:

        if option == "MIPsolverACL":
            from modules.solvers.MIPsolverACL import MIPaclp # pylint: disable=C0415
            return MIPaclp()
        elif option == "QUBO":
            from modules.applications.optimization.ACL.mappings.QUBO import QUBO  # pylint: disable=C0415
            return QUBO()
        else:
            raise NotImplementedError(f"Submodule Option {option} not implemented")

    def get_parameter_options(self):
        return {
            "platforms": {
                "values": [1, 2, 3, 4, 5, 6, 7, 8],
                "description": "How many platforms does the auto carrier have?"
            },
            "vehicles": {
                "values": [1, 2, 3, 4, 5, 6, 7, 8],
                "description": "How many vehicles do you want to load?"
            }
        }

    class Config(TypedDict):
        platforms: int
        vehicles: int

    def generate_problem(self, config: Config):
        vehicles = config['vehicles']
        platforms = config['platforms']

        # Construct sets
        V = set(range(vehicles))
        P = set(range(platforms))

        # In the future: User entry or get from database
        vehicle_weights = [2000, 2100, 2500, 2300, 1900, 1800, 2600, 2000]
        weight_restriction = [2000, 2000, 2500, 2200, 2600, 1800, 2500, 2000]

        # Create the 'prob' variable to contain the problem data
        prob = LpProblem("ACL", LpMaximize)

        # Create variables
        x_pv = pulp.LpVariable.dicts('x', ((p, v) for p in P for v in V), cat='Binary')

        # Objective function
        # Maximize number of vehicles on the truck
        prob += pulp.lpSum(x_pv[p, v] for p in P for v in V)

        # Constraints
        # Assignment constraints
        # (1) Every vehicle can only be assigned to a single platform
        for p in P:
            prob += pulp.lpSum(x_pv[p, v] for v in V) <= 1

        # (2) Every platform can only hold a single vehicle
        for v in V:
            prob += pulp.lpSum(x_pv[p, v] for p in P) <= 1

        # Weight constraints
        # (3) Weight of vehicle has to be less than weight restriction of the platform in is standing on
        for p in P:
            prob += pulp.lpSum(vehicle_weights[v]*x_pv[p, v] for v in V) <= weight_restriction[p]

        # Set the problem sense and name
        problem_instance = prob.to_dict()
        self.application = problem_instance
        return self.application

    def validate(self, solution) -> (bool, float):
        """
        Checks if the solution is a valid solution
:
        :param solution: Proposed solution
        :type solution: any
        :return: bool value if solution is valid and the time it took to validate the solution
        :rtype: tuple(bool, float)

        """
        start = start_time_measurement()
        status = solution["status"]
        if status == 'Optimal':
            return True, end_time_measurement(start)
        else:
            return False, end_time_measurement(start)

    def get_solution_quality_unit(self) -> str:
        return "Number of loaded vehicles"

    def evaluate(self, solution: any) -> (float, float):
        """
        Checks how good the solution is

        :param solution: Provided solution
        :type solution: any
        :return: Evaluation and the time it took to create it
        :rtype: tuple(any, float)

        """
        start = start_time_measurement()
        objective_value = solution["obj_value"]
        return objective_value, end_time_measurement(start)

    def save(self, path: str, iter_count: int) -> None:
        # Convert our problem instance from Dict to an LP problem and then to json
        _, problem_instance = LpProblem.from_dict(self.application)
        # Save problem instance to json
        problem_instance.to_json(f"{path}/ACL_instance.json")

