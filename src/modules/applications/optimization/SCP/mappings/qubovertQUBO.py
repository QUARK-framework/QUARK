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
from qubovert.problems import SetCover
from modules.applications.Mapping import *
from utils import start_time_measurement, end_time_measurement


class QubovertQUBO(Mapping):
    """
    Qubovert formulation of the vehicle-options problem.
    """

    def __init__(self):
        """
        Constructor method
        """
        super().__init__()
        self.submodule_options = ["Annealer"]

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Return requirements of this module

        :return: list of dict with requirements of this module
        :rtype: list[dict]
        """
        return [
            {
                "name": "qubovert",
                "version": "1.2.5"
            }
        ]

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this mapping

        :return:
                 .. code-block:: python

                      return {
                          "penalty_weight": {
                              "values": [2, 5, 10, 25, 50, 100],
                              "custom_input": True,
                              "custom_range": True,
                              "postproc": float,
                              "description": "Please choose the weight of the penalties in the QUBO representation of
                              the problem"
                          }
                      }

        """
        return {
            "penalty_weight": {
                "values": [2, 5, 10, 25, 50, 100],
                "custom_input": True,
                "allow_ranges": True,
                "postproc": float,
                "description": "Please choose the weight of the penalties in the QUBO representation of the problem"
            }
        }

    class Config(TypedDict):
        """
        Attributes of a valid config

        .. code-block:: python

             penalty_weight: float

        """
        penalty_weight: float

    def map(self, problem: tuple, config: Config) -> (dict, float):
        """
        Maps the SCP to a QUBO matrix.

        :param problem: tuple containing the set of all elements of an instance and a list of subsets each covering some
         of these elements
        :type problem: tuple
        :param config: config with the parameters specified in Config class
        :type config: Config
        :return: dict with QUBO matrix, time it took to map it
        :rtype: tuple(dict, float)
        """
        start = start_time_measurement()
        penalty_weight = config['penalty_weight']

        u, v = problem

        self.SCP_problem = SetCover(u, v)  # pylint: disable=W0201
        self.SCP_qubo = self.SCP_problem.to_qubo(penalty_weight)  # pylint: disable=W0201

        logging.info(f"Converted to QUBO with {self.SCP_qubo.num_binary_variables} Variables.")

        # now we need to convert it to the right format to be accepted by Braket / Dwave
        q_dict = {}

        for key, val in self.SCP_qubo.items():
            # interaction (quadratic) terms
            if len(key) == 2:
                if (key[0], key[1]) not in q_dict:
                    q_dict[(key[0], key[1])] = float(val)
                else:
                    q_dict[(key[0], key[1])] += float(val)
            # local (linear) fields
            elif len(key) == 1:
                if (key[0], key[0]) not in q_dict:
                    q_dict[(key[0], key[0])] = float(val)
                else:
                    q_dict[(key[0], key[0])] += float(val)

        return {"Q": q_dict}, end_time_measurement(start)

    def reverse_map(self, solution: dict) -> (set, float):
        """
        Maps the solution of the QUBO to a set of subsets included in the solution.

        :param solution: QUBO matrix in dict form
        :type solution: dict
        :return: tuple with set of subsets that are part of the solution and the time it took to map it
        :rtype: tuple(set, float)
        """
        start = start_time_measurement()
        sol = self.SCP_problem.convert_solution(solution)
        return sol, end_time_measurement(start)

    def get_default_submodule(self, option: str) -> Core:

        if option == "Annealer":
            from modules.solvers.Annealer import Annealer  # pylint: disable=C0415
            return Annealer()
        else:
            raise NotImplementedError(f"Solver Option {option} not implemented")
