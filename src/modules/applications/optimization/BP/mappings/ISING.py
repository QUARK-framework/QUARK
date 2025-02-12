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

import pdb
from typing import TypedDict
import numpy as np

from modules.applications.Mapping import *
import modules.applications.optimization.BP.BP as BPack
from utils import start_time_measurement, end_time_measurement


class Ising(Mapping):
    """
    Ising formulation for the BinPacking-Problem.
    """

    def __init__(self):
        """
        Constructor method
        """
        super().__init__()
        self.submodule_options = ["QAOA", "PennylaneQAOA", "QiskitQAOA"]
        self.key_mapping = None
        self.graph = None
        self.config = None

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Return requirements of this module

        :return: list of dict with requirements of this module
        :rtype: list[dict]
        """
        return [
            {
                "name": "numpy",
                "version": "1.23.5"
            }
        ]

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this mapping

        :return:
                 .. code-block:: python

                     return {
                                "penalty_factor": {
                                    "values": [0.75, 1.0, 1.25],
                                    "description": "By which factor would you like to multiply your lagrange?"
                                },
                                "mapping": {
                                    "values": ["ocean", "qiskit", "pyqubo"],
                                    "description": "Which Ising formulation of the TSP problem should be used?"
                                }
                            }

        """
        return {
            "penalty_factor": {
                "values": [2],  # [1, 2, 5, 10],
                "description": "By which factor would you like to multiply your lagrange?"
            }
        }

    class Config(TypedDict):
        """
        Attributes of a valid config

        .. code-block:: python

             penalty_factor: float

        """
        penalty_factor: float

    def map(self, problem: (list, float, list), config: Config) -> (dict, float):
        """
        Maps the bin packing problem input to an ISING formulation.

        :param problem: bin packing problem instance defined by
                    1. object weights, 2. bin capacity, 3. incompatible objects
        :type problem: (list, float, list)
        :param config: config with the parameters specified in Config class
        :type config: Config
        :return: dict with ISING-matrix, -vector and -offset as well as time it took to map it
        :rtype: tuple(dict, float)
        """
        self.problem = problem
        self.config = config
        start = start_time_measurement()

        # %% create docplex model for the binpacking-problem
        bin_packing_mip = BPack.create_MIP(problem)

        # %% transform the MIP to an Ising formulation
        penalty_factor = config['penalty_factor']
        self.ising_matrix, self.ising_vector, self.ising_offset, self.qubo = BPack.transform_docplex_mip_to_ising(
            bin_packing_mip, penalty_factor)
        # %%

        return {"J": self.ising_matrix, "h": self.ising_vector,
                "c": self.ising_offset, "QUBO": self.qubo}, end_time_measurement(start)

    def reverse_map(self, solution: any) -> (dict, float):
        """
        Maps the solution back to be able to validate and evaluate it

        :param solution: the solution of the QAOA is a numpy-array
        :type solution: numpy-array
        :return: solution mapped accordingly, time it took to map it
        :rtype: tuple(dict, float)
        """
        start = start_time_measurement()

        solution_dict = {}

        variable_names = [var.name for var in self.qubo.variables]

        for idx in range(len(solution)):
            var_name = variable_names[idx]
            var_value = int(solution[len(solution) - 1 - idx])  # QAOA-result bitstring is reversed
            solution_dict[var_name] = var_value

        # pdb.set_trace()
        return solution_dict, end_time_measurement(start)

    @staticmethod
    def _convert_ising_to_qubo(solution: any) -> any:
        solution = np.array(solution)
        with np.nditer(solution, op_flags=['readwrite']) as it:
            for x in it:
                if x == -1:
                    x[...] = 0
        return solution

    def get_default_submodule(self, option: str) -> Core:

        if option == "QAOA":
            from modules.solvers.QAOA import QAOA  # pylint: disable=C0415
            return QAOA()
        elif option == "PennylaneQAOA":
            from modules.solvers.PennylaneQAOA import PennylaneQAOA  # pylint: disable=C0415
            return PennylaneQAOA()
        elif option == "QiskitQAOA":
            from modules.solvers.QiskitQAOA import QiskitQAOA  # pylint: disable=C0415
            return QiskitQAOA()
        else:
            raise NotImplementedError(f"Solver Option {option} not implemented")
