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

import networkx as nx
import numpy as np
from dimod import qubo_to_ising

from quark.modules.applications.Mapping import Mapping, Core
from quark.modules.applications.optimization.PVC.mappings.QUBO import QUBO
from quark.utils import start_time_measurement, end_time_measurement


class Ising(Mapping):
    """
    Ising formulation for the PVC.

    """

    def __init__(self):
        """
        Constructor method.
        """
        super().__init__()
        self.submodule_options = ["QAOA", "PennylaneQAOA"]
        self.key_mapping = None

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Returns requirements of this module.

        :return: List of dictionaries with requirements of this module
        """
        return [
            {"name": "networkx", "version": "3.4.2"},
            {"name": "numpy", "version": "1.26.4"},
            {"name": "dimod", "version": "0.12.18"},
            *QUBO.get_requirements()
        ]

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this mapping.

        :return: Dictionary containing parameter options.
        .. code-block:: python

            return {
                    "lagrange_factor": {
                        "values": [0.75, 1.0, 1.25],
                        "description": "By which factor would you like to multiply your Lagrange?"
                    }
                }
        """
        return {
            "lagrange_factor": {
                "values": [0.75, 1.0, 1.25],
                "description": "By which factor would you like to multiply your Lagrange?"
            }
        }

    class Config(TypedDict):
        """
        Configuration attributes for Ising mapping.

        Attributes:
             lagrange_factor (float): Factor to multiply the Langrange.
        """
        lagrange_factor: float

    def map(self, problem: nx.Graph, config: Config) -> tuple[dict, float]:
        """
        Uses the PVC QUBO formulation and converts it to an Ising representation.

        :param problem: Networkx graph representing the PVC problem
        :param config: Config dictionary with the mapping configuration
        :return: Tuple containing a dictionary with the ising problem and time it took to map it
        """
        start = start_time_measurement()

        # Convert the PVC problem to QUBO
        qubo_mapping = QUBO()
        q, _ = qubo_mapping.map(problem, config)

        # Convert QUBO to ising using dimod
        t, j, _ = qubo_to_ising(q["Q"])

        # Extract unique configuration and tool attributes from the graph
        config = [x[2]['c_start'] for x in problem.edges(data=True)]
        config = list(set(config + [x[2]['c_end'] for x in problem.edges(data=True)]))

        tool = [x[2]['t_start'] for x in problem.edges(data=True)]
        tool = list(set(tool + [x[2]['t_end'] for x in problem.edges(data=True)]))

        # Initialize J matrix and mapping
        timesteps = int((problem.number_of_nodes() - 1) / 2 + 1)
        matrix_size = problem.number_of_nodes() * len(config) * len(tool) * timesteps
        j_matrix = np.zeros((matrix_size, matrix_size), dtype=float)
        self.key_mapping = {}

        # Map J values to a matrix representation
        index_counter = 0
        for key, value in j.items():
            if key[0] not in self.key_mapping:
                self.key_mapping[key[0]] = index_counter
                index_counter += 1
            if key[1] not in self.key_mapping:
                self.key_mapping[key[1]] = index_counter
                index_counter += 1
            u = self.key_mapping[key[0]]
            v = self.key_mapping[key[1]]
            j_matrix[u][v] = value

        return {"J": j_matrix, "t": np.array(list(t.values()))}, end_time_measurement(start)

    def reverse_map(self, solution: dict) -> tuple[dict, float]:
        """
        Maps the solution back to the representation needed by the PVC class for validation/evaluation.

        :param solution: Dictionary containing the solution
        :return: Tuple with the remapped solution and time it took to reverse map
        """
        start = start_time_measurement()
        logging.info(f"Key Mapping: {self.key_mapping}")

        result = {key: 1 if solution[self.key_mapping[key]] == 1 else 0 for key in self.key_mapping}

        return result, end_time_measurement(start)

    def get_default_submodule(self, option: str) -> Core:
        """
        Returns the default submodule based on the provided option.

        :param option: Option specifying the submodule
        :return: Instance of the corresponding submodule
        :raises NotImplementedError: If the option is not recognized
        """
        if option == "QAOA":
            from quark.modules.solvers.QAOA import QAOA  # pylint: disable=C0415
            return QAOA()
        if option == "PennylaneQAOA":
            from quark.modules.solvers.PennylaneQAOA import PennylaneQAOA  # pylint: disable=C0415
            return PennylaneQAOA()
        else:
            raise NotImplementedError(f"Solver Option {option} not implemented")
