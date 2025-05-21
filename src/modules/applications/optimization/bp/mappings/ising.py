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
import numpy as np

from modules.applications.mapping import Mapping
from modules.core import Core
from docplex.mp.model import Model

from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.converters import InequalityToEquality, IntegerToBinary
from qiskit_optimization import QuadraticProgram
from modules.applications.optimization.bp.mappings.qubo import QUBO
from modules.applications.optimization.bp.mappings.mip import MIP
from utils import start_time_measurement, end_time_measurement


class Ising(Mapping):
    """
    Ising formulation for the Bin Packing Problem.
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

        :return: List of dict with requirements of this module
        """
        return [{"name": "numpy", "version": "1.26.4"},
                {"name": "docplex", "version": "2.25.236"}]

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this mapping.

        :return:
        .. code-block:: python

            return {
            "penalty_factor": {
                                "values": [2],
                                "description": "Choose your QUBO-penalty-factor(s).",
                                "custom_input": True,
                                "allow_ranges": True,
                                "postproc": float
                    }
                }
        """
        return {
            "penalty_factor": {
                "values": [2],
                "description": "Choose your QUBO-penalty-factor(s).",
                "custom_input": True,
                "allow_ranges": True,
                "postproc": float  # Since we allow custom input here we need to parse it to float (input is str)
            }
        }

    class Config(TypedDict):
        """
        Attributes of a valid config.

        .. code-block:: python

             penalty_factor: float

        """
        penalty_factor: float

    def map(self, problem: tuple[list, float, list], config: Config) -> tuple[dict, float]:
        """
        Maps the bin packing problem input to an ISING formulation.

        :param problem: Bin packing problem instance defined by
                    1. object weights, 2. bin capacity, 3. incompatible objects
        :param config: Config with the parameters specified in Config class
        :return: Dict with ISING-matrix, -vector and -offset as well as time it took to map it
        """
        self.problem = problem
        self.config = config
        start = start_time_measurement()

        # Create docplex model for the bin packing problem
        bin_packing_mip = MIP.create_mip(self, problem)

        # Transform the MIP to an Ising formulation
        penalty_factor = config['penalty_factor']
        self.ising_matrix, self.ising_vector, self.ising_offset, self.qubo = self.transform_docplex_mip_to_ising(
            bin_packing_mip, penalty_factor
        )

        return {
            "J": self.ising_matrix,
            "t": self.ising_vector,
            "c": self.ising_offset,
            "QUBO": self.qubo
        }, end_time_measurement(start)

    def reverse_map(self, solution: np.ndarray) -> tuple[dict, float]:
        """
        Maps the solution back to be able to validate and evaluate it.

        :param solution: The solution of the QAOA is a numpy-array
        :return: Solution mapped accordingly, time it took to map it
        """
        start = start_time_measurement()

        solution_dict = {}

        variable_names = [var.name for var in self.qubo.variables]

        for idx in range(len(solution)):
            var_name = variable_names[idx]
            var_value = int(solution[len(solution) - 1 - idx])  # QAOA-result bitstring is reversed
            solution_dict[var_name] = var_value

        return solution_dict, end_time_measurement(start)

    @staticmethod
    def _convert_ising_to_qubo(solution: any) -> np.ndarray:
        """
        Converts ISING format solution to QUBO.

        :param solution: Solution in ISING format
        :return: Solution converted to QUBO format
        """
        solution = np.array(solution)
        with np.nditer(solution, op_flags=['readwrite']) as it:
            for x in it:
                if x == -1:
                    x[...] = 0
        return solution

    def transform_docplex_mip_to_ising(self, mip_docplex: Model, penalty_factor) -> (
            tuple)[np.ndarray, np.ndarray, float, QuadraticProgram]:
        """
        Transform a docplex mix-integer-problem to an Ising formulation.

        :param mip_docplex: Docplex-Model
        :param penalty_factor: Penalty factor for transformation
        :return: J-matrix, t-vector and c-offset of the Ising formulation, and the QUBO matrix
        """
        # Generate the QUBO with binary variables in {0; 1}
        qubo_instance = QUBO()
        _, qubo = qubo_instance.transform_docplex_mip_to_qubo(mip_docplex, penalty_factor)

        # Transform it to an Ising formulation
        # --> x in {0; 1} gets transformed the following way via y in {-1; 1}: x = 1/2 * (1 - y)
        #       0 --> 1    and   1 --> -1
        # in the following we construct a matrix J, a vector h and an offset c for the Ising formulation
        # so that we have an equivalent formulation of the QUBO: obj = x J x^T + h x + c
        num_qubits = len(qubo.variables)
        ising_matrix = np.zeros((num_qubits, num_qubits), dtype=np.float64)
        ising_vector = np.zeros(num_qubits, dtype=np.float64)

        ising_offset = qubo.objective.constant

        for idx, coeff in qubo.objective.linear.to_dict().items():
            ising_vector[idx] -= 1 / 2 * coeff
            ising_offset += 1 / 2 * coeff

        for (i, j), coeff in qubo.objective.quadratic.to_dict().items():
            if i == j:
                # Because the quadratic term x_i * x_j reduces to 1 if the x are ising
                # variables in {-1, 1} --> another constant term
                ising_offset += 1 / 2 * coeff
            else:
                ising_matrix[i, j] += 1 / 4 * coeff
                ising_offset += 1 / 4 * coeff

            ising_vector[i] -= 1 / 4 * coeff
            ising_vector[j] -= 1 / 4 * coeff

        return ising_matrix, ising_vector, ising_offset, qubo

    def get_default_submodule(self, option: str) -> Core:
        """
        Returns the default submodule based on the provided option.

        :param option: Option specifying the submodule
        :return: Instance of the corresponding submodule
        :raises NotImplementedError: If the option is not recognized
        """

        if option == "QAOA":
            from modules.solvers.qaoa import QAOA  # pylint: disable=C0415
            return QAOA()
        elif option == "PennylaneQAOA":
            from modules.solvers.pennylane_qaoa import PennylaneQAOA  # pylint: disable=C0415
            return PennylaneQAOA()
        elif option == "QiskitQAOA":
            from modules.solvers.qiskit_qaoa import QiskitQAOA  # pylint: disable=C0415
            return QiskitQAOA()
        else:
            raise NotImplementedError(f"Solver Option {option} not implemented")
