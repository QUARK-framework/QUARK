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

from docplex.mp.model import Model

from modules.applications.mapping import Mapping
from modules.core import Core
from modules.applications.optimization.bp.mappings.mip import MIP

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.converters import InequalityToEquality, IntegerToBinary, LinearEqualityToPenalty
from utils import start_time_measurement, end_time_measurement


class QUBO(Mapping):
    """
    QUBO formulation for the Bin Packing problem.

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
        Return requirements of this module.

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
                                "values": [1],
                                "description": "How do you want to choose your QUBO-penalty-factors?",
                                "custom_input": True,
                                "allow_ranges": True,
                                "postproc": float
                    }
                }
        """
        return {
            "penalty_factor": {
                "values": [1],
                "description": "How do you want to choose your QUBO-penalty-factors?",
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
        Maps the bin packing problem input to a QUBO formulation.

        :param problem: Bin packing problem instance defined by
                    1. object weights, 2. bin capacity, 3. incompatible objects
        :param config: Config with the parameters specified in Config class
        :return: Dict with QUBO, time it took to map it
        """
        self.problem = problem
        self.config = config
        start = start_time_measurement()

        # Create docplex model for the binpacking-problem
        bin_packing_mip = MIP.create_mip(self, problem)

        # Transform docplex model to QUBO
        penalty_factor = config['penalty_factor']
        self.qubo_operator, self.qubo_bin_packing_problem = self.transform_docplex_mip_to_qubo(
            bin_packing_mip, penalty_factor
        )

        return {
            "Q": self.qubo_operator,
            "QUBO": self.qubo_bin_packing_problem
        }, end_time_measurement(start)

    def transform_docplex_mip_to_qubo(self, mip_docplex: Model, penalty_factor: float) -> tuple[dict, QuadraticProgram]:
        """
        Transform a docplex mixed-integer-problem to a QUBO.

        :param mip_docplex: Docplex-Model
        :param penalty_factor: Penalty factor for constraints in QUBO
        :return: The transformed QUBO
        """
        # Transform docplex model to the qiskit-optimization framework
        mip_qiskit = from_docplex_mp(mip_docplex)

        # Transform inequalities to equalities --> with slacks
        mip_ineq2eq = InequalityToEquality().convert(mip_qiskit)

        # Transform integer variables to binary variables -->split up into multiple binaries
        mip_int2bin = IntegerToBinary().convert(mip_ineq2eq)

        # Transform the linear equality constraints to penalties in the objective
        if penalty_factor is None:
            # Normalize the coefficients of the QUBO that results from penalty coefficients = 1
            qubo = LinearEqualityToPenalty(penalty=1).convert(mip_int2bin)
            max_lin_coeff = numpy.max(abs(qubo.objective.linear.to_array()))
            max_quad_coeff = numpy.max(abs(qubo.objective.quadratic.to_array()))
            max_coeff = max(max_lin_coeff, max_quad_coeff)
            penalty_factor = round(1 / max_coeff, 3)
        qubo = LinearEqualityToPenalty(penalty=penalty_factor).convert(mip_int2bin)

        # Squash the quadratic and linear QUBO-coefficients together into a dictionary
        quadr_coeff = qubo.objective.quadratic.to_dict(use_name=True)
        lin_coeff = qubo.objective.linear.to_dict(use_name=True)
        for var, var_value in lin_coeff.items():
            if (var, var) in quadr_coeff.keys():
                quadr_coeff[(var, var)] += var_value
            else:
                quadr_coeff[(var, var)] = var_value
        qubo_operator = quadr_coeff

        return qubo_operator, qubo

    def get_default_submodule(self, option: str) -> Core:
        """
        Returns the default submodule based on the provided option.

        :param option: Option specifying the submodule
        :return: Instance of the corresponding submodule
        :raises NotImplementedError: If the option is not recognized
        """
        if option == "Annealer":
            from modules.solvers.annealer import Annealer  # pylint: disable=C0415
            return Annealer()
        else:
            raise NotImplementedError(f"Solver Option {option} not implemented")
