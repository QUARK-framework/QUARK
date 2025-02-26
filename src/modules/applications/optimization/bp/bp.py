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
import random
import math
import logging

from docplex.mp.model import Model
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.converters import InequalityToEquality, IntegerToBinary, LinearEqualityToPenalty

from modules.applications.Application import Application
from modules.Core import Core
from modules.applications.optimization.bp.mappings.mip import MIP
from modules.applications.optimization.Optimization import Optimization
from utils import start_time_measurement, end_time_measurement


class BP(Optimization):
    """
    The bin packing problem is a classic optimization challenge where items of varying sizes must be efficiently packed 
    into a finite number of bins, each with a fixed capacity, aiming to minimize the number of bins utilized. This problem 
    is computationally NP-hard, meaning that finding an exact solution in a reasonable time frame is often impractical 
    for large datasets. Consequently, various approximation algorithms have been developed to provide near-optimal solutions 
    within acceptable time limits.

    In practical applications, bin packing is prevalent in industries such as logistics and manufacturing. For instance,
    it is used in loading trucks with weight capacity constraints, filling containers to maximize space utilization, and creating
    file backups in media storage. Additionally, it plays a role in technology mapping for FPGA semiconductor chip design, 
    where efficient resource allocation is crucial.

    To address the bin packing problem, several heuristic and approximation methods have been proposed. One common approach is 
    the First-Fit Decreasing (FFD) algorithm, which involves sorting items in descending order by size and then placing each 
    item into the first bin that can accommodate it. While FFD does not always yield an optimal solution, it is effective and 
    widely used due to its simplicity and efficiency. Other advanced techniques, such as Best-Fit and Karmarkar–Karp algorithms,
    offer improved performance for specific scenarios by considering different strategies for item placement and bin selection. 
    (source: https://en.wikipedia.org/wiki/Bin_packing_problem)
    """

    def __init__(self):
        """
        Constructor method
        """
        super().__init__("BinPacking")
        self.submodule_options = ["MIP", "Ising", "QUBO"]

    @staticmethod
    def get_requirements() -> list:
        """
        Return requirements of this module.

        :return: List of dict with requirements of this module
        """
        return [
            {"name": "numpy", "version": "1.26.4"},
            {"name": "qiskit_optimization", "version": "0.6.1"},
            {"name": "docplex", "version": "2.25.236"}
        ]

    def get_solution_quality_unit(self) -> str:
        """
        Returns the unit of measurement for the solution quality.

        :return: Unit of measurement for the solution quality
        """
        return "number_of_bins"

    def get_default_submodule(self, option: str) -> Core:
        """
        Returns the default submodule based on the provided option.

        :param option: Option specifying the submodule
        :return: Instance of the corresponding submodule
        :raises NotImplementedError: If the option is not recognized
        """
        if option == "Ising":
            from modules.applications.optimization.bp.mappings.ising import Ising  # pylint: disable=C0415
            return Ising()
        elif option == "QUBO":
            from modules.applications.optimization.bp.mappings.qubo import QUBO  # pylint: disable=C0415
            return QUBO()
        elif option == "MIP":
            from modules.applications.optimization.bp.mappings.mip import MIP  # pylint: disable=C0415
            return MIP()
        else:
            raise NotImplementedError(f"Mapping Option {option} not implemented")

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this application.

        :return:
        .. code-block:: python

            return {
                "number_of_objects": {
                    "values": list([3,4,5,6,7,8,9,10,15,20]),
                    "description": "How many objects do you want to fit inside the bins?",
                    },
                "instance_creating_mode": {
                    "values": list(["linear weights without incompatibilities",
                                    "linear weights with incompatibilities",
                                    "random weights without incompatibilities",
                                    "random weights with incompatibilities"]),
                    "description": "How do you want to create the object weights?"
                    }
                }
        """
        return {
            "number_of_objects": {
                "values": [3, 4, 5, 6, 7, 8, 9, 10, 15, 20],
                "description": "How many objects do you want to fit inside the bins?",
            },
            "instance_creating_mode": {
                "values": [
                    "linear weights without incompatibilities",
                    "linear weights with incompatibilities",
                    "random weights without incompatibilities",
                    "random weights with incompatibilities"
                ],
                "description": "How do you want to create the object weights?"
            }
        }

    class Config(TypedDict):
        """
        Attributes of a valid config.

        .. code-block:: python

             number_of_objects: int
             instance_creating_mode: str

        """
        number_of_objects: int
        instance_creating_mode: str

    def create_bin_packing_instance(self, number_of_objects: int, mode: str) -> tuple[list, int, list]:
        """
        Generates a bin packing problem instance depending on the mode and the number of objects.

        :param number_of_objects: How many objects should the bin packing problem instance consist of
        :param mode: Declares the mode with which the bin packing problem instance should be created
        :return: Tuple with object_weights, bin_capacity, incompatible_objects
        """
        if mode == "linear weights without incompatibilities":
            object_weights = list(range(1, number_of_objects + 1))
            bin_capacity = max(object_weights)
            incompatible_objects = []

        elif mode == "linear weights with incompatibilities":
            object_weights = list(range(1, number_of_objects + 1))
            bin_capacity = max(object_weights)
            incompatible_objects = []
            # add some incompatible objects via a for-loop
            for i in range(math.floor(number_of_objects / 2)):
                incompatible_objects.append((i, number_of_objects - 1 - i))

        elif mode == "random weights without incompatibilities":
            object_weights = [random.randint(1, number_of_objects) for _ in range(number_of_objects)]
            bin_capacity = max(object_weights)
            incompatible_objects = []

        elif mode == "random weights with incompatibilities":
            object_weights = [random.randint(1, number_of_objects) for _ in range(number_of_objects)]
            bin_capacity = max(object_weights)
            incompatible_objects = []
            for i in range(math.floor(number_of_objects / 2)):
                incompatible_objects.append((i, number_of_objects - i))

        else:
            logging.error("An error occurred. Couldn't create a bin packing instance")
            raise ValueError("forbidden mode during bin-packing-instance-creating-process")

        return object_weights, bin_capacity, incompatible_objects

    def generate_problem(self, config: Config, **kwargs) -> tuple[list, float, list]:
        """
        Generates a bin-packing problem instance with the input configuration.

        :param config: Configuration dictionary with problem settings
        :param kwargs: Optional additional arguments
        :return: Tuple with object_weights, bin_capacity, incompatible_objects
        """
        if config is None:
            config = {
                "number_of_objects": 5,
                "instance_creating_mode": "linear weights without incompatibilities"
            }

        number_of_objects = config['number_of_objects']
        instance_creating_mode = config['instance_creating_mode']

        self.object_weights, self.bin_capacity, self.incompatible_objects = self.create_bin_packing_instance(
            number_of_objects, instance_creating_mode
        )

        return self.object_weights, self.bin_capacity, self.incompatible_objects
    
    @staticmethod
    def detect_mapping_from_solution(solution: dict) -> str:
        # If any key contains '@int_slack@', assume it's a QUBO/Ising solution.
        if any('@int_slack@' in key for key in solution.keys()):
            # Optionally, you could differentiate further between QUBO and Ising if needed.
            return ['QUBO','Ising']
        else:
            return "MIP"

    def validate(self, solution: dict, **kwargs) -> tuple[bool, float]:
        """
        Checks if a given solution is feasible for the problem instance.

        :param solution: List containing the nodes of the solution
        :param kwargs: Optional additional arguments
        :return: Boolean whether the solution is valid, time it took to validate
        """
        start = start_time_measurement()
        mapping = BP.detect_mapping_from_solution(solution)

        if solution is None:
            return False, end_time_measurement(start)
        else:
            # create the MIP to investigate the solution
            problem_instance = (self.object_weights, self.bin_capacity, self.incompatible_objects)
            self.mip_original = MIP.create_MIP(self,problem_instance)
            logging.info(f"Detected mapping 2: {mapping}")
            # MIP
            if mapping == 'MIP':
                # Transform docplex model to the qiskit-optimization framework
                self.mip_qiskit = from_docplex_mp(self.mip_original)
                # Put the solution-values into a list to be able to check feasibility
                solution_list = []
                for key, value in solution.items():
                    solution_list.append(value)
                feasible_or_not = self.mip_qiskit.is_feasible(solution_list)

            # QUBO
            elif mapping == ['QUBO','Ising']:  # QUBO or Ising -->we need the binary equation formulation of the MIP

                # Transform docplex model to the qiskit-optimization framework
                self.mip_qiskit = from_docplex_mp(self.mip_original)
                # Transform inequalities to equalities --> with slacks
                mip_ineq2eq = InequalityToEquality().convert(self.mip_qiskit)
                # Transform integer variables to binary variables -->split up into multiple binaries
                self.mip_qiskit_int2bin = IntegerToBinary().convert(mip_ineq2eq)

                # Re-order the solution-values to be able to check feasibility -> because
                # The variables are muddled in the dictionary
                x_values = []
                y_values = []
                slack_values = []
                for key, value in solution.items():
                    if key[0] == "x":  # bin-variable
                        x_values.append(value)
                    elif key[0] == "y":  # object-assignment-variable
                        y_values.append(value)
                    else:  # slack-variable
                        slack_values.append(value)
                solution_list = x_values + y_values + slack_values
                feasible_or_not = self.mip_qiskit_int2bin.is_feasible(solution_list)
            else:
                logging.error('Error during validation. illegal mapping was used, please check')
                feasible_or_not = 'Please raise error'

            return feasible_or_not, end_time_measurement(start)

    def evaluate(self, solution: dict, **kwargs) -> tuple[float, float]:
        """
        Find the number of used bins for a given solution.

        :param solution: Dictionary containing the solution values
        :param kwargs: Optional additional arguments
        :return: Tour cost and the time it took to calculate it
        """
        start = start_time_measurement()
        mapping = BP.detect_mapping_from_solution(solution)
        logging.info(f"Detected mapping: {mapping}")
        if solution is None:
            return False, end_time_measurement(start)
        else:
            # Put the solution-values into a list
            solution_list = []
            for keys, value in solution.items():
                solution_list.append(value)

            if mapping == 'MIP':
                obj_value = self.mip_qiskit.objective.evaluate(solution_list)

            elif mapping == ['QUBO','Ising']:  # QUBO or Ising -->we need the binary equation formulation of the MIP
                obj_value = self.mip_qiskit_int2bin.objective.evaluate(
                    solution_list)  # mip_int2bin.objective.evaluate(solution)

            else:
                logging.error('Error during validation. illegal mapping was used, please check')
                obj_value = 'Please raise error'

            return obj_value, end_time_measurement(start)

    def save(self, path: str, iter_count: int) -> None:
        pass


