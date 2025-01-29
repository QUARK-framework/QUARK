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
import pickle
import os

from quark.modules.applications.Application import Application
from quark.modules.applications.optimization.Optimization import Optimization
from quark.utils import start_time_measurement, end_time_measurement


class SCP(Optimization):
    """
    The set cover problem (SCP) is a classical combinatorial optimization problem where the objective is to find the
    smallest subset of given elements that covers all required elements in a collection. This can be formulated as
    selecting the minimum number of sets from a collection such that the union of the selected sets contains all
    elements from the universe of the problem instance.

    SCP has widespread applications in various fields, including sensor positioning, resource allocation, and network
    design. For example, in sensor positioning, SCP can help determine the fewest number of sensors required to cover
    a given area. Similarly, in resource allocation, SCP helps to allocate resources in an optimal way, ensuring
    coverage of all demand points while minimizing costs. Network design also uses SCP principles to efficiently place
    routers or gateways in a network to ensure full coverage with minimal redundancy.

    This implementation of SCP provides configurable problem instances of different sizes, such as "Tiny," "Small,"
    and "Large," allowing the user to explore solutions with varying complexities. We employ various quantum-inspired
    methods to solve SCP, including a mapping to the QUBO (Quadratic Unconstrained Binary Optimization) formulation
    using Qubovert. These approaches allow us to explore how different optimization algorithms and frameworks perform
    when applied to this challenging problem, offering insights into both classical and emerging quantum methods.
    """

    def __init__(self):
        """
        Constructor method.
        """
        super().__init__("SCP")
        self.submodule_options = ["qubovertQUBO"]

    def get_solution_quality_unit(self) -> str:
        return "Number of selected subsets"

    def get_default_submodule(self, option: str) -> Application:
        """
        Returns the default submodule based on the provided option.

        :param option: Option specifying the submodule
        :return: Instance of the corresponding submodule
        :raises NotImplementedError: If the option is not recognized
        """
        if option == "qubovertQUBO":
            from quark.modules.applications.optimization.SCP.mappings.qubovertQUBO import QubovertQUBO  # pylint: disable=C0415
            return QubovertQUBO()
        else:
            raise NotImplementedError(f"Mapping Option {option} not implemented")

    def get_parameter_options(self):
        """
        Returns the configurable settings for this application

        :return: Dictionary containing parameter options
        .. code-block:: python

        return {
            "model_select": {
                "values": list(["Tiny", "Small", "Large"]),
                "description": "Please select the problem size(s). Tiny: 4 elements, 3 subsets. Small:
                15 elements, 8 subsets. Large: 100 elements, 100 subsets"
            }
        }
        """
        return {
            "model_select": {
                "values": list(["Tiny", "Small", "Large"]),
                "description": "Please select the problem size(s). Tiny: 4 elements, 3 subsets. Small: 15 elements, "
                               "8 subsets. Large: 100 elements, 100 subsets"
            }
        }

    class Config(TypedDict):
        model_select: str

    def generate_problem(self, config: Config) -> tuple[set, list]:
        """
        Generates predefined instances of the SCP.

        :param config: Config specifying the selected problem instances
        :return: The union of all elements of an instance and a set of subsets, each covering a part of the union
        """
        model_select = config['model_select']
        self.application = {}

        if model_select == "Tiny":
            self.application["elements_to_cover"] = set(range(1, 4))
            self.application["subsets"] = [{1, 2}, {1, 3}, {3, 4}]
        elif model_select == "Small":
            self.application["elements_to_cover"] = set(range(1, 15))
            self.application["subsets"] = [
                {1, 3, 4, 6, 7, 13}, {4, 6, 8, 12}, {2, 5, 9, 11, 13}, {1, 2, 7, 14, 15},
                {3, 10, 12, 14}, {7, 8, 14, 15}, {1, 2, 6, 11}, {1, 2, 4, 6, 8, 12}
            ]

        elif model_select == "Large":
            self.application["elements_to_cover"] = set(range(1, 100))
            self.application["subsets"] = []
            path = os.path.join(os.path.dirname(__file__))
            with open(f"{path}/data/set_cover_data_large.txt") as data:
                while line := data.readline():
                    new_set = []
                    for i in line.split(','):
                        new_set.append(int(i))
                    new_set = set(new_set)
                    self.application["subsets"].append(new_set)

        else:
            raise ValueError(f"Unknown model_select value: {model_select}")

        return self.application["elements_to_cover"], self.application["subsets"]

    def process_solution(self, solution: list) -> tuple[list, float]:
        """
        Returns list of selected subsets and the time it took to process the solution.

        :param solution: Unprocessed solution
        :return: Processed solution and the time it took to process it
        """
        start_time = start_time_measurement()
        selected_subsets = [list(self.application["subsets"][i]) for i in solution]
        return selected_subsets, end_time_measurement(start_time)

    def validate(self, solution: list) -> tuple[bool, float]:
        """
        Checks if the elements of the subsets that are part of the solution cover every element of the instance.

        :param solution: List containing all subsets that are part of the solution
        :return: Boolean whether the solution is valid and time it took to validate
        """
        start = start_time_measurement()
        covered = set.union(*[set(subset) for subset in solution])

        return covered == self.application["elements_to_cover"], end_time_measurement(start)

    def evaluate(self, solution: list) -> tuple[int, float]:
        """
        Calculates the number of subsets that are of the solution.

        :param solution: List containing all subsets that are part of the solution
        :return: Number of subsets and the time it took to calculate it
        """
        start = start_time_measurement()
        selected_num = len(solution)

        return selected_num, end_time_measurement(start)

    def save(self, path: str, iter_count: int) -> None:
        """
        Saves the SCP instance to a file.

        :param path: Path to save the SCP instance
        :param iter_count: Iteration count
        """
        with open(f"{path}/SCP_instance", "wb") as file:
            pickle.dump(self.application, file, pickle.HIGHEST_PROTOCOL)
