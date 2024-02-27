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

from modules.applications.Mapping import *
import modules.applications.optimization.BinPacking.BinPacking as BPack
from utils import start_time_measurement, end_time_measurement


class QUBO(Mapping):
    """
    QUBO formulation for the BinPacking problem.

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
        ]

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this mapping

        :return:
                 .. code-block:: python

                     return {
                                "lagrange_factor": {
                                                    "values": [0.75, 1.0, 1.25],
                                                    "description": "By which factor would you like to multiply your "
                                                                    "lagrange?",
                                                    "custom_input": True,
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
        Attributes of a valid config

        .. code-block:: python

             penalty_factor: float

        """
        penalty_factor: float

    def map(self, problem: (list, float, list), config: Config) -> (dict, float):
        """
        Maps the bin packing problem input to a QUBO formulation.

        :param problem: bin packing problem instance defined by
                    1. object weights, 2. bin capacity, 3. incompatible objects
        :type problem: (list, float, list)
        :param config: config with the parameters specified in Config class
        :type config: Config
        :return: dict with QUBO, time it took to map it
        :rtype: tuple(dict, float)
        """
        self.problem = problem
        self.config = config
        start = start_time_measurement()
        print_models = True
        
        # %% create docplex model for the binpacking-problem
        bin_packing_mip = BPack.create_MIP(problem)
        
        # %% transform docplex model to QUBO  
        penalty_factor = config['penalty_factor']
        self.qubo_operator, self.qubo_bin_packing_problem = BPack.transform_docplex_mip_to_qubo(bin_packing_mip, penalty_factor)
        # %%
        
        return {"Q": self.qubo_operator, "QUBO": self.qubo_bin_packing_problem}, end_time_measurement(start)
    


    def get_default_submodule(self, option: str) -> Core:

        if option == "Annealer":
            from modules.solvers.Annealer import Annealer  # pylint: disable=C0415
            return Annealer()
        else:
            raise NotImplementedError(f"Solver Option {option} not implemented")
