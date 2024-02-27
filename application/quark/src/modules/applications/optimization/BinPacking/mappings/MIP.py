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

from docplex.mp.model import Model

from modules.applications.Mapping import *
import modules.applications.optimization.BinPacking.BinPacking as BPack
from utils import start_time_measurement, end_time_measurement


class MIP(Mapping):
    """
    MIP formulation for the BinPacking-Problem.
    """

    def __init__(self):
        """
        Constructor method
        """
        super().__init__()
        self.submodule_options = ["MIPSolver"]
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
                "name": "docplex",
                "version": "2.25.236"
            }
        ]

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this mapping

        :return:
                 .. code-block:: python

                     return {}

        """
        return {}

    class Config(TypedDict):
        """
        Attributes of a valid config

        .. code-block:: python

             penalty_factor: float
             mapping: str

        """
        modelling_goal: float

    def map(self, problem: (list, float, list), config: Config) -> (Model, float):
        """
        Maps the bin packing problem input to a MIP formulation.

        :param problem: bin packing problem instance defined by
                    1. object weights, 2. bin capacity, 3. incompatible objects
        :type problem: list, float, list
        :param config: config with the parameters specified in Config class
        :type config: Config
        :return: docplex-model, time it took to map it
        :rtype: tuple(dict, float)
        """
        #TODO kwargs -->MIP und QUBO haben jeweils andere bessere Formulierungen
        start = start_time_measurement()
        self.problem = problem
        self.config = config
        
        # create the docplex MIP model
        return BPack.create_MIP(problem), end_time_measurement(start)
    

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
