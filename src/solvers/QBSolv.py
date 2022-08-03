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

import minorminer
import networkx as nx
from dwave.system.composites import FixedEmbeddingComposite
from dwave_qbsolv import QBSolv as dwave_qbsolv

from devices.SimulatedAnnealingSampler import SimulatedAnnealingSampler
from devices.braket.DWave import DWave
from solvers.Solver import *


class QBSolv(Solver):
    """
    QBSolv from D-Wave
    """

    def __init__(self):
        """
        Constructor method
        """
        super().__init__()
        self.device_options = ["arn:aws:braket:::device/qpu/d-wave/DW_2000Q_6",
                               "arn:aws:braket:::device/qpu/d-wave/Advantage_system4"]

    def get_device(self, device_option: str) -> DWave:
        if device_option == "arn:aws:braket:::device/qpu/d-wave/DW_2000Q_6":
            return DWave("DW_2000Q_6", "arn:aws:braket:::device/qpu/d-wave/DW_2000Q_6")
        if device_option == "arn:aws:braket:::device/qpu/d-wave/Advantage_system4":
            return DWave("Advantage_system4", "arn:aws:braket:::device/qpu/d-wave/Advantage_system4")
        else:
            raise NotImplementedError(f"Device Option {device_option}  not implemented")

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this solver

        :return:
                 .. code-block:: python

                              return {
                                        "number_of_reads": {
                                            "values": [100, 250, 500, 750, 1000],
                                            "description": "How many reads do you need?"
                                        },
                                        "solver_limit": {
                                            "values": [40, 50],
                                            "description": "Define size of the sub-problems"
                                        },
                                        "num_repeats": {
                                            "values": [2, 3],
                                            "description": "Define the number of repeats for the hybrid solver to search for the optimal solution."
                                        }
                                    }

        """
        return {
            "number_of_reads": {
                "values": [100, 250, 500, 750, 1000],
                "description": "How many reads do you need?"
            },
            "solver_limit": {
                "values": [40, 50],
                "description": "Define size of the sub-problems"
            },
            "num_repeats": {
                "values": [2, 3],
                "description": "Define the number of repeats for the hybrid solver to search for the optimal solution."
            }
        }

    class Config(TypedDict):
        """
        Attributes of a valid config

        .. code-block:: python

            number_of_reads: int
            solver_limit: int
            num_repeats: int

        """
        number_of_reads: int
        solver_limit: int
        num_repeats: int

    def run(self, mapped_problem: dict, device_wrapper: any, config: Config, **kwargs: dict) -> (dict, float):
        """
        Run QBSolv algorithm on QUBO formulation.

        :param mapped_problem: dictionary with the key 'Q' where its value should be the QUBO
        :type mapped_problem: dict
        :param device_wrapper: instance of an annealer
        :type device_wrapper: any
        :param config: annealing settings
        :type config: Config
        :param kwargs: no additionally settings needed
        :type kwargs: any
        :return: Solution, the time it took to compute it and optional additional information
        :rtype: tuple(list, float, dict)
        """

        Q = mapped_problem['Q']

        device = device_wrapper.get_device()
        start = time() * 1000

        # find embedding of subproblem-sized complete graph to the QPU
        G = nx.complete_graph(config['solver_limit'])
        embedding = minorminer.find_embedding(G.edges, device.edgelist)

        # use the FixedEmbeddingComposite() method with a fixed embedding
        solver = FixedEmbeddingComposite(device, embedding)

        response = dwave_qbsolv().sample_qubo(Q, solver=solver, num_repeats=config['num_repeats'],
                                              solver_limit=config['solver_limit'],
                                              num_reads=config['number_of_reads'])

        time_to_solve = round(time() * 1000 - start, 3)

        # take the result with the lowest energy:
        sample = response.lowest().first.sample
        logging.info("Result:" + str({k: v for k, v in sample.items() if v == 1}))

        return sample, time_to_solve, {}
