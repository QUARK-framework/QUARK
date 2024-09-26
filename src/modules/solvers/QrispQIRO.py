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
import logging
from typing import Tuple
from typing import TypedDict

import os
import numpy as np

# from qiskit_ibm_runtime import QiskitRuntimeService

from modules.solvers.Solver import *
from utils import start_time_measurement, end_time_measurement


class QIROSolver(Solver):
    """
    Qrisp QIRO.
    run the QIRO implementation within the Qrisp local simulator Backend. Further Backends TBD
    """

    def __init__(self):
        """
        Constructor method
        """
        super().__init__()
        # self.submodule_options = ["qasm_simulator", "qasm_simulator_gpu", "ibm_eagle"]
        self.submodule_options = ["qrisp_simulator"]

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Return requirements of this module

        :return: list of dict with requirements of this module
        :rtype: list[dict]
        """
        return [
            {
                "name": "qrisp",
                "version": "0.4.12"
            }
        ]

    def get_default_submodule(self, option: str) -> Core:
        # TBD?
        if option == "qrisp_simulator":
            from modules.devices.HelperClass import HelperClass  # pylint: disable=C0415
            return HelperClass("qrisp_simulator")

        else:
            raise NotImplementedError(f"Device Option {option} not implemented")

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this solver.

        :return:
                 .. code-block:: python

                              return {
                                        "shots": {  # number measurements to make on circuit
                                            "values": list(range(10, 500, 30)),
                                            "description": "How many shots do you need?"
                                        },
                                        "iterations": {  # number measurements to make on circuit
                                            "values": [1, 5, 10, 20, 50, 75],
                                            "description": "How many iterations do you need? Warning: When using\
                                            the IBM Eagle Device you should only choose a lower number of\
                                            iterations, since a high number would lead to a waiting time that\
                                            could take up to mulitple days!"
                                        },
                                        "depth": {
                                            "values": [2, 3, 4, 5, 10, 20],
                                            "description": "How many layers for QAOA (Parameter: p) do you want?"
                                        },
                                        "method": {
                                            "values": ["classic", "vqe", "qaoa"],
                                            "description": "Which Qiskit solver should be used?"
                                        },
                                        "optimizer": {
                                            "values": ["POWELL", "SPSA", "COBYLA"],
                                            "description": "Which Qiskit solver should be used? Warning: When\
                                            using the IBM Eagle Device you should not use the SPSA optimizer,\
                                            since it is not suited for only one evaluation!"
                                        }
                                    }

        """
        return {
            "shots": {  # number measurements to make on circuit
                "values": [10, 500, 1000, 2000, 5000, 10000],
                "description": "How many shots do you need?"
            },
            "iterations": {  # number of optimization iterations
                "values": [5, 10, 20, 50, 75],
                "description": "How many optimization iterations do you need?"
            },
            "depth": { # depth of original QAOA
                "values": [2, 3, 4, 5, 10],
                "description": "How many layers for QAOA (Parameter: p) do you want?"
            },
            "QIRO_reps": { # number of QIRO reps
                "values": [2, 3, 4, 5],
                "description": "How QIRO reps (Parameter: n) do you want? Choose this parameter sensibly in relation to the graph size!"
            }
        }

        ##############FURTHER OPTIONS TO BE INCLUDED (MAYBE)
        """ 
            # do i want to do something here?
            "method": {
                "values": ["classic", "vqe", "qaoa"],
                "description": "Which Qiskit solver should be used?"
            },
            "optimizer": {
                "values": ["not", "yet", "implemented"],
                "description": "Which QIRO solver should be used?"
            } """

    class Config(TypedDict):
        """
        Attributes of a valid config.

        .. code-block:: python

            shots: int
            depth: int
            iterations: int
            layers: int
            method: str

        """
        shots: int
        depth: int
        iterations: int
        layers: int
        method: str



    def run(self, mapped_problem: any, device_wrapper: any, config: Config, **kwargs: dict) -> (any, float):
        """
        Run Qrisp QIRO on the local Qrisp simulator

        :param mapped_problem: dictionary with the keys 'graph' and 't'
        :type mapped_problem: nx.Graph
        :param device_wrapper: instance of device
        :type device_wrapper: any
        :param config:
        :type config: Config
        :param kwargs: no additionally settings needed, may include the measurement kwargs
        :type kwargs: any
        :return: Solution, the time it took to compute it and optional additional information
        :rtype: tuple(list, float, dict)
        """
        
        
        n = config["QIRO_reps"]
        p = config["depth"]
        shots = config["shots"]
        max_iter = config["iterations"]


        G = mapped_problem['graph']
        import networkx as nx
        start = start_time_measurement()
        # imports
        from qrisp.qiro import QIROProblem, qiro_init_function, qiro_RXMixer, create_maxIndep_replacement_routine, create_maxIndep_cost_operator_reduced
        from qrisp.qaoa.problems.maxIndepSetInfrastr import maxIndepSetclCostfct
        from qrisp import QuantumVariable
        
        qiro_instance = QIROProblem(G,
                            replacement_routine=create_maxIndep_replacement_routine,
                            cost_operator= create_maxIndep_cost_operator_reduced,
                            mixer= qiro_RXMixer,
                            cl_cost_function= maxIndepSetclCostfct,
                            init_function= qiro_init_function
                            )
        

        qarg = QuantumVariable(G.number_of_nodes())
        # run actual optimization algorithm
        try:
            # We run the qiro instance and get the results!
            res_qiro = qiro_instance.run_qiro(qarg=qarg, depth = p, n_recursions = n, mes_kwargs={"shots":shots}, max_iter=max_iter)
        except ValueError as e:
            logging.error(f"The following ValueError occurred in module QrispQIRO: {e}")
            logging.error("The benchmarking run terminates with exception.")
            raise Exception("Please refer to the logged error message.") from e
        #print("QIRO best result")
        #best_bitstring  = max(res_qiro, key=res_qiro.get)
        best_bitstring = self._qiro_select_best_state(res_qiro,maxIndepSetclCostfct(G))
        #print(best_bitstring)

        def _translate_state_to_nodes(state:str, nodes:list) -> list:
            return [key for index, key in enumerate(nodes) if state[index] == '1']
        
        winner_state = _translate_state_to_nodes(best_bitstring, nodes = G.nodes())

        #print(winner_state)
        return winner_state, end_time_measurement(start), {}
    

    # post_processing: find the best solution out of the 10 most likely ones.
    # QIRO is an optimization algo, the most_likely solution is mostly of bad quality, unless the problem is complex
    def _qiro_select_best_state(self,  res_qiro, costFunc ) -> str:
        maxfive = sorted(res_qiro, key=res_qiro.get, reverse=True)[:10]
        max_cost = 0 
        best_state = "0" * len(maxfive[0])
        for key in maxfive:
            if costFunc({key:1}) < max_cost:
                best_state = key
                max_cost = costFunc({key:1})
        
        return best_state