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
from typing import TypedDict

from qrisp import QuantumVariable
from qrisp.algorithms.qiro import (
    QIROProblem,
    create_max_indep_replacement_routine,
    create_max_indep_cost_operator_reduced,
    qiro_rx_mixer,
    qiro_init_function
)
from qrisp.qaoa import create_max_indep_set_cl_cost_function

from quark.modules.solvers.Solver import Solver, Core
from quark.utils import start_time_measurement, end_time_measurement


class QIROSolver(Solver):
    """
    Run the QIRO implementation within the Qrisp local simulator backend.
    """

    def __init__(self):
        """
        Constructor method.
        """
        super().__init__()
        self.submodule_options = ["qrisp_simulator"]

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Return requirements of this module.

        :return: List of dict with requirements of this module
        """
        return [{"name": "qrisp", "version": "0.5.2"}]

    def get_default_submodule(self, option: str) -> Core:
        """
        Returns the default submodule based on the given option.

        :param option: The submodule option to select
        :return: Instance of the selected submodule
        :raises NotImplemented: If the provided option is not implemented
        """
        if option == "qrisp_simulator":
            from quark.modules.devices.qrisp_simulator.QrispSimulator import QrispSimulator  # pylint: disable=C0415
            return QrispSimulator()  # pylint: disable=E1102

        else:
            raise NotImplementedError(f"Device Option {option} not implemented")

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this solver.

        :return:
        .. code-block:: python

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
                                "description": "How QIRO reps (Parameter: n) do you want? Choose this
                                                "parameter sensibly in relation to the graph size!"
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
            "depth": {  # depth of original QAOA
                "values": [2, 3, 4, 5, 10],
                "description": "How many layers for QAOA (Parameter: p) do you want?"
            },
            "QIRO_reps": {  # number of QIRO reps
                "values": [2, 3, 4, 5],
                "description": "How QIRO reps (Parameter: n) do you want? Choose this parameter sensibly in relation to"
                               " the graph size!"
            }
        }

    class Config(TypedDict):
        """
        Attributes of a valid config.

        .. code-block:: python

            shots: int
            depth: int
            iterations: int
            QIRO_reps: int

        """
        shots: int
        depth: int
        iterations: int
        QIRO_reps: int

    def run(self, mapped_problem: any, device_wrapper: any, config: Config, **kwargs: dict) -> tuple[any, float, dict]:
        """
        Run Qrisp QIRO on the local Qrisp simulator.

        :param mapped_problem: Dictionary with the keys 'graph' and 't'
        :param device_wrapper: Instance of device
        :param config: Solver configuration settings
        :param kwargs: no additionally settings needed, may include the measurement kwargs
        :return: Solution, the time it took to compute it and optional additional information
        """

        n = config["QIRO_reps"]
        p = config["depth"]
        shots = config["shots"]
        max_iter = config["iterations"]

        g = mapped_problem['graph']
        start = start_time_measurement()

        qiro_instance = QIROProblem(g,
                                    replacement_routine=create_max_indep_replacement_routine,
                                    cost_operator=create_max_indep_cost_operator_reduced,
                                    mixer=qiro_rx_mixer,
                                    cl_cost_function=create_max_indep_set_cl_cost_function,
                                    init_function=qiro_init_function
                                    )

        qarg = QuantumVariable(g.number_of_nodes())

        # Run actual optimization algorithm
        try:
            res_qiro = qiro_instance.run_qiro(qarg=qarg, depth=p, n_recursions=n, mes_kwargs={"shots": shots},
                                              max_iter=max_iter)
        except ValueError as e:
            logging.error(f"The following ValueError occurred in module QrispQIRO: {e}")
            logging.error("The benchmarking run terminates with exception.")
            raise Exception("Please refer to the logged error message.") from e
        best_bitstring = self._qiro_select_best_state(res_qiro, create_max_indep_set_cl_cost_function(g))

        def _translate_state_to_nodes(state: str, nodes: list) -> list:
            return [key for index, key in enumerate(nodes) if state[index] == '1']

        winner_state = _translate_state_to_nodes(best_bitstring, nodes=g.nodes())

        return winner_state, end_time_measurement(start), {}

    def _qiro_select_best_state(self, res_qiro, cost_func) -> str:
        """
        This function is used for post_processing, i.e. finding the best solution out of the 10 most likely ones.
        Since QIRO is an optimization algorithm, the most_likely solution can be of bad quality, depending on
        the problem cost landscape.

        :param res_qiro: Dictionary containing the QIRO optimization routine results, i.e. the final state.
        :param cost_func: classical cost function of the problem instance
        """
        maxfive = sorted(res_qiro, key=res_qiro.get, reverse=True)[:10]
        max_cost = 0
        best_state = "0" * len(maxfive[0])

        for key in maxfive:
            if cost_func({key: 1}) < max_cost:
                best_state = key
                max_cost = cost_func({key: 1})

        return best_state
