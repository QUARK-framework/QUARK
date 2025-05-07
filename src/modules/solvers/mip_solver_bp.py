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
from pathlib import Path
from typing import TypedDict
import pyscipopt as scip_opt

from modules.solvers.solver import *
from utils import start_time_measurement, end_time_measurement


class MIPSolver(Solver):
    """
    QAOA with some parts copied/derived from https://github.com/aws/amazon-braket-examples.
    """

    def __init__(self):
        """
        Constructor method
        """
        super().__init__()
        self.submodule_options = ["Local"]

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Return requirements of this module.

        :return: List of dict with requirements of this module
        """
        return [{"name": "pyscipopt", "version": "5.0.1"}]

    def get_default_submodule(self, option: str) -> Core:
        """
        Returns the default submodule based on the provided option.

        :param option: Option specifying the submodule
        :return: Instance of the corresponding submodule
        :raises NotImplementedError: If the option is not recognized
        """
        if option == "Local":
            from modules.devices.local import Local  # pylint: disable=C0415
            return Local()
        else:
            raise NotImplementedError(f"Device Option {option} not implemented")

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this solver.

        :return:
        .. code-block:: python

                    return {
                        "mip_gap": {  # number measurements to make on circuit
                            "values": [0], #default value 0 means optimal solution is required
                            "description": "What MIP-Gap do you allow?"
                        },
                        "solution_method": {
                            "values": [-1], # for gurobi:  -1=default automatic, 0=primal simplex,
                                    1=dual simplex, 2=barrier, 3=concurrent, 4=deterministic concurrent,
                                    5=deterministic concurrent simplex
                            "description": "Which optimization method do you want?"
                        },
                        "time_limit": {
                            "values": [60*60*2], # default value: 2 hours
                            "description": "How much time may the solving take?"
                        }
                    }
        """
        return {
            "mip_gap": {  # Number measurements to make on circuit
                "values": [0],  # Default value 0 means optimal solution is required
                "custom_input": True,
                "postproc": float,
                "description": "What relative gap to the global optimum do you allow (e.g., '0.01' means solutions "
                               "within one percent of the global optimum are accepted and the optimization terminates)?"
            },
            "time_limit": {
                "values": [60 * 60 * 2],  # Default value: 2 hours
                "custom_input": True,
                "postproc": int,
                "description": "How much time may the solving take (in seconds)?"
            }
        }

    class Config(TypedDict):
        """
        Attributes of a valid config.

        .. code-block:: python

            shots: int
            opt_method: str
            depth: int

        """
        mip_gap: int
        time_limit: int

    def run(self, mapped_problem: any, device_wrapper: any, config: Config, **kwargs: dict) -> tuple[dict, float, dict]:
        """
        Run MIP-solver on an optimization problem.

        :param mapped_problem: Optimization problem
        :param device_wrapper: Instance of device
        :param config: Configuration parameters for the solver
        :param kwargs: No additionally settings needed
        :return: Tuple consisting of solution and the time it took to compute it and  additional solution information
        """
        start = start_time_measurement()

        # Save mapped problem to result folder via lp
        export_path = kwargs['store_dir']
        mapped_problem.export_as_lp(basename="MIP", path=export_path)

        # Read the lp-file to get the model into a SCIP_OPT-model
        scip_model = scip_opt.Model()
        scip_model.readProblem(filename=Path(export_path) / Path("MIP.lp"))

        # Config scip solver
        scip_model.setParam("limits/gap", config["mip_gap"])
        scip_model.setParam("limits/time", config["time_limit"])

        # Start the optimization
        scip_model.optimize()

        # Get the optimization results
        if scip_model.getStatus() == 'infeasible':
            logging.warning('The problem is infeasible.')
            additional_solution_info = {'obj_value': None,
                                        'opt_status': 'infeasible'}
            return {}, end_time_measurement(), additional_solution_info
        else:
            if scip_model.getSols() == []:
                logging.warning('No solution found within time limit')
                additional_solution_info = {'obj_value': None,
                                            'opt_status': 'no solution found within time limit'}
                return {}, end_time_measurement(), additional_solution_info
            else:
                obj_value = scip_model.getObjVal()
                solution = scip_model.getBestSol()
                solution_dict = {}
                for var in scip_model.getVars():
                    var_name = var.__repr__()
                    var_value = solution[var]
                    solution_dict[var_name] = var_value
                additional_solution_info = {'obj_value': obj_value,
                                            'opt_status': 'optimal solution'}
                return solution_dict, end_time_measurement(start), additional_solution_info
