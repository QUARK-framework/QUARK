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
from pathlib import Path
from typing import TypedDict
import pdb
import pyscipopt as scip_opt  # Requires installation of Microsoft Visual C++

from modules.solvers.Solver import *
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
        Return requirements of this module

        :return: list of dict with requirements of this module
        :rtype: list[dict]
        """
        return [
            {
                "name": "pyscipopt",
                "version": "4.3.0"
            }
        ]

    def get_default_submodule(self, option: str) -> Core:

        if option == "Local":
            from modules.devices.Local import Local  # pylint: disable=C0415
            return Local()
        else:
            raise NotImplementedError(f"Device Option {option} not implemented")

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this solver

        :return:
                 .. code-block:: python

                              return {
                                  "mip_gap": {  # number measurements to make on circuit
                                      "values": [0], #default value 0 means optimal solution is required
                                      "description": "What MIP-Gap do you allow?"
                                  },
                                  "solution_method": {
                                      "values": [-1], # for gurobi:  -1=default automatic, 0=primal simplex, 1=dual simplex, 2=barrier, 3=concurrent, 4=deterministic concurrent, 5=deterministic concurrent simplex
                                      "description": "Which optimization method do you want?"
                                  },
                                  "time_limit": {
                                      "values": [60*60*2], # default value: 2 hours
                                      "description": "How much time may the solving take?"
                                  }
                              }

        """
        return {
            "mip_gap": {  # number measurements to make on circuit
                "values": [0], #default value 0 means optimal solution is required
                "description": "What MIP-Gap do you allow?"
            },
            "solution_method": {
                "values": [-1], # for gurobi:  -1=default automatic, 0=primal simplex, 1=dual simplex, 2=barrier, 3=concurrent, 4=deterministic concurrent, 5=deterministic concurrent simplex
                "description": "Which optimization method do you want?"
            },
            "time_limit": {
                "values": [60*60*2], # default value: 2 hours
                "description": "How much time may the solving take?"
            }
        }

    class Config(TypedDict):
        """
        Attributes of a valid config

        .. code-block:: python

            shots: int
            opt_method: str
            depth: int

        """
        mip_gap: int
        solution_method: str
        time_limit: int

    def run(self, mapped_problem: any, device_wrapper: any, config: Config, **kwargs: dict) -> (dict, float, dict):
        """
        Run MIP-solver on an optimization problem

        :param mapped_problem: optimization problem
        :type mapped_problem: any
        :param device_wrapper: instance of device
        :type device_wrapper: any
        :param config:
        :type config: Config
        :param kwargs: no additionally settings needed
        :type kwargs: any
        :return: Tuple consisting of solution as well as the time it took to compute it and  additional solution information
        :rtype: dict, float, dict
        """
        start = start_time_measurement()
        
        #save mapped problem to result folder via lp
        export_path = kwargs['store_dir']
        mapped_problem.export_as_lp(basename="MIP", path=export_path)
        # pdb.set_trace()
        #read the lp-file to get the model into a SCIP_OPT-model
        scip_model = scip_opt.Model()
        #scip_model =
        scip_model.readProblem(filename=Path(export_path) / Path("MIP.lp"))
        
        #start the optimization
        scip_model.optimize()
        
        #get the optimization results
        if scip_model.getStatus() == 'infeasible':
            print('infeasible')
            additional_solution_info = {'obj_value': None,
                               'opt_status': 'infeasible'}
            return {}, end_time_measurement(), additional_solution_info
        else:
            if scip_model.getSols() == []:
                print('no solution found within time limit')
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