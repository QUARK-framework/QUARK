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
import logging

from quark.modules.solvers.Solver import Solver
from quark.modules.Core import Core
from quark.utils import start_time_measurement, end_time_measurement


class Annealer(Solver):
    """
    Class for both quantum and simulated annealing.
    """

    def __init__(self):
        """
        Constructor method.
        """
        super().__init__()
        self.submodule_options = ["Simulated Annealer"]

    def get_default_submodule(self, option: str) -> Core:
        """
        Returns the default submodule based on the provided option.

        :param option: The name of the submodule
        :return: Instance of the default submodule
        """
        if option == "Simulated Annealer":
            from quark.modules.devices.SimulatedAnnealingSampler import SimulatedAnnealingSampler  # pylint: disable=C0415
            return SimulatedAnnealingSampler()
        else:
            raise NotImplementedError(f"Device Option {option}  not implemented")

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this solver.

        :return: Dictionary of parameter options
        .. code-block:: python

            return {
                    "number_of_reads": {
                        "values": [100, 250, 500, 750, 1000],
                        "description": "How many reads do you need?"
                    }
                }

        """
        return {
            "number_of_reads": {
                "values": [100, 250, 500, 750, 1000],
                "description": "How many reads do you need?"
            }
        }

    class Config(TypedDict):
        """
        Attributes of a valid config.

        .. code-block:: python

            number_of_reads: int

        """
        number_of_reads: int

    def run(self, mapped_problem: dict, device_wrapper: any, config: Config, **kwargs: dict) \
            -> tuple[dict, float, dict]:
        """
        Run the annealing solver.

        :param mapped_problem: Dict with the key 'Q' where its value should be the QUBO
        :param device_wrapper: Annealing device
        :param config: Annealing settings
        :param kwargs: Additional keyword arguments
        :return: Solution, the time it took to compute it and optional additional information
        """

        q = mapped_problem['Q']
        additional_solver_information = {}
        device = device_wrapper.get_device()
        start = start_time_measurement()

        if device_wrapper.device_name != "simulated annealer":
            logging.error("Only simulated annealer available at the moment!")
            logging.error("Please select another solver module.")
            logging.error("The benchmarking run terminates with exception.")
            raise Exception("Please refer to the logged error message.")

        response = device.sample_qubo(q, num_reads=config['number_of_reads'])
        time_to_solve = end_time_measurement(start)

        # Take the result with the lowest energy:
        sample = response.lowest().first.sample
        logging.info(f'Annealing finished in {time_to_solve} ms.')

        return sample, time_to_solve, additional_solver_information
