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

from typing import TypedDict, Union
from dwave.system.composites import EmbeddingComposite

from devices.braket.DWave import DWave
from devices.SimulatedAnnealingSampler import SimulatedAnnealingSampler
from solvers.Solver import *


class Annealer(Solver):
    """
    Class for both quantum and simulated annealing.
    """

    def __init__(self):
        """
        Constructor method
        """
        super().__init__()
        self.device_options = ["SimulatedAnnealer", "arn:aws:braket:::device/qpu/d-wave/DW_2000Q_6",
                               "arn:aws:braket:::device/qpu/d-wave/Advantage_system4"]

    def get_device(self, device_option: str) -> Union[DWave, SimulatedAnnealingSampler]:
        if device_option == "arn:aws:braket:::device/qpu/d-wave/DW_2000Q_6":
            return DWave("DW_2000Q_6", "arn:aws:braket:::device/qpu/d-wave/DW_2000Q_6")
        if device_option == "arn:aws:braket:::device/qpu/d-wave/Advantage_system4":
            return DWave("Advantage_system4", "arn:aws:braket:::device/qpu/d-wave/Advantage_system4")
        elif device_option == "SimulatedAnnealer":
            return SimulatedAnnealingSampler()
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

    def run(self, mapped_problem: dict, device_wrapper: any, config: Config, **kwargs: dict) -> (dict, float):
        """
        Annealing Solver.

        :param mapped_problem: dictionary with the key 'Q' where its value should be the QUBO
        :type mapped_problem: dict
        :param device_wrapper: Annealing device
        :type device_wrapper: any
        :param config: Annealing settings
        :type config: Config
        :param kwargs:
        :type kwargs: any
        :return: Solution and the time it took for the annealer to compute it
        :rtype: tuple(dict, float)
        """

        Q = mapped_problem['Q']

        device = device_wrapper.get_device()
        start = time() * 1000
        if device_wrapper.device_name != "simulatedannealer":
            # This is for AWS
            # TODO Improve the embedding
            # EmbeddingComposite automatically maps the problem to the structure of the solver.
            device = EmbeddingComposite(device)
            response = device.sample_qubo(Q, num_reads=config['number_of_reads'], answer_mode="histogram")
        else:
            # This is for D-Wave simulated Annealer
            response = device.sample_qubo(Q, num_reads=config['number_of_reads'])
        time_to_solve = round(time() * 1000 - start, 3)

        # take the result with the lowest energy:
        sample = response.lowest().first.sample
        # logging.info("Result:" + str({k: v for k, v in sample.items() if v == 1}))
        logging.info(f'Annealing finished in {time_to_solve} ms.')

        return sample, time_to_solve
