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
import numpy as np
from quark.modules.applications.qml.generative_modeling.training.TrainingGenerative import TrainingGenerative, Core, GPU


class Inference(TrainingGenerative):
    """
    This module executes a quantum circuit with parameters of a pretrained model.
    """

    def __init__(self):
        """
        Constructor method.
        """
        super().__init__("Inference")

        self.target: np.array
        self.n_states_range: list

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Returns requirements of this module.

        :return: List of dict with requirements of this module.
        """
        return [{"name": "numpy", "version": "1.26.4"}]

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this circuit.

        :return: Configuration settings for the pretrained model
        .. code-block:: python

            return {
                    "pretrained": {
                        "values": [False],
                        "custom_input": True,
                        "postproc": str,
                        "description": "Please provide the parameters of a pretrained model."
                    }
                }
        """
        return {
            "pretrained": {
                "values": [],
                "custom_input": True,
                "postproc": str,
                "description": "Please provide the parameters of a pretrained model."
            }
        }

    class Config(TypedDict):
        """
        Attributes of a valid config.

        .. code-block:: python

            pretrained: str

        """
        pretrained: str

    def get_default_submodule(self, option: str) -> Core:
        """
        Raises ValueError as this module has no submodules.

        :param option: Option name
        :raises ValueError: If called, since this module has no submodules
        """
        raise ValueError("This module has no submodules.")

    def start_training(self, input_data: dict, config: Config, **kwargs: dict) -> dict:
        """
        Method that uses a pretrained model for inference.

        :param input_data: Dictionary with information needed for inference
        :param config: Inference settings
        :param kwargs: Optional additional arguments
        :return: Dictionary including the information of previous modules as well as of this module
        """
        self.n_states_range = range(2 ** input_data['n_qubits'])
        self.target = np.asarray(input_data["histogram_train"])
        execute_circuit = input_data["execute_circuit"]

        parameters = np.load(config["pretrained"])

        pmfs, samples = execute_circuit([parameters.get() if GPU else parameters])
        pmfs = np.asarray(pmfs)
        samples = (
            self.sample_from_pmf(pmf=pmfs[0], n_shots=input_data["n_shots"])
            if samples is None
            else samples[0]
        )

        loss = self.kl_divergence(pmfs.reshape([-1, 1]), self.target)

        input_data["best_parameter"] = parameters.get() if GPU else parameters
        input_data["inference"] = True
        input_data["KL"] = [loss.get() if GPU else loss]
        input_data["best_sample"] = samples.astype(int).get() if GPU else samples.astype(int)

        return input_data
