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

# from modules.applications.Application import *
from modules.applications.qml.classification.data.data_handler.image_data import ImageData
from modules.applications.qml.qml import QML
from utils import end_time_measurement, start_time_measurement


class Classification(QML):
    """
    Image Classification using quantum circuits.
    """

    def __init__(self):
        """
        Constructor method.
        """
        super().__init__("Classification")
        self.submodule_options = ["Image Data"]
        self.data = None

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Returns requirements of this module.

        :return: list of dict with requirements of this module
        """
        # These requirements are used in the accompanying module QuantumModel
        return [
            {"name": "numpy", "version": "1.26.4"},
            {"name": "torch", "version": "2.2.2"},
            {"name": "pennylane", "version": "0.39.0"},
            {"name": "qiskit", "version": "1.3.0"},
            {"name": "qiskit-machine-learning", "version": "0.8.2"},
            {"name": "torchvision", "version": "0.17.2"},
        ]

    def get_solution_quality_unit(self) -> str:
        return "maximal Accuracy"

    def get_default_submodule(self, option: str) -> ImageData:
        """
        Returns the default submodule based on the given option.

        :param option: The submodule option to select
        :return: Instance of the selected submodule
        :raises NotImplemented: If the provided option is not implemented
        """
        if option == "Image Data":
            self.data = ImageData()
        else:
            raise NotImplementedError(f"Transformation Option {option} not implemented")
        return self.data

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this application

        :return: Dictionary of configurable parameters
        .. code-block:: python

            return {
                    "n_qubits": {
                    "values": [4, 6],
                    "description": "How many qubits do you want to use?"
                    }
                }

        """
        return {
            "n_qubits": {
                "values": [4, 6],
                "description": "How many qubits do you want to use?",
            }
        }

    class Config(TypedDict):
        """
        Attributes of a valid config

        .. code-block:: python

            n_qubits: int
        """

        n_qubits: int

    def generate_problem(self, config: dict) -> dict:
        """
        The number of qubits is chosen for this problem.

        :param config:
        :return: Dictionary with the number of qubits
        """

        application_config = {"n_qubits": config["n_qubits"]}
        return application_config

    def preprocess(self, input_data: dict, config: dict, **kwargs: dict) -> tuple[dict, float]:
        """
        Generate the actual problem instance in the preprocess function.

        :param input_data: Usually not used for this method
        :param config: Config for the problem creation
        :param kwargs: Optional additional arguments

        :return: Tuple containing qubit number and the function's computation time
        """
        start = start_time_measurement()
        output = self.generate_problem(config)
        output["store_dir_iter"] = f"{kwargs['store_dir']}/rep_{kwargs['rep_count']}"
        return output, end_time_measurement(start)

    def postprocess(self, input_data: dict, config: dict, **kwargs: dict) -> tuple[dict, float]:
        """
        Process the solution here, then validate and evaluate it.

        :param input_data: Representation of the quantum machine learning model that will be trained
        :param config: Config specifying the parameters of the training
        :param kwargs: Optional keyword arguments
        :return: Tuple with same dictionary like input_data and the time
        """

        start = start_time_measurement()
        return input_data, end_time_measurement(start)
