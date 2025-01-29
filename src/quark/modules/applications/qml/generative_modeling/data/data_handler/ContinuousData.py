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
import logging

import numpy as np
import pkg_resources

from quark.utils import start_time_measurement, end_time_measurement
from quark.modules.applications.qml.generative_modeling.transformations.MinMax import MinMax
from quark.modules.applications.qml.generative_modeling.transformations.PIT import PIT
from quark.modules.applications.qml.generative_modeling.data.data_handler.DataHandlerGenerative import DataHandlerGenerative


class ContinuousData(DataHandlerGenerative):
    """
    A data handler for continuous datasets. This class loads a dataset from a specified path and provides
    methods for data transformation and evaluation.

    """

    def __init__(self):
        """
        The continuous data class loads a dataset from the path
        src/modules/applications/qml/generative_modeling/data
        """
        super().__init__("")
        self.submodule_options = ["PIT", "MinMax"]
        self.transformation = None
        self.dataset = None
        self.n_registers = None
        self.gc = None
        self.n_qubits = None

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Returns requirements of this module.

        :return: List of dict with requirements of this module
        """
        return [{"name": "numpy", "version": "1.26.4"}]

    def get_default_submodule(self, option: str) -> Union[PIT, MinMax]:
        if option == "MinMax":
            self.transformation = MinMax()
        elif option == "PIT":
            self.transformation = PIT()
        else:
            raise NotImplementedError(f"Transformation Option {option} not implemented")
        return self.transformation

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this application.

        :return: Dictionary of parameter options
        .. code-block:: python

            return {
                "data_set": {
                    "values": ["X_2D", "O_2D", "MG_2D", "Stocks_2D"],
                    "description": "Which dataset do you want to use?"
                },

                "train_size": {
                    "values": [0.1, 0.3, 0.5, 0.7, 1.0],
                    "description": "What percentage of the dataset do you want to use for training?"
                }
            }
        """
        return {
            "data_set": {
                "values": ["X_2D", "O_2D", "MG_2D", "Stocks_2D"],
                "description": "Which dataset do you want to use?"
            },

            "train_size": {
                "values": [0.1, 0.3, 0.5, 0.7, 1.0],
                "description": "What percentage of the dataset do you want to use for training?"
            }

        }

    class Config(TypedDict):
        """
        Attributes of a valid config.

        .. code-block:: python

            data_set: str

        """

        data_set: int
        train_size: float

    def data_load(self, gen_mod: dict, config: Config) -> dict:
        """
        The chosen dataset is loaded and split into a training set.

        :param gen_mod: Dictionary with collected information of the previous modules
        :param config: Config specifying the parameters of the data handler
        :return: Dictionary including the mapped problem
        """
        self.dataset_name = config["data_set"]
        self.n_qubits = gen_mod["n_qubits"]

        filename = pkg_resources.resource_filename(
            'modules.applications.qml.generative_modeling.data',
            f"{self.dataset_name}.npy"
        )
        self.dataset = np.load(filename)

        application_config = {
            "dataset_name": self.dataset_name,
            "n_qubits": self.n_qubits,
            "dataset": self.dataset,
            "train_size": config["train_size"],
            "store_dir_iter": gen_mod["store_dir_iter"]}

        return application_config

    def evaluate(self, solution: dict) -> tuple[float, float]:
        """
        Calculates KL in original space.

        :param solution: a dictionary containing the solution data, including histogram_generated_original
                         and histogram_train_original
        :return: Kullback-Leibler (KL) divergence for the generated samples and the time it took to calculate it
        """
        start = start_time_measurement()

        generated = solution["histogram_generated_original"]
        generated[generated == 0] = 1e-8

        histogram_train_original = solution["histogram_train_original"]
        histogram_train_original[histogram_train_original == 0] = 1e-8

        target = histogram_train_original.ravel()
        generated = generated.ravel()
        kl_divergence = self.kl_divergence(target, generated)

        logging.info(f"KL original space: {kl_divergence}")

        return kl_divergence, end_time_measurement(start)

    def kl_divergence(self, target: np.ndarray, q: np.ndarray) -> float:
        """
        Function to calculate KL divergence.

        :param target: Probability mass function of the target distribution
        :param q: Probability mass function generated by the quantum circuit
        :return: Kullback-Leibler divergence
        """
        if q.shape != target.shape:
            q = np.resize(q, target.shape)

        return np.sum(target * np.log(target / q))
