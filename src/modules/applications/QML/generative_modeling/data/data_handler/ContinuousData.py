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

from utils import start_time_measurement, end_time_measurement

from modules.applications.QML.generative_modeling.transformations.MinMax import MinMax
from modules.applications.QML.generative_modeling.transformations.PIT import PIT
from modules.applications.QML.generative_modeling.data.data_handler.DataHandler import *


class ContinuousData(DataHandler):
    """
    A data handler for continuous datasets. This class loads a dataset from a specified path and provides
    methods for data transformation and evaluation.

    """

    def __init__(self):
        """
        The continuous data class loads a dataset from the path 
        src/modules/applications/QML/generative_modeling/data
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
        Returns requirements of this module

        :return: list of dict with requirements of this module
        :rtype: list[dict]
        """
        return [
            {
                "name": "numpy",
                "version": "1.23.5"
            }
        ]

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
        Returns the configurable settings for this application

        :return:

                 .. code-block:: python

                        return {
                            "data_set": {
                                "values": ["X_2D", "O_2D", "MG_2D", "Stocks_2D"],
                                "description": "Which dataset do you want to use?"
                            }
                        }

        """
        return {
            "data_set": {
                "values": ["X_2D", "O_2D", "MG_2D", "Stocks_2D"],
                "description": "Which dataset do you want to use?"
            }
        }

    class Config(TypedDict):
        """
        Attributes of a valid config

        .. code-block:: python

            data_set: str

        """

        data_set: int

    def data_load(self, gen_mod: dict, config: Config) -> dict:

        """
        The chosen dataset is loaded and split into a training set.

        :param gen_mod: Dictionary with collected information of the previous modules
        :type gen_mod: dict
        :param config: Config specifying the paramters of the data handler
        :type config: dict
        :return: Must always return the mapped problem and the time it took to create the mapping
        :rtype: tuple(any, float)
        """
        self.dataset_name = config["data_set"]
        self.n_qubits = gen_mod["n_qubits"]

        filename = pkg_resources.resource_filename('modules.applications.QML.generative_modeling.data',
                                                   f"{self.dataset_name}.npy")
        self.dataset = np.load(filename)

        application_config = {
            "dataset_name": self.dataset_name,
            "n_qubits": self.n_qubits,
            "dataset": self.dataset,
            "store_dir_iter": gen_mod["store_dir_iter"]}

        return application_config

    def evaluate(self, solution: list, **kwargs) -> (float, float):
        """
        Calculate KL in original space.

        :param solution: A dictionary-like object containing the solution data, including histogram_generated_original
                         and histogram_train_original.
        :type solution: list
        :return: KL for the generated samples and the time it took to calculate it.
        :rtype: tuple(float, float)
        """
        start = start_time_measurement()

        generated = solution["histogram_generated_original"]
        generated[generated == 0] = 1e-8

        histogram_train_original = solution["histogram_train_original"]
        histogram_train_original[histogram_train_original == 0] = 1e-8

        # Flatten the arrays for efficient computation
        target = histogram_train_original.ravel()
        generated = generated.ravel()

        # Compute KL divergence using NumPy vectorized operations
        kl_divergence = self.kl_divergence(target, generated)

        logging.info(f"KL original space: {kl_divergence}")

        return kl_divergence, end_time_measurement(start)

    def kl_divergence(self, target, q):
        return np.sum(target * np.log(target / q))