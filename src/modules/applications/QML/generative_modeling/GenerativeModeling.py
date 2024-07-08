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

from typing import Union
from utils import start_time_measurement, end_time_measurement

from modules.applications.Application import *
from modules.applications.QML.QML import QML
from modules.applications.QML.generative_modeling.data.data_handler.DiscreteData import DiscreteData
from modules.applications.QML.generative_modeling.data.data_handler.ContinuousData import ContinuousData


class GenerativeModeling(QML):
    """
    Generative models enable the creation of new data by learning the underlying probability distribution of the
    training data set of interest. More specifically, a generative model attempts to learn an unknown probability
    distribution Q by modeling an approximated probability distribution P(θ) which is parameterized by a set of
    variables θ. Data sampled from Q is used to train a model by tuning θ such that P(θ) more closely approximates Q.
    """

    def __init__(self):
        """
        Constructor method
        """
        super().__init__("GenerativeModeling")
        self.submodule_options = ["Continuous Data", "Discrete Data"]
        self.data = None

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Returns requirements of this module

        :return: list of dict with requirements of this module
        :rtype: list[dict]
        """
        return []

    def get_solution_quality_unit(self) -> str:
        return "minimum KL"

    def get_default_submodule(self, option: str) -> Union[ContinuousData, DiscreteData]:
        if option == "Continuous Data":
            self.data = ContinuousData()
        elif option == "Discrete Data":
            self.data = DiscreteData()
        else:
            raise NotImplementedError(f"Transformation Option {option} not implemented")
        return self.data

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this application

        :return:
                 .. code-block:: python

                      return {
                                "n_qubits": {
                                "values": [4, 6, 8, 10, 12],
                                "description": "How many qubits do you want to use?"
                                }
                            }

        """
        return {
            "n_qubits": {
                "values": [6, 8, 10, 12],
                "description": "How many qubits do you want to use?"
            }
        }

    def generate_problem(self, config: dict) -> dict:

        """
        The number of qubits is chosen for this problem.

        :param config: dictionary including the number of qubits
        :type config: dict
        :return: dictionary with the number of qubits
        :rtype: dict
        """

        application_config = {"n_qubits": config["n_qubits"]}
        return application_config

    def preprocess(self, input_data: dict, config: dict, **kwargs) -> tuple[dict, float]:
        """
        Generate the actual problem instance in the preprocess function.
        :param input_data: Usually not used for this method.
        :type input_data: dict
        :param config: config for the problem creation.
        :type config: dict
        :param kwargs: Optional additional arguments
        :type kwargs: dict
        :param kwargs: optional additional arguments.

        :return: tuple containing qubit number and the function's computation time
        :rtype: tuple[dict, float]
        """
        start = start_time_measurement()
        output = self.generate_problem(config)
        output["store_dir_iter"] = f"{kwargs['store_dir']}/rep_{kwargs['rep_count']}"
        return output, end_time_measurement(start)

    def postprocess(self, input_data: dict, config: dict, **kwargs) -> tuple[dict, float]:
        """
        Process the solution here, then validate and evaluate it.

        :param input_data: A representation of the quantum machine learning model that will be trained
        :type input_data: dict
        :param config: Config specifying the parameters of the training
        :type config: dict
        :param kwargs: optional keyword arguments
        :type kwargs: dict
        :return: tuple with input_data and the function's computation time
        :rtype: (dict, float)
        """

        start = start_time_measurement()
        return input_data, end_time_measurement(start)
