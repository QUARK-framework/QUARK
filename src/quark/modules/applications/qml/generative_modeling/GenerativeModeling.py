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
from quark.utils import start_time_measurement, end_time_measurement

from quark.modules.applications.qml.QML import QML
from quark.modules.applications.qml.generative_modeling.data.data_handler.DiscreteData import DiscreteData
from quark.modules.applications.qml.generative_modeling.data.data_handler.ContinuousData import ContinuousData


class GenerativeModeling(QML):
    """
    Generative models enable the creation of new data by learning the underlying probability distribution of the
    training data set of interest. More specifically, a generative model attempts to learn an unknown probability
    distribution Q by modeling an approximated probability distribution P(θ) which is parameterized by a set of
    variables θ. Data sampled from Q is used to train a model by tuning θ such that P(θ) more closely approximates Q.
    """

    def __init__(self):
        """
        Constructor method.
        """
        super().__init__("GenerativeModeling")
        self.submodule_options = ["Continuous Data", "Discrete Data"]
        self.data = None

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Returns requirements of this module.

        :return: List of dicts with requirements of this module
        """
        return []

    def get_solution_quality_unit(self) -> str:
        return "minimum KL"

    def get_default_submodule(self, option: str) -> Union[ContinuousData, DiscreteData]:
        """
        Returns the default submodule based on the given option.

        :param option: The submodule option to select
        :return: Instance of the selected submodule
        :raises NotImplemented: If the provided option is not implemented
        """
        if option == "Continuous Data":
            self.data = ContinuousData()
        elif option == "Discrete Data":
            self.data = DiscreteData()
        else:
            raise NotImplementedError(f"Transformation Option {option} not implemented")
        return self.data

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this application.

        :return: Dictionary of configurable parameters
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

        :param config: Dictionary including the number of qubits
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

        :param input_data: A representation of the quantum machine learning model that will be trained
        :param config: Config specifying the parameters of the training
        :param kwargs: Optional keyword arguments
        :return: Tuple with input_data and the function's computation time
        """
        start = start_time_measurement()
        return input_data, end_time_measurement(start)
