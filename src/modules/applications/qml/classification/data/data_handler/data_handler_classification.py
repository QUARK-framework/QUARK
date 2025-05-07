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
from abc import ABC, abstractmethod

import pandas as pd
from modules.core import *
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from utils import end_time_measurement, start_time_measurement


class DataHandler(Core, ABC):
    """
    The task of the DataHandler module is to translate the application’s data and problem specification into
    preprocessed format.
    """

    def __init__(self, name):
        """
        Constructor method.
        """
        super().__init__()
        self.dataset_name = name
        self.classification_mark = None

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Returns requirements of this module.

        :return: List of dicts with requirements of this module
        """
        return [
            {"name": "pandas", "version": "2.2.3"},
            {"name": "tensorboard", "version": "2.18.0"},
        ]

    def preprocess(self, input_data: dict, config: dict, **kwargs) -> tuple[dict, float]:
        """
        In this module, the preprocessing step is transforming the data to the correct target format.

        :param input_data: collected information of the benchmarking process
        :param config: config specifying the parameters of the training
        :param kwargs: optional additional settings
        :return: tuple with transformed problem and the time it took to map it
        """
        start = start_time_measurement()
        output = self.data_load(input_data, config)

        if "classification_metrics" in list(output.keys()):
            self.classification_mark = True

        return output, end_time_measurement(start)

    def postprocess(self, input_data: dict, config: dict, **kwargs) -> tuple[dict, float]:
        """
        In this module, the postprocessing step is transforming the data to the correct target format.

        :param input_data: Collected information of the benchmarking process
        :param config: Config specifying the parameters of the training
        :param kwargs: Optional additional settings
        :return: Tuple with an output_dictionary and the time it took
        """
        start = start_time_measurement()
        return input_data, end_time_measurement(start)

    @abstractmethod
    def data_load(self, gen_mod: dict, config: dict) -> dict:
        """
        Helps to ensure that the model can effectively learn the underlying patterns and structure of the data, and
        produce high-quality outputs.

        :param gen_mod: dictionary with collected information of the previous modules
        :param config: config specifying the parameters of the data handler
        :return: mapped problem and the time it took to create the mapping
        """
        raise NotImplementedError

    # def generalisation(self) -> tuple[dict, float]:
    #     """
    #     Compute generalisation metrics.
    #     :return: Evaluation and the time it took to create it
    #     """
    #     metrics = {}
    #     time_taken = 0.0
    #     return metrics, time_taken

    # @abstractmethod
    # def evaluate(self, solution: dict) -> tuple[dict, float]:
    #     """
    #     Computes the best loss values.
    #
    #     :param solution: Dictionary containing the solution data
    #     :return: Evaluated solution and the time it took to evaluate it
    #     """
    #     raise NotImplementedError

    @staticmethod
    def tb_to_pd(logdir: str, rep: str) -> None:
        """
        Converts TensorBoard event files in the specified log directory into a pandas DataFrame and saves it as a pickle
         file.

        :param logdir: Path to the log directory containing TensorBoard event files
        :param rep: Repetition number for naming the output file
        """
        event_acc = EventAccumulator(logdir)
        event_acc.Reload()
        tags = event_acc.Tags()
        tag_data = {}
        for tag in tags["scalars"]:
            data = event_acc.Scalars(tag)
            tag_values = [d.value for d in data]
            tag_data[tag] = tag_values
        data = pd.DataFrame(tag_data, index=[d.step for d in data])
        data.to_pickle(f"{logdir}/data_{rep}.pkl")
        data.to_pickle(f"{logdir}/data_{rep}.pkl")
