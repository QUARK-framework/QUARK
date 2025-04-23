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
from modules.Core import *
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from utils import end_time_measurement, start_time_measurement


class DataHandler(Core, ABC):
    """
    The task of the DataHandler module is to translate the application’s data
    and problem specification into preprocessed format.
    """

    def __init__(self, name):
        """
        Constructor method
        """
        super().__init__()
        self.dataset_name = name
        self.classification_mark = None

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Returns requirements of this module

        :return: list of dict with requirements of this module
        :rtype: list[dict]
        """
        return [
            {"name": "pandas", "version": "1.5.2"},
            {"name": "tensorboard", "version": "2.13.0"},
        ]

    def preprocess(self, input_data: dict, config: dict, **kwargs):
        """
        In this module, the preprocessing step is transforming the data to the correct target format.

        :param input_data: collected information of the benchmarking process
        :type input_data: dict
        :param config: config specifying the parameters of the training
        :type config: dict
        :param kwargs: optional additional settings
        :type kwargs: dict
        :return: tuple with transformed problem and the time it took to map it
        :rtype: (dict, float)
        """
        start = start_time_measurement()
        output = self.data_load(input_data, config)

        if "classification_metrics" in list(output.keys()):
            self.classification_mark = True

        return output, end_time_measurement(start)

    def postprocess(self, input_data: dict, config: dict, **kwargs):
        """
        In this module, the postprocessing step is transforming the data to the correct target format.

        :param input_data: any
        :type input_data: dict
        :param config: config specifying the parameters of the training
        :type config: dict
        :param kwargs: optional additional settings
        :type kwargs: dict
        :return: tuple with an output_dictionary and the time it took
        :rtype: (dict, float)
        """
        start = start_time_measurement()
        return input_data, end_time_measurement(start)

    @abstractmethod
    def data_load(self, gen_mod: dict, config: dict) -> dict:
        """
        Helps to ensure that the model can effectively learn the underlying
        patterns and structure of the data, and produce high-quality outputs.

        :param gen_mod: dictionary with collected information of the previous modules
        :type gen_mod: dict
        :param config: config specifying the parameters of the data handler
        :type config: dict
        :return: mapped problem and the time it took to create the mapping
        :rtype: tuple(any, float)
        """
        raise NotImplementedError

    def generalisation(self) -> tuple[dict, float]:
        """
        Compute generalisation metrics

        :param solution:
        :type solution: any
        :return: Evaluation and the time it took to create it
        :rtype: tuple(any, float)

        """
        metrics = {}
        time_taken = 0.0
        return metrics, time_taken

    @abstractmethod
    def evaluate(self, solution: any) -> tuple[dict, float]:
        """
        Compute best loss values.

        :param solution:
        :type solution: any
        :return: bool and the time it took to create it
        :rtype: tuple(bool, float)

        """
        raise NotImplementedError

    @staticmethod
    def tb_to_pd(logdir: str, rep: str) -> None:
        """
        Converts TensorBoard event files in the specified log directory
        into a pandas DataFrame and saves it as a pickle file.

        :param logdir: path to the log directory containing TensorBoard event files
        :type logdir: str

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
