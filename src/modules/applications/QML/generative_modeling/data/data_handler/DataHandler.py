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

import pickle
import os

import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from modules.Core import *
from utils import start_time_measurement, end_time_measurement


class DataHandler(Core, ABC):
    """
    The task of the DataHandler module is to translate the application’s data 
    and problem specification into preproccesed format.
    """

    def __init__(self, name):
        """
        Constructor method
        """
        super().__init__()
        self.dataset_name = name
        self.generalization_mark = None

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
            },
            {
                "name": "pandas",
                "version": "1.5.2"
            },
            {
                "name": "tensorboard",
                "version": "2.13.0"
            }
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

        if "generalization_metrics" in list(output.keys()):
            self.generalization_mark = True

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
        store_dir_iter = input_data["store_dir_iter"]
        current_directory = os.getcwd()

        if "samples_complete" in list(input_data.keys()):
            with open(f"{store_dir_iter}/samples_complete_{kwargs['rep_count']}.pkl", 'wb') as f:
                pickle.dump(input_data.pop("samples_complete"), f)

        evaluation, _ = self.evaluate(input_data)

        if self.generalization_mark is not None:
            self.metrics.add_metric_batch({"KL_best": evaluation["KL_best"]})
            metrics, _ = self.generalisation()

            # Save generalisation metrics
            with open(f"{store_dir_iter}/record_gen_metrics_{kwargs['rep_count']}.pkl", 'wb') as f:
                pickle.dump(metrics, f)

            self.metrics.add_metric_batch({"generalisation_metrics": metrics})

        else:
            self.metrics.add_metric_batch({"KL_best": evaluation})

        # Save metrics per iteration
        if "inference" not in input_data.keys():
            DataHandler.tb_to_pd(logdir=store_dir_iter, rep=kwargs['rep_count'])
            self.metrics.add_metric_batch(
                {"metrics_pandas": os.path.relpath(f"{store_dir_iter}/data.pkl", current_directory)})
            if self.generalization_mark is not None:
                np.save(f"{store_dir_iter}/histogram_generated.npy", evaluation["histogram_generated"])
            else:
                samples = input_data["best_sample"]
                n_shots = np.sum(samples)
                histogram_generated = np.asarray(samples) / n_shots
                histogram_generated[histogram_generated == 0] = 1e-8
                np.save(f"{store_dir_iter}/histogram_generated.npy", histogram_generated)
            self.metrics.add_metric_batch({"histogram_generated": os.path.relpath(
                f"{store_dir_iter}/histogram_generated.npy_{kwargs['rep_count']}.npy", current_directory)})

            # Save histogram generated dataset
            np.save(f"{store_dir_iter}/histogram_train.npy", input_data.pop("histogram_train"))
            self.metrics.add_metric_batch({"histogram_train": os.path.relpath(
                f"{store_dir_iter}/histogram_train.npy_{kwargs['rep_count']}.npy", current_directory)})

            # Save best parameters
            np.save(f"{store_dir_iter}/best_parameters_{kwargs['rep_count']}.npy", input_data.pop("best_parameter"))
            self.metrics.add_metric_batch({"best_parameter": os.path.relpath(
                f"{store_dir_iter}/best_parameters_{kwargs['rep_count']}.npy", current_directory)})

            # Save training results
            with open(f"{store_dir_iter}/training_results-{kwargs['rep_count']}.pkl", 'wb') as f:
                pickle.dump(input_data, f)

            # Save variables transformed space
            if "Transformation" in list(input_data.keys()):
                self.metrics.add_metric_batch({"KL_best": input_data["KL_best_transformed"]})
                np.save(f"{store_dir_iter}/histogram_train_original.npy", input_data.pop("histogram_train_original"))
                np.save(f"{store_dir_iter}/histogram_generated_original.npy",
                        input_data.pop("histogram_generated_original"))
                with open(f"{store_dir_iter}/best_samples_transformed_{kwargs['rep_count']}.pkl", 'wb') as f:
                    pickle.dump(input_data["transformed_samples"], f)

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
        pass

    def generalisation(self) -> (dict, float):
        """
        Compute generalisation metrics

        :param solution:
        :type solution: any
        :return: Evaluation and the time it took to create it
        :rtype: tuple(any, float)

        """
        # Compute your metrics here
        metrics = {}  # Replace with actual metric calculations
        time_taken = 0.0  # Replace with actual time calculation
        return metrics, time_taken

    @abstractmethod
    def evaluate(self, solution: any) -> (dict, float):
        """
        Compute best loss values.

        :param solution:
        :type solution: any
        :return: bool and the time it took to create it
        :rtype: tuple(bool, float)

        """
        pass

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
        for tag in tags['scalars']:
            data = event_acc.Scalars(tag)
            tag_values = [d.value for d in data]
            tag_data[tag] = tag_values
        data = pd.DataFrame(tag_data, index=[d.step for d in data])
        data.to_pickle(f"{logdir}/data_{rep}.pkl")
