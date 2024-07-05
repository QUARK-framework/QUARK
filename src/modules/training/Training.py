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

import logging
from abc import ABC, abstractmethod
import time

try:
    import cupy as np
    GPU = True
    logging.info("Using CuPy, dataprocessing on GPU")
except ModuleNotFoundError:
    import numpy as np
    GPU = False
    logging.info("CuPy not available, using vanilla numpy, dataprocessing on CPU")

from modules.Core import Core
from utils import start_time_measurement, end_time_measurement

class Training(Core, ABC):
    """
    The Training module is the base class fot both finding (QCBM) and excuting trained models (Inference)
    """

    def __init__(self, name):
        """
        Constructor method
        """
        self.name = name
        super().__init__()

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

    def postprocess(self, input_data: dict, config: dict, **kwargs):
        """
        Here, the actual training of the machine learnning model is done

        :param input_data: Collected information of the benchmarking process
        :type input_data: dict
        :param config: Training settings
        :type config: dict
        :param kwargs: Optional additional arguments
        :type kwargs: dict
        :return: 
        :rtype: 
        """
        start = start_time_measurement()
        logging.info("Start training")
        training_results = self.start_training(
            self.preprocessed_input,
            config,
            **kwargs
        )
        for dict_key in ["backend", "circuit", "execute_circuit"]:
            training_results.pop(dict_key)
        postprocessing_time = end_time_measurement(start)
        logging.info(f"Training finished in {postprocessing_time / 1000} s.")
        return training_results, postprocessing_time

    @abstractmethod
    def start_training(self, input_data, config, **kwargs) -> dict:
        """
        This function starts the training of QML model or deploys a pretrained model.

        :param input_data: A representation of the quntum machine learning model that will be trained
        :type input_data: dict
        :param config: Config specifying the paramters of the training
        :type config: dict
        :param kwargs: optional additional settings
        :type kwargs: dict
        :return: Solution, the time it took to compute it and some optional additional information
        :rtype: dict
        """
        pass

    @staticmethod
    def sample_from_pmf(n_states_range, pmf, n_shots):
        samples = np.random.choice(n_states_range, size=n_shots, p=pmf)
        counts = np.bincount(samples, minlength=len(n_states_range))
        return counts

    def kl_divergence(self, pmf_model, pmf_target):
        pmf_model[pmf_model == 0] = 1e-8
        return np.sum(pmf_target * np.log(pmf_target / pmf_model), axis=1)

    def nll(self, pmf_model, pmf_target):
        pmf_model[pmf_model == 0] = 1e-8
        return -np.sum(pmf_target * np.log(pmf_model), axis=1)

    def mmd(self, pmf_model, pmf_target):
        pmf_model[pmf_model == 0] = 1e-8
        sigma = 1/pmf_model.shape[1] # TODO Improve scaling sigma and revise Formula
        kernel_distance = np.exp((-np.square(pmf_model - pmf_target) / (sigma ** 2)))
        mmd = 2 - 2 * np.mean(kernel_distance, axis=1)
        # The correct formula would take the transformed distances of both distributions into account. Since we are
        # not sampling from the distribution but using the probability mass function we can skip this step since the
        # sum of both, a modified version of the Gaussian kernel is used.
        return mmd

    class Timing:
        """
        This module is an abstraction of time measurement for for both CPU and GPU processes 
        """

        def __init__(self):
            """
            Constructor method
            """

            if GPU:
                self.start_cpu: time.perf_counter
            else:
                self.start_gpu: np.cuda.Event
                self.end_gpu: self.start_cpu

            self.start_recording = self.start_recording_gpu if GPU else self.start_recording_cpu
            self.stop_recording = self.stop_recording_gpu if GPU else self.stop_recording_cpu

        def start_recording_cpu(self):
            self.start_cpu = start_time_measurement()

        def stop_recording_cpu(self):
            return end_time_measurement(self.start_cpu)

        def start_recording_gpu(self):
            self.start_gpu = np.cuda.Event()
            self.end_gpu = np.cuda.Event()
            self.start_gpu.record()

        def stop_recording_gpu(self):
            self.end_gpu.record()
            self.end_gpu.synchronize()
            return np.cuda.get_elapsed_time(self.start_gpu, self.end_gpu)
