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
from abc import ABC
import time

try:
    import cupy as np
    GPU = True
    logging.info("Using CuPy, data processing on GPU")
except ModuleNotFoundError:
    import numpy as np
    GPU = False
    logging.info("CuPy not available, using vanilla numpy, data processing on CPU")

from modules.Core import Core
from modules.applications.qml.Training import Training
from utils import start_time_measurement, end_time_measurement


class TrainingGenerative(Core, Training, ABC):
    """
    The Training module is the base class fot both finding (QCBM) and executing trained models (Inference).
    """

    def __init__(self, name: str):
        """
        Constructor method.

        :param name: Name of the training instance
        """
        self.name = name
        super().__init__()
        self.n_states_range = None

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Returns requirements of this module.

        :return: list of dict with requirements of this module
        """
        return [{"name": "numpy", "version": "1.26.4"}]

    def postprocess(self, input_data: dict, config: dict, **kwargs) -> tuple[dict, float]:
        """
        Perform the actual training of the machine learning model.

        :param input_data: Collected information of the benchmarking process
        :param config: Training settings
        :param kwargs: Optional additional arguments
        :return: Training results and the postprocessing time
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

    def sample_from_pmf(self, pmf: np.ndarray, n_shots: int) -> np.ndarray:
        """
        This function samples from the probability mass function generated by the quantum circuit.

        :param pmf: Probability mass function generated by the quantum circuit
        :param n_shots: Number of shots
        :return: Number of counts in the 2**n_qubits bins
        """
        samples = np.random.choice(self.n_states_range, size=n_shots, p=pmf)
        counts = np.bincount(samples, minlength=len(self.n_states_range))
        return counts

    def kl_divergence(self, pmf_model: np.ndarray, pmf_target: np.ndarray) -> np.ndarray:
        """
        This function calculates the Kullback-Leibler divergence, that is used as a loss function.

        :param pmf_model: Probability mass function generated by the quantum circuit
        :param pmf_target: Probability mass function of the target distribution
        :return: Kullback-Leibler divergence
        """
        pmf_model[pmf_model == 0] = 1e-8
        return np.sum(pmf_target * np.log(pmf_target / pmf_model), axis=1)

    def nll(self, pmf_model: np.ndarray, pmf_target: np.ndarray) -> np.ndarray:
        """
        This function calculates th negative log likelihood, that is used as a loss function.

        :param pmf_model: Probability mass function generated by the quantum circuit
        :param pmf_target: Probability mass function of the target distribution
        :return: Negative log likelihood
        """
        pmf_model[pmf_model == 0] = 1e-8
        return -np.sum(pmf_target * np.log(pmf_model), axis=1)

    def mmd(self, pmf_model: np.ndarray, pmf_target: np.ndarray) -> np.ndarray:
        """
        This function calculates the maximum mean discrepancy, that is used as a loss function.

        :param pmf_model: Probability mass function generated by the quantum circuit
        :param pmf_target: Probability mass function of the target distribution
        :return: Maximum mean discrepancy
        """
        pmf_model[pmf_model == 0] = 1e-8
        sigma = 1 / pmf_model.shape[1]
        kernel_distance = np.exp((-np.square(pmf_model - pmf_target) / (sigma ** 2)))
        mmd = 2 - 2 * np.mean(kernel_distance, axis=1)
        return mmd

    class Timing:
        """
        This module is an abstraction of time measurement for both CPU and GPU processes.
        """

        def __init__(self):
            """
            Constructor method.
            """

            if GPU:
                self.start_cpu: time.perf_counter
            else:
                self.start_gpu: np.cuda.Event
                self.end_gpu: time.perf_counter

            self.start_recording = self.start_recording_gpu if GPU else self.start_recording_cpu
            self.stop_recording = self.stop_recording_gpu if GPU else self.stop_recording_cpu

        def start_recording_cpu(self) -> None:
            """
            This is a function to start time measurement on the CPU.
            """
            self.start_cpu = start_time_measurement()

        def stop_recording_cpu(self) -> float:
            """
            This is a function to stop time measurement on the CPU.

            .return: Elapsed time in milliseconds
            """
            return end_time_measurement(self.start_cpu)

        def start_recording_gpu(self) -> None:
            """
            This is a function to start time measurement on the GPU.
            """
            self.start_gpu = np.cuda.Event()
            self.end_gpu = np.cuda.Event()
            self.start_gpu.record()

        def stop_recording_gpu(self) -> float:
            """
            This is a function to stop time measurement on the GPU.

            :return: Elapsed time in milliseconds
            """
            self.end_gpu.record()
            self.end_gpu.synchronize()
            return np.cuda.get_elapsed_time(self.start_gpu, self.end_gpu)