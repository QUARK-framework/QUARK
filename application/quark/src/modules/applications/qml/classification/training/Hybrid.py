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
import time
from typing import TypedDict

import matplotlib
matplotlib.use("Agg")  # Use a non-interactive backend for matplotlib
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from matplotlib import axes, figure
from modules.applications.qml.classification.QuantumModel import QuantumModel

from modules.applications.qml.MetricsQuantum import MetricsQuantum
from modules.applications.qml.Training import Training
from modules.Core import Core
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from utils import end_time_measurement, start_time_measurement
from utils_mpi import get_comm, is_running_mpi

try:
    import cupy as np

    GPU = True
    logging.info("Using CuPy, data processing on GPU")
except ModuleNotFoundError:
    import numpy as np

    GPU = False
    logging.info("CuPy not available, using vanilla numpy, data processing on CPU")

MPI = is_running_mpi()
comm = get_comm()


class Hybrid(Core, Training):
    def __init__(self):
        """
        Constructor method.
        """
        super().__init__()  # "Hybrid")

        self.n_states_range: list
        self.target: np.array
        self.study_generalization: bool
        self.generalization_metrics: dict
        self.submodule_options = []
        self.writer: SummaryWriter
        self.loss_func: callable
        self.fig: figure
        self.ax: axes.Axes

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Returns requirements of this module.

        :return: list of dict with requirements of this module
        :rtype: list[dict]
        """
        return [
            {"name": "matplotlib", "version": "3.7.5"},
            {"name": "torch", "version": "2.2.2"},
            {"name": "tqdm", "version": "4.67.1"},
            {"name": "numpy", "version": "1.26.4"},
            {"name": "tensorboardX", "version": "2.6.2.2"},
        ]

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for the Quantum Circuit Born Machine.

        :return:
            .. code-block:: python

                return {
                    "n_epochs": {
                        "values": [1, 3, 7, 10, 20],
                        "description": "How many epochs do you want to train?"
                    },
                    "n_reduced_features": {
                        "values": [4],
                        "description": "Number of reduced features, also is the number of qubits in quantum layer"
                    },
                }
        """
        return {
            "n_epochs": {
                "values": [1, 3, 7, 10, 20],
                "description": "How many epochs do you want to train?",
            },
            "n_reduced_features": {
                "values": [4],
                "description": "Number of reduced features, also is the number of qubits in quantum layer",
            },
        }

    class Config(TypedDict):
        """
        Attributes of a valid config.

        .. code-block:: python

            n_epochs: int
            n_reduced_features: int

        """

        n_epochs: int
        n_reduced_features: int

    def get_default_submodule(self, option: str) -> None:
        """
        Raises ValueError as this module has no submodules.

        :param option: Option name
        :raises ValueError: If called, since this module has no submodules
        """
        raise ValueError("This module has no submodules.")

    def postprocess(self, input_data: dict, config: dict, **kwargs) -> tuple[dict, float]:
        """
        Here, the actual training of the machine learning model is done.

        :param input_data: Collected information of the benchmarking process
        :param config: Training settings
        :param kwargs: Optional additional arguments
        :return: Training results and computation time of postprocessing
        """
        start = start_time_measurement()
        logging.info("Start training")
        training_results = self.start_training(self.preprocessed_input, config, **kwargs)

        postprocessing_time = end_time_measurement(start)
        logging.info(f"Training finished in {postprocessing_time / 1000} s.")
        return training_results, postprocessing_time

    def setup_training(self, input_data: dict) -> None:
        """
        Sets up the training configuration.

        :param input_data: Dictionary with the variables needed to start the training
        """
        logging.info(f"Running config: [n_qubits={input_data['n_qubits']}]")

        self.study_classification = "classification_metrics" in list(input_data.keys())
        if self.study_classification:
            self.classification_metrics = input_data["classification_metrics"]

        self.writer = SummaryWriter(input_data["store_dir_iter"])

    def start_training(self, input_data: dict, config: Config, **kwargs: dict,) -> dict:
        """
        This function starts the hybrid training.

        :param input_data: Dataset to be trained on
        :param config: Config specifying the parameters of the training
        :param kwargs: Optional additional settings
        :return: Dictionary including the information of previous modules as well as of the training
        """
        self.model = QuantumModel(
            n_reduced_features=config["n_reduced_features"],
            dataset_name=input_data["dataset_name"],
            n_classes=input_data["n_classes"],
        )
        quantum_metrics_class = MetricsQuantum()
        circuit, params = self.model.get_quantum_circuit()
        print("Quantum layer:\n", circuit.draw(output="text"))
        quantum_metrics = quantum_metrics_class.get_metrics(circuit, params)
        self.metrics.add_metric_batch(
            {
                "meyer-wallach": quantum_metrics["meyer-wallach"],
                "expressibility_jsd": quantum_metrics["expressibility_jsd"],
            }
        )
        logging.info("Quantum metrics: %s", quantum_metrics)

        train_dataloader = input_data["dataset_train"]
        val_dataloader = input_data["dataset_val"]

        loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        lr_scheduler_step = 100
        lr_scheduler_gamma = 0
        lr_scheduler = StepLR(self.optimizer, step_size=lr_scheduler_step, gamma=lr_scheduler_gamma)

        self.setup_training(input_data)

        size = None
        input_data["MPI_size"] = size
        input_data["store_dir_iter"] += f"_{input_data['dataset_name']}_qubits{input_data['n_qubits']}"
        n_epochs = config["n_epochs"]

        self.n_states_range = range(2 ** input_data["n_qubits"])
        timing = self.Timing()

        for parameter in [
            "time_circuit",
            "time_loss",
            "train_accuracy",
            "train_loss",
            "validation_accuracy",
            "validation_loss",
        ]:
            input_data[parameter] = []

        best_loss = float("inf")
        train_loss = float("inf")
        time_circ = 0
        time_loss = 0
        self.fig, self.ax = plt.subplots()

        for epoch_idx in tqdm(range(n_epochs)):
            y_trues = []
            y_preds = []
            train_losses = []

            for val in tqdm(train_dataloader):
                if len(val) == 3:
                    x, y_true, _ = val
                elif len(val) == 2:
                    x, y_true = val
                else:
                    raise ValueError("Unexpected data format")
                timing.start_recording()
                y_raw = self.model(x)
                time_circ = timing.stop_recording()
                y_logits = nn.Softmax(dim=1)(y_raw)
                y_pred = np.argmax(y_logits.detach().numpy(), axis=1)

                timing.start_recording()
                train_loss = loss_fn(y_raw, y_true)
                train_losses.append(train_loss)

                y_trues = y_trues + list(y_true)
                y_preds = y_preds + list(y_pred)

                time_loss = timing.stop_recording()

                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()

            lr_scheduler.step()

            if train_loss < best_loss:
                best_loss = train_loss

            self.writer.add_scalar("loss/train", sum(train_losses) / len(train_losses), epoch_idx)
            input_data["train_loss"].append(float(train_loss))

            val_losses = []
            val_accuracies = []

            for val in tqdm(val_dataloader):
                if len(val) == 3:
                    x, y_true, _ = val
                elif len(val) == 2:
                    x, y_true = val
                else:
                    raise ValueError("Unexpected data format")
                y_raw = self.model(x)
                time_circ = timing.stop_recording()
                y_logits = nn.Softmax(dim=1)(y_raw)
                y_pred = np.argmax(y_logits.detach().numpy(), axis=1)

                val_losses.append(loss_fn(y_raw, y_true).detach().numpy())
                val_accuracies.append(sum(y_pred == y_true.detach().numpy()) / len(y_pred))

            val_loss = np.mean(val_losses)
            val_accuracy = np.mean(val_accuracies)

            self.writer.add_scalar("metrics/accuracy/validation", val_accuracy, epoch_idx)
            self.writer.add_scalar("loss/validation", val_loss, epoch_idx)

            input_data["validation_loss"].append(float(val_loss))
            input_data["validation_accuracy"].append(float(val_accuracy))

            metrics = self.classification_metrics.get_metrics(y_preds, y_trues)
            train_accuracy = 0
            for metric_name, metric_value in metrics.items():
                self.writer.add_scalar(f"metrics/{metric_name}/train", metric_value, epoch_idx)
                if metric_name == "accuracy":
                    train_accuracy = metric_value
                else:
                    train_accuracy = metrics["accuracy"]

            input_data["train_accuracy"].append(float(train_accuracy))

            logging.info(
                f"[Epoch {epoch_idx}/{n_epochs}] "
                f"[Train Loss: {train_loss:.5f}] "
                f"[Train Accuracy: {train_accuracy:.5f}] "
                f"[Validation Loss: {val_loss:.5f}] "
                f"[Validation Accuracy: {val_accuracy:.5f}] "
                f"[Circuit processing: {time_circ:.3f} ms] "
                f"[Loss processing: {time_loss:.3f} ms]"
            )

        self.trained = True

        self.metrics.add_metric_batch({"val_accuracy": val_accuracy, "train_accuracy": train_accuracy})

        plt.close()
        self.writer.flush()
        self.writer.close()

        return input_data

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
