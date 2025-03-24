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
from typing import TypedDict

import matplotlib.pyplot as plt
from cma import CMAEvolutionStrategy
from matplotlib import axes, figure
from modules.applications.qml.Training import *
from modules.Core import Core
from tensorboardX import SummaryWriter
from utils import start_time_measurement, end_time_measurement
from utils_mpi import get_comm, is_running_mpi

MPI = is_running_mpi()
comm = get_comm()


# completely copied from QCBM, to edit
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

from typing import TypedDict
import logging
from tensorboardX import SummaryWriter
from matplotlib import figure, axes
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from modules.applications.qml.classification.QuantumModel import QuantumModel
from modules.applications.qml.classification.training.ClassifierTraining import *
from utils_mpi import is_running_mpi, get_comm
from torch.optim.lr_scheduler import StepLR
from modules.applications.qml.MetricsQuantum import MetricsQuantum

MPI = is_running_mpi()
comm = get_comm()

class Hybrid(Core, Training, ABC):

    def __init__(self):
        """
        Constructor method
        """
        super().__init__()#"Hybrid")

        self.n_states_range: list
        self.target: np.array
        self.study_generalization: bool
        self.generalization_metrics: dict
        self.sub_options = ["Hybrid"] 
        self.writer: SummaryWriter
        self.loss_func: callable
        self.fig: figure
        self.ax: axes.Axes

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Returns requirements of this module

        :return: list of dict with requirements of this module
        :rtype: list[dict]
        """
        return [
            {
                "name": "pandas",
                "version": "2.2.2"
            },
            {
                "name": "qiskit",
                "version": "0.40.0"
            },
            {
                "name": "qiskit-machine-learning",
                "version": "0.6.1"
            },
            {
                "name": "qleet",
                "version": "0.2.0.1"
            },
            {
                "name": "pennylane",
                "version": "0.35.1"
            },
            {
                "name": "pillow",
                "version": "10.3.0"
            },
            {
                "name": "scikit-learn",
                "version": "1.4.2"
            },
            {
                "name": "torch",
                "version": "2.2.2"
            },
            {
                "name": "torchvision",
                "version": "0.17.2"
            },
            {
                "name": "tqdm",
                "version": "4.63.0"
            },
            {
                "name": "numpy",
                "version": "1.26.4"
            },
            {
                "name": "tensorboard",
                "version": "2.11.0"
            },
            {
                "name": "tensorboardX",
                "version": "2.5.0"
            },
    
            {
                "name": "cirq",
                "version": "0.13.1"
            },
            {
                "name": "plotly",
                "version": "5.1.0"
            },
            {
                "name": "cma",
                "version": "3.3.0"
            },
            {
                "name": "tensorflow",
                "version": "2.7.0"
            },
            {
                "name": "tensorflow-quantum",
                "version": "0.7.2"
            },
            {
                "name": "protobuf",
                "version": "3.17.3"
            }
        ]

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for the quantum circuit born machine

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
                "description": "How many epochs do you want to train?"
            },
            "n_reduced_features": {
                "values": [4],
                "description": "Number of reduced features, also is the number of qubits in quantum layer"
            },
        }

    class Config(TypedDict):
        """
        Attributes of a valid config

        .. code-block:: python

            n_epochs: int
            n_reduced_features: int

        """
        usecase_name: str
        n_classes: int
        n_epochs: int
        n_reduced_features: int
        n_images_per_class: int
        dataset_name: str
        
        

    def get_default_submodule(self, option: str) -> Core:
        raise ValueError("This module has no submodules.")

    def postprocess(self, input_data: dict, config: dict, **kwargs):
        """
        Here, the actual training of the machine learning model is done

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

        postprocessing_time = end_time_measurement(start)
        logging.info(f"Training finished in {postprocessing_time / 1000} s.")
        return training_results, postprocessing_time
    
    def setup_training(self, input_data, config) -> tuple:
        logging.info(
            f"Running config: [n_qubits={input_data['n_qubits']}]")

        self.study_classification = "classification_metrics" in list(input_data.keys())
        if self.study_classification:
            self.classification_metrics = input_data["classification_metrics"]
            
        self.writer = SummaryWriter(input_data["store_dir_iter"])

    def start_training(self, input_data: dict, config: Config, **kwargs: dict) -> (dict, float):
        """

        :param input_data: the dataset to be trained on
        :type input_data: dict
        :param config: Config specifying the paramters of the training
        :type config: dict
        :param kwargs: optional additional settings
        :type kwargs: dict
        :return: Dictionary including the information of previous modules as well as of the training
        :rtype: dict
        """
        self.model = QuantumModel(n_reduced_features=config['n_reduced_features'], dataset_name=config["input_data"], n_classes=config["n_classes"])
        quantum_metrics_class = MetricsQuantum()
        circuit, params = self.model.get_quantum_circuit()
        print("Quantum layer:\n", circuit.draw(output='text'))
        quantum_metrics = quantum_metrics_class.get_metrics(circuit, params)
        self.metrics.add_metric_batch({
            "meyer-wallach": quantum_metrics["meyer-wallach"],
            "expressibility_jsd": quantum_metrics["expressibility_jsd"]
        })
        logging.info('Quantum metrics: %s', quantum_metrics)

        train_dataloader = input_data["dataset_train"]
        val_dataloader = input_data["dataset_val"]
        test_dataloader = input_data["dataset_test"]

        loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        lr_scheduler_step = 100
        lr_scheduler_gamma = 0
        lr_scheduler = StepLR(
            self.optimizer, step_size=lr_scheduler_step, gamma=lr_scheduler_gamma
        )

        self.setup_training(input_data, config)

        size = None
        input_data['MPI_size'] = size
        input_data["store_dir_iter"] += f"_{input_data['dataset_name']}_qubits{input_data['n_qubits']}"
        n_epochs = config['n_epochs']

        self.n_states_range = range(2 ** input_data['n_qubits'])
        timing = self.Timing()

        for parameter in ["time_circuit", "time_loss", "train_accuracy", "train_loss", "validation_accuracy",
                          "validation_loss"]:
            input_data[parameter] = []

        best_loss = float("inf")
        self.fig, self.ax = plt.subplots()

        for epoch_idx in tqdm(range(n_epochs)):
            y_trues = []
            y_preds = []
            train_losses = []
            val_losses = []

            for val in tqdm(train_dataloader):
                if len(val) == 3: 
                    X, y_true, _ = val
                elif len(val) == 2:
                    X, y_true = val
                timing.start_recording()
                y_raw = self.model(X)
                time_circ = timing.stop_recording()
                y_logits = nn.Softmax(dim=1)(y_raw)
                y_pred = np.argmax(
                    y_logits.detach().numpy(), axis=1
                )

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

            self.writer.add_scalar(f"loss/train", sum(train_losses)/len(train_losses), epoch_idx)
            input_data["train_loss"].append(float(train_loss))

            val_losses = []
            val_accuracies = []

            for val in tqdm(val_dataloader):
                if len(val) == 3:
                    X, y_true, _ = val
                elif len(val) == 2:
                    X, y_true = val
                y_raw = self.model(X)
                time_circ = timing.stop_recording()
                y_logits = nn.Softmax(dim=1)(y_raw)
                y_pred = np.argmax(
                    y_logits.detach().numpy(), axis=1
                )

                val_losses.append(loss_fn(y_raw, y_true).detach().numpy())
                val_accuracies.append(sum(y_pred == y_true.detach().numpy()) / len(
                    y_pred
                ))


            val_loss = np.mean(val_losses)
            val_accuracy = np.mean(val_accuracies)

            self.writer.add_scalar(f"metrics/accuracy/validation", val_accuracy, epoch_idx)
            self.writer.add_scalar(f"loss/validation", val_loss, epoch_idx)
            
            input_data["validation_loss"].append(float(val_loss))
            input_data["validation_accuracy"].append(float(val_accuracy))

            metrics = self.classification_metrics.get_metrics(y_preds, y_trues)
            for (metric_name, metric_value) in metrics.items():
                self.writer.add_scalar(f"metrics/{metric_name}/train", metric_value, epoch_idx)
                if metric_name == "accuracy":
                    train_accuracy = metric_value

            input_data["train_accuracy"].append(float(train_accuracy))
            
            logging.info(
                f"[Epoch {epoch_idx}/{n_epochs}] "
                f"[Train Loss: {train_loss:.5f}] "\
                f"[Train Accuracy: {train_accuracy:.5f}] "\
                f"[Validation Loss: {val_loss:.5f}] "\
                f"[Validation Accuracy: {val_accuracy:.5f}] "\
                f"[Circuit processing: {(time_circ):.3f} ms] "\
                f"[Loss processing: {(time_loss):.3f} ms]")

        self.trained = True
        
        self.metrics.add_metric_batch({
                "val_accuracy": val_accuracy,
                "train_accuracy": train_accuracy
            })

        plt.close()
        self.writer.flush()
        self.writer.close()

        return input_data

class QCBM(Training):
    """
    This module optmizes the paramters of quantum circuit using CMA-ES.
    This training method is referred to as quantum circuit born machine (QCBM).
    """

    def __init__(self):
        """
        Constructor method
        """
        super().__init__("QCBM")

        self.n_states_range: list
        self.target: np.array
        self.study_generalization: bool
        self.generalization_metrics: dict
        self.writer: SummaryWriter
        self.loss_func: callable
        self.fig: figure
        self.ax: axes.Axes

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Returns requirements of this module

        :return: list of dict with requirements of this module
        :rtype: list[dict]
        """
        return [
            {"name": "numpy", "version": "1.23.5"},
            {"name": "cma", "version": "3.3.0"},
            {"name": "tensorboard", "version": "2.13.0"},
            {"name": "tensorboardX", "version": "2.6.2"},
        ]

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for the quantum circuit born machine

        :return:
            .. code-block:: python

                return {

                    "population_size": {
                        "values": [5, 10, 100, 200, 10000],
                        "description": "What population size do you want?"
                    },

                    "max_evaluations": {
                        "values": [100, 1000, 20000, 100000],
                        "description": "What should be the maximum number of evaluations?"
                    },

                    "sigma": {
                        "values": [0.01, 0.5, 1, 2],
                        "description": "Which sigma would you like to use?"
                    },

                    "pretrained": {
                        "values": [False],
                        "custom_input": True,
                        "postproc": str,
                        "description": "Please provide the parameters of a pretrained model?"
                    },

                    "loss": {
                        "values": ["KL", "NLL"],
                        "description": "Which loss function do you want to use?"
                    }
                }
        """
        return {
            "population_size": {
                "values": [5, 10, 100, 200, 10000],
                "description": "What population size do you want?",
            },
            "max_evaluations": {
                "values": [100, 1000, 20000, 100000],
                "description": "What should be the maximum number of evaluations?",
            },
            "sigma": {
                "values": [0.01, 0.5, 1, 2],
                "description": "Which sigma would you like to use?",
            },
            "pretrained": {
                "values": [False],
                "custom_input": True,
                "postproc": str,
                "description": "Please provide the parameters of a pretrained model?",
            },
            "loss": {
                "values": ["KL", "NLL"],
                "description": "Which loss function do you want to use?",
            },
        }

    class Config(TypedDict):
        """
        Attributes of a valid config

        .. code-block:: python

            population_size: int
            max_evaluations: int
            sigma: float
            pretrained: str
            loss: str

        """

        population_size: int
        max_evaluations: int
        sigma: float
        pretrained: str
        loss: str

    def get_default_submodule(self, option: str) -> Core:
        raise ValueError("This module has no submodules.")

    def setup_training(self, input_data, config) -> tuple:
        """
        Method to configure the training setup including CMA-ES and tensorboard.

        :param input_data: A representation of the quntum machine learning model that will be trained
        :type input_data: dict
        :param config: Config specifying the parameters of the training
        :type config: dict
        :return: Updated input_data
        :rtype: dict
        """

        logging.info(
            f"Running config: [backend={input_data['backend']}] [n_qubits={input_data['n_qubits']}] "
            f"[population_size={config['population_size']}]"
        )

        self.study_generalization = "generalization_metrics" in list(input_data.keys())
        if self.study_generalization:
            self.generalization_metrics = input_data["generalization_metrics"]
            self.generalization_metrics.n_shots = input_data["n_shots"]
            input_data["store_dir_iter"] += (
                f"_alpha{self.generalization_metrics.train_size}_depth{input_data['depth']}"
            )

        self.writer = SummaryWriter(input_data["store_dir_iter"])

        if config["loss"] == "KL":
            self.loss_func = self.kl_divergence
        elif config["loss"] == "NLL":
            self.loss_func = self.nll
        else:
            raise NotImplementedError("Loss function not implemented")

        n_params = len(input_data["circuit"].parameters)
        x0 = (np.random.rand(n_params) - 0.5) * np.pi
        if config["pretrained"] != "False":
            parameters = np.load(config["pretrained"])
            x0[: len(parameters)] = parameters
            logging.info(
                f"Training starting from parameters in path {config['pretrained']}"
            )

        options = {
            "bounds": [n_params * [-np.pi], n_params * [np.pi]],
            "maxfevals": config["max_evaluations"],
            "popsize": config["population_size"],
            "verbose": -3,
            "tolfun": 1e-12,
        }

        return x0, options

    def start_training(
        self, input_data: dict, config: Config, **kwargs: dict
    ) -> (dict, float):
        """
        This function finds the best parameters of the circuit on a transformed problem instance and returns a solution.

        :param input_data: A representation of the quntum machine learning model that will be trained
        :type input_data: dict
        :param config: Config specifying the paramters of the training
        :type config: dict
        :param kwargs: optional additional settings
        :type kwargs: dict
        :return: Dictionary including the information of previous modules as well as of the training
        :rtype: dict
        """

        size = None  # TODO: define correct mpi size
        input_data["MPI_size"] = size
        input_data["store_dir_iter"] += (
            f"_{input_data['dataset_name']}_qubits{input_data['n_qubits']}"
        )
        x0, options = self.setup_training(input_data, config)

        if comm.Get_rank() == 0:
            self.target = np.asarray(input_data["histogram_train"])
            self.target[self.target == 0] = 1e-8
        self.n_states_range = range(2 ** input_data["n_qubits"])
        execute_circuit = input_data["execute_circuit"]
        timing = self.Timing()

        es = CMAEvolutionStrategy(x0.get() if GPU else x0, config["sigma"], options)
        for parameter in [
            "best_parameters",
            "time_circuit",
            "time_loss",
            "KL",
            "best_sample",
        ]:
            input_data[parameter] = []

        best_loss = float("inf")
        self.fig, self.ax = plt.subplots()
        while not es.stop():
            solutions = es.ask()
            epoch = es.result[4]
            sigma = es.sigma

            timing.start_recording()
            pmfs_model, samples = execute_circuit(solutions)
            pmfs_model = np.asarray(pmfs_model)
            time_circ = timing.stop_recording()

            timing.start_recording()
            if comm.Get_rank() == 0:
                loss_epoch = self.loss_func(
                    pmfs_model.reshape([config["population_size"], -1]), self.target
                )
            else:
                loss_epoch = np.empty(config["population_size"])
            comm.Bcast(loss_epoch, root=0)
            comm.Barrier()

            time_loss = timing.stop_recording()

            es.tell(solutions, loss_epoch.get() if GPU else loss_epoch)

            if es.result[1] < best_loss:
                best_loss = es.result[1]
                best_pmf = self.data_visualization(
                    loss_epoch, pmfs_model, samples, epoch
                )

            input_data["best_parameters"].append(es.result[0])
            input_data["KL"].append(float(es.result[1]))

            logging.info(
                f"[Iteration {es.result[4]}] "
                f"[{config['loss']}: {es.result[1]:.5f}] "
                f"[Circuit processing: {(time_circ):.3f} ms] "
                f"[{config['loss']} processing: {(time_loss):.3f} ms] "
                f"[sigma: {sigma:.5f}]"
            )

        plt.close()
        self.writer.flush()
        self.writer.close()

        input_data["best_parameter"] = es.result[0]
        best_sample = self.sample_from_pmf(
            self.n_states_range,
            best_pmf.get() if GPU else best_pmf,
            n_shots=input_data["n_shots"],
        )
        input_data["best_sample"] = best_sample.get() if GPU else best_sample  # pylint: disable=E1101

        return input_data

    def data_visualization(self, loss_epoch, pmfs_model, samples, epoch):
        index = loss_epoch.argmin()
        best_pmf = pmfs_model[index] / pmfs_model[index].sum()
        if self.study_generalization:
            if samples is None:
                counts = self.sample_from_pmf(
                    n_states_range=self.n_states_range,
                    pmf=best_pmf.get() if GPU else best_pmf,
                    n_shots=self.generalization_metrics.n_shots,
                )
            else:
                counts = samples[int(index)]

            metrics = self.generalization_metrics.get_metrics(
                counts.get() if GPU else counts
            )
            for key, value in metrics.items():
                self.writer.add_scalar(f"metrics/{key}", value, epoch)

        nll = self.nll(best_pmf.reshape([1, -1]), self.target)
        kl = self.kl_divergence(best_pmf.reshape([1, -1]), self.target)
        self.writer.add_scalar("metrics/NLL", nll.get() if GPU else nll, epoch)
        self.writer.add_scalar("metrics/KL", kl.get() if GPU else kl, epoch)

        self.ax.clear()
        self.ax.imshow(
            best_pmf.reshape(
                int(np.sqrt(best_pmf.size)), int(np.sqrt(best_pmf.size))
            ).get()
            if GPU
            else best_pmf.reshape(
                int(np.sqrt(best_pmf.size)), int(np.sqrt(best_pmf.size))
            ),
            cmap="binary",
            interpolation="none",
        )
        self.ax.set_title(f"Iteration {epoch}")
        self.writer.add_figure("grid_figure", self.fig, global_step=epoch)

        return best_pmf