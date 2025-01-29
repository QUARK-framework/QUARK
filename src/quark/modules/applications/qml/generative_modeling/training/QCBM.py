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
import numpy as np
from cma import CMAEvolutionStrategy
from tensorboardX import SummaryWriter
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import figure, axes

from quark.modules.applications.qml.generative_modeling.training.TrainingGenerative import TrainingGenerative, Core, GPU
from quark.utils_mpi import is_running_mpi, get_comm

matplotlib.use('Agg')
MPI = is_running_mpi()
comm = get_comm()


class QCBM(TrainingGenerative):
    """
    This module optimizes the parameters of quantum circuit using CMA-ES.
    This training method is referred to as quantum circuit born machine (QCBM).
    """

    def __init__(self):
        """
        Constructor method.
        """
        super().__init__("QCBM")

        self.n_states_range: list
        self.target: np.ndarray
        self.study_generalization: bool
        self.generalization_metrics: dict
        self.writer: SummaryWriter
        self.loss_func: callable
        self.fig: figure
        self.ax: axes.Axes

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Returns requirements of this module.

        :return: List of dict with requirements of this module
        """
        return [
            {"name": "numpy", "version": "1.26.4"},
            {"name": "cma", "version": "4.0.0"},
            {"name": "matplotlib", "version": "3.9.3"},
            {"name": "tensorboard", "version": "2.18.0"},
            {"name": "tensorboardX", "version": "2.6.2.2"}
        ]

    def get_parameter_options(self) -> dict:
        """
        This function returns the configurable settings for the quantum circuit born machine.

        :return: Configuration settings for QCBM
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
                    "description": "Please provide the parameters of a pretrained model."
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
                "description": "Please provide the parameters of a pretrained model."
            },

            "loss": {
                "values": ["KL", "NLL"],
                "description": "Which loss function do you want to use?"
            }
        }

    class Config(TypedDict):
        """
        Attributes of a valid config.

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
        """
        Raises ValueError as this module has no submodules.

        :param option: Option name
        :raises ValueError: If called, since this module has no submodules
        """
        raise ValueError("This module has no submodules.")

    def setup_training(self, input_data: dict, config: Config) -> tuple[float, dict]:
        """
        Method to configure the training setup including CMA-ES and tensorboard.

        :param input_data: A representation of the quantum machine learning model that will be trained
        :param config: Config specifying the parameters of the training
        :return: Random initial parameter and options for CMA-ES
        """

        logging.info(
            f"Running config: [backend={input_data['backend']}] "
            f"[n_qubits={input_data['n_qubits']}] "
            f"[population_size={config['population_size']}]"
        )

        self.study_generalization = "generalization_metrics" in list(input_data.keys())
        if self.study_generalization:
            self.generalization_metrics = input_data["generalization_metrics"]
            self.generalization_metrics.n_shots = input_data["n_shots"]

        self.writer = SummaryWriter(input_data["store_dir_iter"])

        if config['loss'] == "KL":
            self.loss_func = self.kl_divergence
        elif config['loss'] == "NLL":
            self.loss_func = self.nll
        elif config['loss'] == "MMD":
            self.loss_func = self.mmd
        else:
            raise NotImplementedError("Loss function not implemented")

        n_params = input_data["n_params"]
        x0 = (np.random.rand(n_params) - 0.5) * np.pi
        if config["pretrained"] != "False":
            parameters = np.load(config["pretrained"])
            x0[:len(parameters)] = parameters
            logging.info(f'Training starting from parameters in path {config["pretrained"]}')

        options = {
            'bounds': [n_params * [-np.pi], n_params * [np.pi]],
            'maxfevals': config['max_evaluations'],
            'popsize': config['population_size'],
            'verbose': -3,
            'tolfun': 1e-12
        }

        return x0, options

    def start_training(self, input_data: dict, config: dict, **kwargs: dict) -> dict:
        """
        This function finds the best parameters of the circuit on a transformed problem instance and returns a solution.

        :param input_data: A representation of the quantum machine learning model that will be trained
        :param config: Config specifying the parameters of the training
        :param kwargs: Optional additional settings
        :return: Dictionary including the information of previous modules as well as of the training
        """

        size = None
        input_data['MPI_size'] = size
        input_data["store_dir_iter"] += f"_{input_data['dataset_name']}_qubits{input_data['n_qubits']}"
        x0, options = self.setup_training(input_data, config)

        is_master = comm.Get_rank() == 0
        if is_master:
            self.target = np.asarray(input_data["histogram_train"])
            self.target[self.target == 0] = 1e-8

        self.n_states_range = range(2 ** input_data['n_qubits'])
        execute_circuit = input_data["execute_circuit"]
        timing = self.Timing()

        es = CMAEvolutionStrategy(x0.get() if GPU else x0, config['sigma'], options)

        for parameter in ["best_parameters", "time_circuit", "time_loss", "KL", "best_sample"]:
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
            if is_master:
                loss_epoch = self.loss_func(pmfs_model.reshape([config['population_size'], -1]), self.target)
            else:
                loss_epoch = np.empty(config["population_size"])

            comm.Bcast(loss_epoch, root=0)
            comm.Barrier()

            time_loss = timing.stop_recording()

            es.tell(solutions, loss_epoch.get() if GPU else loss_epoch)

            if es.result[1] < best_loss:
                best_loss = es.result[1]
                best_pmf = self.data_visualization(loss_epoch, pmfs_model, samples, epoch)

            input_data["best_parameters"].append(es.result[0])
            input_data["KL"].append(float(es.result[1]))

            logging.info(
                f"[Iteration {epoch}] "
                f"[{config['loss']}: {es.result[1]:.5f}] "
                f"[Circuit processing: {(time_circ):.3f} ms] "
                f"[{config['loss']} processing: {(time_loss):.3f} ms] "
                f"[sigma: {sigma:.5f}]"
            )

        plt.close()
        self.writer.flush()
        self.writer.close()

        input_data["best_parameter"] = es.result[0]
        best_sample = self.sample_from_pmf(best_pmf.get() if GPU else best_pmf,  # pylint: disable=E0606
                                           n_shots=input_data["n_shots"])
        input_data["best_sample"] = best_sample.get() if GPU else best_sample  # pylint: disable=E1101

        return input_data

    def data_visualization(self, loss_epoch: np.ndarray, pmfs_model: np.ndarray, samples: any, epoch: int) -> (
            np.ndarray):
        """
        Visualizes the data and metrics for training.

        :param loss_epoch: Loss for the current epoch
        :param pmfs_model: The probability mass functions from the model
        :param samples: The samples from the model
        :param epoch: The current epoch number
        :return: Best probability mass function for visualization
        """
        index = loss_epoch.argmin()
        best_pmf = pmfs_model[index] / pmfs_model[index].sum()

        if self.study_generalization:
            if samples is None:
                counts = self.sample_from_pmf(
                    pmf=best_pmf.get() if GPU else best_pmf,
                    n_shots=self.generalization_metrics.n_shots)
            else:
                counts = samples[int(index)]

            metrics = self.generalization_metrics.get_metrics(counts if GPU else counts)
            for key, value in metrics.items():
                self.writer.add_scalar(f"metrics/{key}", value, epoch)

        nll = self.nll(best_pmf.reshape([1, -1]), self.target)
        kl = self.kl_divergence(best_pmf.reshape([1, -1]), self.target)
        mmd = self.mmd(best_pmf.reshape([1, -1]), self.target)

        self.writer.add_scalar("metrics/NLL", nll.get() if GPU else nll, epoch)
        self.writer.add_scalar("metrics/KL", kl.get() if GPU else kl, epoch)
        self.writer.add_scalar("metrics/MMD", mmd.get() if GPU else mmd, epoch)

        self.ax.clear()
        self.ax.imshow(
            best_pmf.reshape(int(np.sqrt(best_pmf.size)), int(np.sqrt(best_pmf.size))).get()
            if GPU else best_pmf.reshape(int(np.sqrt(best_pmf.size)), int(np.sqrt(best_pmf.size))),
            cmap='binary',
            interpolation='none'
        )
        self.ax.set_title(f'Iteration {epoch}')
        self.writer.add_figure('grid_figure', self.fig, global_step=epoch)

        return best_pmf
