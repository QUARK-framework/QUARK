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
from cma import CMAEvolutionStrategy
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

from modules.applications.QML.training.Training import *
from utils_mpi import is_running_mpi, get_comm

MPI = is_running_mpi()
comm = get_comm()


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
                "name": "cma",
                "version": "3.3.0"
            },
            {
                "name": "tensorboard",
                "version": "2.13.0"
            },
            {
                "name": "tensorboardX",
                "version": "2.6.2"
            }
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

    def setup_training(self, input_data, config) -> dict:
        """
        Method to configure the training setup including CMA-ES and tensorboard.

        :param input_data: A representation of the quntum machine learning model that will be trained
        :type input_data: dict
        :param config: Config specifying the parameters of the training
        :type config: dict
        :return: Updated input_data 
        :rtype: dict
        """
        size = None
        input_data['MPI_size'] = size
        input_data["store_dir_iter"] += f"_{input_data['dataset_name']}_qubits{input_data['n_qubits']}"
        logging.info(
            f"Running config: [backend={input_data['backend']}] [n_qubits={input_data['n_qubits']}] [n_shots={input_data['n_shots']}] [population_size={config['population_size']}]")

        if comm.Get_rank() == 0:
            self.target = np.asarray(input_data["histogram_train"])
            self.target[self.target == 0] = 1e-8
        self.n_states_range = range(2 ** input_data['n_qubits'])
        self.n_shots = input_data["n_shots"]
        self.execute_circuit = input_data["execute_circuit"]
        self.timing = self.Timing(gpu=gpu)

        self.study_generalization = "generalization_metrics" in list(input_data.keys())
        if self.study_generalization:
            self.generalization_metrics = input_data["generalization_metrics"]
            self.generalization_metrics.n_shots = input_data["n_shots"]
            input_data["store_dir_iter"] += f"_alpha{self.generalization_metrics.train_size}_depth{input_data['depth']}"

        self.writer = SummaryWriter(input_data["store_dir_iter"])
        epochs = int(config['max_evaluations'] / config['population_size'])
        self.image_epochs = range(0, epochs, 1) if int(epochs / 100) < 1 else range(0, epochs, int(epochs / 100))
        self.tb_images = "2D" in input_data["dataset_name"]

        if config['loss'] == "KL":
            self.loss_func = self.kl_divergence
        elif config['loss'] == "NLL":
            self.loss_func = self.nll
        else:
            raise NotImplementedError("Loss function not implemented")

        self.n_params = len(input_data["circuit"].parameters)
        self.x0 = (np.random.rand(self.n_params) - 0.5) * np.pi

        if config["pretrained"] != "False":
            parameters = np.load(config["pretrained"])
            self.x0[:len(parameters)] = parameters
            logging.info(f'Training starting from parameters in path {config["pretrained"]}')

        self.options = {
            'bounds': [self.n_params * [-np.pi], self.n_params * [np.pi]],
            'maxfevals': config['max_evaluations'],
            'popsize': config['population_size'],
            'verbose': -3,
            'tolfun': 1e-12
        }

        return input_data

    def start_training(self, input_data: dict, config: Config, **kwargs: dict) -> (dict, float):
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
        input_data = self.setup_training(input_data, config)

        es = CMAEvolutionStrategy(self.x0.get() if gpu else self.x0, config['sigma'], self.options)
        for parameter in ["best_parameters", "time_circuit", "time_loss", "KL", "best_sample"]:
            input_data[parameter] = []

        best_loss = float("inf")
        fig, ax = plt.subplots()
        while not es.stop():
            solutions = es.ask()
            epoch = es.result[4]
            sigma = es.sigma

            self.timing.start_recording()
            pmfs_model, samples = self.execute_circuit(solutions)
            pmfs_model = np.asarray(pmfs_model)
            time_circ = self.timing.stop_recording()

            self.timing.start_recording()
            if comm.Get_rank() == 0:
                loss_epoch = self.loss_func(pmfs_model.reshape([config['population_size'], -1]))
            else:
                loss_epoch = np.empty(config["population_size"])
            comm.Bcast(loss_epoch, root=0)
            comm.Barrier()

            time_loss = self.timing.stop_recording()

            es.tell(solutions, loss_epoch.get() if gpu else loss_epoch)

            if es.result[1] < best_loss:
                best_loss = es.result[1]
                index = loss_epoch.argmin()
                best_pmf = pmfs_model[index] / pmfs_model[index].sum()
                if self.study_generalization:
                    counts = self.sample_from_pmf(pmf=best_pmf.get() if gpu else best_pmf,
                                                  n_shots=self.generalization_metrics.n_shots) if samples is None else \
                        samples[int(index)]
                    metrics = self.generalization_metrics.get_metrics(counts)
                    for (key, value) in metrics.items():
                        self.writer.add_scalar(f"metrics/{key}", value, epoch)

                self.writer.add_scalar("metrics/NLL", self.nll(best_pmf.reshape([1, -1])).get() if gpu else self.nll(
                    best_pmf.reshape([1, -1])), epoch)
                self.writer.add_scalar("metrics/KL", self.kl_divergence(
                    best_pmf.reshape([1, -1])).get() if gpu else self.kl_divergence(best_pmf.reshape([1, -1])), epoch)
                self.writer.add_scalar("time/circuit", time_circ, epoch)
                self.writer.add_scalar("time/loss", time_loss, epoch)
            input_data["best_parameters"].append(es.result[0])

            input_data["KL"].append(float(es.result[1]))

            if self.tb_images and epoch in self.image_epochs:
                ax.clear()
                ax.imshow(
                    best_pmf.reshape(int(np.sqrt(best_pmf.size)), int(np.sqrt(best_pmf.size))).get() if gpu
                    else best_pmf.reshape(int(np.sqrt(best_pmf.size)),
                                          int(np.sqrt(best_pmf.size))),
                    cmap='binary',
                    interpolation='none')
                ax.set_title(f'Iteration {epoch}')
                self.writer.add_figure('grid_figure', fig, global_step=epoch)

            logging.info(
                f"[Iteration {es.result[4]}] [{config['loss']}: {es.result[1]:.5f}] [Circuit processing: {(time_circ):.3f} ms] [{config['loss']} processing: {(time_loss):.3f} ms] [sigma: {sigma:.5f}] ")

        plt.close()
        self.writer.flush()
        self.writer.close()

        input_data["best_parameter"] = es.result[0]
        input_data["best_sample"] = self.sample_from_pmf(best_pmf.get() if gpu else best_pmf,
                                                         n_shots=input_data["n_shots"])

        return input_data
