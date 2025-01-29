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

import itertools
import logging
from pprint import pformat
from typing import TypedDict

import numpy as np

from quark.modules.applications.qml.generative_modeling.circuits.CircuitCardinality import CircuitCardinality
from quark.modules.applications.qml.generative_modeling.data.data_handler.DataHandlerGenerative import DataHandlerGenerative
from quark.modules.applications.qml.generative_modeling.metrics.MetricsGeneralization import MetricsGeneralization
from quark.utils import start_time_measurement, end_time_measurement


class DiscreteData(DataHandlerGenerative):
    """
    A data handler for discrete datasets with cardinality constraints.
    This class creates a dataset with a cardinality constraint and provides
    methods for generalization metrics computing and evaluation.
    """

    def __init__(self):
        super().__init__("")
        self.submodule_options = ["CircuitCardinality"]
        self.n_registers = None
        self.n_qubits = None
        self.train_size = None
        self.histogram_train = None
        self.histogram_solution = None
        self.generalization_metrics = None
        self.samples = None
        self.train_set = None
        self.solution_set = None

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Returns requirements of this module.

        :return: List of dict with requirements of this module
        """
        return [{"name": "numpy", "version": "1.26.4"}]

    def get_default_submodule(self, option: str) -> CircuitCardinality:
        """
        Get the default submodule based on the given option.

        :param option: Submodule option
        :return: Corresponding submodule
        """
        if option == "CircuitCardinality":
            return CircuitCardinality()
        else:
            raise NotImplementedError(f"Circuit Option {option} not implemented")

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this application.

        :return: A dictionary of parameter options
        .. code-block:: python

        return {
            "train_size": {
            "values": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
            "description": "What percentage of the dataset do you want to use for training?"
            }
        }
        """
        return {
            "train_size": {
                "values": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                "description": "What percentage of the dataset do you want to use for training?"
            }
        }

    class Config(TypedDict):
        """
        Attributes of a valid config.

        .. code-block:: python

            train_size: int

        """

        train_size: int

    def data_load(self, gen_mod: dict, config: Config) -> dict:
        """
        The cardinality constrained dataset is created and split into a training set.

        :param gen_mod: Dictionary with collected information of the previous modules
        :param config: Config specifying the parameters of the data handler
        :return: Dictionary including the mapped problem
        """
        dataset_name = "Cardinality_Constraint"
        self.n_qubits = gen_mod["n_qubits"]
        self.train_size = config["train_size"]
        num_ones = self.n_qubits // 2

        # Generate all possible binary permutations of length n_qubits using NumPy arrays
        search_space = np.array(list(itertools.product([0, 1], repeat=self.n_qubits)))
        search_space = np.apply_along_axis(lambda x: ''.join(str(bit) for bit in x), 1, search_space)

        # Filter the binary permutations based on the cardinality constraint using np.char.count
        cardinality_constraint = np.char.count(search_space, '1') == num_ones
        solution_set = search_space[cardinality_constraint]

        # Use np.random.choice to select the desired number of rows randomly
        size_train = int(self.train_size * len(solution_set))
        train_set_indices = np.random.choice(len(solution_set), size_train, replace=False)
        train_set = solution_set[train_set_indices]

        # Create the histogram solution data
        self.histogram_solution = np.zeros(2 ** self.n_qubits)
        self.solution_set = np.array([int(i, 2) for i in solution_set])
        self.histogram_solution[self.solution_set] = 1 / len(self.solution_set)

        # Create the histogram training data
        self.histogram_train = np.zeros(2 ** self.n_qubits)
        self.train_set = np.array([int(i, 2) for i in train_set])
        self.histogram_train[self.train_set] = 1 / len(self.train_set)

        train_set_binary = np.array([list(map(int, s)) for s in train_set])
        solution_set_binary = np.array([list(map(int, s)) for s in solution_set])

        application_config = {
            "dataset_name": dataset_name,
            "binary_train": train_set_binary,
            "binary_solution": solution_set_binary,
            "train_size": self.train_size,
            "n_qubits": self.n_qubits,
            "n_registers": 2,
            "histogram_solution": self.histogram_solution,
            "histogram_train": self.histogram_train,
            "store_dir_iter": gen_mod["store_dir_iter"]
        }

        if self.train_size != 1:
            self.generalization_metrics = MetricsGeneralization(
                train_set=self.train_set,
                train_size=self.train_size,
                solution_set=self.solution_set,
                n_qubits=self.n_qubits,
            )
            application_config["generalization_metrics"] = self.generalization_metrics

        return application_config

    def generalization(self) -> tuple[dict, float]:
        """
        Calculate generalization metrics for the generated.

        :return: Tuple containing a dictionary of generalization metrics and the execution time
        """
        start = start_time_measurement()
        results = self.generalization_metrics.get_metrics(self.samples)

        logging.info(pformat(results))

        return results, end_time_measurement(start)

    def evaluate(self, solution: dict) -> tuple[dict, float]:
        """
        Evaluates a given solution and calculates the histogram of generated samples and the minimum KL divergence
        value.

        :param solution: Dictionary containing the solution data, including generated samples and KL divergence values
        :return: Tuple containing a dictionary with the histogram of generated samples and the minimum KL divergence
                 value, and the time it took to evaluate the solution
        """
        start = start_time_measurement()
        self.samples = solution["best_sample"]
        n_shots = np.sum(self.samples)

        histogram_generated = np.asarray(self.samples) / n_shots
        histogram_generated[histogram_generated == 0] = 1e-8

        kl_list = solution["KL"]
        kl_best = min(kl_list)

        evaluate_dict = {"histogram_generated": histogram_generated, "KL_best": kl_best}

        return evaluate_dict, end_time_measurement(start)
