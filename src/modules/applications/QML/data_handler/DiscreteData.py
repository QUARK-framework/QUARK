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

from typing import TypedDict, Union
import itertools
import logging
from pprint import pformat

import numpy as np

from modules.applications.QML.data_handler.DataHandler import *
from modules.applications.QML.circuits.CircuitCardinality import CircuitCardinality
from modules.applications.QML.data_handler.MetricsGeneralization import MetricsGeneralization


class DiscreteData(DataHandler):
    """
    A data handler for discrete datasets with cardinality constraints.
    This class creates a dataset with a cardinality constraint and provides
    methods for generalisation metrics computing and evaluation.
    """

    def __init__(self):
        super().__init__("")
        self.submodule_options = ["CircuitCardinality"]
        self.n_registers = None
        self.n_qubits = None
        self.train_size = None
        self.histogram_train  = None
        self.histogram_solution = None
        self.generalization_metrics = None
        self.samples = None


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

    def get_default_submodule(self, option: str) -> CircuitCardinality:

        if option == "CircuitCardinality":
            return CircuitCardinality()
        else:
            raise NotImplementedError(
                f"Circuit Option {option} not implemented")

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this application

        :return:
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
            },
        }

    class Config(TypedDict):
        """
        Attributes of a valid config

        .. code-block:: python

            n_qubits: int
            train_size: int

        """

        n_qubits: int
        train_size: int

    def data_load(self, gen_mod: dict, config: Config) -> dict:
        """
        The cardinality constrained dataset is created and split into a training set.

        :param gen_mod: Dictionary with collected information of the previous modules
        :type gen_mod: dict
        :param config: Config specifying the paramters of the data handler
        :type config: dict
        :return: Must always return the mapped problem and the time it took to create the mapping
        :rtype: tuple(any, float)
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
        solution_set = np.array([int(i, 2) for i in solution_set])
        self.histogram_solution[solution_set] = 1 / len(solution_set)

        # Create the histogram training data
        self.histogram_train = np.zeros(2 ** self.n_qubits)
        train_set = np.array([int(i, 2) for i in train_set])
        self.histogram_train[train_set] = 1 / len(train_set)

        self.generalization_metrics = MetricsGeneralization(
            train_set=train_set,
            train_size=self.train_size,
            solution_set=solution_set,
            n_qubits=self.n_qubits,
        )

        application_config = {
            "dataset_name": dataset_name,
            "n_qubits": self.n_qubits,
            "histogram_solution": self.histogram_solution,
            "histogram_train": self.histogram_train,
            "generalization_metrics": self.generalization_metrics,
            "store_dir_iter": gen_mod["store_dir_iter"]}

        return application_config

    def generalisation(self, solution: list) -> (dict, float):
        """
        Calculate generalization metrics for the generated.

        :param solution: A list representing the solution to be evaluated.
        :type solution: list
        :return: A tuple containing a dictionary of generalization metrics 
                and the execution time in seconds.
        :rtype: tuple
        """
        start = start_time_measurement()
        results = self.generalization_metrics.get_metrics(self.samples)

        logging.info(pformat(results))

        return results, end_time_measurement(start)

    def evaluate(self, solution: list) -> (dict, float):
        """
        Evaluates a given solution and calculates the histogram of generated samples 
        and the minimum KL divergence value.

        :param solution: A dictionary-like object containing the solution data,
                        including generated samples and KL divergence values.
        :type solution: list
        :return: A tuple containing a dictionary with the histogram of generated samples 
                and the minimum KL divergence value, and the time it took to evaluate
                the solution in milliseconds.
        :rtype: (dict, float)
        """
        start = start_time_measurement()
        self.samples = solution["best_sample"]
        n_shots = np.sum(self.samples)

        histogram_generated = np.asarray(self.samples) / n_shots
        histogram_generated[histogram_generated == 0] = 1e-8

        KL_list = solution["KL"]
        KL_best = min(KL_list)

        evaluate_dict = {"histogram_generated": histogram_generated, "KL_best": KL_best}

        return evaluate_dict, end_time_measurement(start)
