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

import math
import numpy as np


class MetricsGeneralization:
    """
    A class to compute generalization metrics for generated samples based on train and solution sets.
    """

    def __init__(self, train_set: np.array, train_size: float, solution_set: np.array, n_qubits: int):
        """
        :param train_set: Set of queries in the training set
        :param train_size: The fraction of queries used for training
        :param solution_set: Set of queries in the solution set
        :param n_qubits: The number of qubits.
        """
        self.train_set = train_set
        self.train_size = train_size
        self.solution_set = solution_set
        self.n_states = 2 ** n_qubits
        self.n_shots = 10000

        self.mask_new, self.mask_sol = self.get_masks()

    def get_masks(self) -> tuple[np.array, np.array]:
        """
        Method to determine the masks, on which the generalization metrics are based.

        :return: Masks needed to determine the generalization metrics for a given train and solution set
        """
        mask_new = np.ones(self.n_states, dtype=bool)
        mask_new[self.train_set] = 0

        mask_sol = np.zeros(self.n_states, dtype=bool)
        mask_sol[self.solution_set] = 1
        mask_sol[self.train_set] = 0

        return mask_new, mask_sol

    def get_metrics(self, generated: np.array) -> dict:
        """
        Method that determines all generalization metrics of a given multiset of generated samples.

        :param generated: Generated samples
        :return: Dictionary with generalization metrics
        """
        g_new = np.sum(generated[self.mask_new])
        g_sol = np.sum(generated[self.mask_sol])
        g_sol_unique = generated[self.mask_sol & self.mask_new]
        g_sol_unique = g_sol_unique[g_sol_unique != 0].size
        g_train = generated[self.train_set]

        results = {
            "fidelity": self.fidelity(g_new, g_sol),
            "exploration": self.exploration(g_new),
            "coverage": self.coverage(g_sol_unique),
            "normalized_rate": self.normalized_rate(g_sol),
            "precision": self.precision(g_sol, g_train)

        }
        return results

    def fidelity(self, g_new: float, g_sol: float) -> float:
        """
        Method to determine the fidelity.

        :param g_new: Multi-subset of unseen queries (noisy or valid)
        :param g_sol: Multi-subset of unseen and valid queries
        :return: Fidelity
        """
        return g_sol / g_new

    def coverage(self, g_sol_unique: float) -> float:
        """
        Method to determine the coverage.

        :param g_sol_unique: Subset of unique unseen and valid queries
        :return: Coverage
        """
        return g_sol_unique / (math.ceil(1 - self.train_size) * len(self.solution_set))

    def normalized_rate(self, g_sol: float) -> float:
        """
        Method to determine the normalized_rate.

        :param g_sol: Multi-subset of unseen and valid queries
        :return: Normalized_rate
        """
        return g_sol / ((1 - self.train_size) * self.n_shots)

    def exploration(self, g_new: float) -> float:
        """
        Method to determine the exploration.

        :param g_new: Multi-subset of unseen queries (noisy or valid)
        :return: Exploration
        """
        return g_new / self.n_shots

    def precision(self, g_sol: float, g_train: float) -> float:
        """
        Method to determine the precision.

        :param g_sol: Multi-subset of unseen and valid queries
        :param g_train: Number of queries that were memorized from the training set
        :return: Precision
        """
        return (np.sum(g_sol) + np.sum(g_train)) / self.n_shots
