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


import numpy as np
import sklearn.metrics


class MetricsClassification:
    """
    A class to compute classification metrics for generated samples
    """

    def __init__(self) -> None:
        """
        Constructor method.
        """
        self.submodule_options = []
        pass

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Returns requirements of this module.

        :return: List of dicts with requirements of this module
        :rtype: list[dict]
        """
        return [
            {"name": "numpy", "version": "1.26.4"},
            {"name": "scikit-learn", "version": "1.4.2"},
        ]

    def get_metrics(self, y_pred: np.array, y_true: np.array) -> dict[str, float]:
        """
        Method that determines all classification metrics.

        :param y_pred: Predicted labels
        :param y_true: Real labels
        :return: Dictionary with classification metrics
        """

        results = {
            "accuracy": self.accuracy(y_pred, y_true),
            "recall": self.recall(y_pred, y_true),
            "precision": self.precision(y_pred, y_true),
            "f1_score": self.f1_score(y_pred, y_true),
        }

        return results

    def accuracy(self, y_pred: np.array, y_true: np.array) -> float:
        """
        Method to determine the accuracy.

        :param y_pred: Predicted labels
        :param y_true: Real labels
        :return: Accuracy
        """
        accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
        return accuracy

    def recall(self, y_pred: np.array, y_true: np.array) -> float:
        """
        Method to determine the recall.

        :param y_pred: Predicted labels
        :param y_true: Real labels
        :return: Recall
        """
        recall = sklearn.metrics.recall_score(y_true, y_pred, zero_division=1.0, average="macro")
        return recall

    def precision(self, y_pred: np.array, y_true: np.array) -> float:
        """
        Method to determine the precision.

        :param y_pred: Predicted labels
        :param y_true: Real labels
        :return: Precision
        """
        precision = sklearn.metrics.precision_score(y_true, y_pred, zero_division=1.0, average="macro")
        return precision

    def f1_score(self, y_pred: np.array, y_true: np.array) -> float:
        """
        Method to determine the F1 score.

        :param y_pred: Predicted labels
        :param y_true: Real labels
        :return: F1_score
        """
        f1_score = sklearn.metrics.f1_score(y_true, y_pred, zero_division=1.0, average="macro")
        return f1_score
