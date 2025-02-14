import copy
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from QuantumModel import *
from sklearn.metrics import classification_report, confusion_matrix
from torch.optim.lr_scheduler import StepLR


class BinaryClassifier:
    """
    Wrapper for pytorch models to facilitate training and testing.
    """

    def __init__(self, model, loss_fn, optimizer, lr_scheduler):
        self.model = model
        self.initial_model_state_dict = copy.deepcopy(self.model.state_dict())
        self.optimizer = optimizer
        self.initial_optimizer_state_dict = copy.deepcopy(self.optimizer.state_dict())
        self.lr_scheduler = lr_scheduler
        self.initial_lr_scheduler_state_dict = copy.deepcopy(
            self.lr_scheduler.state_dict()
        )
        self.loss_fn = loss_fn

    def train(self, dataloader: torch.utils.data.DataLoader, n_epochs: int = 10):
        """
        Args:
            dataloader: A DataLoader object that provides training data and labels.
            n_epochs: The number of epochs for training.
        """
        self.model.train()
        self.n_epochs = n_epochs
        self.val_size = None

        es = EarlyStopping(patience=5)

        self.training_loss = []
        self.training_accuracy = []

        for epoch in range(n_epochs):
            print(f"Epoch: {epoch}")
            for X, y, paths in dataloader:
                pred = self.model(X)
                pred_with_softmax = nn.Softmax(dim=1)(pred)
                predicted_classes = np.argmax(
                    pred_with_softmax.detach().numpy(), axis=1
                )
                loss = self.loss_fn(pred, y)
                self.training_loss.append(loss.detach().numpy())
                accuracy = sum(predicted_classes == y.detach().numpy()) / len(
                    predicted_classes
                )
                self.training_accuracy.append(accuracy)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.lr_scheduler.step()
            print(f"Loss: {loss}")

            if es.step(loss):
                break

            print(f"Accuracy: {accuracy}")
        self.trained = True

    def predict(self, dataloader: torch.utils.data.DataLoader) -> pd.DataFrame:
        """
        Args:
            dataloader: A DataLoader object that provides data.

        Returns:
            A pandas Dataframe with columns "target", "true_label",
                "score" (the probability of class 1), "path".
        """
        self.model.eval()
        prediction_scores = []
        predicted_classes = []
        target_classes = []
        file_paths = []
        for X, y, path in dataloader:
            target_classes.extend(list(y.detach().numpy()))
            pred = self.model(X)
            pred_with_softmax = nn.Softmax(dim=1)(pred)
            predicted_classes.extend(
                list(np.argmax(pred_with_softmax.detach().numpy(), axis=1))
            )
            prediction_scores.extend(list(pred_with_softmax.detach().numpy()[:, 1]))
            file_paths.extend(path)
        results = []
        for idx in range(len(target_classes)):
            results.append(
                {
                    "true_label": target_classes[idx],
                    "prediction": predicted_classes[idx],
                    "score": prediction_scores[idx],
                    "path": file_paths[idx],
                }
            )
        results_df = pd.DataFrame(results)
        return results_df

    def test(self, dataloader: torch.utils.data.DataLoader):
        """
        Args:
            dataloader: A DataLoader object that provides test data.
        """
        self.model.eval()
        self.test_loss = []
        num_batches = len(dataloader.dataset)
        test_loss = 0
        predicted_classes = []
        target_classes = []
        test_file_paths = []
        for X, y, path in dataloader:
            target_classes.extend(list(y.detach().numpy()))
            pred = self.model(X)
            predicted_classes.extend(list(np.argmax(pred.detach().numpy(), axis=1)))
            loss = self.loss_fn(pred, y)
            test_loss += loss
            self.test_loss.append(loss.detach().numpy())
            test_file_paths.extend(path)

        misclassified_files = np.array(test_file_paths)[
            np.array(target_classes) != np.array(predicted_classes)
        ]

        test_loss = test_loss / num_batches
        print(f"Test loss: {test_loss}")
        print("Classification report")
        print(classification_report(target_classes, predicted_classes))
        print("Confusion matrix")
        print(confusion_matrix(target_classes, predicted_classes))
        print("Incorrect predictions")
        print(misclassified_files)

    def save_to_disk(self, filepath: str) -> None:
        if self.val_size is not None:
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "epoch": self.n_epochs,
                    "train_loss": self.training_loss,
                    "train_accuracy": self.training_accuracy,
                    "val_loss": self.validation_loss,
                    "val_accuracy": self.validation_accuracy,
                    "avg_train_loss": self.get_avg_training_loss_kfold(),
                    "foldperf": self.foldperf,
                },
                filepath,
            )
        else:
            torch.save(
                {
                    "epoch": self.n_epochs,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "train_loss": self.training_loss,
                    "train_accuracy": self.training_accuracy,
                },
                filepath,
            )

    def load_from_disk(self, filepath: str) -> None:
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.n_epochs = checkpoint["epoch"]

        self.training_loss = checkpoint["train_loss"]
        self.training_accuracy = checkpoint["train_accuracy"]
        if "val_loss" in checkpoint:
            self.validation_loss = checkpoint["val_loss"]
        if "val_accuracy" in checkpoint:
            self.validation_accuracy = checkpoint["val_accuracy"]

        if "avg_train_loss" in checkpoint:
            self.avg_training_loss = checkpoint["avg_train_loss"]
        if "foldperf" in checkpoint:
            self.foldperf = checkpoint["foldperf"]

        self.model.eval()

    def get_training_loss(self) -> List:
        assert hasattr(self, "trained"), "The model must be trained first."
        return self.training_loss

    def get_training_accuracy(self) -> List:
        assert hasattr(self, "trained"), "The model must be trained first."
        return self.training_accuracy


class QuantumTraining(BinaryClassifier):
    def __init__(
        self,
        learning_rate: float = 0.001,
        lr_scheduler_step: int = 100,
        lr_scheduler_gamma: float = 0,
        n_reduced_features: int = 4,
    ) -> None:
        """
        Args:
            n_reduced_features: Number of output neurons in the first layer, responsible for dimensionality reduction.
            learning_rate: Learning rate of the model
            lr_scheduler_step: Adjust learning rate after step epochs.
            lr_scheduler_gamma: Adjust learning rate by gamma.
            quantum_device: An instance of pennylane.Device on which the quantum circuit will be executed.
        """
        self.learning_rate = learning_rate

        model = QuantumModel(n_reduced_features=n_reduced_features)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        lr_scheduler = StepLR(
            optimizer, step_size=lr_scheduler_step, gamma=lr_scheduler_gamma
        )
        super(QuantumTraining, self).__init__(model, loss_fn, optimizer, lr_scheduler)

    def summary(self):
        for p in self.model.parameters():
            print(p)
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def draw(self):
        print("Drawing the quantum layers of the hybrid neural network:")
        layers = [m for m in self.model.children()]
        layers[1].draw()
        return layers[1]
