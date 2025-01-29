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

import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as funct
from tensorboardX import SummaryWriter
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from quark.modules.applications.qml.generative_modeling.training.TrainingGenerative import TrainingGenerative, Core
from quark.utils_mpi import is_running_mpi, get_comm

matplotlib.use('Agg')
MPI = is_running_mpi()
comm = get_comm()


class QGAN(TrainingGenerative):  # pylint: disable=R0902
    """
    Class for QGAN
    """

    def __init__(self):
        """
        Constructor method.
        """
        super().__init__("QGAN")

        self.beta_1 = None
        self.real_label = None
        self.fake_label = None
        self.n_qubits = None
        self.n_registers = None
        self.n_shots = None
        self.train_size = None
        self.execute_circuit = None
        self.device = None
        self.n_epochs = None
        self.batch_size = None
        self.learning_rate_generator = None
        self.n_bins = None
        self.n_states_range = None
        self.timing = None
        self.writer = None
        self.bins_train = None
        self.bins_train = None
        self.study_generalization = None
        self.generalization_metrics = None
        self.target = None
        self.n_params = None
        self.discriminator = None
        self.params = None
        self.generator = None
        self.accuracy = None
        self.criterion = None
        self.optimizer_discriminator = None
        self.real_labels = None
        self.fake_labels = None
        self.dataloader = None
        self.loss_func = None
        self.params = None
        self.discriminator_weights = None

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Returns requirements of this module.

        :return: List of dict with requirements of this module
        """
        return [
            {"name": "numpy", "version": "1.26.4"},
            {"name": "torch", "version": "2.5.1"},
            {"name": "matplotlib", "version": "3.9.3"},
            {"name": "tensorboard", "version": "2.18.0"},
            {"name": "tensorboardX", "version": "2.6.2.2"}
        ]

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this circuit.

        :return: Configuration settings for QGAN
        .. code-block:: python
        return {
            "epochs": {
                "values": [2, 100, 200, 10000],
                "description": "How many epochs do you want?"
            },
            "batch_size": {
                "values": [10, 20, 100, 2000],
                "description": "What batch size do you want?"
            },
            "learning_rate_generator": {
                "values": [0.1, 0.2],
                "description": "What learning rate do you want to set for the generator?"
            },
            "learning_rate_discriminator": {
                "values": [0.1, 0.05],
                "description": "What learning rate do you want to set for the discriminator?"
            },
            "device": {
                "values": ["cpu", "gpu"],
                "description": "Where do you want to run the discriminator?"
            },
            "pretrained": {
                "values": [True, False],
                "description": "Do you want to use parameters of a pretrained model?"
            },
            "loss": {
                "values": ["KL", "NLL"],
                "description": "Which loss function do you want to use?"
            }
                    }
        """
        return {

            "epochs": {
                "values": [2, 100, 200, 10000],
                "description": "How many epochs do you want?"
            },
            "batch_size": {
                "values": [10, 20, 100, 2000],
                "description": "What batch size do you want?"
            },
            "learning_rate_generator": {
                "values": [0.1, 0.2],
                "description": "What learning rate do you want to set for the generator?"
            },
            "learning_rate_discriminator": {
                "values": [0.1, 0.05],
                "description": "What learnig rate do you want to set for the discriminator?"
            },
            "device": {
                "values": ["cpu", "gpu"],
                "description": "Where do you want to run the discriminator?"
            },
            "pretrained": {
                "values": [True, False],
                "description": "Do you want to use parameters of a pretrained model?"
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

            epochs: int
            batch_size: int
            learning_rate_generator: int
            learning_rate_discriminator: int
            device: str
            loss: str

        """

        epochs: int
        batch_size: int
        learning_rate_generator: float
        learning_rate_discriminator: float
        device: str
        loss: str

    def get_default_submodule(self, option: str) -> Core:
        """
        Raises ValueError as this module has no submodules.

        :param option: Option name
        :raises ValueError: If called, since this module has no submodules
        """
        raise ValueError("This module has no submodules.")

    def setup_training(self, input_data: dict, config: dict) -> None:
        """
        Sets up the training configuration.

        :param input_data: dictionary with the variables from the circuit needed to start the training
        :param config: Configurations for the QGAN training.
        """
        self.beta_1 = 0.5
        self.real_label = 1.
        self.fake_label = 0.

        self.n_qubits = input_data['n_qubits']
        self.n_registers = input_data['n_registers']
        self.n_shots = input_data["n_shots"]
        self.train_size = input_data["train_size"]
        self.execute_circuit = input_data["execute_circuit"]

        self.device = config["device"]
        self.n_epochs = config["epochs"]
        self.batch_size = config["batch_size"]
        self.learning_rate_generator = config["learning_rate_generator"]

        n = 2 ** (self.n_qubits // self.n_registers)
        self.n_bins = n ** self.n_registers
        self.n_states_range = range(2 ** self.n_qubits)

        self.timing = self.Timing()
        self.writer = SummaryWriter(input_data["store_dir_iter"])

        self.bins_train = input_data["binary_train"]
        if input_data["dataset_name"] == "Cardinality_Constraint":
            new_size = 1000
            self.bins_train = np.repeat(self.bins_train, new_size, axis=0)

        self.study_generalization = "generalization_metrics" in list(input_data.keys())
        if self.study_generalization:
            self.generalization_metrics = input_data["generalization_metrics"]

        self.target = np.asarray(input_data["histogram_train"])
        self.target[self.target == 0] = 1e-8

        self.n_params = input_data["n_params"]

        self.discriminator = Discriminator(self.n_qubits).to(self.device)
        self.discriminator.apply(Discriminator.weights_init)

        self.params = np.random.rand(self.n_params) * np.pi
        self.generator = QuantumGenerator(self.n_qubits, self.execute_circuit, self.batch_size)

        self.accuracy = []
        self.criterion = torch.nn.BCELoss()
        self.optimizer_discriminator = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=config["learning_rate_discriminator"],
            betas=(self.beta_1, 0.999)
        )

        self.real_labels = torch.full((self.batch_size,), 1.0, dtype=torch.float, device=self.device)
        self.fake_labels = torch.full((self.batch_size,), 0.0, dtype=torch.float, device=self.device)

        self.dataloader = DataLoader(self.bins_train, batch_size=self.batch_size, shuffle=True, drop_last=True)

        if config['loss'] == "KL":
            self.loss_func = self.kl_divergence
        elif config['loss'] == "NLL":
            self.loss_func = self.nll
        else:
            raise NotImplementedError("Loss function not implemented")

    def start_training(self, input_data: dict, config: dict, **kwargs: dict) -> dict:  # pylint: disable=R0915
        """
        This function starts the training of the QGAN.

        :param input_data: Dictionary with the variables from the circuit needed to start the training
        :param config: Training settings
        :param kwargs: Optional additional arguments
        :return: Dictionary including the solution
        """
        self.setup_training(input_data, config)
        generator_losses = []
        discriminator_losses = []
        best_kl_divergence = float('inf')
        best_generator_params = None
        pmfs_model = None
        best_sample = None

        n_batches = len(self.dataloader)

        for epoch in range(self.n_epochs):
            for batch, data in enumerate(self.dataloader):
                # Training the discriminator
                # Data from real distribution for training the discriminator
                real_data = data.float().to(self.device)
                self.discriminator.zero_grad()
                out_d_real = self.discriminator(real_data).view(-1)
                err_d_real = self.criterion(out_d_real, self.real_labels)
                err_d_real.backward()

                # Use Quantum Variational Circuit to generate fake samples
                fake_data, _ = self.generator.execute(self.params, self.batch_size)
                fake_data = fake_data.float().to(self.device)
                out_d_fake = self.discriminator(fake_data).view(-1)
                err_d_fake = self.criterion(out_d_fake, self.fake_labels)
                err_d_fake.backward()

                err_d = err_d_real + err_d_fake
                self.optimizer_discriminator.step()

                out_d_fake = self.discriminator(fake_data).view(-1)
                err_g = self.criterion(out_d_fake, self.real_labels)
                fake_data, _ = self.generator.execute(self.params, self.batch_size)
                gradients = self.generator.compute_gradient(
                    self.params,
                    self.discriminator,
                    self.criterion,
                    self.real_labels,
                    self.device
                )

                updated_params = self.params - self.learning_rate_generator * gradients
                self.params = updated_params

                self.discriminator_weights = self.discriminator.state_dict()
                generator_losses.append(err_g.item())
                discriminator_losses.append(err_d.item())

                # Calculate loss
                _, pmfs_model = self.generator.execute(self.params, self.n_shots)
                pmfs_model = np.asarray(pmfs_model.copy())

                loss = self.loss_func(pmfs_model[None,], self.target)
                self.accuracy.append(loss)

                self.writer.add_scalar("metrics/KL", loss, epoch * n_batches + batch)
                circuit_evals = (epoch * n_batches + batch) * self.batch_size * (2 * self.n_params + 1)
                self.writer.add_scalar("metrics/KL_circuit_evals", loss, circuit_evals)

                # Calculate and log the loss values at the end of each epoch
                self.writer.add_scalar('Loss/GAN_Generator', err_g.item(), circuit_evals)
                self.writer.add_scalar('Loss/GAN_Discriminator', err_d.item(), circuit_evals)

                if loss < best_kl_divergence:
                    best_kl_divergence = loss
                    best_generator_params = self.params.copy()  # Make a copy of the parameters
                _, best_pdf = self.generator.execute(best_generator_params, self.n_shots)
                best_pdf = np.asarray(best_pdf)
                best_pdf = best_pdf / best_pdf.sum()
                best_sample = self.sample_from_pmf(pmf=best_pdf, n_shots=self.n_shots)

                # Log the training progress
                log_message = (
                    f"Epoch: {epoch + 1}/{self.n_epochs}, "
                    f"Batch: {batch + 1}/{len(self.bins_train) // self.batch_size}, "
                    f"Discriminator Loss: {err_d.item()}, Generator Loss: {err_g.item()}, KL Divergence: {loss} "
                )

                logging.info(log_message)

            fig, ax = plt.subplots()
            ax.imshow(
                pmfs_model.reshape((2 ** (self.n_qubits // 2), 2 ** (self.n_qubits // 2))),
                cmap='binary',
                interpolation='none'
            )
            ax.set_title(f'Iteration {epoch}')
            self.writer.add_figure('grid_figure', fig, global_step=epoch)

            ax.clear()
            ax.imshow(
                self.target.reshape((2 ** (self.n_qubits // 2), 2 ** (self.n_qubits // 2))),
                cmap='binary',
                interpolation='none'
            )
            # Log the figure in TensorBoard
            ax.set_title("train")
            self.writer.add_figure("train", fig)

            # Plot the generator and discriminator losses on the existing figure
            ax.clear()
            ax.plot(generator_losses, label='Generator Loss', color='blue')
            ax.plot(discriminator_losses, label='Discriminator Loss', color='red')
            ax.legend()

            # Save the updated loss plot to TensorBoard
            self.writer.add_figure('Loss_Plot', fig, global_step=epoch)

        plt.close()
        self.writer.flush()
        self.writer.close()

        input_data["best_parameter"] = best_generator_params
        input_data["best_sample"] = best_sample
        input_data["KL"] = self.accuracy
        input_data["generator_loss"] = generator_losses
        input_data["discriminator_loss"] = discriminator_losses

        return input_data


class Discriminator(nn.Module):
    """
    This class defines the discriminator of the QGAN.
    """

    def __init__(self, input_length: int):
        super().__init__()
        self.dense1 = nn.Linear(int(input_length), 2 * int(input_length))
        self.dense2 = nn.Linear(2 * int(input_length), 1)

    def forward(self, x: torch.Tensor) -> float:
        """
        Initializes the weight tensor of the linear layers with values using a Xavier uniform distribution.

        :param x: Input of the discriminator
        :type x: torch.Tensor
        :return: Probability fake/real sample
        :rtype: float
        """
        h = funct.leaky_relu(self.dense1(x))
        h = funct.leaky_relu(self.dense2(h))
        return funct.sigmoid(h)

    @staticmethod
    def weights_init(m: nn.Linear) -> None:
        """
        Initializes the weight tensor of the linear
        layers with values using a Xavier uniform distribution.

        :param m: Neural network layer
        """
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data, gain=10)
            nn.init.constant_(m.bias.data, 1)


class QuantumGenerator:
    """
    This class defines the generator of the QGAN.
    """

    def __init__(self, n_qubits, execute_circuit, batch_size):
        self.n_qubits = n_qubits
        self.execute_circuit = execute_circuit
        self.batch_size = batch_size

    def execute(self, params: np.ndarray, n_shots: int) -> tuple[torch.Tensor, np.ndarray]:
        """
        Forward pass of the generator.

        :param params: Parameters of the quantum circuit
        :param n_shots: Number of shots
        :return: samples and the probability distribution generated by the quantum circuit
        """

        # Call the quantum circuit and obtain probability distributions
        pdfs, _ = self.execute_circuit(np.expand_dims(params, axis=0))
        pdfs = pdfs.flatten()

        # Sample from the provided probability distribution for the specified batch size
        sampling = torch.multinomial(torch.from_numpy(pdfs), n_shots, replacement=True)

        # Convert the sampling tensor to a list of integers
        binary_samples = [(sampling >> i) & 1 for i in range(self.n_qubits)]
        binary_samples = binary_samples[::-1]  # Reverse the order to match your expected format

        # Convert binary samples to a PyTorch tensor
        samples = torch.stack(binary_samples, dim=1).float()

        return samples, pdfs

    # pylint: disable=R0917
    def compute_gradient(self, params: np.ndarray, discriminator: torch.nn.Module, criterion: callable,
                         label: torch.Tensor, device: str) -> np.ndarray:
        """
        This function defines the forward pass of the generator.

        :param params: Parameters of the quantum circuit
        :param discriminator: Discriminator of the QGAN
        :param criterion: Loss function
        :param label: Label indicating of sample is true or fake
        :param device: Torch device (e.g., CPU or CUDA)
        :return: Samples and the probability distribution generated by the quantum circuit
        """
        shift = 0.5 * np.pi
        gradients = np.zeros(len(params))  # Initialize gradients as an array of zeros

        for i in range(len(params)):
            # Compute shifts for the i-th parameter
            positive_shifted_params = params.copy()
            negative_shifted_params = params.copy()
            positive_shifted_params[i] += shift
            negative_shifted_params[i] -= shift

            # Generate samples with shifts
            positive_samples, _ = self.execute(positive_shifted_params, self.batch_size)
            negative_samples, _ = self.execute(negative_shifted_params, self.batch_size)

            # Convert positive_samples and negative_samples to tensors
            positive_samples = positive_samples.to(device)
            negative_samples = negative_samples.to(device)

            # Compute discriminator outputs for all samples
            forward_outputs = discriminator(positive_samples).view(-1)
            backward_outputs = discriminator(negative_samples).view(-1)

            # Compute criterion differences for all samples
            forward_diff = criterion(forward_outputs, label)
            backward_diff = criterion(backward_outputs, label)

            # Calculate the gradient for the i-th parameter
            gradients[i] = 0.5 * (forward_diff.item() - backward_diff.item())

        return gradients
