import copy
from typing import List, Optional

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from torchvision import models


class EarlyStopping(object):
    def __init__(self, mode="min", min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if not percentage:
            if mode == "min":
                self.is_better = lambda a, best: a < best - min_delta
            if mode == "max":
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == "min":
                self.is_better = lambda a, best: a < best - (best * min_delta / 100)
            if mode == "max":
                self.is_better = lambda a, best: a > best + (best * min_delta / 100)


class QuantumLayer(nn.Module):
    """
    Torch module implementing a quantum layer.
    """

    def __init__(
        self, nqubits=4, circuit_depth=4, quantum_device: Optional[qml.Device] = None
    ) -> None:
        """
        Define the quantum circuit.
        """
        super(QuantumLayer, self).__init__()
        self.n_qubits = nqubits
        if quantum_device is None:
            self.quantum_device = qml.device("default.qubit", wires=self.n_qubits)
        else:
            self.quantum_device = quantum_device
        self.num_variational_layers = circuit_depth
        self.model = self.build_circuit()

    def H_layer(self, nqubits):
        """
        Layer of single-qubit Hadamard gates.
        """
        for idx in range(nqubits):
            self.circuit.h(idx)

    def RY_layer(self, w):
        """
        Layer of parametrized qubit rotations around the y axis.
        """
        for idx, element in enumerate(w):
            self.circuit.ry(element, idx)

    def entangling_layer(self, nqubits):
        """
        Layer of CNOTs followed by another shifted layer of CNOT.
        """
        for i in range(0, nqubits - 1, 2):
            self.circuit.cx(i, i + 1)
        for i in range(1, nqubits - 1, 2):
            self.circuit.cx(i, i + 1)

    def generate_pauliz_observables(self, nqubits):
        """
        PauliZ gates for each qubit of the circuit
        """
        observables = []
        for i in range(nqubits):
            pauli_string = ["I"] * nqubits
            pauli_string[i] = "Z"
            observables.append(SparsePauliOp.from_list([("".join(pauli_string), 1.0)]))
        return observables

    def build_circuit(self):
        """
        Define the quantum node, i.e. variational quantum circuit plus device on which the circuit is executed.

        Args:
            q_input_features: The input features for the circuit.
            q_weigths_flat: The trainable variational parameters of the circuit.

        Returns:
            The expectation value of a measurement.
        """

        self.circuit = QuantumCircuit(self.n_qubits)

        input_params = [Parameter(f"input_{i}") for i in range(self.n_qubits)]
        var_params = [
            Parameter(f"var_{i}")
            for i in range(self.num_variational_layers * self.n_qubits)
        ]
        self.params = input_params + var_params

        self.H_layer(self.n_qubits)
        self.RY_layer(input_params)

        for i in range(self.num_variational_layers):
            self.entangling_layer(self.n_qubits)
            for j in range(self.n_qubits):
                self.circuit.ry(var_params[i * self.n_qubits + j], j)

        observables = self.generate_pauliz_observables(self.n_qubits)

        self.qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=observables,
            input_params=input_params,
            weight_params=var_params,
            input_gradients=True,
        )

        model = TorchConnector(self.qnn)
        return model

    def forward(self, x):
        x = torch.tanh(x) * np.pi / 2.0
        return self.model(x)

    def draw(self):
        """
        Print the quantum layer/circuit using the qiskit built-in functionality
        """
        print(self.circuit.draw(output="text"))


class QuantumModel(nn.Module):
    def __init__(
        self,
        quantum_device=None,
        n_reduced_features=4,
        circuit_depth=1,
        dataset_name="Concrete_Crack",
        n_classes=2,
    ):
        super(QuantumModel, self).__init__()

        self.dataset_name = dataset_name
        self.n_classes = n_classes

        if quantum_device is None:
            quantum_device = qml.device("default.qubit", wires=n_reduced_features)

        self.quantum_layer = QuantumLayer(
            nqubits=n_reduced_features,
            quantum_device=quantum_device,
            circuit_depth=circuit_depth,
        )

        if self.dataset_name == "Concrete_Crack":
            self.image_features = 512
            self.model = nn.Sequential(
                nn.Linear(self.image_features, n_reduced_features),
                self.quantum_layer,
                nn.Linear(n_reduced_features, self.n_classes),
            )

        elif self.dataset_name == "MNIST":
            self.image_features = 28
            self.resnet18 = models.resnet18(pretrained=True)
            self.fc_inputs = self.resnet18.fc.in_features
            self.resnet18.fc = nn.Identity()

            self.model = nn.Sequential(
                nn.Linear(self.fc_inputs, n_reduced_features),
                self.quantum_layer,
                nn.Linear(n_reduced_features, self.n_classes),
            )
        else:
            raise Exception(f"No valid dataset name {self.dataset_name}.")

    def get_quantum_circuit(self):
        return self.quantum_layer.circuit, self.quantum_layer.params

    def forward(self, x):
        if self.dataset_name == "MNIST":
            with torch.no_grad():
                x = self.resnet18(x)

        return self.model(x)
