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

"""Module to draw samples from the circuit.
Used for computing properties of the circuit like Entanglability and Expressibility.
"""

import typing

import numpy as np
import qiskit

from qiskit_aer.noise import NoiseModel as qiskitNoiseModel
from ..interface.circuit import CircuitDescriptor
from qiskit import transpile
from qiskit_aer import AerSimulator


class CircuitSimulator:
    """The interface for users to execute their CircuitDescriptor objects"""

    def __init__(
        self,
        circuit: CircuitDescriptor,
        noise_model: typing.Union[
            qiskitNoiseModel, None
        ] = None,
    ) -> None:
        """Initialize the state simulator
        :type circuit: CircuitDescriptor
        :param circuit: the target circuit to simulate
        :type noise_model: Noise model as a dict or in the library format
        :param noise_model: the noise model as dict or empty dict for density matrix simulations,
            None if performing state vector simulations
        """
        self.circuit = circuit
        self.noise_model = noise_model
        self._result = None

    @property
    def result(
        self,
    ) -> typing.Optional[np.ndarray]:
        """Get the results stored from the circuit simulator
        :return: stored result of the circuit simulation if it has been performed, else None.
        :rtype: np.array or None
        """
        return self._result

    def simulate(
        self,
        param_resolver: typing.Dict[qiskit.circuit.Parameter, float],
        shots: int = 1024,
    ) -> np.ndarray:
        """Simulate to get the state vector or the density matrix
        :type param_resolver: Dict to resolve all parameters to a static float value
        :param param_resolver: a dictionary of all the symbols/parameters mapping to their values
        :type shots: int
        :param shots: number of times to run the qiskit density matrix simulator
        :returns: state vector or density matrix resulting from the simulation
        :rtype: np.array
        :raises NotImplementedError: if circuit simulation is not supported for a backend
        """

        if self.circuit.default_backend == "qiskit":
            backend = AerSimulator()  # .get_backend('aer_simulator_statevector')

            # Use assign_parameters instead of bind_parameters
            circuit = self.circuit.qiskit_circuit.assign_parameters(param_resolver)

            # Check if a noise model is provided
            if self.noise_model is not None:
                # Add a snapshot for density matrix
                circuit.save_statevector(label='statevector')

                # Transpile the circuit for the backend
                new_circuit = transpile(circuit, backend)

                # Execute the circuit with noise model
                job = backend.run(
                    new_circuit,
                    shots=shots,
                    noise_model=self.noise_model,
                    backend_options={
                        "method": "density_matrix"})

                # Get the result
                result = job.result()

                result_data = result.data(0)["statevector"]
            else:
                # Add a snapshot for statevector
                circuit.save_statevector(label='statevector')

                # Transpile the circuit for the backend
                new_circuit = transpile(circuit, backend)

                # Execute the circuit without noise
                job = backend.run(new_circuit, shots=shots)

                # Get the result
                result = job.result()

                result_data = result.data(0)["statevector"]

        else:
            raise NotImplementedError(
                "Parametrized circuit simulation is not implemented for this backend."
            )

        self._result = result_data
        return result_data
