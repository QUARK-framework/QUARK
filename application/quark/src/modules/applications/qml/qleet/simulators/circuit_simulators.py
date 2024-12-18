"""Module to draw samples from the circuit.
Used for computing properties of the circuit like Entanglability and Expressibility.
"""

import typing

import cirq
import numpy as np
import qiskit

from qiskit_aer.noise import NoiseModel as qiskitNoiseModel
from cirq.devices.noise_model import NoiseModel as cirqNoiseModel
from pyquil.noise import NoiseModel as pyquilNoiseModel

from ..interface.circuit import CircuitDescriptor
from qiskit import transpile
from qiskit_aer import AerSimulator

class CircuitSimulator:
    """The interface for users to execute their CircuitDescriptor objects"""

    def __init__(
        self,
        circuit: CircuitDescriptor,
        noise_model: typing.Union[
            cirqNoiseModel, qiskitNoiseModel, pyquilNoiseModel, None
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
        # if self.circuit.default_backend == "qiskit":
        #     circuit = self.circuit.qiskit_circuit.bind_parameters(param_resolver)
        #     if self.noise_model is not None:
        #         circuit.snapshot("final", snapshot_type="density_matrix")
        #         result = qiskit.execute(
        #             circuit,
        #             qiskit.Aer.get_backend("qasm_simulator"),
        #             shots=shots,
        #             noise_model=self.noise_model,
        #             backend_options={"method": "density_matrix"},
        #         ).result()
        #         result_data = result.data(0)["snapshots"]["density_matrix"]["final"][0][
        #             "value"
        #         ]
        #     else:
        #         circuit.snapshot("final", snapshot_type="statevector")
        #         result = qiskit.execute(
        #             circuit, qiskit.Aer.get_backend("aer_simulator_statevector")
        #         ).result()
        #         result_data = result.data(0)["snapshots"]["statevector"]["final"][0]

        if self.circuit.default_backend == "qiskit":
            backend = AerSimulator() #.get_backend('aer_simulator_statevector')

            # Use assign_parameters instead of bind_parameters
            circuit = self.circuit.qiskit_circuit.assign_parameters(param_resolver)

            # Check if a noise model is provided
            if self.noise_model is not None:
                # Add a snapshot for density matrix
                circuit.save_statevector(label='statevector')
                
                # Transpile the circuit for the backend
                new_circuit = transpile(circuit, backend)

                # Execute the circuit with noise model
                job = backend.run(new_circuit, shots=shots, noise_model=self.noise_model, backend_options={"method": "density_matrix"})
                
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


        elif self.circuit.default_backend == "cirq":

            circuit = self.circuit.cirq_circuit
            non_unitary_flag = False
            for op in circuit.all_operations():
                op_name = str(op).split("(")[0]
                if op_name in [
                    "phase_flip",
                    "phase_damp",
                    "amplitude_damp",
                    "depolarize",
                    "asymmetric_depolarize",
                ]:
                    non_unitary_flag = True
                    break

            if self.noise_model is None and not non_unitary_flag:
                simulator = cirq.Simulator()  # type: ignore
                result = simulator.simulate(self.circuit.cirq_circuit, param_resolver)
                result_data = result.final_state_vector
            else:
                simulator = cirq.DensityMatrixSimulator(noise=self.noise_model)  # type: ignore
                result = simulator.simulate(self.circuit.cirq_circuit, param_resolver)
                result_data = result.final_density_matrix

        else:
            raise NotImplementedError(
                "Parametrized circuit simulation is not implemented for this backend."
            )

        self._result = result_data
        return result_data
