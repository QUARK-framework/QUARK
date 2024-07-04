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
from typing import Union
import logging
from time import perf_counter

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.compiler import transpile, assemble
from qiskit.transpiler import CouplingMap
from qiskit.providers import Backend
from qiskit import Aer
from qiskit_aer import AerSimulator
from qiskit_aer import noise
from qiskit_aer.noise import NoiseModel
# from qiskit_ibm_runtime import QiskitRuntimeService
import numpy as np

from modules.training.QCBM import QCBM
from modules.training.Inference import Inference
from modules.applications.QML.generative_modeling.mappings.Library import Library

logging.getLogger("NoisyQiskit").setLevel(logging.WARNING)


def split_string(s):
    return s.split(' ', 1)[0]


class CustomQiskitNoisyBackend(Library):
    """
    This module maps a library-agnostic gate sequence to a qiskit circuit
    """

    def __init__(self):
        """
        Constructor method
        """
        super().__init__("NoisyQiskit")
        self.submodule_options = ["QCBM", "Inference"]

    circuit_transpiled = None

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Returns requirements of this module

        :return: list of dict with requirements of this module
        :rtype: list[dict]
        """
        return [
            {
                "name": "qiskit",
                "version": "0.45.0"
            },
            # {
            #    "name": "qiskit_ibm_runtime",
            #      "version": "0.10.0"
            # },
            {
                "name": "qiskit_aer",
                "version": "0.11.2"
            },
            {
                "name": "numpy",
                "version": "1.23.5"
            },
            {
                "name": "qiskit-ibmq-provider",
                "version": "0.19.2"
            }
        ]

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for the Qiskit Library.

        :return:
                 .. code-block:: python

                        return {
                            "backend": {
                                "values": ["aer_statevector_simulator_gpu", "aer_statevector_simulator_cpu",
                                           "cusvaer_simulator (only available in cuQuantum applicance)",
                                           "aer_simulator_gpu",
                                           "aer_simulator_cpu", "ionQ_Harmony", "Amazon_SV1"],
                                "description": "Which backend do you want to use? (aer_statevector_simulator
                                                uses the measurement probability vector, the others are shot based)"
                            },

                            "n_shots": {
                                "values": [100, 1000, 10000, 1000000],
                                "description": "How many shots do you want use for estimating the PMF of the model?
                                                (If the aer_statevector_simulator selected,
                                                only relevant for studying generalization)"
                            }
                        }

        """
        value_list = []
        value_list.append('Custom configurations')
        value_list.append('No noise')
        return {
            "backend": {
                "values": ["aer_simulator_gpu", "aer_simulator_cpu"],
                "description": "Which backend do you want to use? "
                               "In the NoisyQiskit module only aer_simulators can be used."
            },

            "simulation_method": {
                "values": ["automatic", "statevector", "density_matrix", "cpu_mps"],  # TODO Change names!
                "description": "What simulation method should be used?"
            },

            "n_shots": {
                "values": [100, 1000, 10000, 1000000],
                "description": "How many shots do you want use for estimating the PMF of the model?"
                # (If the aer_statevector_simulator selected, only relevant for studying generalization)"
            },

            "transpile_optimization_level": {
                "values": [1, 2, 3, 0],
                "description": "Switch between different optimization levels in the Qiskit transpile routine. "
                               "1: light optimization, 2: heavy optimization, 3: even heavier optimization, "
                               "0: no optimization. Level 1 recommended as standard option."
            },

            "noise_configuration": {
                "values": value_list,
                "description": "What noise configuration do you want to use?"
            },
            "custom_readout_error": {
                "values": [0, 0.005, 0.01, 0.02, 0.05, 0.07, 0.1, 0.2],
                "description": "Add a custom readout error."
            },
            "two_qubit_depolarizing_errors": {
                "values": [0, 0.005, 0.01, 0.02, 0.05, 0.07, 0.1, 0.2]
                ,
                "description": "Add a custom 2-qubit gate depolarizing error."
            },
            "one_qubit_depolarizing_errors": {
                "values": [0, 0.0001, 0.0005, 0.001, 0.005, 0.007, 0.01, 0.02]
                ,
                "description": "Add a 1-qubit gate depolarizing error."
            },
            "qubit_layout": {
                # "values": [None, 'linear', 'circle', 'fully_connected', 'ibm_brisbane'],
                "values": [None, 'linear', 'circle', 'fully_connected'],
                "description": "How should the qubits be connected in the simulated chip: coupling_map "
            }
        }

    def get_default_submodule(self, option: str) -> Union[QCBM, Inference]:

        if option == "QCBM":
            return QCBM()
        elif option == "Inference":
            return Inference()
        else:
            raise NotImplementedError(f"Option {option} not implemented")

    def sequence_to_circuit(self, input_data: dict) -> dict:
        """
        Maps the gate sequence, that specifies the architecture of a quantum circuit
        to its Qiskit implementation.

        :param input_data: Collected information of the benchmarking process
        :type input_data: dict
        :return: Same dictionary but the gate sequence is replaced by it Qiskit implementation
        :rtype: dict
        """
        n_qubits = input_data["n_qubits"]
        gate_sequence = input_data["gate_sequence"]
        circuit = QuantumCircuit(n_qubits, n_qubits)
        param_counter = 0
        for gate, wires in gate_sequence:
            if gate == "Hadamard":
                circuit.h(wires[0])
            elif gate == "X":
                circuit.x(wires[0])
            elif gate == "SX":
                circuit.sx(wires[0])
            elif gate == "RZ_PI/2":
                circuit.rz(np.pi / 2, wires[0])
            elif gate == "CNOT":
                circuit.cx(wires[0], wires[1])
            elif gate == "ECR":
                circuit.ecr(wires[0], wires[1])
            elif gate == "RZ":
                circuit.rz(Parameter(f"x_{param_counter:03d}"), wires[0])
                param_counter += 1
            elif gate == "RX":
                circuit.rx(Parameter(f"x_{param_counter:03d}"), wires[0])
                param_counter += 1
            elif gate == "RY":
                circuit.ry(Parameter(f"x_{param_counter:03d}"), wires[0])
                param_counter += 1
            elif gate == "RXX":
                circuit.rxx(Parameter(f"x_{param_counter:03d}"), wires[0], wires[1])
                param_counter += 1
            elif gate == "RYY":
                circuit.ryy(Parameter(f"x_{param_counter:03d}"), wires[0], wires[1])
                param_counter += 1
            elif gate == "RZZ":
                circuit.rzz(Parameter(f"x_{param_counter:03d}"), wires[0], wires[1])
                param_counter += 1
            elif gate == "CRY":
                circuit.cry(Parameter(f"x_{param_counter:03d}"), wires[0], wires[1])
                param_counter += 1
            elif gate == "Barrier":
                circuit.barrier()
            elif gate == "Measure":
                circuit.measure(wires[0], wires[0])
            else:
                raise NotImplementedError(f"Gate {gate} not implemented")

        input_data["circuit"] = circuit
        input_data.pop("gate_sequence")
        logging.info(param_counter)
        return input_data

    @staticmethod
    def select_backend(config: str) -> dict:
        """
        This method configures the backend

        :param config: Name of a backend
        :type config: str
        :return: Configured qiskit backend
        :rtype: qiskit.providers.Backend
        """

        if config == "aer_simulator_gpu":
            # from qiskit import Aer  # pylint: disable=C0415
            backend = Aer.get_backend("aer_simulator")
            backend.set_options(device="GPU")

        elif config == "aer_simulator_cpu":
            # from qiskit import Aer  # pylint: disable=C0415
            backend = Aer.get_backend("aer_simulator")
            backend.set_options(device="CPU")

        else:
            raise NotImplementedError(f"Device Configuration {config} not implemented")

        return backend

    def get_execute_circuit(self, circuit: QuantumCircuit, backend: Backend, config: str, config_dict: dict) \
            -> callable:  # pylint: disable=W0221
        """
        This method combines the qiskit circuit implementation and the selected backend and returns a function,
        that will be called during training.

        :param circuit: Qiskit implementation of the quantum circuit
        :type circuit: qiskit.circuit.QuantumCircuit
        :param backend: Configured qiskit backend
        :type backend: qiskit.providers.Backend
        :param config: Name of a backend
        :type config: str
        :param config_dict: Contains information about config
        :type config_dict: dict
        :return: Method that executes the quantum circuit for a given set of parameters
        :rtype: callable
        """
        n_shots = config_dict["n_shots"]
        n_qubits = circuit.num_qubits
        start = perf_counter()

        backend = self.decompile_noisy_config(config_dict, n_qubits)
        logging.info(f'Backend in Use: {backend=}')
        optimization_level = self.get_transpile_routine(config_dict['transpile_optimization_level'])
        seed_transp = 42 # Remove seed if wanted
        logging.info(f'Using {optimization_level=} with seed: {seed_transp}')
        circuit_transpiled = transpile(circuit, backend=backend, optimization_level=optimization_level,
                                       seed_transpiler=seed_transp)
        logging.info(f'Circuit operations before transpilation: {circuit.count_ops()}')
        logging.info(f'Circuit operations before transpilation: {circuit_transpiled.count_ops()}')
        logging.info(perf_counter() - start)


        if config in ["aer_simulator_cpu", "aer_simulator_gpu"]:
            def execute_circuit(solutions):

                all_circuits = [circuit_transpiled.bind_parameters(solution) for solution in solutions]
                qobjs = assemble(all_circuits, backend=backend)
                jobs = backend.run(qobjs, shots=n_shots)
                samples_dictionary = [jobs.result().get_counts(circuit).int_outcomes() for circuit in all_circuits]
                samples = []
                for result in samples_dictionary:
                    target_iter = np.zeros(2 ** n_qubits)
                    result_keys = list(result.keys())
                    result_vals = list(result.values())
                    target_iter[result_keys] = result_vals
                    target_iter = np.asarray(target_iter)
                    samples.append(target_iter)
                samples = np.asarray(samples)
                pmfs = samples / n_shots

                return pmfs, samples

        return execute_circuit, circuit_transpiled


    @staticmethod
    def split_string(s):
        return s.split(' ', 1)[0]

    def decompile_noisy_config(self, config_dict, num_qubits):
        backend_config = config_dict['backend']
        device = 'GPU' if 'gpu' in backend_config else 'CPU'
        simulation_method, device = self.get_simulation_method_and_device(device, config_dict['simulation_method'])

        backend = self.get_custom_config(config_dict, num_qubits) \
            if config_dict['noise_configuration'] == "Custom configurations" else Aer.get_backend("aer_simulator")

        backend.set_options(device=device, method=simulation_method)
        self.log_backend_options(backend)

        return backend

    def get_simulation_method_and_device(self, device, simulation_config):
        simulation_method = {
            "statevector": "statevector",
            "density_matrix": "density_matrix",
            "cpu_mps": "matrix_product_state"
        }.get(simulation_config, 'automatic')

        if simulation_config == "cpu_mps":
            device = 'CPU'

        return simulation_method, device

    def get_transpile_routine(self, transpile_config):
        return transpile_config if transpile_config in [0, 1, 2, 3] else 1

    def get_custom_config(self, config_dict, num_qubits):
        noise_model = self.build_noise_model(config_dict)
        coupling_map = self.get_coupling_map(config_dict, num_qubits)
        backend = AerSimulator(noise_model=noise_model,
                               coupling_map=coupling_map) if coupling_map is not None else AerSimulator(
            noise_model=noise_model)
        return backend

    def build_noise_model(self, config_dict):
        noise_model = NoiseModel()
        if config_dict['custom_readout_error']:
            readout_error = config_dict['custom_readout_error']
            noise_model.add_all_qubit_readout_error(
                [[1 - readout_error, readout_error], [readout_error, 1 - readout_error]])

        self.add_quantum_errors(noise_model, config_dict)
        return noise_model

    def add_quantum_errors(self, noise_model, config_dict):
        if config_dict['two_qubit_depolarizing_errors'] is not None:
            two_qubit_error = noise.depolarizing_error(config_dict['two_qubit_depolarizing_errors'], 2)
            for gate in ['cx', 'ecr', 'rxx']:
                noise_model.add_all_qubit_quantum_error(two_qubit_error, gate)

        if config_dict['one_qubit_depolarizing_errors'] is not None:
            one_qubit_error = noise.depolarizing_error(config_dict['one_qubit_depolarizing_errors'], 1)
            for gate in ['sx', 'x', 'rx', 'ry', 'rz', 'h', 's']:
                noise_model.add_all_qubit_quantum_error(one_qubit_error, gate)

    def get_coupling_map(self, config_dict, num_qubits):
        layout = config_dict['qubit_layout']
        if layout == 'linear':
            return CouplingMap.from_line(num_qubits)
        elif layout == 'circle':
            return CouplingMap.from_ring(num_qubits)
        elif layout == 'fully_connected':
            return CouplingMap.from_full(num_qubits)
        # elif layout == "ibm_brisbane":
            # service = QiskitRuntimeService()
             #backend = service.backend("ibm_brisbane")
            # logging.info(f'Loaded with IBMQ Account {backend.name}, {backend.version}, {backend.num_qubits}')
            # return backend.coupling_map
        elif layout is None:
            logging.info('No coupling map specified, using default.')
            return None
        else:
            raise ValueError(f"Unknown qubit layout: {layout}")

    def log_backend_options(self, backend):
        logging.info(f'Backend configuration: {backend.configuration()}')
        logging.info(f'Simulation method: {backend.options.method}')
