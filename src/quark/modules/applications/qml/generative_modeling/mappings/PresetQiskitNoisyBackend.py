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
import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.providers import Backend
from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2
from qiskit_aer import Aer, AerSimulator
from qiskit_aer.noise import NoiseModel

from quark.modules.applications.qml.generative_modeling.training.QCBM import QCBM
from quark.modules.applications.qml.generative_modeling.training.Inference import Inference
from quark.modules.applications.qml.generative_modeling.mappings.LibraryGenerative import LibraryGenerative

logging.getLogger("NoisyQiskit").setLevel(logging.WARNING)


class PresetQiskitNoisyBackend(LibraryGenerative):
    """
    This module maps a library-agnostic gate sequence to a qiskit circuit.
    """

    def __init__(self):
        """
        Constructor method.
        """
        super().__init__("PresetQiskitNoisyBackend")
        self.submodule_options = ["QCBM", "Inference"]

    circuit_transpiled = None

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Returns requirements of this module.

        :return: List of dict with requirements of this module
        """
        return [
            {"name": "qiskit", "version": "1.3.0"},
            {"name": "qiskit_ibm_runtime", "version": "0.33.2"},
            {"name": "qiskit_aer", "version": "0.15.1"},
            {"name": "numpy", "version": "1.26.4"}
        ]

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for the Qiskit Library.

            :return: Dictionary with configurable settings.
            .. code-block:: python

                {
                "backend": {
                    "values": ["aer_simulator_gpu", "aer_simulator_cpu"],
                    "description": "Which backend do you want to use? "
                                   "In the NoisyQiskit Module only aer_simulators can be used."
                },

                "simulation_method": {
                    "values": ["automatic", "statevector", "density_matrix", "cpu_mps"],  # TODO Change names
                    "description": "What simulation methode should be used"
                },

                "n_shots": {
                    "values": [100, 1000, 10000, 1000000],
                    "description": "How many shots do you want use for estimating the PMF of the model?"
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
                }
            }
        """
        provider = FakeProviderForBackendV2()
        backends = provider.backends()
        value_list = []
        value_list.append('No noise')
        for backend in backends:
            if backend.num_qubits >= 6:
                value_list.append(f'{backend.name} V{backend.version} {backend.num_qubits} Qubits')

        return {
            "backend": {
                "values": ["aer_simulator_gpu", "aer_simulator_cpu"],
                "description": "Which backend do you want to use? "
                               "In the NoisyQiskit Module only aer_simulators can be used."
            },

            "simulation_method": {
                "values": ["automatic", "statevector", "density_matrix", "cpu_mps"],  # TODO Change names
                "description": "What simulation methode should be used"
            },

            "n_shots": {
                "values": [100, 1000, 10000, 1000000],
                "description": "How many shots do you want use for estimating the PMF of the model?"
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
            }
        }

    def get_default_submodule(self, option: str) -> Union[QCBM, Inference]:
        """
        Returns the default submodule based on the given option.

        :param option: The submodule option to select
        :return: Instance of the selected submodule
        :raises NotImplemented: If the provided option is not implemented
        """
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
        :return: Same dictionary but the gate sequence is replaced by it Qiskit implementation
        """
        # TODO: Identical to CustomQiskitNoisyBackend.sequence_to_circuit -> move to Library
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
        input_data["n_params"] = len(circuit.parameters)
        return input_data

    @staticmethod
    def select_backend(config: str, n_qubits: int) -> Backend:
        """
        This method configures the backend.

        :param config: Name of a backend
        :param n_qubits: Number of qubits
        :return: Configured qiskit backend
        """
        # TODO: Identical to CustomQiskitNoisyBackend.select_backend -> move to Library
        if config == "aer_simulator_gpu":
            backend = Aer.get_backend("aer_simulator")
            backend.set_options(device="GPU")
        elif config == "aer_simulator_cpu":
            backend = Aer.get_backend("aer_simulator")
            backend.set_options(device="CPU")
        else:
            raise NotImplementedError(f"Device Configuration {config} not implemented")

        return backend

    def get_execute_circuit(self, circuit: QuantumCircuit, backend: Backend,  # pylint: disable=W0221
                            config: str, config_dict: dict) -> tuple[any, any]:
        """
        This method combines the qiskit circuit implementation and the selected backend and returns a function,
        that will be called during training.

        :param circuit: Qiskit implementation of the quantum circuit
        :param backend: Configured qiskit backend
        :param config: Name of a backend
        :param config_dict: Contains information about config
        :return: Tuple that contains a method that executes the quantum circuit for a given set of parameters and the
        transpiled circuit
        """
        # TODO: Identical to CustomQiskitNoisyBackend.get_execute_circuit -> move to Library
        n_shots = config_dict["n_shots"]
        n_qubits = circuit.num_qubits
        start = perf_counter()

        backend = self.decompile_noisy_config(config_dict, n_qubits)
        logging.info(f'Backend in Use: {backend=}')
        optimization_level = self.get_transpile_routine(config_dict['transpile_optimization_level'])
        seed_transp = 42  # Remove seed if wanted
        logging.info(f'Using {optimization_level=} with seed: {seed_transp}')
        circuit_transpiled = transpile(circuit, backend=backend, optimization_level=optimization_level,
                                       seed_transpiler=seed_transp)
        logging.info(f'Circuit operations before transpilation: {circuit.count_ops()}')
        logging.info(f'Circuit operations after transpilation: {circuit_transpiled.count_ops()}')
        logging.info(perf_counter() - start)

        if config in ["aer_simulator_cpu", "aer_simulator_gpu"]:
            def execute_circuit(solutions):
                all_circuits = [circuit_transpiled.assign_parameters(solution) for solution in solutions]
                jobs = backend.run(all_circuits, shots=n_shots)
                samples_dictionary = [jobs.result().get_counts(c).int_outcomes() for c in all_circuits]
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

        else:
            logging.error(f"Unknown backend option selected: {config}")
            logging.error("Run terminates with Exception error.")
            raise Exception

        return execute_circuit, circuit_transpiled

    @staticmethod
    def split_string(s):
        return s.split(' ', 1)[0]

    def decompile_noisy_config(self, config_dict: dict, num_qubits: int) -> Backend:
        """
        This method processes a configuration dictionary.
        If a custom noise configuration is specified, it creates a custom backend configuration; otherwise, it defaults
        to the 'aer_simulator' backend. It returns the configured backend.

        :param config_dict: Contains information about config
        :param num_qubits: Number of qubits
        :return: Configured qiskit backend
        """
        backend_config = config_dict['backend']
        device = 'GPU' if 'gpu' in backend_config else 'CPU'
        simulation_method, device = self.get_simulation_method_and_device(device, config_dict['simulation_method'])
        backend = self.select_backend_configuration(config_dict['noise_configuration'], num_qubits)

        self.configure_backend(backend, device, simulation_method)
        self.log_backend_info(backend)

        return backend

    def select_backend_configuration(self, noise_configuration: str, num_qubits: int) -> Backend:
        """
        This method selects the backend configuration based on the provided noise configuration.

        :param noise_configuration: Noise configuration type
        :param num_qubits: Number of qubits
        :return: Selected backend configuration
        """
        if "fake" in noise_configuration:
            return self.get_FakeBackend(noise_configuration, num_qubits)
        elif noise_configuration == "No noise":
            return Aer.get_backend("aer_simulator")
        elif noise_configuration in ['ibm_brisbane 127 Qubits', 'ibm_osaka 127 Qubits']:
            logging.warning("Not yet implemented. Please check upcoming QUARK versions.")
            raise ValueError(f"Noise configuration '{noise_configuration}' not yet implemented.")
        else:
            raise ValueError(f"Unknown noise configuration: {noise_configuration}")

    def configure_backend(self, backend: Backend, device: str, simulation_method: str) -> None:
        """
        This method configures the backend with the specified device and simulation method.

        :param backend: Backend to be configured
        :param device: Device type (CPU/GPU)
        :param simulation_method: Simulation method
        """
        backend.set_options(device=device)
        backend.set_options(method=simulation_method)

    def log_backend_info(self, backend: Backend):
        logging.info(f'Backend configuration: {backend.configuration()}')
        logging.info(f'Simulation method: {backend.options.method}')

    def get_simulation_method_and_device(self, device: str, simulation_config: str) -> tuple[str, str]:
        """
        This method determines the simulation method and device based on the provided configuration.

        :param device: Contains information about processing unit
        :param simulation_config: Contains information about qiskit simulation method
        :return: Tuple containing the simulation method and device
        """
        simulation_methods = {
            "statevector": "statevector",
            "density_matrix": "density_matrix",
            "cpu_mps": "matrix_product_state"
        }
        simulation_method = simulation_methods.get(simulation_config, 'automatic')
        if simulation_config == "cpu_mps":
            device = 'CPU'
        return simulation_method, device

    def get_transpile_routine(self, transpile_config: int) -> int:
        """
        This method returns the transpile routine based on the provided configuration.

        :param transpile_config: Configuration for transpile routine
        :return: Transpile routine level
        """
        return transpile_config if transpile_config in [0, 1, 2, 3] else 1

    def get_FakeBackend(self, noise_configuration: str, num_qubits: int) -> Backend:
        """
        This method returns a fake backend based on the provided noise configuration and number of qubits.

        :param noise_configuration: Noise configuration type
        :param num_qubits: Number of qubits
        :return: Fake backend simulator
        """
        backend_name = str(self.split_string(noise_configuration))
        provider = FakeProviderForBackendV2()
        try:
            backend = provider.backend(name=backend_name)
        except TypeError:
            logging.info("qiskit.providers.fake_provider.FakeProviderForBackendV2.get_backend overwritten. "
                         "Will be addressed with upcoming qiskit upgrade.")
            filtered_backends = [backend for backend in provider._backends if  # pylint: disable=W0212
                                 backend.name == backend_name]
            if not filtered_backends:
                raise FileNotFoundError()  # pylint: disable=W0707
            backend = filtered_backends[0]

        if num_qubits > backend.num_qubits:
            logging.warning(f'Requested number of qubits ({num_qubits}) exceeds the backend capacity. '
                            f'Using default aer_simulator.')
            return Aer.get_backend("aer_simulator")

        noise_model = NoiseModel.from_backend(backend)
        logging.info(f'Using {backend_name} with coupling map: {backend.coupling_map}')
        logging.info(f'Using {backend_name} with noise model: {noise_model}')
        return AerSimulator.from_backend(backend)
