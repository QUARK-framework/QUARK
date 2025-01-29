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

from time import sleep
from typing import TypedDict
import logging

import numpy as np
from braket.circuits import Circuit
from braket.aws import AwsDevice
from scipy.optimize import minimize

from quark.modules.solvers.Solver import Solver
from quark.modules.Core import Core
from quark.utils import start_time_measurement, end_time_measurement


class QAOA(Solver):
    """
    QAOA with some parts copied/derived from https://github.com/aws/amazon-braket-examples.
    """

    def __init__(self):
        """
        Constructor method.
        """
        super().__init__()
        self.submodule_options = [
            "LocalSimulator", "arn:aws:braket:::device/quantum-simulator/amazon/sv1",
            "arn:aws:braket:::device/quantum-simulator/amazon/tn1",
            "arn:aws:braket:us-east-1::device/qpu/ionq/Harmony",
            "arn:aws:braket:us-west-1::device/qpu/rigetti/Aspen-M-3"
        ]

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Return requirements of this module.

        :return: List of dict with requirements of this module
        """
        return [
            {"name": "amazon-braket-sdk", "version": "1.88.2"},
            {"name": "scipy", "version": "1.12.0"},
            {"name": "numpy", "version": "1.26.4"}
        ]

    def get_default_submodule(self, option: str) -> Core:
        """
        Returns the default submodule based on the provided option.

        :param option: The name of the submodule
        :return: Instance of the default submodule
        """

        if option == "arn:aws:braket:us-east-1::device/qpu/ionq/Harmony":
            from quark.modules.devices.braket.Ionq import Ionq  # pylint: disable=C0415
            return Ionq("ionQ", "arn:aws:braket:us-east-1::device/qpu/ionq/Harmony")
        elif option == "arn:aws:braket:::device/quantum-simulator/amazon/sv1":
            from quark.modules.devices.braket.SV1 import SV1  # pylint: disable=C0415
            return SV1("SV1", "arn:aws:braket:::device/quantum-simulator/amazon/sv1")
        elif option == "arn:aws:braket:::device/quantum-simulator/amazon/tn1":
            from quark.modules.devices.braket.TN1 import TN1  # pylint: disable=C0415
            return TN1("TN1", "arn:aws:braket:::device/quantum-simulator/amazon/tn1")
        elif option == "arn:aws:braket:us-west-1::device/qpu/rigetti/Aspen-M-3":
            from quark.modules.devices.braket.Rigetti import Rigetti  # pylint: disable=C0415
            return Rigetti("Rigetti Aspen-9", "arn:aws:braket:us-west-1::device/qpu/rigetti/Aspen-M-3")
        elif option == "LocalSimulator":
            from quark.modules.devices.braket.LocalSimulator import LocalSimulator  # pylint: disable=C0415
            return LocalSimulator("LocalSimulator")
        else:
            raise NotImplementedError(f"Device Option {option} not implemented")

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this solver.

        :return: Dictionary of parameter settings
        .. code-block:: python

                    return {
                            "shots": {  # number measurements to make on circuit
                                "values": list(range(10, 500, 30)),
                                "description": "How many shots do you need?"
                            },
                            "opt_method": {
                                "values": ["Powell", "Nelder-Mead"],
                                "description": "Which optimization method do you want?"
                            },
                            "depth": {
                                "values": [3],
                                "description": "Which circuit depth for QAOA do you want?"
                            }
                        }

        """
        return {
            "shots": {  # number of measurements to make on circuit
                "values": list(range(10, 500, 30)),
                "description": "How many shots do you need?"
            },
            "opt_method": {
                "values": ["Powell", "Nelder-Mead"],
                "description": "Which optimization method do you want?"
            },
            "depth": {
                "values": [3],
                "description": "Which circuit depth for QAOA do you want?"
            }
        }

    class Config(TypedDict):
        """
        Attributes of a valid config.

        .. code-block:: python

            shots: int
            opt_method: str
            depth: int

        """
        shots: int
        opt_method: str
        depth: int

    def run(self, mapped_problem: dict, device_wrapper: any, config: Config, **kwargs: dict) -> tuple[any, float, dict]:
        """
        Run QAOA algorithm on Ising.

        :param mapped_problem: Dict containing problem parameters mapped to the Ising model
        :param device_wrapper: Instance of device
        :param config: Solver configuration settings
        :param kwargs: No additionally settings needed
        :return: Solution, the time it took to compute it and optional additional information
        """
        j = mapped_problem['J']
        if np.any(np.iscomplex(j)):
            logging.warning("The problem matrix of the QAOA solver contains imaginary numbers."
                            "This may lead to an error later in the run.")
        else:
            j = np.real(j)

        # Set up the problem
        n_qubits = j.shape[0]

        # User-defined hypers
        depth = config['depth']
        opt_method = config['opt_method']  # SLSQP, COBYLA, Nelder-Mead, BFGS, Powell, ...

        # Initialize reference solution (simple guess)
        bitstring_init = -1 * np.ones([n_qubits])
        energy_init = np.dot(bitstring_init, np.dot(j, bitstring_init))

        # Set tracker to keep track of results
        tracker = {
            'count': 1,  # Elapsed optimization steps
            'optimal_energy': energy_init,  # Global optimal energy
            'opt_energies': [],  # Optimal energy at each step
            'global_energies': [],  # Global optimal energy at each step
            'optimal_bitstring': bitstring_init,  # Global optimal bitstring
            'opt_bitstrings': [],  # Optimal bitstring at each step
            'costs': [],  # Cost (average energy) at each step
            'res': None,  # Quantum result object
            'params': []  # Track parameters
        }

        # Set options for classical optimization
        options = {'disp': True, 'maxiter': 100}
        # options = {'disp': True, 'ftol': 1e-08, 'maxiter': 100, 'maxfev': 50}  # example options

        ##################################################################################
        # Run QAOA optimization on graph
        ##################################################################################

        logging.info(f"Circuit depth hyperparameter:{depth}")
        logging.info(f"Problem size:{n_qubits}")

        # Kick off training
        start = start_time_measurement()
        _, _, tracker = train(
            device=device_wrapper.get_device(),
            options=options,
            p=depth, ising=j,
            n_qubits=n_qubits,
            n_shots=config['shots'],
            opt_method=opt_method,
            tracker=tracker,
            s3_folder=device_wrapper.s3_destination_folder,
            verbose=True
        )
        time_to_solve = end_time_measurement(start)

        # Log optimized results
        logging.info(f"Optimal energy: {tracker['optimal_energy']}")
        logging.info(f"Optimal classical bitstring: {tracker['optimal_bitstring']}")

        # TODO maybe save this plot
        # plt.plot(cycles, optim_classical)
        # plt.xlabel('optimization cycle')
        # plt.ylabel('best classical minimum')
        # plt.show()

        return tracker['optimal_bitstring'], time_to_solve, {}


# QAOA utils (source:
# https://github.com/aws/amazon-braket-examples/blob/main/examples/hybrid_quantum_algorithms/QAOA/utils_qaoa.py)

# Function to implement ZZ gate using CNOT gates
def zz_gate(q1: any, q2: any, gamma: float) -> Circuit:
    """
    Function that returns a circuit implementing exp(-i \\gamma Z_i Z_j) using CNOT gates if ZZ not supported.

    :param q1: Qubit 1 (control)
    :param q2: Qubit 2 (target)
    :param gamma: Gamma parameter (angle)
    :return: ZZ gate
    """
    # Get a circuit
    circ_zz = Circuit()

    # Construct decomposition of ZZ
    circ_zz.cnot(q1, q2).rz(q2, gamma).cnot(q1, q2)

    return circ_zz


# Function to implement evolution with driver Hamiltonian
def driver(beta: float, n_qubits: int) -> Circuit:
    """
    Returns circuit for driver Hamiltonian U(Hb, beta).

    :param beta: Beta parameter (angle)
    :param n_qubits: Number of qubits
    :return: Circuit with rotated qubits
    """
    # Instantiate circuit object
    circ = Circuit()

    # Apply parametrized rotation around x to every qubit
    for qubit in range(n_qubits):
        gate = Circuit().rx(qubit, 2 * beta)
        circ.add(gate)

    return circ


# Helper function for evolution with cost Hamiltonian
def cost_circuit(gamma: float, ising: np.ndarray, device: AwsDevice) -> Circuit:
    """
    Returns circuit for evolution with cost Hamiltonian.

    :param gamma: Gamma parameter (angle)
    :param ising: Ising matrix
    :param device: Device to run the circuit on
    :return: Circuit representing the cost Hamiltonian
    """
    # Instantiate circuit object
    circ = Circuit()

    # Get all non-zero entries (edges) from Ising matrix
    idx = ising.nonzero()
    edges = list(zip(idx[0], idx[1]))

    # Apply ZZ gate for every edge (with corresponding interaction strength)
    for qubit_pair in edges:
        # Get interaction strength from Ising matrix
        int_strength = ising[qubit_pair[0], qubit_pair[1]]
        # For Rigetti we decompose ZZ using CNOT gates
        if device.name in ["Rigetti", "Aspen-9"]:  # TODO make this more flexible
            gate = zz_gate(qubit_pair[0], qubit_pair[1], gamma * int_strength)
        # Classical simulators and IonQ support ZZ gate
        else:
            gate = Circuit().zz(qubit_pair[0], qubit_pair[1], angle=2 * gamma * int_strength)
        circ.add(gate)

    return circ


# Function to build the QAOA circuit with depth p
def circuit(params: np.array, device: AwsDevice, n_qubits: int, ising: np.ndarray) -> Circuit:
    """
    Function to return the full QAOA circuit; depends on device as ZZ implementation depends on gate set of backend.

    :param params: Array containing the beta and gamma parameters
    :param device: Device to run the circuit on
    :param n_qubits: Number of qubits
    :param ising: Ising matrix
    :return: QAOA Circuit
    """

    # Initialize QAOA circuit with first Hadamard layer
    circ = Circuit()
    x_on_all = Circuit().x(range(0, n_qubits))
    circ.add(x_on_all)
    h_on_all = Circuit().h(range(0, n_qubits))
    circ.add(h_on_all)

    # Setup two parameter families
    circuit_length = int(len(params) / 2)
    gammas = params[:circuit_length]
    betas = params[circuit_length:]

    # Add QAOA circuit layer blocks
    for mm in range(circuit_length):
        circ.add(cost_circuit(gammas[mm], ising, device))
        circ.add(driver(betas[mm], n_qubits))

    return circ


# Function that computes cost function for given params
# pylint: disable=R0917
# pylint: disable=R0913
def objective_function(params: np.array, device: AwsDevice, ising: np.ndarray, n_qubits: int, n_shots: int,
                       tracker: dict, s3_folder: tuple[str, str], verbose: bool) -> float:
    """
    Objective function takes a list of variational parameters as input,
    and returns the cost associated with those parameters.

    :param params: Array containing beta and gamma parameters
    :param device: Device to run the circuit on
    :param ising: Ising matrix
    :param n_qubits: Number of qubits
    :param n_shots: Number of measurements to make on the circuit
    :param tracker: Keeps track of the runs on the circuit
    :param s3_folder: AWS S3 bucket
    :param verbose: Controls degree of detail in logs
    :return: Energy expectation value
    """

    if verbose:
        logging.info("==================================" * 2)
        logging.info(f"Calling the quantum circuit. Cycle: {tracker['count']}")

    # Get a quantum circuit instance from the parameters
    qaoa_circuit = circuit(params, device, n_qubits, ising)

    # Classically simulate the circuit
    # Execute the correct device.run call depending on whether the backend is local or cloud based
    if device.name in ["DefaultSimulator", "StateVectorSimulator"]:
        task = device.run(qaoa_circuit, shots=n_shots)
    else:
        task = device.run(qaoa_circuit, s3_folder, shots=n_shots, poll_timeout_seconds=3 * 24 * 60 * 60)

        # Get ID and status of submitted task
        task_id = task.id
        status = task.state()
        logging.info(f"ID of task: {task_id}")
        logging.info(f"Status of task: {status}")

        # Wait for job to complete
        while status != 'COMPLETED':
            status = task.state()
            logging.info(f"Status: {status}")
            sleep(10)

    # Get result for this task
    result = task.result()
    logging.info(result)

    # Convert results (0 and 1) to ising (-1 and 1)
    meas_ising = result.measurements
    meas_ising[meas_ising == 0] = -1

    # Get all energies (for every shot): (n_shots, 1) vector
    all_energies = np.diag(np.dot(meas_ising, np.dot(ising, np.transpose(meas_ising))))

    # Find minimum and corresponding classical string
    energy_min = np.min(all_energies)
    tracker["opt_energies"].append(energy_min)
    optimal_string = meas_ising[np.argmin(all_energies)]
    tracker["opt_bitstrings"].append(optimal_string)
    logging.info(tracker["optimal_energy"])

    # Store optimal (classical) result/bitstring
    if energy_min < tracker["optimal_energy"]:
        tracker.update({"optimal_energy": energy_min, "optimal_bitstring": optimal_string})

    # Store global minimum
    tracker["global_energies"].append(tracker["optimal_energy"])

    # Energy expectation value
    energy_expect = np.sum(all_energies) / n_shots

    if verbose:
        logging.info(f"Minimal energy: {energy_min}")
        logging.info(f"Optimal classical string: {optimal_string}")
        logging.info(f"Energy expectation value (cost): {energy_expect}")

    # Update tracker
    tracker.update({"count": tracker["count"] + 1, "res": result})
    tracker["costs"].append(energy_expect)
    tracker["params"].append(params)

    return energy_expect


# The function to execute the training: run classical minimization.
# pylint: disable=R0917
def train(device: AwsDevice, options: dict, p: int, ising: np.ndarray, n_qubits: int, n_shots: int, opt_method: str,
          tracker: dict, s3_folder: tuple[str, str], verbose: bool = True) -> tuple[float, np.ndarray, dict]:
    """
    Function to run QAOA algorithm for given, fixed circuit depth p.

    :param device: Device to run the circuit on
    :param options: Dict containing parameters of classical part of the QAOA
    :param p: Circuit depth
    :param ising: Ising matrix
    :param n_qubits: Number of qubits
    :param n_shots: Number of measurements to make on the circuit
    :param opt_method: Controls degree of detail in logs
    :param tracker: Keeps track of the runs on the circuit
    :param s3_folder: AWS S3 bucket
    :param verbose: Controls degree of detail in logs
    :return: Results of the training as a tuple of the energy, the angle and the tracker
    """
    logging.info("Starting the training.")
    logging.info("==================================" * 2)
    logging.info(f"OPTIMIZATION for circuit depth p={p}")

    if not verbose:
        logging.info('Param "verbose" set to False. Will not print intermediate steps.')
        logging.info("==================================" * 2)

    # Initialize
    cost_energy = []

    # Randomly initialize variational parameters within appropriate bounds
    gamma_initial = np.random.uniform(0, 2 * np.pi, p).tolist()
    beta_initial = np.random.uniform(0, np.pi, p).tolist()
    params0 = np.array(gamma_initial + beta_initial)

    # Set bounds for search space
    bnds_gamma = [(0, 2 * np.pi) for _ in range(int(len(params0) / 2))]
    bnds_beta = [(0, np.pi) for _ in range(int(len(params0) / 2))]
    bnds = bnds_gamma + bnds_beta

    tracker["params"].append(params0)
    print(f"Qubit count: {n_qubits}")

    # Run classical optimization (example: method='Nelder-Mead')
    try:
        result = minimize(
            objective_function,
            params0,
            args=(device, ising, n_qubits, n_shots, tracker, s3_folder, verbose),
            options=options,
            method=opt_method,
            bounds=bnds,
        )
    except ValueError as e:
        logging.error(f"The following ValueError occurred in module QAOA: {e}")
        logging.error("The benchmarking run terminates with exception.")
        raise Exception("Please refer to the logged error message.") from e

    # Store result of classical optimization
    result_energy = result.fun
    cost_energy.append(result_energy)
    logging.info(f"Final average energy (cost): {result_energy}")
    result_angle = result.x
    logging.info(f"Final angles: {result_angle}")
    logging.info("Training complete.")

    return result_energy, result_angle, tracker
