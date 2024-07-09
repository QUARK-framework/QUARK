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

import numpy as np
from braket.circuits import Circuit
from scipy.optimize import minimize

from modules.solvers.Solver import *
from utils import start_time_measurement, end_time_measurement


class QAOA(Solver):
    """
    QAOA with some parts copied/derived from https://github.com/aws/amazon-braket-examples.
    """

    def __init__(self):
        """
        Constructor method
        """
        super().__init__()
        self.submodule_options = ["LocalSimulator", "arn:aws:braket:::device/quantum-simulator/amazon/sv1",
                                  "arn:aws:braket:::device/quantum-simulator/amazon/tn1",
                                  "arn:aws:braket:us-east-1::device/qpu/ionq/Harmony",
                                  "arn:aws:braket:us-west-1::device/qpu/rigetti/Aspen-M-3"]

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Return requirements of this module

        :return: list of dict with requirements of this module
        :rtype: list[dict]
        """
        return [
            {
                "name": "amazon-braket-sdk",
                "version": "1.35.1"
            },
            {
                "name": "scipy",
                "version": "1.10.1"
            },
            {
                "name": "numpy",
                "version": "1.23.5"
            }
        ]

    def get_default_submodule(self, option: str) -> Core:

        if option == "arn:aws:braket:us-east-1::device/qpu/ionq/Harmony":
            from modules.devices.braket.Ionq import Ionq  # pylint: disable=C0415
            return Ionq("ionQ", "arn:aws:braket:us-east-1::device/qpu/ionq/Harmony")
        elif option == "arn:aws:braket:::device/quantum-simulator/amazon/sv1":
            from modules.devices.braket.SV1 import SV1  # pylint: disable=C0415
            return SV1("SV1", "arn:aws:braket:::device/quantum-simulator/amazon/sv1")
        elif option == "arn:aws:braket:::device/quantum-simulator/amazon/tn1":
            from modules.devices.braket.TN1 import TN1  # pylint: disable=C0415
            return TN1("TN1", "arn:aws:braket:::device/quantum-simulator/amazon/tn1")
        elif option == "arn:aws:braket:us-west-1::device/qpu/rigetti/Aspen-M-3":
            from modules.devices.braket.Rigetti import Rigetti  # pylint: disable=C0415
            return Rigetti("Rigetti Aspen-9", "arn:aws:braket:us-west-1::device/qpu/rigetti/Aspen-M-3")
        elif option == "LocalSimulator":
            from modules.devices.braket.LocalSimulator import LocalSimulator  # pylint: disable=C0415
            return LocalSimulator("LocalSimulator")
        else:
            raise NotImplementedError(f"Device Option {option} not implemented")

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this solver

        :return:
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

    class Config(TypedDict):
        """
        Attributes of a valid config

        .. code-block:: python

            shots: int
            opt_method: str
            depth: int

        """
        shots: int
        opt_method: str
        depth: int

    def run(self, mapped_problem: any, device_wrapper: any, config: Config, **kwargs: dict) -> (any, float):
        """
        Run QAOA algorithm on Ising.

        :param mapped_problem: dictionary with the keys 'J' and 't'
        :type mapped_problem: any
        :param device_wrapper: instance of device
        :type device_wrapper: any
        :param config:
        :type config: Config
        :param kwargs: no additionally settings needed
        :type kwargs: any
        :return: Solution, the time it took to compute it and optional additional information
        :rtype: tuple(list, float, dict)
        """

        j = mapped_problem['J']
        if np.any(np.iscomplex(j)):
            logging.warning("The problem matrix of the QAOA solver contains imaginary numbers."
                            "This may lead to an error later in the run.")
        else:
            j = np.real(j)

        # set up the problem
        n_qubits = j.shape[0]

        # User-defined hypers
        depth = config['depth']  # circuit depth for QAOA
        opt_method = config['opt_method']  # SLSQP, COBYLA, Nelder-Mead, BFGS, Powell, ...

        # initialize reference solution (simple guess)
        bitstring_init = -1 * np.ones([n_qubits])
        energy_init = np.dot(bitstring_init, np.dot(j, bitstring_init))

        # set tracker to keep track of results
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

        # set options for classical optimization
        options = {'disp': True, 'maxiter': 100}
        # options = {'disp': True, 'ftol': 1e-08, 'maxiter': 100, 'maxfev': 50}  # example options

        ##################################################################################
        # run QAOA optimization on graph
        ##################################################################################

        logging.info(f"Circuit depth hyperparameter:{depth}")
        logging.info(f"Problem size:{n_qubits}")

        # kick off training
        start = start_time_measurement()
        # result_energy, result_angle, tracker
        _, _, tracker = train(
            device=device_wrapper.get_device(), options=options, p=depth, ising=j, n_qubits=n_qubits,
            n_shots=config['shots'],
            opt_method=opt_method, tracker=tracker, s3_folder=device_wrapper.s3_destination_folder, verbose=True)
        time_to_solve = end_time_measurement(start)

        # print execution time
        # logging.info('Code execution time [sec]: ' + (end - start))

        # print optimized results
        logging.info(f"Optimal energy: {tracker['optimal_energy']}")
        logging.info(f"Optimal classical bitstring: {tracker['optimal_bitstring']}")

        # visualize the optimization process
        # cycles = np.arange(1, tracker['count'])
        # optim_classical = tracker['global_energies']

        # TODO maybe save this plot
        # plt.plot(cycles, optim_classical)
        # plt.xlabel('optimization cycle')
        # plt.ylabel('best classical minimum')
        # plt.show()

        return tracker['optimal_bitstring'], time_to_solve, {}


# QAOA utils (source:
# https://github.com/aws/amazon-braket-examples/blob/main/examples/hybrid_quantum_algorithms/QAOA/utils_qaoa.py)

# function to implement ZZ gate using CNOT gates
def ZZgate(q1, q2, gamma):
    """
    function that returns a circuit implementing exp(-i \\gamma Z_i Z_j) using CNOT gates if ZZ not supported
    """

    # get a circuit
    circ_zz = Circuit()

    # construct decomposition of ZZ
    circ_zz.cnot(q1, q2).rz(q2, gamma).cnot(q1, q2)

    return circ_zz


# function to implement evolution with driver Hamiltonian
def driver(beta, n_qubits):
    """
    Returns circuit for driver Hamiltonian U(Hb, beta)
    """
    # instantiate circuit object
    circ = Circuit()

    # apply parametrized rotation around x to every qubit
    for qubit in range(n_qubits):
        gate = Circuit().rx(qubit, 2 * beta)
        circ.add(gate)

    return circ


# helper function for evolution with cost Hamiltonian
def cost_circuit(gamma, n_qubits, ising, device):
    """
    returns circuit for evolution with cost Hamiltonian
    """
    # instantiate circuit object
    circ = Circuit()

    # get all non-zero entries (edges) from Ising matrix
    idx = ising.nonzero()
    edges = list(zip(idx[0], idx[1]))

    # apply ZZ gate for every edge (with corresponding interaction strength)
    for qubit_pair in edges:
        # get interaction strength from Ising matrix
        int_strength = ising[qubit_pair[0], qubit_pair[1]]
        # for Rigetti we decompose ZZ using CNOT gates
        if device.name in ["Rigetti", "Aspen-9"]:  # TODO make this more flexible
            gate = ZZgate(qubit_pair[0], qubit_pair[1], gamma * int_strength)
            circ.add(gate)
        # classical simulators and IonQ support ZZ gate
        else:
            gate = Circuit().zz(qubit_pair[0], qubit_pair[1], angle=2 * gamma * int_strength)
            circ.add(gate)

    return circ


# function to build the QAOA circuit with depth p
def circuit(params, device, n_qubits, ising):
    """
    function to return full QAOA circuit; depends on device as ZZ implementation depends on gate set of backend
    """

    # initialize qaoa circuit with first Hadamard layer: for minimization start in |->
    circ = Circuit()
    X_on_all = Circuit().x(range(0, n_qubits))
    circ.add(X_on_all)
    H_on_all = Circuit().h(range(0, n_qubits))
    circ.add(H_on_all)

    # setup two parameter families
    circuit_length = int(len(params) / 2)
    gammas = params[:circuit_length]
    betas = params[circuit_length:]

    # add QAOA circuit layer blocks
    for mm in range(circuit_length):
        circ.add(cost_circuit(gammas[mm], n_qubits, ising, device))
        circ.add(driver(betas[mm], n_qubits))

    return circ


# function that computes cost function for given params
def objective_function(params, device, ising, n_qubits, n_shots, tracker, s3_folder, verbose):
    """
    objective function takes a list of variational parameters as input,
    and returns the cost associated with those parameters
    """

    if verbose:
        logging.info("==================================" * 2)
        logging.info(f"Calling the quantum circuit. Cycle: {tracker['count']}")

    # get a quantum circuit instance from the parameters
    qaoa_circuit = circuit(params, device, n_qubits, ising)

    # classically simulate the circuit
    # execute the correct device.run call depending on whether the backend is local or cloud based
    if device.name in ["DefaultSimulator", "StateVectorSimulator"]:
        task = device.run(qaoa_circuit, shots=n_shots)
    else:
        task = device.run(
            qaoa_circuit, s3_folder, shots=n_shots, poll_timeout_seconds=3 * 24 * 60 * 60
        )

        # get ID and status of submitted task
        task_id = task.id
        status = task.state()
        logging.info(f"ID of task: {task_id}")
        logging.info(f"Status of task: {status}")
        # wait for job to complete
        while status != 'COMPLETED':
            status = task.state()
            logging.info(f"Status: {status}")
            sleep(10)

    # get result for this task
    result = task.result()
    logging.info(result)

    # get metadata
    # metadata = result.task_metadata

    # convert results (0 and 1) to ising (-1 and 1)
    meas_ising = result.measurements
    meas_ising[meas_ising == 0] = -1

    # get all energies (for every shot): (n_shots, 1) vector
    all_energies = np.diag(np.dot(meas_ising, np.dot(ising, np.transpose(meas_ising))))

    # find minimum and corresponding classical string
    energy_min = np.min(all_energies)
    tracker["opt_energies"].append(energy_min)
    optimal_string = meas_ising[np.argmin(all_energies)]
    tracker["opt_bitstrings"].append(optimal_string)
    logging.info(tracker["optimal_energy"])

    # store optimal (classical) result/bitstring
    if energy_min < tracker["optimal_energy"]:
        tracker.update({"optimal_energy": energy_min})
        tracker.update({"optimal_bitstring": optimal_string})

    # store global minimum
    tracker["global_energies"].append(tracker["optimal_energy"])

    # energy expectation value
    energy_expect = np.sum(all_energies) / n_shots

    if verbose:
        logging.info(f"Minimal energy: {energy_min}")
        logging.info(f"Optimal classical string: {optimal_string}")
        logging.info(f"Energy expectation value (cost): {energy_expect}")

    # update tracker
    tracker.update({"count": tracker["count"] + 1, "res": result})
    tracker["costs"].append(energy_expect)
    tracker["params"].append(params)

    return energy_expect


# The function to execute the training: run classical minimization.
# pylint: disable=R0913
def train(device, options, p, ising, n_qubits, n_shots, opt_method, tracker, s3_folder, verbose=True):
    """
    function to run QAOA algorithm for given, fixed circuit depth p
    """
    logging.info("Starting the training.")

    logging.info("==================================" * 2)
    logging.info(f"OPTIMIZATION for circuit depth p={p}")

    if not verbose:
        logging.info('Param "verbose" set to False. Will not print intermediate steps.')
        logging.info("==================================" * 2)

    # initialize
    cost_energy = []

    # randomly initialize variational parameters within appropriate bounds
    gamma_initial = np.random.uniform(0, 2 * np.pi, p).tolist()
    beta_initial = np.random.uniform(0, np.pi, p).tolist()
    params0 = np.array(gamma_initial + beta_initial)

    # set bounds for search space
    bnds_gamma = [(0, 2 * np.pi) for _ in range(int(len(params0) / 2))]
    bnds_beta = [(0, np.pi) for _ in range(int(len(params0) / 2))]
    bnds = bnds_gamma + bnds_beta

    tracker["params"].append(params0)

    # run classical optimization (example: method='Nelder-Mead')
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

    # store result of classical optimization
    result_energy = result.fun
    cost_energy.append(result_energy)
    logging.info(f"Final average energy (cost): {result_energy}")
    result_angle = result.x
    logging.info(f"Final angles: {result_angle}")
    logging.info("Training complete.")

    return result_energy, result_angle, tracker
