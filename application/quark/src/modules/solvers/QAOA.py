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
from matplotlib import pyplot as plt
import numpy as np
import math
import pdb
import pickle
from collections import Counter

from braket.circuits import Circuit
from scipy.optimize import minimize
from scipy.stats import gaussian_kde

from modules.solvers.Solver import *
from utils import start_time_measurement, end_time_measurement


class QAOA(Solver):
    """
    QAOA-algorithm from the Amazon-Braket-framework with some parts copied/derived from
    https://github.com/aws/amazon-braket-examples.
    """

    def __init__(self):
        """
        Constructor method
        """
        super().__init__()
        self.submodule_options = ["LocalSimulator", 
                                  "arn:aws:braket:::device/quantum-simulator/amazon/sv1",
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
                                  "shots_for_training": {  # number measurements to make on circuit during parameter training
                                      "values": [50, 100, 500, 1000],
                                      "description": "How many shots do you need for parameter training?"
                                  },
                                  "shots_for_sampling": {  # number measurements to make on circuit during sampling with optimal parameters
                                      "values": [1000, 10000, 100000],
                                      "description": "How many shots do you need for solution sampling?"
                                  },
                                  "opt_method": {
                                      "values": ["Powell", "Nelder-Mead"],
                                      "description": "Which optimization method do you want?"
                                  },
                                  "depth": {
                                      "values": [1], #TODO andere Tiefen zulassen
                                      "description": "Which circuit depth for QAOA do you want?"
                                  }
                              }

        """
        return {
            "shots_for_training": {  # number measurements to make on circuit during parameter training
                "values": [1000],#[50, 100, 500, 1000],
                "description": "How many shots do you need for parameter training?"
            },
            "shots_for_sampling": {  # number measurements to make on circuit during sampling with optimal parameters
                "values": [100000],#[1000, 10000, 100000],
                "description": "How many shots do you need for solution sampling?"
            },
            "opt_method": {
                "values": ["Powell"],#["Powell", "Nelder-Mead"],
                "description": "Which optimization method do you want?"
            },
            "depth": {
                "values": [1, 2], #TODO andere Tiefen zulassen
                "description": "Which circuit depth for QAOA do you want?"
            },
            "grid_search": {
                "values": [True],#[True, False],
                "description": "Do you want to do a grid search for depth 1?"
            }
        }

    class Config(TypedDict):
        """
        Attributes of a valid config

        .. code-block:: python

            shots_for_training: int
            shots_for_sampling: int
            opt_method: str (-> for params)
            depth: int (->circuit depth)
            grid_search: str 

        """
        shots_for_training: int
        shots_for_sampling: int
        opt_method: str
        depth: int
        grid_search: bool

    def run(self, mapped_problem: any, device_wrapper: any, config: Config, **kwargs: dict) -> (any, float, any):
        """
        Run QAOA algorithm on Ising.

        :param mapped_problem: dictionary with the keys 'J', 'h', and 'c'
        :type mapped_problem: any
        :param device_wrapper: instance of device
        :type device_wrapper: any
        :param config:
        :type config: Config
        :param kwargs: no additionally settings needed
        :type kwargs: any
        :return: Counts of solution sampling, the time it took to compute it and optional additional information
        :rtype: tuple(list, float, dict)
        """
        # Ising-problem objective can be gotten with obj = x J x^T + h x + c
        j = mapped_problem['J'] if 'J' in mapped_problem.keys() else [] # J-matrix of the Ising formulation
        h = mapped_problem['h'] if 'J' in mapped_problem.keys() else [] # h-vector of the Ising formulation
        c = mapped_problem['c'] if 'c' in mapped_problem.keys() else 0# c-offset of the Ising formulation
        
        # %% pre-QAOA
        n_qubits = j.shape[0]
        depth = config['depth']  # circuit depth for QAOA
        opt_method = config['opt_method']  # SLSQP, COBYLA, Nelder-Mead, BFGS, Powell, ...
        bitstring_init = -1 * np.ones([n_qubits]) # initialize reference solution (simple guess)
        energy_init = np.dot(bitstring_init, np.dot(j, bitstring_init)) # energy value with initial guess
        # set tracker to keep track of results
        tracker = {
                    'count': 1,  # Elapsed optimization steps
                    'optimal_energy': energy_init,  # Global optimal energy
                    'optimal_bitstring': bitstring_init,  # Global optimal bitstring
                    'optimal_params': None, # Global optimal parameters
                    'per_cycle_optimal_bitstring': [],  # Optimal bitstring at each step
                    'per_cycle_optimal_energy': [],  # Optimal energy at each step
                    'per_cycle_energy_expectation': [], # energy expectation at each step
                    'per_cycle_params': []  # Track parameters
                }

        # set options for classical optimization
        options = {'disp': True, 'maxiter': 100}
        # options = {'disp': True, 'ftol': 1e-08, 'maxiter': 100, 'maxfev': 50}  # example options
        
        # %% run QAOA to train the parameters beta and gamma
        logging.info(f"Circuit depth hyperparameter:{depth}")
        logging.info(f"Problem size:{n_qubits}")

        # kick off training
        start = start_time_measurement()
        # result_energy, result_angle, tracker
        _, _, tracker = train(
                                device = device_wrapper.get_device(), 
                                options = options, 
                                p = depth, 
                                ising = (j, h, c), 
                                n_qubits = n_qubits,
                                n_shots = config['shots_for_training'],
                                opt_method = opt_method, 
                                tracker = tracker, 
                                s3_folder = device_wrapper.s3_destination_folder
                                )
        
        time_to_solve = end_time_measurement(start)
        # visualize the parameter optimization process
        visualize_parameter_optimization_process(
                                count=tracker['count'],
                                energy_expectation=tracker['per_cycle_energy_expectation'], 
                                store_dir=kwargs['store_dir']
                                )
        
        # %% do a grid search for circuit depth p=1
        if config['grid_search'] == True:
            grid_search_beta, grid_search_gamma = grid_search_for_optimal_params_for_depth_1(
                                grid_size=100,
                                device = device_wrapper.get_device(), 
                                n_qubits = n_qubits,
                                ising = (j, h, c),
                                n_shots = config['shots_for_training'],
                                s3_folder = device_wrapper.s3_destination_folder,
                                store_dir=kwargs['store_dir']
                                )            
        
        # %% run QAOA circuit with optimal parameters to hopefully get feasible solutions into a best-bitstring-dictionary
        
        best_sample_qaoa, best_energy_qaoa, all_energies_qaoa = sample_qaoa_circuit(
                                params = tracker['optimal_params'],
                                device = device_wrapper.get_device(),
                                n_qubits = n_qubits,
                                n_shots = config['shots_for_sampling'],
                                ising = (j, h, c),
                                s3_folder = device_wrapper.s3_destination_folder
                                )

        counts_qaoa = {}
        for key, value in Counter(all_energies_qaoa).items():
            counts_qaoa[int(key)] = value
        
        # %% do grid search circuit sampling --> run the QAOA circuit with the best parameters from the grid search
        if config['grid_search'] == True:
            best_sample_grid_search, best_energy_grid_search, all_energies_grid_search = sample_qaoa_circuit(
                                    params = [grid_search_gamma, grid_search_beta], # no rotations in the quantum circuit -->random generator
                                    device = device_wrapper.get_device(),
                                    n_qubits = n_qubits,
                                    n_shots = config['shots_for_sampling'],
                                    ising = (j, h, c),
                                    s3_folder = device_wrapper.s3_destination_folder
                                    ) 
            counts_grid_search = {}
            for key, value in Counter(all_energies_grid_search).items():
                counts_grid_search[int(key)] = value
        else:
            best_sample_grid_search, best_energy_grid_search, all_energies_grid_search = None, None, []
        
        # %% do random sampling --> run the QAOA with parameters [0, 0] --> no rotations
        best_sample_random, best_energy_random, all_energies_random = sample_qaoa_circuit(
                                params = [0, 0], # no rotations in the quantum circuit -->random generator
                                device = device_wrapper.get_device(),
                                n_qubits = n_qubits,
                                n_shots = config['shots_for_sampling'],
                                ising = (j, h, c),
                                s3_folder = device_wrapper.s3_destination_folder
                                ) 
        counts_random = {}
        for key, value in Counter(all_energies_random).items():
            counts_random[int(key)] = value
        
        # %% result visualisation
        #compare the probability densitys of energies during QAOA sampling vs random sampling vs grid-search-sampling
        visualize_distribution_of_energies(all_energies_qaoa, kwargs['store_dir'], all_energies_random, all_energies_grid_search)
        #compare the distribution of the best energy values during sampling with a barplot 
        barplot_distribution_of_energies(counts_qaoa, kwargs['store_dir'], 2, counts_grid_search, counts_random)
        pdb.set_trace()
        # %%
        return best_sample_qaoa, time_to_solve, {}


# The function to execute the training: run classical minimization.
def train(device: any, options: dict, p: int, ising: any, n_qubits: int, n_shots: int, opt_method: str, tracker: dict, s3_folder: any) -> (float, float, dict):
    """
    function to run QAOA algorithm for given, fixed circuit depth p
    
    :param device: the device the QAOA should be run on
    :type device: any
    :param options: dictionary that specifies certain settings for the optimization of the parameters
    :type options: dict
    :param p: depth of the QAOA circuit --> number of alternating QAOA-layers
    :type p: int
    :param ising: the ising formulation of the problem
    :type : any
    :param n_qubits: number of qubits the full QAOA circuit acts on
    :type : int
    :param n_shots: number of times the circuit should be run on the device
    :type n_shots: int
    :param opt_method: the classical method to optimize the parameters beta and gamma
    :type opt_method:
    :param tracker: dictionary that keeps track of the parameter optimization process
    :type tracker: dict
    :param s3_folder: ???
    :type s3_folder: ???p
    :return: the expected energy with the current parameters
    :rtype: float
    """
    logging.info("Starting the training.")

    logging.info("==================================" * 2)
    logging.info(f"OPTIMIZATION for circuit depth p={p}")
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
    
    # initialize the params-tracker
    tracker["params_start"] = params0
    
    # run classical optimization (example: method='Nelder-Mead')
    result = minimize(
                    objective_function,
                    params0,
                    args=(device, ising, n_qubits, n_shots, tracker, s3_folder),
                    options=options,
                    method=opt_method,
                    bounds=bnds,
                    )

    # store result of classical optimization
    result_energy = result.fun
    result_params = result.x
    tracker["optimal_params"] = result_params
    logging.info("==================================" * 2)
    logging.info(f"Final average energy (cost): {result_energy}")
    logging.info(f"optimal parameters: {result_params}")
    logging.info("Training complete.")

    return result_energy, result_params, tracker


def objective_function(params: list, device: any, ising: tuple, n_qubits: int, n_shots: int, tracker: dict, s3_folder: any) -> float:
    """
    objective function takes a list of variational parameters as input,
    and returns the energy expectation value of the circuit that result of those
    input parameters.
    
    :param params: list of parameters that contain the beta and gamma angles -> are classically optimized during QAOA
    :type : list
    :param device: the device the QAOA should be run on
    :type device: any
    :param ising: the ising formulation of the problem
    :type : tuple(matrix, vector, constant)
    :param n_qubits: number of qubits the full QAOA circuit acts on
    :type : int
    :param n_shots: number of times the circuit should be run on the device
    :type n_shots: int
    :param tracker: dictionary that keeps track of the parameter optimization process
    :type tracker: dict
    :param s3_folder: ???
    :type s3_folder: ???
    :return: the expected energy with the current parameters
    :rtype: float
    """
    logging.info("==================================" * 2)
    logging.info(f"Calling the quantum circuit. Cycle: {tracker['count']}")

    # get a quantum circuit instance from the parameters
    qaoa_circuit = circuit(params, device, n_qubits, ising)

    # classically simulate the circuit
    result, task_id, status = run_circuit(
                            device = device,
                            qaoa_circuit = qaoa_circuit, 
                            s3_folder = s3_folder, 
                            n_shots = n_shots
                            )

    meas_ising = result.measurements
    # convert results (0 and 1) to ising (1 and -1)
    meas_ising[meas_ising == 1] = -1 
    meas_ising[meas_ising == 0] = 1
    all_energies = calc_ising_energy(ising[0], ising[1], ising[2], meas_ising)

    # find minimum energy and corresponding classical string
    energy_min = np.min(all_energies)
    optimal_string = meas_ising[np.argmin(all_energies)]
    energy_expect = np.sum(all_energies) / n_shots
    
    #update the tracker
    tracker["per_cycle_optimal_energy"].append(energy_min)    
    tracker["per_cycle_optimal_bitstring"].append(optimal_string)
    tracker["per_cycle_energy_expectation"].append(energy_expect)
    if energy_min < tracker["optimal_energy"]: # possibly update the optimal energy and bitstring
        tracker.update({"optimal_energy": energy_min})
        tracker.update({"optimal_bitstring": optimal_string})
    tracker.update({"count": tracker["count"] + 1, "res": result})
    tracker["per_cycle_params"].append(params)
    
    #log the calculation state
    logging.info(f"Minimal energy during cycle: {energy_min}")
    logging.info(f"Optimal classical string during cycle: {optimal_string}")
    logging.info(f"Energy expectation value during cycle: {energy_expect}")

    return energy_expect


def run_circuit(device: any, qaoa_circuit: Circuit, s3_folder: any, n_shots: int) -> (any, int, str):
    """
    function to run a QAOA circuit on a device with n shots
    
    :param device: the device the QAOA should be run on
    :type device: any
    :param qaoa_circuit: the QAOA circuit that should be run on the device
    :type qaoa_circuit: Circuit
    :param s3_folder: ???
    :type s3_folder: ???
    :param n_shots: number of times the circuit should be run on the device
    :type n_shots: int
    :return: the task-result, the task-id and the task-status
    :rtype: (any, int, str)
    """
    # execute the correct device.run call depending on whether the backend is local or cloud based
    if device.name in ["DefaultSimulator", "StateVectorSimulator"]:
        task = device.run(qaoa_circuit, shots=n_shots)
        task_id = 0
        status = 'COMPLETED'
    else:
        task = device.run(qaoa_circuit, s3_folder, shots=n_shots, poll_timeout_seconds=3 * 24 * 60 * 60)
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
    
    return result, task_id, status


def sample_qaoa_circuit(params: list, device: any, n_qubits: int, n_shots: int, ising: tuple, s3_folder: any) -> dict:
    """
    function that samples a qaoa circuit with the goal to get good solutions for a given ising
    
    :param params: list of parameters that contain the beta and gamma angles -> are classically optimized during QAOA
    :type : list
    :param device: the device the QAOA should be run on
    :type device: any
    :param n_qubits: number of qubits the full QAOA circuit acts on
    :type : int
    :param n_shots: number of times the circuit should be run on the device
    :type n_shots: int
    :param ising: the ising formulation of the problem
    :type : tuple(matrix, vector, constant)
    :param s3_folder: ???
    :type s3_folder: ???
    
    :return: a dictionary with the best bitstrings and their corresponding counts and energy values
    :rtype: dict
    """
    # create the circuit with the optimal beta and gamma
    qaoa_circuit = circuit(params, device, n_qubits, ising)
    
    # run the optimal circuit
    result, task_id, status = run_circuit(device, qaoa_circuit, s3_folder, n_shots)    
    
    # analyse the results of the circuit sampling
    meas_ising = result.measurements
    # convert results (0 and 1) to ising (1 and -1)
    meas_ising[meas_ising == 1] = -1 
    meas_ising[meas_ising == 0] = 1
    
    best_energy = 100000000000
    best_sample = ''
    all_energies = []
    
    for row in meas_ising: # iterate through the array to filter out the rows with high counts
        # calculate all energies: (n, 1) vector: E = measurements * ising * measurements T
        energy = calc_ising_energy(ising[0], ising[1], ising[2], row)
        all_energies.append(energy)
        if energy < best_energy: # then update energy
            row[row == 1] = 0
            row[row == -1] = 1
            best_sample = np.array2string(row, separator='')[1:-1]
            best_energy = energy
    
    return best_sample, best_energy, all_energies


def circuit(params: list, device: any, n_qubits: int, ising: tuple) -> Circuit:
    """
    function to return full QAOA circuit
    
    :param params: list of parameters that contain the beta and gamma angles -> are classically optimized during QAOA
    :type : list[gammas, betas]
    :param device: the device the QAOA should be run on
    :type : any
    :param n_qubits: number of qubits the full QAOA circuit acts on
    :type : int
    :param ising: the ising formulation of the problem
    :type : tuple(matrix, vector, constant)
    :return: a circuit that implemented the whole QAOA circuits on the n_qubits
    :rtype: Circuit from braket.Circuits 
    """

    circ = Circuit()
    # add Hadamard to all qubits to initialize the |+> state
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


def cost_circuit(gamma: float, n_qubits: int, ising: tuple, device: any) -> Circuit:
    """
    returns circuit for evolution with cost Hamiltonian
    :param gamma: plays a part in the rotation angle of the zz/z-gates
    :type : float
    :param n_qubits: number of qubits the cost Hamiltonian acts on
    :type : int
    :param ising: the ising formulation of the problem
    :type : tuple(matrix, vector, constant)
    :param device: the device the QAOA should be run on
    :type : any
    :return: a circuit that implemented the cost Hamiltonian on the n_qubits
    :rtype: Circuit from braket.Circuits 
    """
    # instantiate circuit object
    circ = Circuit()

    # get all non-zero entries (edges) from Ising matrix
    idx = ising[0].nonzero()
    edges = list(zip(idx[0], idx[1]))

    # apply gate for every edge (with corresponding interaction strength)
    for (qubit1, qubit2) in edges:
        # get interaction strength from Ising matrix
        interaction_strength = ising[0][qubit1, qubit2]
        # for Rigetti we decompose ZZ using CNOT gates
        if device.name in ["Rigetti", "Aspen-9"]:  # TODO make this more flexible
            gate = ZZgate(qubit1, qubit2, gamma * interaction_strength)
            circ.add(gate)
        # classical simulators and IonQ support ZZ gate
        else:
            gate = Circuit().zz(qubit1, qubit2, angle = 2 * gamma * interaction_strength)
            circ.add(gate)
                        
    # get all non-zero entries (edges) from Ising vector
    idx = ising[1].nonzero()
    edges = list(zip(idx[0]))

    # apply gate for every edge (with corresponding interaction strength)
    for (qubit1) in edges:
        # get interaction strength from Ising vector
        interaction_strength = ising[1][qubit1]
        gate = Circuit().rz(qubit1, angle = 2 * gamma * interaction_strength)
        circ.add(gate)

    return circ


def driver(beta: float, n_qubits: int) -> Circuit:
    """
    Returns circuit for driver Hamiltonian U(Hb, beta) --> rx-gates on the qubits
    :param beta: rotation angle of the x-rotation
    :type beta: float
    :param n_qubits: the number of qubits the driver circuit acts on
    :type n_qubits: int
    :return: a circuit that implemented a driver circuit on the n_qubits
    :rtype: Circuit from braket.Circuits        
    """
    # instantiate circuit object
    circ = Circuit()

    # apply parametrized rotation around x to every qubit
    for qubit in range(n_qubits):
        gate = Circuit().rx(qubit, 2 * beta)
        circ.add(gate)

    return circ


def ZZgate(q1: int, q2: int, gamma: float) -> Circuit:
    """
    function that returns a circuit implementing exp(-i \\gamma Z_i Z_j) using 
    CNOT gates if ZZ gates are not supported by a device
    
    :param q1: 1st qubit the zz-gate acts on
    :type q1: int
    :param q2: 2nd qubit the zz-gate acts on
    :type q2: int
    :param gamma: angle of the zz-gate
    :type gamma: float
    :return: a circuit that implemented a zz gate via cnot gates on q1 and q2
    :rtype: Circuit from braket.Circuits
    """
    # get a circuit
    circ_zz = Circuit()

    # construct decomposition of ZZ
    circ_zz.cnot(q1, q2).rz(q2, gamma).cnot(q1, q2)

    return circ_zz


def calc_ising_energy(ising_matrix: np.array, ising_vector: np.array, ising_offset: float, bitstring_result: np.array) -> any:
    """
    function that calculated the ising energy of bitstring measurement/s
    
    :param ising_matrix: the quadratic part of the ising formulation
    :type ising_matrix: numpy array
    :param ising_vector: the linear part of the ising formulation
    :type ising_vector: numpy array
    :param ising_offset: the constant part of the ising formulation
    :type ising_offset: float
    :param bitstring_result: the measurement result/s
    :type bitstring_result: numpy-array
    :return: the ising energy/ies of the measurement result/s
    :rtype: float if one measurement and numpy array for multiple measurements
    """
    energy_from_vector = np.dot(bitstring_result, ising_vector)
    if len(bitstring_result.shape) == 1: #if it is only one measurement -->if dim(bitstring_result) = n x 1
        energy_from_matrix = np.dot(bitstring_result, np.dot(ising_matrix, np.transpose(bitstring_result)))
        energy_from_offset = ising_offset
    else: # if it is multiple measurements -->if dim(bitstring_result) = n x m
        energy_from_matrix = np.diag(np.dot(bitstring_result, np.dot(ising_matrix, np.transpose(bitstring_result))))
        energy_from_offset = ising_offset * np.ones(bitstring_result.shape[0])
        
    return energy_from_matrix + energy_from_vector + energy_from_offset


def grid_search_for_optimal_params_for_depth_1(grid_size: int, device: any, n_qubits: int, ising: any, n_shots: int, s3_folder: any, store_dir: str) -> (float, float):
    '''
    function that tries to find optimal qaoa parameters for circuit depth p=1 
    via a grid search
    
    :param grid_size: the size of the grid for the grid search
    :type grid_size: int
    :param device: the device the QAOA circuit should be run on
    :type device: any
    :param n_qubits: number of qubits of the QAOA circuit
    :type n_qubits: any
    :param ising: tuple consisting of ising matrix, vector and offset
    :type ising: (numpy-array, numpy-array, float)
    :param n_shots: number of times the circuit should be run on the device for each grid element
    :type n_shots: int
    :param s3_folder: ???
    :type s3_folder: ???
    :param store_dir: the directory the graphic should be saved in
    :type store_dir: str
    :return: the proposal of the grid search for beta and gamma for the QAOA with p=1
    :rtype: tuple(float, float)

    '''
    #initialize a numpy-array from which the grid graphic will be filled
    grid_results = np.zeros((grid_size, grid_size)) 
    range_beta = math.pi / 2
    range_gamma = 2 * math.pi
    
    calc_state_old = 0
    for beta_index in range(grid_size):
        beta = beta_index * ( range_beta / grid_size )
        
        for gamma_index in range(grid_size):
            gamma = gamma_index * ( range_gamma / grid_size )
            
            energy_expectation = get_energy_expectation_of_qaoa_circuit_depth_1(ising, beta, gamma)
            grid_results[beta_index, gamma_index] = energy_expectation
                        
            #log the state of the grid search
            calc_state_new = round(100 * (beta_index * grid_size + gamma_index + 1) / (grid_size**2), 0)
            if calc_state_new != calc_state_old:
                calc_state_old = calc_state_new
                logging.info("State of the grid search for beta and gamma: " + str(calc_state_old) + "%")
                            
    #create a heatmap graphic and save this graphic to the export-path
    plt.figure(figsize=(20,5))
    plt.imshow(grid_results, cmap='viridis', extent=[0, range_gamma, 0, range_beta])
    plt.colorbar(label = 'Expectation value of objective')
    plt.ylabel('Beta')
    plt.xlabel('Gamma')
    plt.title('Result of Gridsearch')
    plt.savefig(store_dir + "\gridsearch_for_circuit_depth_one.png")
    plt.clf()
    
    # find the index with minimum value in grid_results
    pickle.dump(grid_results, open(store_dir + "\grid_results.pkl", 'wb'))
    (beta_optimal_index, gamma_optimal_index) = np.unravel_index(np.argmin(grid_results), grid_results.shape)
    grid_search_beta = beta_optimal_index * ( range_beta / grid_size )
    grid_search_gamma = gamma_optimal_index * ( range_gamma / grid_size )
    
    return grid_search_beta, grid_search_gamma


def get_energy_expectation_of_qaoa_circuit_depth_1(ising: tuple, beta: float, gamma: float) -> float:
    '''
    function that calculates the energy expectation value of a qaoa circuit
    with depth 1 only by its ising and beta/gamma-parameters.
    It was derived from formula 13 from the following paper:
    https://iopscience.iop.org/article/10.1088/2058-9565/ac9013/pdf
    
    :param ising: tuple consisting of ising matrix, vector and offset
    :type ising: tuple(np.array, np.array, float)
    :param beta: beta parameter for the mixing circuit in the QAOA-circuit
    :type beta: float
    :param gamma: gamma parameter for the cost circuit in the QAOA-circuit
    :type gamma: float
    
    :return: the energy expectation of the qaoa-circuit
    :rtype: float
    ''' 
    #initialize the expectation value, things will be added
    expectation = 0 
    
    ising_mat = ising[0] #ising-matrix
    ising_vec = ising[1] #ising-vector
    ising_offset = ising[2] #ising-offset
    
    n_qubits = ising_mat.shape[0]
    assert (ising_mat == np.triu(ising_mat, k=1)).all() #check if the ising matrix is an upper diagonal matrix
    ising_mat = np.transpose(ising_mat) + ising_mat # the formula was defined for edges ij on a graph, so we put all the values above the diagonal also under the diagonal
    
    #iterate through ising-matrix
    for coords_mat, coeff_mat in np.ndenumerate(ising_mat):
        if coeff_mat != 0 and coords_mat[0]<coords_mat[1]:
            term_to_add_part1 = (coeff_mat * math.sin(4*beta)) / 2 * math.sin(2*gamma*coeff_mat)
            
            term_to_add_part2 = math.cos(2*gamma*ising_vec[coords_mat[0]])
            for idx in range(n_qubits):
                if idx not in coords_mat: # k != i,j
                    term_to_add_part2 *= math.cos(2*gamma*ising_mat[coords_mat[0], idx])
            
            term_to_add_part3 = math.cos(2*gamma*ising_vec[coords_mat[1]])
            for idx in range(n_qubits):
                if idx not in coords_mat: # k != i,j
                    term_to_add_part3 *= math.cos(2*gamma*ising_mat[coords_mat[1], idx])
            
            term_to_add_part4 = coeff_mat / 2 * (math.sin(2*beta))**2
            
            term_to_add_part5 = math.cos(2*gamma*(ising_vec[coords_mat[0]] + ising_vec[coords_mat[1]]))
            
            term_to_add_part6 = 1 #initial product value
            for idx in range(n_qubits):
                if idx not in coords_mat:  # k != i,j
                    term_to_add_part6 *= math.cos(2*gamma*(ising_mat[coords_mat[0], idx] + ising_mat[coords_mat[1], idx]))
            
            term_to_add_part7 = math.cos(2*gamma*(ising_vec[coords_mat[0]] - ising_vec[coords_mat[1]]))
            
            term_to_add_part8 = 1 #initial product value
            for idx in range(n_qubits):
                if idx not in coords_mat:  # k != i,j
                    term_to_add_part8 *= math.cos(2*gamma*(ising_mat[coords_mat[0], idx] - ising_mat[coords_mat[1], idx]))
                            
            expectation += term_to_add_part1 * (term_to_add_part2 + term_to_add_part3) - \
                           term_to_add_part4 * (term_to_add_part5 * term_to_add_part6 - term_to_add_part7 * term_to_add_part8)
        else:
            continue
        
    #iterate through ising-vector
    for coords_vec, coeff_vec in np.ndenumerate(ising_vec):
        if coeff_vec != 0:
            
            term_to_add_part9 = coeff_vec * math.sin(2*beta) * math.sin(2*gamma*coeff_vec)
            
            term_to_add_part10 = 1 #initial product value
            for idx in range(n_qubits):
                if idx not in coords_vec: # k != i
                    term_to_add_part10 *= math.cos(2*gamma*ising_mat[coords_vec[0], idx])                    
                    
            expectation += term_to_add_part9 * term_to_add_part10
        else:
            continue
    
    #add the ising-constant
    expectation += ising_offset
    
    return expectation


def visualize_parameter_optimization_process(count: int, energy_expectation: list, store_dir: str):
    '''
    Function to draw the parameter optimization process during the QAOA.
    Plots the expected energies per optimization cycle
    
    :param count: number of cycles during parameter optimization
    :type count: list
    :param energy_expectation: list of energy-values 
    :type energy_expectation: list 
    :param store_dir: the directory the graphic should be saved in
    :type store_dir: str
    :return : No return, the graphic is just saved during this function
    '''
    cycles = np.arange(1, count)
    # generate values for the plot
    plt.plot(cycles, energy_expectation)
    # create and save the plot
    plt.xlabel('optimization cycle')
    plt.ylabel('energy expectation value for each optimization cycle')
    plt.savefig(store_dir + "/qaoa_process_classical_optimization.jpg")
    plt.clf()
    
    return


def visualize_distribution_of_energies(energies_1: list, store_dir: str, energies_2=[], energies_3=[]):
    '''
    Function to draw a probability density of 2 energy lists and save the resulting graphic.
    This has the goal to get a better understanding of the energy distribution.
    It is possibly helpful to benchmark QAOA results vs random sampling
    
    :param energies_1: list 1 of energy values
    :type energies_1: list
    :param store_dir: the directory the graphic should be saved in
    :type store_dir: str
    :param energies_2: list 2 of energy-values  
    :type energies_2: list
    :param energies_3: list 2 of energy-values 
    :type energies_3: list
    :return : No return, the graphic is just saved during this function
    '''
    pdf_1 = gaussian_kde(energies_1) # pdf = probability density function
    pdf_2 = gaussian_kde(energies_2) # pdf = probability density function
    pdf_3 = gaussian_kde(energies_3) # pdf = probability density function
    
    # generate values for the plot1
    x_whole_support = np.linspace(min(min(energies_1), min(energies_2)), max(max(energies_1), max(energies_2)), num=1000)
    y_1_whole_support = pdf_1(x_whole_support)
    y_2_whole_support = pdf_2(x_whole_support)
    y_3_whole_support = pdf_3(x_whole_support)
    
    # create and save plot 1 - covers the whole support set
    plt.plot(x_whole_support, y_1_whole_support, label='energies from QAOA sampling')
    if energies_2 != []:
        plt.plot(x_whole_support, y_2_whole_support, label='energies from random sampling')
    if energies_3 != []:
        plt.plot(x_whole_support, y_3_whole_support, label='energies from circuit with grid search parameters')
    plt.xlabel('energy')
    plt.ylabel('probability')
    plt.title('probability density of energy of QAOA- vs. random- vs. grid-search-circuit-sampling')
    plt.legend()
    plt.savefig(store_dir + "/qaoa_vs_random_vs_gridsearch_sampling.jpg")
    plt.clf()
    
    # generate values for the plot1
    x_zoomed_in = np.linspace(min(min(energies_1), min(energies_2)), 10 * min(min(energies_1), min(energies_2)), num=1000)
    y_1_zoomed_in = pdf_1(x_zoomed_in)
    y_2_zoomed_in = pdf_2(x_zoomed_in)
    y_3_zoomed_in = pdf_3(x_zoomed_in)
    
    # create and save plot 2 - zooms in on the optimal values
    plt.plot(x_zoomed_in, y_1_zoomed_in, label='energies from QAOA sampling')
    if energies_2 != []:
        plt.plot(x_zoomed_in, y_2_zoomed_in, label='energies from random sampling')
    if energies_3 != []:
        plt.plot(x_zoomed_in, y_3_zoomed_in, label='energies from circuit with grid search parameters')
    plt.xlabel('energy')
    plt.ylabel('probability')
    plt.title('probability density of energy of QAOA- vs. random-sampling')
    plt.legend()
    plt.savefig(store_dir + "/qaoa_vs_random_vs_gridsearch_sampling_ZoomedIn.jpg")
    plt.clf()
    
    return


def barplot_distribution_of_energies(counts_1: dict, store_dir: str, num_bars=5, counts_2={}, counts_3={}):
    '''
    Function to draw bar plots of counts of energy values to get a 
    better understanding of the distribution of energy counts
    
    :param counts_1: dict 1 of counts values
    :type counts_1: dict
    :param store_dir: the directory the graphic should be saved in
    :type store_dir: str
    :param num_bars: number of bars that should be plotted for each counts-dict
    :type num_bars: int
    :param counts_2: dict 2 of counts values
    :type counts_2: dict
    :param counts_3: dict 3 of counts values
    :type counts_3: dict
    :return : No return, the graphic is just saved during this function
    '''
    #get the 10 best energy sample values 
    best_energy_values_dict1 = list(counts_1.keys())
    best_energy_values_dict1.sort()
    best_energy_values_dict1 = best_energy_values_dict1[:num_bars]
    
    best_energy_values_dict2 = list(counts_2.keys())
    best_energy_values_dict2.sort()
    best_energy_values_dict2 = best_energy_values_dict2[:num_bars]
    
    best_energy_values_dict3 = list(counts_3.keys())
    best_energy_values_dict3.sort()
    best_energy_values_dict3 = best_energy_values_dict3[:num_bars]
    
    #unionize the keys from the lists
    common_best_energy_values = set(best_energy_values_dict1) | set(best_energy_values_dict2) | set(best_energy_values_dict3)
    x_values = list(common_best_energy_values)
    
    energy_counts_dict1 = [counts_1.get(energy, 0) for energy in x_values]
    energy_counts_dict2 = [counts_2.get(energy, 0) for energy in x_values]
    energy_counts_dict3 = [counts_3.get(energy, 0) for energy in x_values]
    
    bar_width = 0.1
    
    #plot bars for the energy_counts 
    plt.bar(range(len(x_values)), energy_counts_dict1, bar_width, label='energy counts qaoa', color='blue')
    plt.bar([x + bar_width for x in range(len(x_values))], energy_counts_dict2, bar_width, label='energy counts grid search', color='green')
    plt.bar([x + 2*bar_width for x in range(len(x_values))], energy_counts_dict3, bar_width, label='energy counts random', color='orange')
    
    #legend of the plot
    plt.xlabel('energy')
    plt.ylabel('counts')
    plt.title('barplot distribution of best energies')
    plt.xticks([x + bar_width/2 for x in range(len(x_values))], x_values)
    plt.legend()
    plt.savefig(store_dir + "/barplot_distribution_best_energies_qaoa_vs_random_vs_gridsearch_sampling.jpg")
    plt.clf()
    
    return
   
