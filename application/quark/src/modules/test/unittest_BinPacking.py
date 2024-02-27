#  Copyright 2021 The QUARK Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import unittest
import os
import numpy as np
import pdb
import sys

from qiskit import QuantumCircuit, Aer


#put the \src path to the system path variables
install_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_dir)


from src import utils
from src.modules.applications.optimization.BinPacking import BinPacking as BPack
from src.modules.solvers.MIPSolver import MIPSolver
from src.modules.solvers.QAOA import get_energy_expectation_of_qaoa_circuit_depth_1, run_circuit, circuit, calc_ising_energy


class Unittest_BinPacking(unittest.TestCase):
    """
    This unittesting class contains methods to test the BinPacking-Application
    """

    def __init__(self):
        """
        Constructor method executes all tests
        """
        # initialize some basic variables of the unittest.TestCase-class
        super().__init__() 
        
        # initialize the path for test-results
        self.export_path_for_tests = os.path.dirname(os.path.abspath(__file__)) + "\\export\\"
        
        # create a problem on which the tests should be done on
        self.problem_instance = self.examplary_problem_instance()
        
        # verify the MIP-solver result
        self.test_mip_small_problem_instance()
        
        # verify the QUBO to Ising mapping
        self.test_qubo_to_ising()
        
        # verify the implemented function to calculate the expected energy of an ising model
        self.test_expected_energy_of_ising()
        return

    class examplary_problem_instance():
        def __init__(self):
            self.object_weights = [1, 2, 3]
            self.bin_capacity = 3
            self.incompatible_objects = [(0, 2)]
            self.problem = (self.object_weights, self.bin_capacity, self.incompatible_objects)
            self.penalty_factor_for_qubo = 5
            self.optimal_solutions = [
                                 {'x_0': 1, 'x_1': 1, 'x_2': 0, 'y_0_0': 1, 'y_0_1': 0, 'y_0_2': 0, 'y_1_0': 1, 'y_1_1': 0, 'y_1_2': 0, 'y_2_0': 0, 'y_2_1': 1, 'y_2_2': 0},
                                 {'x_0': 1, 'x_1': 1, 'x_2': 0, 'y_0_0': 0, 'y_0_1': 1, 'y_0_2': 0, 'y_1_0': 0, 'y_1_1': 1, 'y_1_2': 0, 'y_2_0': 1, 'y_2_1': 0, 'y_2_2': 0},
                                 {'x_0': 1, 'x_1': 0, 'x_2': 1, 'y_0_0': 1, 'y_0_1': 0, 'y_0_2': 0, 'y_1_0': 1, 'y_1_1': 0, 'y_1_2': 0, 'y_2_0': 0, 'y_2_1': 0, 'y_2_2': 1},
                                 {'x_0': 1, 'x_1': 0, 'x_2': 1, 'y_0_0': 0, 'y_0_1': 0, 'y_0_2': 1, 'y_1_0': 0, 'y_1_1': 0, 'y_1_2': 1, 'y_2_0': 1, 'y_2_1': 0, 'y_2_2': 0},
                                 {'x_0': 0, 'x_1': 1, 'x_2': 1, 'y_0_0': 0, 'y_0_1': 1, 'y_0_2': 0, 'y_1_0': 0, 'y_1_1': 1, 'y_1_2': 0, 'y_2_0': 0, 'y_2_1': 0, 'y_2_2': 1},
                                 {'x_0': 0, 'x_1': 1, 'x_2': 1, 'y_0_0': 0, 'y_0_1': 0, 'y_0_2': 1, 'y_1_0': 0, 'y_1_1': 0, 'y_1_2': 1, 'y_2_0': 0, 'y_2_1': 1, 'y_2_2': 0}
                                 ]
            self.bitstrings_to_evaluate = [     # ( 0_1_bitstring,  -1_1_bitstring )
                                                (
                                                [int(bit) for bit in "000000000000000000"][::-1], # [::-1] inverts the string
                                                [-1 if bit == '1' else 1 for bit in "000000000000000000"][::-1] # [::-1] inverts the string
                                                ),
                                                (
                                                [int(bit) for bit in "000000001100000101"][::-1], # [::-1] inverts the string
                                                [-1 if bit == '1' else 1 for bit in "000000001100000101"][::-1] # [::-1] inverts the string
                                                ),
                                                (
                                                [int(bit) for bit in "011001001010000000"],
                                                [-1 if bit == '1' else 1 for bit in "011001001010000000"]
                                                )
                                            ]                       
            return
    
    def test_mip_small_problem_instance(self):
        
        self.bin_packing_mip = BPack.create_MIP(self.problem_instance.problem)
        solver = MIPSolver()
        solution, _, _ = solver.run(self.bin_packing_mip, [], [], store_dir=self.export_path_for_tests)
        
        self.assertIn(solution, self.problem_instance.optimal_solutions)
            
    
    def test_qubo_to_ising(self):
        self.ising_matrix, self.ising_vector, self.ising_offset, self.qubo = \
                         BPack.transform_docplex_mip_to_ising(self.bin_packing_mip, penalty_factor=self.problem_instance.penalty_factor_for_qubo)
        
        for (qubo_bitstring, ising_bitstring) in self.problem_instance.bitstrings_to_evaluate:
            
            # calc qubo objective value via qiskit-built-in-function
            qubo_obj_value = self.qubo.objective.evaluate(qubo_bitstring)
            
            # calc ising objective value: x^T H x + h x + c
            ising_obj_value = np.dot(np.transpose(ising_bitstring), np.dot(self.ising_matrix, ising_bitstring)) + \
                              np.dot(self.ising_vector, ising_bitstring) + \
                              self.ising_offset
            
            ising_obj_value_by_function = calc_ising_energy(self.ising_matrix, self.ising_vector, self.ising_offset, np.array(ising_bitstring))
            
            self.assertEqual(qubo_obj_value, ising_obj_value)
            self.assertEqual(ising_obj_value, ising_obj_value_by_function)
            

    def test_expected_energy_of_ising(self):
        beta_for_test = 1#1# 13/30 * (math.pi / 2)
        gamma_for_test = 1# 7/30 * (2 * math.pi)
        isings = []
        isings.append((self.ising_matrix, self.ising_vector, self.ising_offset))
        isings.append((np.array([[0, 0], [0, 0]]), np.array([2, 3]), 4))
        isings.append((np.array([[0,1,0,0],[0,0,0,0],[0,0,0,1],[0,0,0,0]]), np.array([0,0,0,0]), 4))
        isings.append((np.array([[0,1,0,0],[0,0,0,0],[0,0,0,1],[0,0,0,0]]), np.array([0,1,2,3]), 4))
        isings.append((np.array([[0,0,1,0],[0,0,0,0],[0,0,0,1],[0,0,0,0]]), np.array([0,0,0,0]), 4))
        isings.append((np.array([[0,1,2,3],[0,0,4,5],[0,0,0,6],[0,0,0,0]]), np.array([0,0,0,0]), 4))
        
        #iterate through all the test-cases
        for ising in isings:
            # %% calculate the expected ising energy with the implemented function
            expectation_by_function = get_energy_expectation_of_qaoa_circuit_depth_1(ising, beta_for_test, gamma_for_test)
            
            # %% simulate the corresponding QAOA circuit
            module = "src.modules.devices.braket.LocalSimulator"
            class_name = "LocalSimulator"
            device_name = "LocalSimulator"
            clazz = utils._import_class(module, class_name, None)
            device_wrapper = clazz(device_name)
            
            qaoa_circuit = circuit( 
                                   params=[gamma_for_test, beta_for_test],
                                   device=device_wrapper.get_device(),
                                   n_qubits=len(ising[1]),
                                   ising=ising
                                   )
            simulation_result, _, _ = run_circuit(
                                        device=device_wrapper.get_device(),
                                        qaoa_circuit=qaoa_circuit,
                                        s3_folder=device_wrapper.s3_destination_folder,
                                        n_shots = 100000
                                        )
            meas_ising = simulation_result.measurements
            meas_ising[meas_ising == 1] = -1
            meas_ising[meas_ising == 0] = 1
            all_energies = []
            for row in meas_ising: # iterate through the array to filter out the rows with high counts
                # calculate all energies: (n, 1) vector: E = measurements * ising * measurements T
                energy = calc_ising_energy(ising[0], ising[1], ising[2], row)
                all_energies.append(energy)
            expectation_by_simulation = sum(all_energies) / len(all_energies)
            
            # %% simulate the corresponding Qiskit circuit
            circ = circuit_qiskit(ising, beta_for_test, gamma_for_test)
            n_shots = 100000
            counts = run_circuit_qiskit(circuit=circ, shots=n_shots)
            expectation_by_simulation_qiskit = 0
            for bitstring, count in counts.items():
                bitstring_list = [-1 if bit == '1' else 1 for bit in bitstring][::-1] 
                energy = np.dot(np.transpose(bitstring_list), np.dot(ising[0], bitstring_list)) + \
                                  np.dot(ising[1], bitstring_list) + \
                                  ising[2]
                expectation_by_simulation_qiskit += energy * (count/n_shots)
                
            # %% check the validity
            allowed_diff = expectation_by_simulation / 100        
            self.assertAlmostEqual(
                            first=expectation_by_function,
                            second=expectation_by_simulation,
                            delta=allowed_diff
                            )
            self.assertAlmostEqual(
                            first=expectation_by_simulation, 
                            second=expectation_by_simulation_qiskit,
                            delta=allowed_diff
                            )
        # %%


def circuit_qiskit(ising, beta, gamma):   
    
    ising_mat = ising[0] #ising-matrix
    ising_vec = ising[1] #ising-vector
    
    num_qubits = ising_mat.shape[0]
    
    # the circuit that will be the final QAOA ciruit
    qc_qaoa = QuantumCircuit(num_qubits) 
    
    # create the initial superposition state with Hadamards
    quantumcircuit_initial = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        quantumcircuit_initial.h(i)
    qc_qaoa.append(quantumcircuit_initial, range(num_qubits))
    
    # create the problem Unitary:
    quantumcircuit_problem = QuantumCircuit(num_qubits)    
    for coords_mat, coeff_mat in np.ndenumerate(ising_mat):
        if coeff_mat != 0:
            quantumcircuit_problem.rzz(2 * coeff_mat * gamma, coords_mat[0], coords_mat[1])
    for coords_vec, coeff_vec in np.ndenumerate(ising_vec):
        if coeff_vec != 0:
            quantumcircuit_problem.rz(2 * coeff_vec * gamma, coords_vec[0])
    qc_qaoa.append(quantumcircuit_problem, range(num_qubits))
    
    # create the mixing Unitary
    quantumcircuit_mixing = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        quantumcircuit_mixing.rx(2 * beta, i)
    qc_qaoa.append(quantumcircuit_mixing, range(num_qubits))
    
    #add a measure to all qubits in the QAOA-graph 
    qc_qaoa.measure_all()
    
    return qc_qaoa.decompose()


def run_circuit_qiskit(circuit, shots):
    
    # create the backend-simulator
    backend = Aer.get_backend('qasm_simulator')
    backend.shots=1000
    #simulate the QAOA-graph
    result = backend.run(circuit, seed_simulator=10, shots=shots).result()
    counts = result.get_counts()
    
    return counts


a=Unittest_BinPacking()























