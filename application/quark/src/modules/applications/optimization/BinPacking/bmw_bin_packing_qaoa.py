# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 14:43:00 2023

@author: stopfer
"""

# %% import modules
import logging
import os
import time
import pdb
import pandas as pd
import math
import pickle
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np


from qiskit import QuantumCircuit, Aer
from qiskit.circuit import Parameter

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.converters import InequalityToEquality, IntegerToBinary, LinearEqualityToPenalty

from docplex.mp.model import Model

from scipy.optimize import minimize

import logging_util


# %% class to initialise some settings

class configuration:
    def __init__(self):
        
        #path settings
        self.path_import = os.path.dirname(os.path.dirname(__file__)) + '\\data\\import\\'  # is the path of this settings.py-File;  dirname returns the directory name the path
        counter = 0
        self.path_export = os.path.dirname(os.path.dirname(__file__)) + '\\data\\export\\%s_Result_BenchQC_Opt_%s\\' %(time.strftime("%Y%m%d"),counter)   # create a string for a path with a timestamp for calculation output; the %s in the string gets specified by the timefunction in the end
        while os.path.exists(self.path_export):  # as long as the output path already exists
            self.path_export = os.path.dirname(os.path.dirname(__file__)) + '\\data\\export\\%s_Result_BenchQC_Opt_%s\\' %(time.strftime("%Y%m%d"),counter) 
            counter += 1
        
        #logging settings
        self.loglevel_logfile = logging.INFO  # logging.INFO=20, .DEBUG=10, .WARNING=30, .ERROR=40, .CRITICAL=50
        self.loglevel_console = logging.INFO  # logging.INFO=20, .DEBUG=10, .WARNING=30, .ERROR=40, .CRITICAL=50
        self.loglevel_high = logging.CRITICAL
        self.start_time = time.time()
        self.log_calculation_states = True
        
        #report settings
        self.print_models = False
        self.draw_circuits = False
        self.draw_final_circuits = True
        
        #QAOA-settings
        self.shots = 1000           # indicates how often the QAOA-ciruit is simulated in each step
        self.shots_to_get_feasible_solution = 100000   # indicates how often the QAOA-circuit is simulated to find a feasible solution
        self.theta = [0.9424777960769379, 0.3141592653589793]#[0,0] #[1.0210176, 0.314159]#[0.5, 0.5] #[0.47855772 1.22433022] , [0.5, 0.5, 0.5, 0.5] #[1, 0.8, 0.6, 0.4, 0.2, 0, 0, 0.2, 0.4, 0.6, 0.8, 1], #[1.47134578, 0.46918538, 0.45777638, 0.57720568, 0.4193936,  0.49818981, 0.49901606, 0.72013397, 0.38308082, 0.47477518, 0.50641615, 0.49430488, 0.50483722, 1.49484834, 0.48462727, 0.49708976, 0.51027346, 0.4958481 ] #[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5] #[1, 1] #[0.5, 0.5, 0.5, 0.5] #((1, 0.8, 0.6, 0.4, 0.2, 0), (0, 0.2, 0.4, 0.6, 0.8, 1)) # (beta,gamma) indicate the parameters for the layers of the alternating-unitary-layers of the QAOA-circuit
        self.qubo_penaltyfactors = 1
        self.normed_penalty_coefficients = False
        self.optimize_theta = True
        
        #grid-search parameters
        self.do_grid_search = False # if true then a gridsearch for the parameter theta will be done --> to see how the expectation value behaves for certain theta --> theta has to be two dimensional
        self.grid_size = 20           # the size of the grid of the gridsearch for the parameter theta
        self.range_beta = math.pi /2        # indicates the range of beta that will be covered in the grid-search --> [0, range_beta]
        self.range_gamma = 2 * math.pi      # indicates the range of gamma that will be covered in the grid-search --> [0, range_gamma]
        return


# %% class to create a bin-packing-problem instance with coherent and incompatible objects

class bin_packing_problem_docplex:
    def __init__(self, settings, bin_capacity, object_weights, coherent_objects, incompatible_objects):
        
        #clean the input-data for the bin-packing-problem
        logging.info("Some data cleaning now is done:")
        object_weights, incompatible_objects = self.join_coherent_objects(object_weights, coherent_objects, incompatible_objects)
        logging.info("we joined coherent objects")
        incompatible_objects = self.clean_incompatible_objects(incompatible_objects)
        logging.info("duplicates in the incompatible objects list have been cleaned")
        max_number_of_bins = self.get_max_number_of_bins(bin_capacity, object_weights)
        logging.info("we calculated the maximum number of bins \n")
        num_of_objects = len(object_weights)
        
        #create model
        self.model = Model("Bin_Packing")
        
        #add variables
        self.bin_variables = self.model.binary_var_list(keys=range(max_number_of_bins), name=[f"x_{i}" for i in range(max_number_of_bins)])        
        logging.info("added binary variables x_i --> =1 if bin i is used, =0 if not")
        self.object_to_bin_variables = self.model.binary_var_matrix(keys1=range(num_of_objects), keys2=range(max_number_of_bins), name="y")#lambda i,j: f'x{j}{i}')
        logging.info("added binary variables y_j_i --> =1 if object j is put into bin i, =0 if not")
        
        #add objective
        self.objective = self.model.minimize(self.model.sum([self.bin_variables[i] for i in range(max_number_of_bins)]))
        logging.info("added the objective with goal to minimize")
        
        #add constraints
        self.assignment_constraints = self.model.add_constraints((self.model.sum(self.object_to_bin_variables[o, i] for i in range(max_number_of_bins)) == 1 for o in range(num_of_objects)), ["assignment_constraint_object_%d" % i for i in range(num_of_objects)])
        logging.info("added constraints so that each object gets assigned to a bin")
        self.capacity_constraints = self.model.add_constraints((self.model.sum(object_weights[o] * self.object_to_bin_variables[o, i] for o in range(num_of_objects)) <= bin_capacity * self.bin_variables[i] for i in range(max_number_of_bins)), ["capacity_constraint_bin_%d" % i for i in range(max_number_of_bins)])
        logging.info("added constraints so that the bin-capacity isn't violated")
        self.incompatibility_constraints = self.model.add_constraints((self.object_to_bin_variables[o1,i] + self.object_to_bin_variables[o2,i] <= 1 for (o1,o2) in incompatible_objects for i in range(max_number_of_bins)), ["incompatibility_constraint_%d" % i for i in range(max_number_of_bins * len(incompatible_objects))])
        logging.info("added constraints so that incompatible objects aren't put in the same bin \n")
        
        logging.info("finished the creation of the bin-packing-object \n\n")
        
        if settings.print_models == True:
            print(self.model.export_as_lp_string())
        
        return

    def get_max_number_of_bins(self, bin_capacity, object_weights):
    
        max_number_of_bins = len(object_weights)
        
        return max_number_of_bins
    
    
    # a pair of coherent objects can be viewed as one object 
    def join_coherent_objects(self, object_weights, coherent_objects, incompatible_objects):
        
        for (cohe_o1, cohe_o2) in coherent_objects:
            #check:
            if cohe_o1 >= cohe_o2:
                logging.error("ERROR coherent-object tuples have to ordered (smaller-number, bigger-number)")
                break
            
            object_weights[cohe_o1] += object_weights[cohe_o2]
            del object_weights[cohe_o2]
            
            #adjust the incompatible_objects
            incompatible_counter = 0
            for (incomp_o1, incomp_o2) in incompatible_objects:
                #check:
                if incomp_o1 >= incomp_o2:
                    logging.error("ERROR incompatible-object tuples have to ordered (smaller-number, bigger-number)")
                    break
                
                if incomp_o1 >= cohe_o2:
                    incomp_o1 -= 1
                    incomp_o2 -= 1
                elif incomp_o2 >= cohe_o2:
                    incomp_o2 -= 1
                
                incompatible_objects[incompatible_counter] = (incomp_o1, incomp_o2)
                incompatible_counter += 1
            
            #adjust the remaining coherent_objects
            coherent_counter = 0
            for (other_cohe_o1, other_cohe_o2) in coherent_objects:
                if other_cohe_o1 >= cohe_o2:
                    other_cohe_o1 -= 1
                    other_cohe_o2 -= 1
                elif other_cohe_o2 >= cohe_o2:
                    other_cohe_o2 -= 1
                
                coherent_objects[coherent_counter] = (other_cohe_o1, other_cohe_o2)
                coherent_counter += 1
                                
        return object_weights, incompatible_objects
    
    def clean_incompatible_objects(self, incompatible_objects):
        # throw out double entries in the incompatible_objects-list
        incompatible_objects = list(set(incompatible_objects))
        return incompatible_objects
    
    def fix_variables(self):
        
        return



# %% function to transform the bin-packing-problem to a binary-QUBO

def transform_to_qubo(settings, bin_packing_problem):
    bin_packing_start = from_docplex_mp( bin_packing_problem.model)
    if settings.print_models == True:
        logging.info("The Bin-Packing-MIP: \n\n\n " + bin_packing_start.export_as_lp_string() + "\n\n")
        
    ineq2eq = InequalityToEquality()
    bin_packing_problem_ineq2eq = ineq2eq.convert(bin_packing_start)
    if settings.print_models == True:
        logging.info("The Bin-Packing-MIP with slacks: \n\n\n " + bin_packing_problem_ineq2eq.export_as_lp_string() + "\n\n")
        
    int2bin = IntegerToBinary()
    bin_packing_problem_int2bin = int2bin.convert(bin_packing_problem_ineq2eq)
    if settings.print_models == True:
        logging.info("The Bin-Packing-MIP with binary slacks: \n\n\n " + bin_packing_problem_int2bin.export_as_lp_string() + "\n\n")
        
    if settings.normed_penalty_coefficients == True:
        #to norm all the penalty-coefficients, we derive a penaltyfactor, so that the largest coefficient in the objective will be 
        lineq2penalty = LinearEqualityToPenalty(penalty = settings.qubo_penaltyfactors)
        qubo_bin_packing_problem = lineq2penalty.convert(bin_packing_problem_int2bin)
        max_lin_coeff = np.max(abs(qubo_bin_packing_problem.objective.linear.to_array()))
        max_quad_coeff = np.max(abs(qubo_bin_packing_problem.objective.quadratic.to_array()))
        max_coeff = max(max_lin_coeff, max_quad_coeff)
        new_penalty_factor = round(1 / max_coeff, 3)
    else:  # if the penalty coefficients of the QUBO don't have to be normed
        new_penalty_factor = settings.qubo_penaltyfactors
    lineq2penalty = LinearEqualityToPenalty(penalty = new_penalty_factor)
    qubo_bin_packing_problem = lineq2penalty.convert(bin_packing_problem_int2bin)
    
    if settings.print_models == True:
        logging.info("The Bin-Packing-QUBO: \n\n\n " + qubo_bin_packing_problem.export_as_lp_string() + "\n\n")
        pdb.set_trace()
    
    return qubo_bin_packing_problem, bin_packing_problem_int2bin



# %% function to create the QAOA-circuit

def create_qaoa_circuit(settings, qubo_bin_packing_problem, num_qubits_needed, theta):
    qc_qaoa = QuantumCircuit(num_qubits_needed)
    
    #create the initial superposition state
    quantumcircuit_initial = QuantumCircuit(num_qubits_needed)
    for i in range(num_qubits_needed):
        quantumcircuit_initial.h(i)
    if settings.draw_circuits == True:
        quantumcircuit_initial.draw()
    #add the initialising_circuit to the QAOA-graph
    qc_qaoa.append(quantumcircuit_initial, range(num_qubits_needed))
    
    #add the problem-unitarys and mixing-unitarys to the QAOA-graph
    n_layers = int(len(theta)/2)
    for i in range(n_layers):
        # create the problem Unitary:
            
        quantumcircuit_problem = QuantumCircuit(num_qubits_needed)
        #gamma = Parameter("$\\gamma$")
        gamma = theta[n_layers+i]
        # add R_z-rotation for linear term in objective
        for var, coefficient in qubo_bin_packing_problem.objective.linear.to_dict().items():
            quantumcircuit_problem.rz(2 * coefficient * gamma, var)
        # add R_zz-rotation for quadratic term in objective
        for (var1, var2), coefficient in qubo_bin_packing_problem.objective.quadratic.to_dict().items():
            if var1 == var2:
                quantumcircuit_problem.rz(2 * coefficient * gamma, var1) # R_z-rotation, because x^2 = x for binary x --> is actually a linear term
            else:
                # quantumcircuit_problem.cx(var1, var2)
                # quantumcircuit_problem.rz(-1*coefficient*gamma, var2)
                # quantumcircuit_problem.cx(var1, var2)
                quantumcircuit_problem.rzz(2 * coefficient * gamma, var1, var2) # R_zz-rotation
        if settings.draw_circuits == True:
            quantumcircuit_problem.decompose().draw()
        
        
        # create the mixing Unitary
        
        quantumcircuit_mixing = QuantumCircuit(num_qubits_needed)
        #beta = Parameter("$\\beta$")
        beta = theta[i]
        for j in range(num_qubits_needed):
            quantumcircuit_mixing.rx(2 * beta, j)
        if settings.draw_circuits == True:
            quantumcircuit_mixing.draw()
        
        #add the parts of the graph to the full QAOA-graph
        qc_qaoa.append(quantumcircuit_problem, range(num_qubits_needed))
        qc_qaoa.append(quantumcircuit_mixing, range(num_qubits_needed))
        
        if settings.log_calculation_states == True:
            logging.info("created layer %s of %s alternating unitary-layers"%(i+1, n_layers))
        
    #add a measure to all qubits in the QAOA-graph 
    qc_qaoa.measure_all()
    
    if settings.draw_final_circuits == True:
        qc_qaoa.decompose().decompose().draw()
    
    return qc_qaoa.decompose()


# %% function to get the expectation value for the objective of the QUBO for a given counts-dictionary of a simulated QAOA-circuit

def get_expectation_from_counts(settings, qubo_bin_packing_problem, counts):
    expectation = 0 # initial value for expectation
    #now go through the counts-dictionary and calculate each objective value for each variable combination
    for var_combination, count in counts.items():
        var_combination_list = [float(i) for i in var_combination]
        objective_value = qubo_bin_packing_problem.objective.evaluate(var_combination_list)
        expectation += (count * objective_value)/len(counts)
    return expectation


# %% function to simulate the QAOA-algorithm for a bin-packing-problem-instance

def execute_the_qaoa(settings, qubo_bin_packing_problem, num_qubits_needed):
    
    def simulate_qaoa(theta):
        
        # create the backend-simulator
        backend = Aer.get_backend('qasm_simulator')
        backend.shots = settings.shots
        
        #create the QAOA-graph
        qaoa_circuit = create_qaoa_circuit(settings, qubo_bin_packing_problem, num_qubits_needed, theta)
        
        #simulate the QAOA-graph
        counts = backend.run(qaoa_circuit, seed_simulator=10, shots=settings.shots).result().get_counts()
        
        #analyse the results of the simulation and calculate the expected objective value    
        expectation_value = get_expectation_from_counts(settings, qubo_bin_packing_problem, counts)
        logging.info("expectation value for current theta: %s"%expectation_value)
        
        return expectation_value
    
    return simulate_qaoa





def run():
    # %% set the settings via the settings-class
    settings = configuration()
    os.mkdir(settings.path_export) # creates a new directory for the export of this run
    logger = logging_util.config_logger(settings.path_export, settings.loglevel_logfile, settings.loglevel_console)
    
    
    # %% create the input-data for the bin-packing-problem
    
    logger.info("create the input-data for the bin-packing-problem \n")
    
    # %%% small instance
    
    # bin_capacity = 3
    # object_weights = list(range(4))
    # coherent_objects = [(0,1),(1,2)]
    # incompatible_objects = [(1,3),(2,3)]
    
    # %%% medium instance
    
    bin_capacity = 4
    object_weights = list(range(5))
    coherent_objects = [(0,1),(1,2)]
    incompatible_objects = []#(1,3),(2,3),(1,4)]
    
    # %%%
    logger.info("we have a bin-packing-instance with: \n\t bin_capacity: %s; \n\t object_weights: %s; \n\t coherent_objects: %s; \n\t incompatible_objects: %s \n"%(bin_capacity, object_weights, coherent_objects, incompatible_objects))
    
    
    # %% create a bin-packing-problem-object and transform it to a QUBO for the QAOA-algorithm
    
    # create the bin-packing-problem-object
    logger.info("create the bin-packing-problem-object \n")
    bin_packing_problem = bin_packing_problem_docplex(settings, bin_capacity, object_weights, coherent_objects, incompatible_objects)
    
    # transform to qubo
    logger.info("transform the bin-packing-problem-object to QUBO \n")
    qubo_bin_packing_problem, bin_packing_problem_int2bin = transform_to_qubo(settings, bin_packing_problem)
    variables_qubo = qubo_bin_packing_problem.variables
    variables_qubo_names = []
    for var in variables_qubo:
        variables_qubo_names.append(var.name)
    num_qubits_needed = qubo_bin_packing_problem.get_num_vars()
    logger.info("%s qubits are needed for the calculation \n"%num_qubits_needed)
        
    
    # %% 3 possibilities to choose the theta for a simulation of the QAOA-circuit
    
    # %%% 1. use a classical optimizer to get the optimal theta that minimizes the expected objective value of the counts
    if settings.optimize_theta == True:
        logger.info("create and simulate the QAOA-circuit")
        logger.info("The goal is to get a theta that minimizes the expected objective value of the counts of the QAOA-circuit-simulation \n")
        expectation_value = execute_the_qaoa(settings, qubo_bin_packing_problem, num_qubits_needed) 
        theta_optimal_dict = minimize(expectation_value, settings.theta, method='COBYLA')
        theta_optimal = theta_optimal_dict['x']
        logger.info("the optimal theta from the COBYLA optimization is %s \n \n"%theta_optimal)
        
    # %%% 2.do a grid search to get a better understanding of the parameter-landscape of theta = (beta, gamma)
    elif settings.do_grid_search == True:
        
        logger.setLevel(settings.loglevel_high)
        backend = Aer.get_backend('qasm_simulator')
        logger.setLevel(settings.loglevel_console) 
        
        logger.info("Do a grid search to take a look at the objective landscape for a 2-dimensional theta \n")
        grid_size = settings.grid_size
        grid_results = np.zeros((grid_size, grid_size))
        last_time = time.time() 
        for beta_index in range(grid_size):
            beta = beta_index * ( settings.range_beta / grid_size )
            
            for gamma_index in range(grid_size):
                gamma = gamma_index * ( settings.range_gamma / grid_size )
                
                qaoa_circuit = create_qaoa_circuit(settings, qubo_bin_packing_problem, num_qubits_needed, [beta, gamma])
                counts = backend.run(qaoa_circuit, seed_simulator=10, shots=settings.shots).result().get_counts()
                expectation_value = get_expectation_from_counts(settings, qubo_bin_packing_problem, counts)
                grid_results[beta_index, gamma_index] = expectation_value
                
                if settings.log_calculation_states == True:
                    #log the simulation state
                    calc_state = round(100 * (beta_index * grid_size + gamma_index + 1) / (grid_size**2), 2)
                    time_tracker = time.time() - last_time
                    last_time = time.time()
                    remaining_time = round((time_tracker * (grid_size**2) * (100 - calc_state) / 100 ) / 60, 2) # in min
                    #logger.debug("State of the grid search for theta: " + str(calc_state) + "%")
                    logger.info("State of the grid search for theta: " + str(calc_state) + "%")
                    logger.info("Remaining time: %s min"%remaining_time)
                    
        #create a graphic and save this graphic to the export-path
        plt.figure(figsize=(18, 18))
        plt.imshow(grid_results, cmap='viridis', extent=[0, settings.range_gamma, 0, settings.range_beta])
        plt.colorbar(label = 'Expectation value of objective')
        plt.ylabel('Beta')
        plt.xlabel('Gamma')
        plt.title('Result of Gridsearch')
        plt.savefig(settings.path_export + "gridsearch_2_dimensional_theta.png")
        plt.show()
        
        #pdb.set_trace()
        # find the index with minimum value in grid_results
        pickle.dump(grid_results, open(settings.path_export + "grid_results", 'wb'))
        (beta_optimal_index, gamma_optimal_index) = np.unravel_index(np.argmin(grid_results), grid_results.shape)
        theta_optimal = [beta_optimal_index * ( settings.range_beta / grid_size ),
                         gamma_optimal_index * ( settings.range_gamma / grid_size )]
        logging.info("The optimal theta from the 2-dim. gridsearch is %s \n"%theta_optimal)
        
    # %%% 3. just choose the input parameter
    else: # set it as the input-theta
        theta_optimal = settings.theta
        logging.info("We use the input theta %s for the creation of the QAOA-circuit \n"%theta_optimal)
    
        
    
    # %% try to get feasible solutions by simulating the QAOA-Circuit with the optimal theta
    
    logger.info("Now the goal is to get feasible solutions")
    logger.info("To do that, we use the QAOA-circuit with the calculated optimal theta")
    
    # create the QAOA with the optimal theta
    logger.info("create the QAOA-circuit with the optimal theta \n")
    qaoa_circuit = create_qaoa_circuit(settings, qubo_bin_packing_problem, num_qubits_needed, theta_optimal)
    logger.setLevel(settings.loglevel_high)
    backend = Aer.get_backend('qasm_simulator')
    logger.setLevel(settings.loglevel_console)
    
    # simulate the QAOA-circuit
    logger.info("Simulate the QAOA-circuit %s times \n"%settings.shots_to_get_feasible_solution)
    counts = backend.run(qaoa_circuit, seed_simulator=10, shots=settings.shots_to_get_feasible_solution).result().get_counts()
    
    # evaluate the counts for feasible solutions
    logger.info("Now evaluate the results from the circuit-simulation. We calculate the objective values of the count-objects")
    feasible_solutions = pd.DataFrame()
    solution_results = pd.DataFrame()
    
    counts_counter = 0 # needed for the logging
    calc_perc = 0 # initial value for the logging 
    
    for var_combination, count in counts.items():
        var_combination_list = [float(i) for i in var_combination]  # transform to a list with numbers instead of strings
        inv_var_combination_list = var_combination_list[::-1]       # invert the list because qiskit variables are from right to left
        
        #calculate the objective value of this certain variable combination and write it into a result dataframe
        objective_value = qubo_bin_packing_problem.objective.evaluate(inv_var_combination_list)
        solution_results.at[var_combination, 'objective value'] = objective_value
        solution_results.at[var_combination, 'count'] = count
        
        #we want to derive feasible solutions, check on the transformed bin-packing-problem before the qubo-transformation on feasibility
        feasible_or_not = bin_packing_problem_int2bin.is_feasible(inv_var_combination_list)
        if feasible_or_not == True: # then it is a feasible solution
            feasible_solutions.at[var_combination, 'objective value'] = objective_value
            feasible_solutions.at[var_combination, 'count'] = count
            variables_with_value_1 = []
            for index in range(len(inv_var_combination_list)):
                if inv_var_combination_list[index] == 1:
                    variables_with_value_1.append(variables_qubo_names[index])
            feasible_solutions.at[var_combination, 'variables with value 1'] = ', '.join(variables_with_value_1)
        
        #log the calculation state
        counts_counter += 1
        old_calc_perc = calc_perc
        calc_perc = round(100 * counts_counter/len(counts), 1)
        if calc_perc != old_calc_perc: 
            logger.info("State of Counts-Evaluation: " + str(calc_perc) + "%")
    
    solution_results_sorted = solution_results.sort_values(by='objective value')
    if not feasible_solutions.empty:
        feasible_solutions_sorted = feasible_solutions.sort_values(by='objective value')
        pickle.dump(feasible_solutions_sorted, open(settings.path_export + "feasible_solutions_sorted", 'wb'))
    
    logger.info("Finished the QAOA. it took %s seconds"%(time.time() - settings.start_time))
    if feasible_solutions.empty:
        logger.info("By simulating the QAOA-circuit with the optimal theta, there were no feasible solutions found")
    else:
        logger.info("By simulating the QAOA-circuit with the optimal theta, we found %s feasible solutions"%feasible_solutions.shape[0])
    
    pdb.set_trace()
    
    
    
    return

if __name__ == "__main__":
    run()




# anhand kleiner instanz penaltyfaktoren variieren und schauen, wie viele zulässige lösungen rauskommen und wie gut die lösungen sind.
# penaltyfaktoren für x1 x2 auch durch maxcoeff 






# statevector simulator nachlesen für qiskit