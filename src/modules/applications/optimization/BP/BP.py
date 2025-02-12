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

from typing import TypedDict
import numpy as np
import random
import pdb
import math

from docplex.mp.model import Model
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.converters import InequalityToEquality, IntegerToBinary, LinearEqualityToPenalty

from modules.applications.Application import *
from modules.applications.optimization.Optimization import Optimization
from utils import start_time_measurement, end_time_measurement


class BP(Optimization):
    """
    \"The bin packing problem is an optimization problem, in which items of different size
    s must be packed into a finite number of bins or containers, each of a fixed given capacity,
    in a way that minimizes the number of bins used. The problem has many applications, such as 
    filling up containers, loading trucks with weight capacity constraints, creating file backups 
    in media, and technology mapping in FPGA semiconductor chip design.\"
    (source: https://en.wikipedia.org/wiki/Bin_packing_problem)
    """

    def __init__(self):
        """
        Constructor method
        """
        super().__init__("BinPacking")
        self.submodule_options = ["MIP", "Ising", "QUBO"]

    @staticmethod
    def get_requirements() -> list:
        return [
            {
                "name": "numpy",
                "version": "1.23.5"
            },
            {
                "name": "qiskit_optimization",
                "version": "0.5.0"
            },
            {
                "name": "docplex",
                "version": "2.25.236"
            }
        ]

    def get_solution_quality_unit(self) -> str:
        return "number_of_bins"

    def get_default_submodule(self, option: str) -> Core:

        if option == "Ising":
            from modules.applications.optimization.TSP.mappings.ISING import Ising  # pylint: disable=C0415
            return Ising()
        elif option == "QUBO":
            from modules.applications.optimization.TSP.mappings.QUBO import QUBO  # pylint: disable=C0415
            return QUBO()
        elif option == "MIP":
            from modules.applications.optimization.BP.mappings.MIP import MIP  # pylint: disable=C0415
            return MIP()
        else:
            raise NotImplementedError(f"Mapping Option {option} not implemented")

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this application

        :return:
                 .. code-block:: python

                      return {
                          "number_of_objects": {
                              "values": list([3,4,5,6,7,8,9,10,15,20]),
                              "description": "How many objects do you want to fit inside the bins?",
                              },
                          "instance_creating_mode": {
                              "values": list(["linear weights without incompatibilities",
                                              "linear weights with incompatibilities",
                                              "random weights without incompatibilities",
                                              "random weights with incompatibilities"]),
                              "description": "How do you want to create the object weights?"
                              }
                            }

        """
        return {
            "number_of_objects": {
                "values": list([3,4,5,6,7,8,9,10,15,20]),
                "description": "How many objects do you want to fit inside the bins?",
                },
            "instance_creating_mode": {
                "values": list(["linear weights without incompatibilities",
                                "linear weights with incompatibilities",
                                "random weights without incompatibilities",
                                "random weights with incompatibilities"]),
                "description": "How do you want to create the object weights?"
                }
            }

    class Config(TypedDict):
        """
        Attributes of a valid config

        .. code-block:: python

             number_of_objects: int
             instance_creating_mode: str

        """
        number_of_objects: int
        instance_creating_mode: str
        
    def create_bin_packing_instance(self, number_of_objects: int, mode: str) -> (list, int, list):
        """
        generates a bin-packing problem instance depending on the mode and the number of objects

        :param number_of_objects: how many objects should the bin packing problem instance consist of
        :type number_of_objects: int
        :param mode: declares the mode with which the bin packing problem instance should be created
        :type mode: str
        :return: tuple with object_weights, bin_capacity, incompatible_objects
        rtype: tuple(list, int, list)
        """
        
        if mode == "linear weights without incompatibilities":
            object_weights = list(range(1, number_of_objects+1))
            bin_capacity = max(object_weights)
            incompatible_objects = []
        
        elif mode == "linear weights with incompatibilities":
            object_weights = list(range(1, number_of_objects+1))
            bin_capacity = max(object_weights)
            incompatible_objects = []
            #add some incompatible objects via a for-loop
            for i in range(math.floor(number_of_objects/2)):
                incompatible_objects.append((i, number_of_objects - 1 - i))
        
        elif mode == "random weights without incompatibilities":
            object_weights = [random.randint(1, number_of_objects) for _ in range(number_of_objects)]
            bin_capacity = max(object_weights)
            incompatible_objects = []
        
        elif mode == "random weights with incompatibilities":
            object_weights = [random.randint(1, number_of_objects) for _ in range(number_of_objects)]
            bin_capacity = max(object_weights)
            incompatible_objects = []
            for i in range(math.floor(number_of_objects/2)):
                incompatible_objects.append((i, number_of_objects - i))
        
        else:
            logging.error("An error occurred. Couldn't create a bin packing instance")
            raise ValueError("forbidden mode during bin-packing-instance-creating-process")
        
        return object_weights, bin_capacity, incompatible_objects
   
    
    def generate_problem(self, config: Config, **kwargs) -> (list, float, list):
        """
        generates a bin-packing problem instance with the input configuration

        :param config:
        :type config: Config
        :param kwargs: Optional additional arguments
        :type kwargs: dict
        :return: tuple with object_weights, bin_capacity, incompatible_objects
        :rtype: tuple(list, int, list)
        """

        if config is None:
            config = {"number_of_objects": 5,
                      "instance_creating_mode": "linear weights without incompatibilities"}

        number_of_objects = config['number_of_objects']
        instance_creating_mode = config['instance_creating_mode']

        self.object_weights, self.bin_capacity, self.incompatible_objects = self.create_bin_packing_instance(number_of_objects, instance_creating_mode)
        
        return self.object_weights, self.bin_capacity, self.incompatible_objects  
    

    def validate(self, solution: dict, **kwargs) -> (bool, float):
        """
        Checks if a given solution is feasible for the problem instance

        :param solution: list containing the nodes of the solution
        :type solution: list
        :param kwargs: Optional additional arguments
        :type kwargs: dict
        :return: Boolean whether the solution is valid, time it took to validate
        :rtype: tuple(bool, float)
        """
        start = start_time_measurement()
        config_summary = kwargs['configuration_summary']
        mapping = config_summary['mapping']
            
        if solution is None:
            return False, end_time_measurement(start)
        else:            
            #create the MIP to investigate the solution
            problem_instance = (self.object_weights, self.bin_capacity, self.incompatible_objects)
            self.mip_original = create_MIP(problem_instance)
            
            # %% MIP
            if mapping == 'MIP':         
                #transform docplex model to the qiskit-optimization framework
                self.mip_qiskit = from_docplex_mp(self.mip_original)
                #put the solution-values into a list to be able to check feasibility
                solution_list = []
                for key, value in solution.items():
                    solution_list.append(value)
                feasible_or_not = self.mip_qiskit.is_feasible(solution_list)
            
            # %% QUBO
            elif mapping in ['QUBO', 'Ising']: # QUBO or Ising -->we need the binary equation formulation of the MIP
                
                #transform docplex model to the qiskit-optimization framework
                self.mip_qiskit = from_docplex_mp(self.mip_original)
                #transform inequalities to equalities --> with slacks
                mip_ineq2eq = InequalityToEquality().convert(self.mip_qiskit)
                #transform integer variables to binary variables -->split up into multiple binaries
                self.mip_qiskit_int2bin = IntegerToBinary().convert(mip_ineq2eq)
                
                #re-order the solution-values to be able to check feasibility -> because the variables are muddled in the dictionary
                x_values = []
                y_values = []
                slack_values = []
                for key, value in solution.items():
                    if key[0] == "x":   #bin-variable
                        x_values.append(value)
                    elif key[0] == "y": #object-assignment-variable
                        y_values.append(value)
                    else:               #slack-variable
                        slack_values.append(value)
                solution_list = x_values + y_values + slack_values
                feasible_or_not = self.mip_qiskit_int2bin.is_feasible(solution_list) 
            else:
                logging.error('Error during validation. illegal mapping was used, please check')
                feasible_or_not = 'Please raise error'
            
            pdb.set_trace()
            return feasible_or_not, end_time_measurement(start)

    def evaluate(self, solution: dict, **kwargs) -> (float, float):
        """
        Find the number of used bins for a given solution

        :param solution:
        :type solution: list
        :param kwargs: Optional additional arguments
        :type kwargs: dict
        :return: Tour cost and the time it took to calculate it
        :rtype: tuple(float, float)
        """
        start = start_time_measurement()
        config_summary = kwargs['configuration_summary']
        mapping = config_summary['mapping']
            
        if solution is None:
            return False, end_time_measurement(start)
        else:        
            #put the solution-values into a list
            solution_list = []
            for keys, value in solution.items():
                solution_list.append(value)
                        
            if mapping == 'MIP': 
                obj_value = self.mip_qiskit.objective.evaluate(solution_list)
                
            elif mapping in ['QUBO', 'Ising']: # QUBO or Ising -->we need the binary equation formulation of the MIP
                obj_value = self.mip_qiskit_int2bin.objective.evaluate(solution_list) #mip_int2bin.objective.evaluate(solution)
            
            else:
                logging.error('Error during validation. illegal mapping was used, please check')
                obj_value = 'Please raise error'
            
            #pdb.set_trace()
            return obj_value, end_time_measurement(start)

    def save(self, path: str, iter_count: int) -> None:
        pass


def create_MIP(problem: (list, float, list), **kwargs) -> (Model, float): 
    """
    generates a bin-packing problem docplex model depending on a certain instance

    :param problem: bin packing problem instance defined by
                1. object weights, 2. bin capacity, 3. incompatible objects
    :type problem: list, float, list
    :param kwargs: Optional additional arguments
    :type kwargs: dict
    :return: the resulting bin packing model
    :rtype: docplex model
    """
    #TODO MIP or QUBO/Ising in kwargs -->different constraints
    
    # %% initialize the problem data
    object_weights = problem[0]
    bin_capacity = problem[1]
    incompatible_objects = problem[2]
    
    # %% create the docplex model
    binpacking_mip = Model("BinPacking")
    logging.info("start the creation of the bin-packing-MIP \n")
    
    #define the necessary variables for the creation
    max_number_of_bins = len(object_weights)
    num_of_objects = len(object_weights)
    
    # %% add model variables
    bin_variables = binpacking_mip.binary_var_list(keys=range(max_number_of_bins), name=[f"x_{i}" for i in range(max_number_of_bins)])        
    logging.info("added binary variables x_i --> =1 if bin i is used, =0 if not")
    object_to_bin_variables = binpacking_mip.binary_var_matrix(keys1=range(num_of_objects), keys2=range(max_number_of_bins), name="y")#lambda i,j: f'x{j}{i}')
    logging.info("added binary variables y_j_i --> =1 if object j is put into bin i, =0 if not")
    
    # %% add model objective --> minimize sum of x_i variables
    objective = binpacking_mip.minimize(binpacking_mip.sum([bin_variables[i] for i in range(max_number_of_bins)]))
    logging.info("added the objective with goal to minimize")
    
    # %% add model constraints
    assignment_constraints = binpacking_mip.add_constraints((binpacking_mip.sum(object_to_bin_variables[o, i] for i in range(max_number_of_bins)) == 1 for o in range(num_of_objects)), ["assignment_constraint_object_%d" % i for i in range(num_of_objects)])
    logging.info("added constraints so that each object gets assigned to a bin")
    capacity_constraints = binpacking_mip.add_constraints((binpacking_mip.sum(object_weights[o] * object_to_bin_variables[o, i] for o in range(num_of_objects)) <= bin_capacity * bin_variables[i] for i in range(max_number_of_bins)), ["capacity_constraint_bin_%d" % i for i in range(max_number_of_bins)])
    logging.info("added constraints so that the bin-capacity isn't violated")
    # the following is good for the QUBO formulation because we don't need to introduce slack variables
    incompatibility_constraints = binpacking_mip.add_quadratic_constraints(object_to_bin_variables[o1, i] * object_to_bin_variables[o2, i] == 0 for (o1, o2) in incompatible_objects for i in range(max_number_of_bins))#, ["incompatibility_constraint_%d" % i for i in range(max_number_of_bins * len(incompatible_objects))])
    #TODO the following is equivalent, but better suited for a MIP Solver because it is linear
    # incompatibility_constraints = binpacking_mip.add_constraints((object_to_bin_variables[o1,i] + object_to_bin_variables[o2,i] <= 1 for (o1,o2) in incompatible_objects for i in range(max_number_of_bins)), ["incompatibility_constraint_%d" % i for i in range(max_number_of_bins * len(incompatible_objects))])
    logging.info("added constraints so that incompatible objects aren't put in the same bin \n")
    
    logging.info("finished the creation of the bin-packing-MIP \n\n")
    
    return binpacking_mip

def transform_docplex_mip_to_qubo(mip_docplex: Model, penalty_factor) -> (dict, QuadraticProgram):
    '''
    transform a docplex mixed-integer-problem to a QUBO
    
    :param mip_docplex : Docplex-Model
    :type mip_docplex: Docplex-Model
    :param config: config with the parameters specified in Config class
    :type config: Config
    :return: the transformed QUBO
    :rtype:  QuadraticProgram from qiskit_optimization
    '''
    #transform docplex model to the qiskit-optimization framework
    mip_qiskit = from_docplex_mp(mip_docplex)
    
    #transform inequalities to equalities --> with slacks
    mip_ineq2eq = InequalityToEquality().convert(mip_qiskit)
    
    #transform integer variables to binary variables -->split up into multiple binaries
    mip_int2bin = IntegerToBinary().convert(mip_ineq2eq)
    
    #transform the linear equality constraints to penalties in the objective
    if penalty_factor == None:
        #normalize the coefficients of the QUBO that results from penalty coefficients = 1
        qubo = LinearEqualityToPenalty(penalty=1).convert(mip_int2bin)
        max_lin_coeff = numpy.max(abs(qubo.objective.linear.to_array()))
        max_quad_coeff = numpy.max(abs(qubo.objective.quadratic.to_array()))
        max_coeff = max(max_lin_coeff, max_quad_coeff)
        penalty_factor = round(1 / max_coeff, 3)
    qubo = LinearEqualityToPenalty(penalty=penalty_factor).convert(mip_int2bin)
    
    # squash the quadratic and linear QUBO-coefficients together into a dictionary
    quadr_coeff = qubo.objective.quadratic.to_dict(use_name=True)
    lin_coeff = qubo.objective.linear.to_dict(use_name=True)                
    for var, var_value in lin_coeff.items():
        if (var,var) in quadr_coeff.keys():
            quadr_coeff[(var,var)] += var_value
        else:
            quadr_coeff[(var,var)] = var_value
    qubo_operator = quadr_coeff 
    
    return qubo_operator, qubo

def transform_docplex_mip_to_ising(mip_docplex: Model, penalty_factor) -> (np.array, np.array, float, QuadraticProgram):
    '''
    transform a docplex mix-integer-problem to an Ising formulation
    
    :param mip_docplex : Docplex-Model
    :type mip_docplex: Docplex-Model
    :param config: config with the parameters specified in Config class
    :type config: Config
    :return: J-matrix, h-vector and c-offset of the Ising formulation
    :rtype:  dict
    '''
    # %% generate the QUBO with binary variables in {0; 1}
    _, qubo = transform_docplex_mip_to_qubo(mip_docplex, penalty_factor)
    
    # %% transform it to an Ising formulation
    # --> x in {0; 1} gets transormed the following way via y in {-1; 1}: x = 1/2 * (1 - y)
    #       0 --> 1    and   1 --> -1
    # in the following we construct a matrix J, a vector h and an offset c for the Ising formulation
    # so that we have an equivalent formulation of the QUBO: obj = x J x^T + h x + c
    num_qubits = len(qubo.variables)
    ising_matrix = np.zeros((num_qubits, num_qubits), dtype=np.float64)
    ising_vector = np.zeros(num_qubits, dtype=np.float64)
    
    ising_offset = qubo.objective.constant
    
    for idx, coeff in qubo.objective.linear.to_dict().items():
        ising_vector[idx] -= 1/2 * coeff
        ising_offset += 1/2 * coeff
    
    for (i, j), coeff in qubo.objective.quadratic.to_dict().items():
        if i == j:
            ising_offset += 1/2 * coeff #because the quadratic term x_i * x_j reduces to 1 if the x are ising variables in {-1, 1} --> another constant term
        else:
            ising_matrix[i,j] += 1/4 * coeff   
            ising_offset += 1/4 * coeff
        ising_vector[i] -= 1/4 * coeff
        ising_vector[j] -= 1/4 * coeff
        
    return ising_matrix, ising_vector, ising_offset, qubo


