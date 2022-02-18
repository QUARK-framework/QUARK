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

import logging
from typing import TypedDict, Union
from applications.Mapping import *
from solvers.ClassicalSAT import ClassicalSAT
from solvers.RandomClassicalSAT import RandomSAT
import io
from nnf import And
from nnf.dimacs import dump
from pysat.formula import CNF, WCNF


class Direct(Mapping):
    """
    Maps the problem from nnf to pysat.
    """

    def __init__(self):
        """
        Constructor method
        """
        super().__init__()
        self.solver_options = ["ClassicalSAT", "RandomSAT"]

    def get_parameter_options(self):
        """
        Returns empty dict as this mapping has no configurable settings.

        :return: empty dict
        :rtype: dict
        """
        return {

        }

    class Config(TypedDict):
        """
        Empty config as this solver has no configurable settings.
        """
        pass

    def map(self, problem: (And, list), config: Config) -> (WCNF, float):
        """
        We map from the nnf library into the python-sat library.

        :param problem:
        :type problem: (nnf.And, list)
        :param config: empty dict
        :type config: Config
        :return: mapped problem and the time it took to map it
        :rtype: tuple(WCNF, float)
        """
        start = time() * 1000
        hard_constraints, soft_constraints = problem
        # get number of vars. The union is required in case not all vars are present in either tests/constraints.
        nr_vars = len(hard_constraints.vars().union(And(soft_constraints).vars()))
        # create a var_labels dictionary that will be used when mapping to pysat
        litdic = {f'L{i - 1}': i for i in range(1, nr_vars + 1)}
        # The most convenient way to map between nnf and pysat was to use the native nnf dump function, which exports
        # the problem as a string, which we can then quickly reload from a buffer.
        # create buffers for dumping:
        hard_buffer = io.StringIO()
        soft_buffer = io.StringIO()
        # dump constraints and tests to their respective buffers
        dump(hard_constraints, hard_buffer, var_labels=litdic, mode='cnf')
        # tests have to be conjuncted, since we will add them as soft constraints.
        dump(And(soft_constraints), soft_buffer, var_labels=litdic, mode='cnf')

        # load the cnfs from the buffers:
        hard_cnf = CNF(from_string=hard_buffer.getvalue())
        soft_cnf = CNF(from_string=soft_buffer.getvalue())
        # create wcnf instance.
        total_wcnf = WCNF()
        # add hard constraints:
        total_wcnf.extend(hard_cnf)
        # add soft constraints, with weights.
        total_wcnf.extend(soft_cnf, weights=[1] * len(soft_cnf.clauses))
        logging.info(f'Generated pysat wcnf with {len(total_wcnf.hard)} constraints and {len(total_wcnf.soft)} tests.')
        return total_wcnf, round(time() * 1000 - start, 3)

    def get_solver(self, solver_option: str) -> Union[ClassicalSAT, RandomSAT]:

        if solver_option == "ClassicalSAT":
            return ClassicalSAT()
        elif solver_option == "RandomSAT":
            return RandomSAT()
        else:
            raise NotImplementedError(f"Solver Option {solver_option} not implemented")

    def reverse_map(self, solution: list) -> (dict, float):
        """
        Maps the solution returned by the pysat solver into the reference format.

        :param solution: dictionary containing the solution
        :type solution: list
        :return: solution mapped accordingly, time it took to map it
        :rtype: tuple(dict, float)
        """

        start = time() * 1000
        # converts from (3 / -3) -> (L2 : True / L2: False)
        mapped_sol = {f'L{abs(lit) - 1}': (lit > 0) for lit in solution}
        return mapped_sol, round(time() * 1000 - start, 3)
