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

import io
from typing import TypedDict
import logging

from nnf import And
from nnf.dimacs import dump
from pysat.formula import CNF, WCNF

from quark.modules.applications.Mapping import Mapping, Core
from quark.utils import start_time_measurement, end_time_measurement


class Direct(Mapping):
    """
    Maps the problem from nnf to pysat.
    """

    def __init__(self):
        """
        Constructor method.
        """
        super().__init__()
        self.submodule_options = ["ClassicalSAT", "RandomSAT"]

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Return requirements of this module.

        :return: List of dict with requirements of this module
        """
        return [
            {"name": "nnf", "version": "0.4.1"},
            {"name": "python-sat", "version": "1.8.dev13"}
        ]

    def get_parameter_options(self) -> dict:
        """
        Returns empty dict as this mapping has no configurable settings.

        :return: Empty dict
        """
        return {}

    class Config(TypedDict):
        """
        Empty config as this solver has no configurable settings.
        """
        pass

    def map(self, problem: tuple[And, list], config: Config) -> tuple[WCNF, float]:
        """
        Map from the nnf library into the python-sat library.

        :param problem: SAT problem
        :param config: Config with the parameters specified in Config class
        :return: Mapped problem and the time it took to map it
        """
        start = start_time_measurement()
        hard_constraints, soft_constraints = problem

        # Get number of vars. The union is required in case not all vars are present in either tests/constraints.
        nr_vars = len(hard_constraints.vars().union(And(soft_constraints).vars()))

        # Create a var_labels dictionary that will be used when mapping to pysat
        litdic = {f'L{i - 1}': i for i in range(1, nr_vars + 1)}

        # The most convenient way to map between nnf and pysat was to use the native nnf dump function, which exports
        # the problem as a string, which we can then quickly reload from a buffer.

        # Create buffers for dumping:
        hard_buffer = io.StringIO()
        soft_buffer = io.StringIO()

        # Dump constraints and tests to their respective buffers
        dump(hard_constraints, hard_buffer, var_labels=litdic, mode='cnf')
        # tests have to be conjoined, since we will add them as soft constraints.
        dump(And(soft_constraints), soft_buffer, var_labels=litdic, mode='cnf')

        # Load the cnfs from the buffers:
        hard_cnf = CNF(from_string=hard_buffer.getvalue())
        soft_cnf = CNF(from_string=soft_buffer.getvalue())

        # Create wcnf instance.
        total_wcnf = WCNF()

        # Add hard constraints:
        total_wcnf.extend(hard_cnf)

        # Add soft constraints, with weights.
        total_wcnf.extend(soft_cnf, weights=[1] * len(soft_cnf.clauses))

        logging.info(
            f'Generated pysat wcnf with {len(total_wcnf.hard)} constraints and {len(total_wcnf.soft)} tests.'
        )

        return total_wcnf, end_time_measurement(start)

    def get_default_submodule(self, option: str) -> Core:
        """
        Returns the default submodule based on the provided option.

        :param option: Option specifying the submodule
        :return: Instance of the corresponding submodule
        :raises NotImplementedError: If the option is not recognized
        """
        if option == "ClassicalSAT":
            from quark.modules.solvers.ClassicalSAT import ClassicalSAT  # pylint: disable=C0415
            return ClassicalSAT()
        elif option == "RandomSAT":
            from quark.modules.solvers.RandomClassicalSAT import RandomSAT  # pylint: disable=C0415
            return RandomSAT()
        else:
            raise NotImplementedError(f"Solver Option {option} not implemented")

    def reverse_map(self, solution: list) -> tuple[dict, float]:
        """
        Maps the solution returned by the pysat solver into the reference format.

        :param solution: List containing the solution
        :return: Solution mapped accordingly, time it took to map it
        """
        start = start_time_measurement()
        # converts from (3 / -3) -> (L2 : True / L2: False)
        mapped_sol = {f'L{abs(lit) - 1}': (lit > 0) for lit in solution}
        return mapped_sol, end_time_measurement(start)
