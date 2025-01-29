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

from itertools import combinations, product
from typing import TypedDict
import logging

from nnf import Var, And

from quark.modules.applications.Mapping import Mapping, Core
from quark.utils import start_time_measurement, end_time_measurement


class ChoiQUBO(Mapping):
    """
    QUBO formulation for SAT problem by Choi (1004.2226).
    """

    def __init__(self):
        """
        Constructor method.
        """
        super().__init__()
        self.submodule_options = ["Annealer"]
        self.nr_vars = None
        self.reverse_dict = None

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Return requirements of this module.

        :return: List of dict with requirements of this module
        """
        return [{"name": "nnf", "version": "0.4.1"}]

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this mapping.

        :return: Dictionary with parameter options
        .. code-block:: python

            return {
                    "hard_reward": {
                        "values": [0.1, 0.5, 0.9, 0.99],
                        "description": "What Bh/A ratio do you want? (How strongly to enforce hard constraints)"
                    },
                    "soft_reward": {
                        "values": [0.1, 1, 2],
                        "description": "What Bh/Bs ratio do you want? This value is multiplied with the "
                        "number of tests."
                    }
                }
        """
        return {
            "hard_reward": {
                "values": [0.1, 0.5, 0.9, 0.99],
                "description": (
                    "What Bh/A ratio do you want?"
                    "(How strongly to enforce hard constraints)"
                )
            },
            "soft_reward": {
                "values": [0.1, 1, 2],
                "description": (
                    "What Bh/Bs ratio do you want?"
                    "This value is multiplied with the number of tests."
                )
            }
        }

    class Config(TypedDict):
        """
        Attributes of a valid config.

        .. code-block:: python

             hard_reward: float
             soft_reward: float

        """
        hard_reward: float
        soft_reward: float

    def map(self, problem: tuple[And, list], config: Config) -> tuple[dict, float]:
        """
        Converts a MaxSAT instance with hard and soft constraints into a graph problem --
        solving MaxSAT then corresponds to solving an instance of the Maximal Independent Set problem.
        See Andrew Lucas (2014), or the original publication by Choi (1004.2226).

        :param problem: A tuple containing hard and soft constraints
        :param config: Config with the parameters specified in Config class
        :return: Dictionary containing the QUBO representation and the time taken
        """
        start = start_time_measurement()

        hard_constraints, soft_constraints = problem
        a = 1
        bh = config['hard_reward'] * a
        # divide Bh by the number of test clauses, such that fulfilling a test result is less favourable than
        # satisfying a constraint, which aim to prioritize.
        bs = bh * config['soft_reward'] / len(soft_constraints)

        # Count the number of different variables that appear in the vehicle options problem:
        self.nr_vars = len(hard_constraints.vars().union(And(soft_constraints).vars()))
        # Edges variable holds all edges in the resulting graph
        edges = {}
        # lit_occur is a dictionary which will store the information in which clause a certain literal will occur.
        lit_occur = {}

        def _add_clause(clause, curr_edges, curr_lit_occ, pos):
            literals = [f"{el}-{pos}" for el in clause.children]
            # Connect the literals within one clause
            for cmb in combinations(literals, 2):
                # Add a weight for each edge within clause
                curr_edges[cmb] = a
            # Add the occurrences of the variables to the occurrences dictionary
            for var in clause.children:
                if var.name not in curr_lit_occ.keys():
                    curr_lit_occ[var.name] = {True: [], False: []}
                # Add occurrences and mark that they correspond to hard constraints
                curr_lit_occ[var.name][var.true].append(pos)
            return curr_edges, curr_lit_occ

        # Convert the hard constraints into the graph
        for idx, hard_constraint in enumerate(hard_constraints):
            edges, lit_occur = _add_clause(hard_constraint, edges, lit_occur, idx)

        # Save the current total clause count:
        constraints_max_ind = len(hard_constraints)
        # Repeat the procedure for the soft constraints:
        for idx, soft_constraint in enumerate(soft_constraints):
            edges, lit_occur = _add_clause(soft_constraint, edges, lit_occur, idx + constraints_max_ind)

        # Connect conflicting clauses using the lit_occur dict:
        for literal, positions_dict in lit_occur.items():
            # for every literal lit, we check its occurrences and connect the non-negated and negated occurrences.
            for pos_true, pos_false in product(positions_dict[True], positions_dict[False]):
                if pos_true != pos_false:
                    # Employ the notation from nnf, where the tilde symbol ~ corresponds to negation.
                    lit_true, lit_false = f"{literal}-{pos_true}", f"~{literal}-{pos_false}"
                    # Add a penalty for each such edge:
                    edges[(lit_true, lit_false)] = a

        # Collect all different nodes that we have in our graph, omitting repetitions:
        node_set = set([])
        for nodes in edges.keys():
            node_set = node_set.union(set(nodes))

        node_list = sorted(node_set)
        # Fix a mapping (node -> binary variable)
        relabel_dict = {v: i for i, v in enumerate(node_list)}
        # Save the reverse mapping, which is later used to decode the solution.
        self.reverse_dict = dict(enumerate(node_list))

        def _remap_pair(pair):
            """Small helper function that maps the nodes of an edge to binary variables"""
            return relabel_dict[pair[0]], relabel_dict[pair[1]]

        # Save the QUBO corresponding to the graph.
        q = {_remap_pair(key): val for key, val in edges.items()}

        for v in node_list:
            # Add different energy rewards depending on whether it is a hard or a soft constraint
            if int(v.split('-')[-1]) < constraints_max_ind:
                # if hard cons, add -Bh as the reward
                q[_remap_pair((v, v))] = -bh
            else:
                # for soft constraints, add -Bs
                q[_remap_pair((v, v))] = -bs

        logging.info(f"Converted to Choi QUBO with {len(node_list)} binary variables. Bh={config['hard_reward']},"
                     f" Bs={bs}.")
        return {'Q': q}, end_time_measurement(start)

    def reverse_map(self, solution: dict) -> tuple[dict, float]:
        """
        Maps the solution back to the representation needed by the SAT class for validation/evaluation.

        :param solution: Dictionary containing the solution
        :return: Solution mapped accordingly, time it took to map it
        """
        start = start_time_measurement()
        # We define the literals list, so that we can check the self-consistency of the solution. That is, we save all
        # assignments proposed by the annealer, and see if there is no contradiction. (In principle a solver
        # could mandate L3 = True and L3 = False, resulting in a contradiction.)
        literals = []
        # Assignments saves the actual solution
        assignments = []

        for node, tf in solution.items():
            # Check if node is included in the set (i.e. if tf is True (1))
            if tf:
                # Convert back to the language of literals
                lit_str = self.reverse_dict[node]
                # Check if the literal is negated:
                if lit_str.startswith('~'):
                    # Remove the negation symbol
                    lit_str = lit_str.replace('~', '')
                    # Save a negated literal object, will be used for self-consistency check
                    lit = Var(lit_str).negate()
                    # Add the negated literal to the assignments, removing the (irrelevant) position part
                    assignments.append(Var(lit_str.split('-')[0]).negate())
                else:
                    # If literal is true, no ~ symbol needs to be removed:
                    lit = Var(lit_str)
                    assignments.append(Var(lit_str.split('-')[0]))
                literals.append(lit)

        # Check for self-consistency of solution; Check that the assignments of all literals are consistent:
        if not And(set(literals)).satisfiable():
            logging.error('Generated solution is not self-consistent!')
            raise ValueError("Inconsistent solution for the ChoiQubo returned.")

        # If the solution is consistent, find and add potentially missing variables:
        assignments = sorted(set(assignments))
        # Find missing vars, or more precisely, their labels:
        missing_vars = set(range(self.nr_vars)) - {int(str(a).replace('L', '').replace('~', '')) for a in assignments}

        # Add the variables that found were missing:
        for nr in missing_vars:
            assignments.append(Var(f'L{nr}'))

        return {list(v.vars())[0]: v.true for v in sorted(assignments)}, end_time_measurement(start)

    def get_default_submodule(self, option: str) -> Core:
        """
        Returns the default submodule based on the provided option.

        :param option: Option specifying the submodule
        :return: Instance of the corresponding submodule
        :raises NotImplementedError: If the option is not recognized
        """
        if option == "Annealer":
            from quark.modules.solvers.Annealer import Annealer  # pylint: disable=C0415
            return Annealer()
        else:
            raise NotImplementedError(f"Solver Option {option} not implemented")
