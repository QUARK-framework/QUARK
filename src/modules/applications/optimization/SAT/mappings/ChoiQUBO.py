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

from nnf import Var, And

from modules.applications.Mapping import *
from utils import start_time_measurement, end_time_measurement


class ChoiQUBO(Mapping):
    """
    QUBO formulation for SAT problem by Choi (1004.2226).
    """

    def __init__(self):
        """
        Constructor method
        """
        super().__init__()
        self.submodule_options = ["Annealer"]
        self.nr_vars = None
        self.reverse_dict = None

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Return requirements of this module

        :return: list of dict with requirements of this module
        :rtype: list[dict]
        """
        return [
            {
                "name": "nnf",
                "version": "0.4.1"
            }
        ]

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this mapping

        :return:
                 .. code-block:: python

                     return {
                                "hard_reward": {
                                    "values": [0.1, 0.5, 0.9, 0.99],
                                    "description": "What Bh/A ratio do you want? (How strongly to enforce hard cons.)"
                                },
                                "soft_reward": {
                                    "values": [0.1, 1, 2],
                                    "description": "What Bh/Bs ratio do you want? This value is multiplied with the"
                                    " number of tests."
                                }
                            }

        """
        return {
            "hard_reward": {
                "values": [0.1, 0.5, 0.9, 0.99],
                "description": "What Bh/A ratio do you want? (How strongly to enforce hard cons.)"
            },
            "soft_reward": {
                "values": [0.1, 1, 2],
                "description": "What Bh/Bs ratio do you want? This value is multiplied with the number of tests."
            }
        }

    class Config(TypedDict):
        """
        Attributes of a valid config

        .. code-block:: python

             hard_reward: float
             soft_reward: float

        """
        hard_reward: float
        soft_reward: float

    def map(self, problem: (And, list), config) -> (dict, float):
        """
        Converts a MaxSAT instance with hard and soft constraints into a graph problem -- solving MaxSAT then
        corresponds to solving an instance of the Maximal Independent Set problem. See Andrew Lucas (2014),
        or the original publication by Choi (1004.2226).

        :param problem:
        :type problem: (nnf.And, list)
        :param config: config with the parameters specified in Config class
        :type config: Config
        :return:
        :rtype: tuple(dict, float)
        """
        start = start_time_measurement()

        hard_constraints, soft_constraints = problem
        # in principle, one could use a different value of A -- it shouldn't play a role though.
        A = 1
        Bh = config['hard_reward'] * A
        # we divide Bh by the number of test clauses, such that fulfilling a test result is less favourable than
        # satisfying a constraint, which we aim to prioritize.
        Bs = Bh * config['soft_reward'] / len(soft_constraints)
        # we count the number of different variables that appear in the vehicle options problem:
        self.nr_vars = len(hard_constraints.vars().union(And(soft_constraints).vars()))
        # edges variable holds all edges in the resulting graph
        edges = {}
        # lit_occur is a dictionary which will store the information in which clause a certain literal will occur.
        lit_occur = {}

        def _add_clause(clause, curr_edges, curr_lit_occ, pos):
            # iterating through the clauses, we add nodes corresponding to each literal
            # the format is as follows: L12-5, means that literal 12 is present in clause nr. 5.
            literals = [f"{el}-{pos}" for el in clause.children]
            # we connect the literals within one clause
            for cmb in combinations(literals, 2):
                # we add a weight for each edge within clause
                curr_edges[cmb] = A
            # we add the occurrences of the variables to the occurrences dictionary
            for var in clause.children:
                if var.name not in curr_lit_occ.keys():
                    curr_lit_occ[var.name] = {True: [], False: []}
                # we add occurrences and mark that they correspond to hard constraints
                curr_lit_occ[var.name][var.true].append(pos)
            return curr_edges, curr_lit_occ

        # first convert the hard constraints into the graph
        for idx, hard_constraint in enumerate(hard_constraints):
            edges, lit_occur = _add_clause(hard_constraint, edges, lit_occur, idx)

        # we save the current total clause count:
        constraints_max_ind = len(hard_constraints)
        # we repeat the procedure for the soft constraints:
        for idx, soft_constraint in enumerate(soft_constraints):
            edges, lit_occur = _add_clause(soft_constraint, edges, lit_occur, idx + constraints_max_ind)

        # we connect conflicting clauses using the lit_occur dict:
        for literal, positions_dict in lit_occur.items():
            # for every literal lit, we check its occurrences and connect the non-negated and negated occurrences.
            for pos_true, pos_false in product(positions_dict[True], positions_dict[False]):
                # we ensure that we do not add a penalty for contradicting literals in the
                if pos_true != pos_false:
                    # we employ the notation from nnf, where the tilde symbol ~ corresponds to negation.
                    lit_true, lit_false = f"{literal}-{pos_true}", f"~{literal}-{pos_false}"
                    # we add a penalty for each such edge:
                    edges[(lit_true, lit_false)] = A

        # we collect all different nodes that we have in our graph, omitting repetitions:
        node_set = set([])
        for nodes in edges.keys():
            node_set = node_set.union(set(nodes))

        node_list = sorted(node_set)
        # we fix a mapping (node -> binary variable)
        relabel_dict = {v: i for i, v in enumerate(node_list)}
        # we save the reverse mapping, which is later used to decode the solution.
        self.reverse_dict = dict(enumerate(node_list))

        def _remap_pair(pair):
            """Small helper function that maps the nodes of an edge to binary variables"""
            return relabel_dict[pair[0]], relabel_dict[pair[1]]

        # we save the Qubo corresponding to the graph.
        Q = {_remap_pair(key): val for key, val in edges.items()}

        for v in node_list:
            # we add different energy rewards depending on whether it is a hard or a soft constraint
            # soft cons. have lower rewards, since we prioritize satisfying hard constraints.
            if int(v.split('-')[-1]) < constraints_max_ind:
                # if hard cons, we add -Bh as the reward
                Q[_remap_pair((v, v))] = -Bh
            else:
                # for soft constraints we add -Bs
                Q[_remap_pair((v, v))] = -Bs

        logging.info(f"Converted to Choi Qubo with {len(node_list)} binary variables. Bh={config['hard_reward']},"
                     f" Bs={Bs}.")
        return {'Q': Q}, end_time_measurement(start)

    def reverse_map(self, solution: dict) -> (dict, float):
        """
        Maps the solution back to the representation needed by the SAT class for validation/evaluation.

        :param solution: dictionary containing the solution
        :type solution: dict
        :return: solution mapped accordingly, time it took to map it
        :rtype: tuple(dict, float)
        """
        start = start_time_measurement()
        # we define the literals list, so that we can check the self-consistency of the solution. That is, we save all
        # assignments proposed by the annealer, and see if there is no contradiction. (In principle a solver
        # could mandate L3 = True and L3 = False, resulting in a contradiction.)
        literals = []
        # assignments saves the actual solution
        assignments = []
        for node, tf in solution.items():
            # we check if node is included in the set (i.e. if tf is True (1))
            if tf:
                # convert back to the language of literals
                lit_str = self.reverse_dict[node]
                # we check if the literal is negated:
                if lit_str.startswith('~'):
                    # remove the negation symbol
                    lit_str = lit_str.replace('~', '')
                    # save a negated literal object, will be used for self-consistency check
                    lit = Var(lit_str).negate()
                    # add the negated literal to the assignments, removing the (irrelevant) position part
                    assignments.append(Var(lit_str.split('-')[0]).negate())
                else:
                    # if literal is true, no ~ symbol needs to be removed:
                    lit = Var(lit_str)
                    assignments.append(Var(lit_str.split('-')[0]))
                literals.append(lit)
        # we check for self-consistency of solution; we check that the assignments of all literals are consistent:
        if not And(set(literals)).satisfiable():
            logging.error('Generated solution is not self-consistent!')
            raise ValueError("Inconsistent solution for the ChoiQubo returned.")

        # If the solution is consistent, we have to find and add potentially missing variables:
        assignments = sorted(set(assignments))
        # find missing vars, or more precisely, their labels:
        missing_vars = set(range(self.nr_vars)) - {int(str(a).replace('L', '').replace('~', '')) for a in assignments}

        # add the variables that we found were missing:
        for nr in missing_vars:
            assignments.append(Var(f'L{nr}'))

        return {list(v.vars())[0]: v.true for v in sorted(assignments)}, end_time_measurement(start)

    def get_default_submodule(self, option: str) -> Core:

        if option == "Annealer":
            from modules.solvers.Annealer import Annealer  # pylint: disable=C0415
            return Annealer()
        else:
            raise NotImplementedError(f"Solver Option {option} not implemented")
