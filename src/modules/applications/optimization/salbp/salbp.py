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
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple, TypedDict

import networkx as nx

from modules.applications.application import Application
from modules.applications.optimization.optimization import Optimization
from modules.core import Core
from utils import end_time_measurement, start_time_measurement


# --- TYPINGS ---
TaskId = int
StationId = int


class Task(NamedTuple):
    """
    A Task for an Assembly Line Balancing Problem.

    Attributes:
        id: ID of this task
        time: Time that is needed to complete this task
    """

    id: TaskId
    time: int


TaskAssignment = dict[StationId, list[Task]]


@dataclass
class SALBPInstance:
    """
    An instance of the Simple Assembly Line Balancing Problem, version 1 (SALBP-1).

    Attributes:
        cycle_time: Time that is available for a station
        tasks: Tasks in this problem
        preceding_tasks: Known tasks' precedence relations
    """

    cycle_time: int
    tasks: frozenset[Task]
    preceding_tasks: frozenset[tuple[Task, Task]] = field(default_factory=frozenset)

    @property
    def number_of_tasks(self) -> int:
        """
        Return number of tasks.

        :return: The total number of tasks
        """
        return len(self.tasks)

    def get_task(self, task_id: TaskId) -> Task:
        """
        Get task for given task_id.

        :param task_id: The ID of the task to retrieve
        :return: The corresponding Task object
        """
        return next(task for task in self.tasks if task.id == task_id)


# --- FACTORY FUNCTION ---
def salbp_factory(
        tasks: list[Task], precedence_relations: list[tuple[Task, Task]], cycle_time: int) -> SALBPInstance:
    """
    Create an SALBP-1 instance given a list of tasks and their precedence relations.
    Do validity checking on the input data and raise a ValueError if the data is invalid.

    :param tasks: The tasks to be assigned to stations
    :param precedence_relations: The tasks' precedence relations
    :param cycle_time: The cycle time of a station (the same for every station)
    :return: An instance of the SALBP-1
    """
    if len(tasks) == 0:
        raise ValueError("No tasks registered! Trivial instance (no stations needed).")

    task_ids: list[TaskId] = [task.id for task in tasks]
    if not len(task_ids) == len(set(task_ids)):
        raise ValueError(f"Some tasks have the same taskId ({tasks})")

    if not all(x >= 0 for x in [task.time for task in tasks]):
        raise ValueError(f"Some tasks have a negative task time ({tasks})")

    if not all(task.time <= cycle_time for task in tasks):
        raise ValueError(f"Cycle time ({cycle_time}) is too short for some tasks!.")

    task_set = frozenset(tasks)
    if not all(t1 in task_set and t2 in task_set for (t1, t2) in precedence_relations):
        raise ValueError(
            f"Preceding tasks ({precedence_relations}) do not match registered tasks ({tasks})."
        )

    precedence_graph = nx.DiGraph(precedence_relations)
    if not nx.is_directed_acyclic_graph(precedence_graph):
        raise ValueError("Precedence graph contains cycles!")

    return SALBPInstance(
        cycle_time=cycle_time,
        tasks=task_set,
        preceding_tasks=frozenset(precedence_relations),
    )


# --- VALIDITY CHECKS FOR TASK ASSIGNMENT ---
def has_overloaded_station(cycle_time: int, task_assignment: TaskAssignment) -> bool:
    """
    Return if a station in the given task_assignment is overloaded wrt the given cycle time.

    :param cycle_time: The maximum time a station can take
    :param task_assignment: The assignment of tasks to stations
    :return: True if at least one station is overloaded, False otherwise
    """
    return any(
        sum(task.time for task in tasks) > cycle_time
        for tasks in task_assignment.values()
    )


def has_unique_assignment_for_every_task(
    tasks: frozenset[Task], task_assignment: TaskAssignment
) -> bool:
    """
    Return if each task is assigned to exactly one station.

    :param tasks: Set of all tasks in the SALBP-1 instance
    :param task_assignment: The assignment of tasks to stations
    :return: True if all tasks are uniquely assigned, False otherwise
    """
    tasks_in_solution = [task for tasks in task_assignment.values() for task in tasks]
    number_of_tasks_correct = len(tasks_in_solution) == len(tasks)
    only_unique_tasks = len(set(tasks_in_solution)) == len(tasks_in_solution)
    return number_of_tasks_correct and only_unique_tasks


def respects_precedences(
    preceding_tasks: frozenset[tuple[Task, Task]], task_assignment: TaskAssignment
) -> bool:
    """
    Return if the given task_assignment respects the given precedences.

    :param preceding_tasks: Set of precedence constraints between tasks
    :param task_assignment: The assignment of tasks to stations
    :return: True if all precedence constraints are satisfied, False otherwise
    """
    task_to_station_assignment = {
        task: station_id
        for station_id, tasks in task_assignment.items()
        for task in tasks
    }
    for task_1, task_2 in preceding_tasks:
        if task_to_station_assignment[task_1] > task_to_station_assignment[task_2]:
            return False
    return True


# --- PARSING HELPER FUNCTIONS ---
# Parser for Assembly Line Balancing Benchmark Datasets by Otto et al. (2013)
# URL: https://assembly-line-balancing.de/

def parse_number_of_tasks(number_task: str) -> int:
    """
    Parse the number n of tasks in this problem.

    :param number_task: The number of tasks as a string
    :return: The number of tasks as an integer
    """
    return int(number_task)


def parse_cycle_time(cycle_time: str) -> int:
    """
    Parse the available cycle time for one station.

    :param cycle_time: The cycle time as a string
    :return: The cycle time as an integer
    """
    return int(cycle_time)


def parse_order_strength(order_strength: str) -> float:
    """
    Parse the order strength of the precedence graph.

    :param order_strength: The order strength as a string
    :return: The order strength as a float
    """
    return float(order_strength.replace(",", "."))


def parse_task(task_times: str) -> Task:
    """
    Parse a task and its time requirement.

    :param task_times: A string containing the task ID and time
    :return: A Task instance
    """
    task_id, task_time = task_times.split(" ")
    return Task(int(task_id), int(task_time))


def parse_precedence_relation(relation: str) -> tuple[TaskId, ...]:
    """
    The precedence relations define constraints on the order in which
    tasks are performed. A priority relation of task i to task j means
    that task i must be completed before task j can be started.
    (Task i, Task j).

    :param relation: A string containing task IDs that define precedence constraints
    :return: A tuple of task IDs representing precedence constraints
    """
    return tuple(int(task) for task in relation.split(","))


TOKEN_PARSER_DISPATCHER = {
    "<number of tasks>": parse_number_of_tasks,
    "<cycle time>": parse_cycle_time,
    "<order strength>": parse_order_strength,
    "<task times>": parse_task,
    "<precedence relations>": parse_precedence_relation,
    "<end>": None,
}


def read_data(file_path: Path) -> list[str]:
    """
    Read scenario files in .alb format.

    :param file_path: Path to `.alb` file
    :return: List of lines given in data
    """
    with open(file=str(file_path), mode="r", encoding="utf-8") as alb_file:
        return list(
            filter(
                lambda s: s != "", list(map(lambda s: s.strip(), alb_file.readlines()))
            )
        )


def get_indices(lines: list[str], keywords: list[str]) -> dict[str, int]:
    """
    Find the positions of the keywords in the list.

    :param lines: List of lines
    :param keywords: Keywords to look for
    :return: Dictionary with keyword and their position in lines
    """
    return {keyword: lines.index(keyword) for keyword in keywords}


def split_lines_to_areas(
    lines: list[str], token_indices: dict[str, int]
) -> dict[str, list[str]]:
    """
    Group the list into keywords and their corresponding values.

    Each group is introduced by a keyword and ends when another keyword follows.

    :param lines: List of lines
    :param token_indices: Keywords and their position in lines
    :return: Dictionary with keyword and corresponding values
    """
    token_and_range = zip(
        token_indices.keys(), itertools.pairwise(token_indices.values())
    )
    return {t: lines[start + 1: stop] for t, (start, stop) in token_and_range}


def convert_preceding_task_ids_to_tasks(
    tasks: list[Task], preceding_task_ids: list[tuple[TaskId, TaskId]]
) -> list[tuple[Task, Task]]:
    """
    Convert a list of preceding task IDs into a list of preceding tasks.

    :param tasks: List of Task objects
    :param preceding_task_ids: List of tuples representing task precedence constraints
    :return: List of tuples representing task precedence constraints with Task objects
    """
    return [
        (
            next(task for task in tasks if task.id == i),
            next(task for task in tasks if task.id == j),
        )
        for i, j in preceding_task_ids
    ]


def create_salbp_from_file(file_path: Path) -> SALBPInstance:
    """
    Read data from a file and create an SALBP-1 instance.

    :param file_path: Path to the `.alb` file
    :return: An instance of SALBP-1 created from the input file
    """
    file_content = read_data(file_path)
    token_to_index = get_indices(file_content, list(TOKEN_PARSER_DISPATCHER.keys()))
    content = split_lines_to_areas(file_content, token_to_index)
    content_parsed = {}
    for (
            k,
            v,
    ) in content.items():
        if parser := TOKEN_PARSER_DISPATCHER.get(k):
            content_parsed[k] = [parser(vii) for vii in v]

    return salbp_factory(
        tasks=content_parsed["<task times>"],
        precedence_relations=convert_preceding_task_ids_to_tasks(
            content_parsed["<task times>"], content_parsed["<precedence relations>"]
        ),
        cycle_time=content_parsed["<cycle time>"][0],
    )


# --- QUARK OPTIMIZATION APPLICATION ---
class SALBP(Optimization):
    """
    The Simple Assembly Line Balancing Problem (SALBP) is a special bin packing problem with precedence relations among
    the items. Given a set of tasks, each with a processing time, precedence relations among those tasks, and a cycle
    time, this problem's goal is to assign every task to a station s.t. the total number of stations is minimized while
    the cycle time per station and the task precedences are respected. This version of the SALBP is commonly referred to
    as version 1 (SALBP-1) in the literature.

    The SALBP-1 finds applications in manufacturing, logistics, and industrial automation, where optimizing assembly line
    operations can significantly enhance efficiency and cost-effectiveness. By balancing workloads across stations,
    industries can minimize idle time, reduce bottlenecks, and improve overall productivity.

    This problem is especially relevant in automobile production, electronics assembly, and large-scale manufacturing,
    where tasks must follow strict precedence relations, meaning some tasks must be completed before others can begin.
    """

    def __init__(self):
        """
        Constructor method
        """
        super().__init__("SALBP")
        self.salbp = None
        self.task_assignment = None
        self.submodule_options = ["MIP"]

    @staticmethod
    def get_requirements() -> list:
        return [
            {"name": "docplex", "version": "2.25.236"},
            {"name": "networkx", "version": "3.4.2"},
        ]

    def get_solution_quality_unit(self) -> str:
        """
        Return the measurement unit for solution quality.

        :return: The unit as a string
        """
        return "Number of used stations"

    def get_default_submodule(self, option: str) -> Core:
        """
        Returns the default submodule based on the provided option.

        :param option: The submodule option.
        :return: The corresponding submodule instance.
        :raises NotImplementedError: If the option is not recognized.
        """
        if option == "MIP":
            from modules.applications.optimization.salbp.mappings.mip import MIP  # pylint: disable=C0415
            return MIP()
        raise NotImplementedError(f"Mapping Option {option} not implemented")

    def get_parameter_options(self) -> dict:
        """
        Return the configurable settings for this application.

        :return: Dictionary containing parameter options
        .. code-block:: python

            return {
                "instance": {
                    "values": list(
                        [
                            "example_instance_n=3.alb",
                            "example_instance_n=5.alb",
                            "example_instance_n=10.alb",
                            "example_instance_n=20.alb",
                        ]
                    ),
                    "description": "Which SALBP-1 instance do you want to solve?",
                },
            }
        """
        return {
            "instance": {
                "values": list(
                    [
                        "example_instance_n=3.alb",
                        "example_instance_n=5.alb",
                        "example_instance_n=10.alb",
                        "example_instance_n=20.alb",
                    ]
                ),
                "description": "Which SALBP-1 instance do you want to solve?",
            },
        }

    class Config(TypedDict):
        """
        Attributes of a valid config.
        """

        instance: str

    def generate_problem(self, config: Config, **kwargs) -> SALBPInstance:
        """
        Generate an SALBP-1 instance using the input configuration.

        :param config: Configuration settings
        :return: Generated SALBP-1 instance
        """
        if config is None:
            config = {
                "instance": "example_instance_n=3.alb",
            }

        try:
            salbp = create_salbp_from_file(
                file_path=Path(__file__).parent / "data" / config["instance"]
            )
            self.salbp = salbp
        except ValueError as err:
            raise err

        return self.salbp

    def validate(self, solution: dict[int, list[Task]]) -> tuple[bool, float]:
        """
        Validate if a given solution is feasible for the problem instance.

        :param solution: The task assignment
        :return: Whether the solution is valid, time it took to validate
        """
        start = start_time_measurement()

        if solution is None:
            return False, end_time_measurement(start)
        self.task_assignment = solution

        return (
            not has_overloaded_station(self.salbp.cycle_time, self.task_assignment)
            and has_unique_assignment_for_every_task(self.salbp.tasks, self.task_assignment)
            and respects_precedences(self.salbp.preceding_tasks, self.task_assignment)
        ), end_time_measurement(start)

    def evaluate(self, solution: dict[int, list[Task]]) -> tuple[float, float]:
        """
        Determine objective value of the solution, i.e., the number of used stations.

        :param solution: The task assignment
        :return: Objective value, time it took to evaluate
        """
        start = start_time_measurement()

        if solution is None:
            return False, end_time_measurement(start)
        obj_value = len([s for s in self.task_assignment.values() if len(s) > 0])

        return obj_value, end_time_measurement(start)

    def save(self, path: str, iter_count: int) -> None:
        pass
