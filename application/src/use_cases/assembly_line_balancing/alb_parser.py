"""
Parser for Assembly Line Balancing Benchmark Datasets by Otto et al. (2013)
URL: https://assembly-line-balancing.de/
"""
import itertools
from pathlib import Path

from application.src.use_cases.assembly_line_balancing.alb_factory import salbp_1_factory
from application.src.use_cases.assembly_line_balancing.alb_typing import Task, TaskId, SALBP1


def parse_number_of_tasks(number_task: str) -> int:
    """number n of tasks in this problem"""
    return int(number_task)


def parse_cycle_time(cycle_time: str) -> int:
    """Time that is available for one station"""
    return int(cycle_time)


def parse_order_strength(order_strength: str) -> float:
    """ " """
    return float(order_strength.replace(",", "."))


def parse_task(task_times: str) -> Task:
    """time that is needed for a task (TaskId, time)"""
    task_id, task_time = task_times.split(" ")
    return Task(int(task_id), int(task_time))


def parse_precedence_relation(relation: str) -> tuple[TaskId, ...]:
    """The precedence relation define constraints on the order in which
    tasks are performed. A priority relation of task i to task j means
    that task i must be completed before task j can be started.
    (Task i, Task j)"""
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
    """Read scenario files in .alb format.

    :param file_path: path to `.alb` file
    :return: list of lines given in data
    """
    with open(file=str(file_path), mode="r", encoding="utf-8") as alb_file:
        return list(
            filter(
                lambda s: s != "", list(map(lambda s: s.strip(), alb_file.readlines()))
            )
        )


def get_indices(lines: list[str], keywords: list[str]) -> dict[str, int]:
    """find the positions of the keywords in the list

    :param lines: list of lines
    :param keywords: keywords to look for
    :return: dictionary with keyword and their position in lines
    """
    return {keyword: lines.index(keyword) for keyword in keywords}


def split_lines_to_areas(
    lines: list[str], token_indices: dict[str, int]
) -> dict[str, list[str]]:
    """Group the list into keywords and their corresponding values.

    Each group is introduced by a keyword and ends when another keyword follows.

    :param lines: list of lines
    :param token_indices: keywords and their position in lines
    :return: dictionary with keyword and corresponding values
    """
    token_and_range = zip(
        token_indices.keys(), itertools.pairwise(token_indices.values())
    )
    return {t: lines[start + 1: stop] for t, (start, stop) in token_and_range}


def convert_preceding_task_ids_to_tasks(
    tasks: list[Task], preceding_task_ids: list[tuple[TaskId, TaskId]]
) -> list[tuple[Task, Task]]:
    """Convert a list of preceding task IDs into a list of preceding tasks."""
    return [
        (
            next(task for task in tasks if task.id == i),
            next(task for task in tasks if task.id == j),
        )
        for i, j in preceding_task_ids
    ]


def parse_content_to_salbp_1_problem(token_and_content: dict[str, list[str]]) -> SALBP1:
    """Create Assembly Line Balancing Problem from content.

    :param token_and_content: token and list of corresponding values
    :return: an SALBP-1 instance
    """

    content_parsed = {}
    for (
        k,
        v,
    ) in token_and_content.items():
        if parser := TOKEN_PARSER_DISPATCHER.get(k):
            content_parsed[k] = [parser(vii) for vii in v]

    return salbp_1_factory(
        tasks=content_parsed["<task times>"],
        precedence_relations=convert_preceding_task_ids_to_tasks(
            content_parsed["<task times>"], content_parsed["<precedence relations>"]
        ),
        cycle_time=content_parsed["<cycle time>"][0],
    )
