"""A factory creating an instance of SALBP-1"""
from application.src.use_cases.assembly_line_balancing.alb_typing import TaskId, Task, SALBP1


def check_unique_tasks_ids(tasks: list[Task]) -> None:
    """Check if all registered tasks have unique IDs."""
    task_ids: list[TaskId] = [task.id for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(f"Some tasks have the same taskId ({tasks})")


def check_non_negativity(tasks: list[Task]) -> None:
    """Check whether all task times are nonnegative."""
    for task in tasks:
        if task.time < 0:
            raise ValueError(f"{task} has a negative task time")


def check_valid_precedence_relations(
    tasks: frozenset[Task], preceding_tasks: list[tuple[Task, Task]]
) -> None:
    """Check if all tasks with precedence relations are registered.

    :param tasks: a set of tasks
    :param preceding_tasks: a list of tuples of tasks that have a precedence relation
    """
    for task_1, task_2 in preceding_tasks:
        if task_1 not in tasks or task_2 not in tasks:
            raise ValueError(f"Preceding Task {task_1} or {task_2} is not registered.")


def salbp_1_factory(
    tasks: list[Task], precedence_relations: list[tuple[Task, Task]], cycle_time: int
) -> SALBP1:
    """creates an SALBP-1 instance given a list of tasks and their precedence relations

    :param tasks: the tasks to be assigned to stations
    :param precedence_relations: the tasks' precedence relations
    :param cycle_time: the cycle time of a station (the same for every station)
    :return: an instance of a simple assembly line balancing problem
    """
    check_unique_tasks_ids(tasks=tasks)
    check_non_negativity(tasks=tasks)
    task_set = frozenset(tasks)
    check_valid_precedence_relations(
        tasks=task_set, preceding_tasks=precedence_relations
    )

    return SALBP1(
        cycle_time=cycle_time,
        tasks=task_set,
        preceding_tasks=frozenset(precedence_relations),
    )
