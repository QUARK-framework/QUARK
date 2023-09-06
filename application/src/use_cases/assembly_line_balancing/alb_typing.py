"""Define typing for an Assembly Line Balancing Problem"""
import logging.config
from typing import NamedTuple
from dataclasses import dataclass, field

from application.src.utils import get_logging_path

TaskId = int
StationId = int

logging.config.fileConfig(get_logging_path())
logger = logging.getLogger("salbp1")


class Task(NamedTuple):
    """A Task for an Assembly Line Balancing Problem.

    Attributes:
        id: ID of this task
        time: Time that is needed to complete this task
    """

    id: TaskId
    time: int


@dataclass(kw_only=True)
class SALBP1:
    """An instance of a Simple Assembly Line Balancing Problem Version 1 (SALBP-1)

    Attributes:
        cycle_time: time that is available for a station
        tasks: tasks in this problem
        preceding_tasks: known tasks' precedence relations
        task_assignment: a mapping of stations to the assigned tasks
    """

    cycle_time: int
    tasks: frozenset[Task]
    preceding_tasks: frozenset[tuple[Task, Task]] = field(default_factory=frozenset)
    task_assignment: dict[StationId, set[Task]] = field(default_factory=dict)

    @property
    def task_ids(self) -> set[int]:
        """Return all Task IDs."""
        return set(task.id for task in self.tasks)

    @property
    def number_of_tasks(self) -> int:
        """Return number of tasks."""
        return len(self.tasks)

    def get_task(self, task_id: TaskId) -> Task:
        """Get task for given task_id."""
        return next(task for task in self.tasks if task.id == task_id)

    @property
    def has_valid_assignment(self) -> bool:
        """Check if current task_assignment is feasible."""
        return (
            not self.a_station_is_overloaded
            and self.tasks_distribution_is_valid
            and self.preceding_tasks_can_be_done
        )

    @property
    def a_station_is_overloaded(self) -> bool:
        """A station is overloaded if time needed for all tasks in station
        is larger than given cycle time."""
        for station, tasks in self.task_assignment.items():
            if sum(task.time for task in tasks) > self.cycle_time:
                logger.warning(f"Station {station} is overloaded.")
                return True
        return False

    @property
    def tasks_distribution_is_valid(self) -> bool:
        """Each task should be assigned to exactly one station."""
        tasks_in_solution = [
            task for tasks in self.task_assignment.values() for task in tasks
        ]
        number_of_tasks_correct = len(tasks_in_solution) == len(self.tasks)
        only_unique_tasks = len(set(tasks_in_solution)) == len(tasks_in_solution)
        return number_of_tasks_correct and only_unique_tasks

    @property
    def preceding_tasks_can_be_done(self) -> bool:
        """If there is a predecessor relation between two tasks, the preceding
        task must not be scheduled on the following station."""
        for task_1, task_2 in self.preceding_tasks:
            station_task_1 = next(
                station
                for station, tasks in self.task_assignment.items()
                for task in tasks
                if task_1 == task
            )
            station_task_2 = next(
                station
                for station, tasks in self.task_assignment.items()
                for task in tasks
                if task_2 == task
            )
            if station_task_1 > station_task_2:
                logger.warning(
                    f"Task {task_1} is distributed on station after task {task_2}."
                )
                return False
        return True

    def print_distribution(self) -> None:
        """Print task distribution per station."""
        logger.info("Distribution:")
        empty_station = []
        for station_id, tasks in self.task_assignment.items():
            task_ids = [task.id for task in tasks]
            if task_ids:
                logger.info(f"\tStation {station_id} -> {task_ids}")
            else:
                empty_station.append(station_id)
        if empty_station:
            logger.info(f"\tEmpty stations: {empty_station}")
