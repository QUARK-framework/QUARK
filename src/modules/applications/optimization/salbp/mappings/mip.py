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
from typing import TypedDict

from docplex.mp.dvar import Var
from docplex.mp.model import Model

from modules.applications.mapping import Mapping
from modules.core import Core
from modules.applications.optimization.salbp.salbp import Task, SALBPInstance
from utils import end_time_measurement, start_time_measurement


class MIP(Mapping):

    def __init__(self):
        """
        Constructor method
        """
        super().__init__()
        self.submodule_options = ["MIPSolver"]
        self.salbp = None
        self.config = None

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Return requirements of this module.

        :return: List of dict with requirements of this module
        """
        return [{"name": "docplex", "version": "2.25.236"}]

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this mapping.
        """
        return {}

    class Config(TypedDict):
        """
        Attributes of a valid config.
        """

    def map(self, salbp: SALBPInstance, config: Config) -> tuple[Model, float]:
        """
        Map the SALBP-1 instance to its MIP formulation.

        :param salbp: SALBP-1 instance
        :param config: Configuration for the mapping
        :return: A tuple containing the generated MIP model and the time taken to create it
        """
        self.salbp = salbp
        start = start_time_measurement()
        logging.info("Start the creation of the SALBP-1-MIP")

        max_num_stations: int = salbp.number_of_tasks
        # Create the docplex MIP model
        salbp_mip: Model = Model("SALBP")

        # Add variables
        station_vars, task_station_vars = self._add_variables(
            salbp, salbp_mip, max_num_stations
        )

        # Add constraints
        self._add_one_station_per_task_constraints(
            salbp_mip, salbp.tasks, max_num_stations, task_station_vars
        )
        self._add_cycle_time_constraints(
            salbp_mip, salbp, max_num_stations, task_station_vars, station_vars
        )
        self._add_preceding_tasks_constraints(
            salbp_mip, salbp, max_num_stations, task_station_vars
        )
        self._add_consecutive_stations_constraints(
            salbp_mip, max_num_stations, station_vars
        )

        # Add objective
        salbp_mip.minimize(salbp_mip.sum(y for y in station_vars))

        logging.info("Finished the creation of the SALBP-1-MIP")

        return salbp_mip, end_time_measurement(start)

    @staticmethod
    def _add_variables(
        salbp: SALBPInstance,
        salbp_mip: Model,
        max_num_stations: int,
    ) -> tuple[list[Var], dict[tuple[int, int], Var]]:
        """
        Add two types of binary variables to the SALBP-1 MIP:
        - one binary variable y_s per potential station s (y_s = 1 iff s is used)
        - one binary variable x_t_s per task-station pair (x_t_s = 1 iff t assigned to s)

        :param salbp: SALBP-1 instance
        :param salbp_mip: MIP model for the SALBP-1 instance
        :param max_num_stations: Maximum number of stations needed
        :return:
            - A list of binary variables representing station usage (y_s)
            - A dictionary of binary variables representing task assignments (x_t_s)
        """
        return (
            salbp_mip.binary_var_list(keys=range(max_num_stations), name="y"),
            salbp_mip.binary_var_matrix(
                keys1=(task.id for task in salbp.tasks),
                keys2=(station for station in range(max_num_stations)),
                name=(
                    f"x_{t.id}_{j}"
                    for t in salbp.tasks
                    for j in range(max_num_stations)
                ),
            ),
        )

    @staticmethod
    def _add_one_station_per_task_constraints(
        salbp_mip: Model,
        tasks: frozenset[Task],
        max_num_stations: int,
        task_station_vars: dict[tuple[int, int], Var],
    ) -> None:
        """
        Add constraints to ensure that each task is assigned to exactly one station.

        :param salbp_mip: MIP model for the SALBP-1 instance
        :param tasks: Tasks in SALBP-1 instance
        :param max_num_stations: Maximum number of stations needed
        :param task_station_vars: Binary variables for task-station pairs
        """
        salbp_mip.add_constraints(
            (
                salbp_mip.sum(
                    task_station_vars[t.id, s] for s in range(max_num_stations)
                )
                == 1
                for t in tasks
            ),
            names=(f"one_station_per_task_{t.id}" for t in tasks),
        )

    @staticmethod
    def _add_cycle_time_constraints(
        salbp_mip: Model,
        salbp: SALBPInstance,
        max_num_stations: int,
        task_station_vars: dict[tuple[int, int], Var],
        station_vars: list[Var],
    ) -> None:
        """
        Add constraints to ensure that no station exceeds the cycle time.

        :param salbp_mip: MIP model for the SALBP-1 instance
        :param salbp: SALBP-1 instance
        :param max_num_stations: Maximum number of stations needed
        :param task_station_vars: Binary variables for task-station pairs
        :param station_vars: Binary variables for stations
        """
        salbp_mip.add_constraints(
            (
                salbp_mip.sum(
                    t.time * task_station_vars[t.id, s] for t in salbp.tasks
                )
                <= station_vars[s] * salbp.cycle_time
                for s in range(max_num_stations)
            ),
            names=(f"station_{s}_cycle_time" for s in range(max_num_stations)),
        )

    @staticmethod
    def _add_preceding_tasks_constraints(
        salbp_mip: Model,
        salbp: SALBPInstance,
        max_num_stations: int,
        task_station_vars: dict[tuple[int, int], Var],
    ) -> None:
        """
        Add constraints to ensure that preceding tasks are done before their succeeding tasks.

        :param salbp_mip: MIP model for the SALBP-1 instance
        :param salbp: SALBP-1 instance
        :param max_num_stations: Maximum number of stations needed
        :param task_station_vars: Binary variables for task-station pairs
        """
        salbp_mip.add_constraints(
            (
                salbp_mip.sum(
                    s * task_station_vars[t1.id, s] for s in range(max_num_stations)
                )
                <= salbp_mip.sum(
                    s * task_station_vars[t2.id, s] for s in range(max_num_stations)
                )
                for (t1, t2) in salbp.preceding_tasks
            ),
            names=(
                f"task_{t1.id}_before_task_{t2.id}"
                for (t1, t2) in salbp.preceding_tasks
            ),
        )

    @staticmethod
    def _add_consecutive_stations_constraints(
        salbp_mip: Model, max_num_stations: int, station_vars: list[Var]
    ) -> None:
        """
        Add constraints to ensure that stations are used in consecutive order. This avoids empty stations in between.

        :param salbp_mip: MIP model for the SALBP-1 instance
        :param max_num_stations: Maximum number of stations needed
        :param station_vars: Binary variables for stations
        """
        salbp_mip.add_constraints(
            (
                station_vars[s] >= station_vars[s + 1]
                for s in range(max_num_stations - 1)
            ),
            names=(f"consecutive_stations_{s}_{s + 1}" for s in range(max_num_stations)),
        )

    def reverse_map(self, solution: dict[str, int]) -> tuple[dict[int, list[Task]], float]:
        """
        Map the solution of the MIP to a task assignment.

        :param solution: A dict mapping a variable name to its solution value
        :return: The task assignment and the time it took to create it
        """
        start = start_time_measurement()
        logging.info(
            "Start the reverse mapping of the MIP solution to a task assignment"
        )
        used_stations: list[int] = [
            int(var[2:])
            for var, val in solution.items()
            if val > 0 and var.startswith("y")
        ]
        task_assignment: dict[int, list[Task]] = {s: [] for s in used_stations}
        for var_name, value in solution.items():
            if var_name.startswith("x") and value > 0:
                task, station = var_name.split("_")[1:3]
                task_assignment[int(station)].append(self.salbp.get_task(int(task)))

        return task_assignment, end_time_measurement(start)

    def get_default_submodule(self, option: str) -> Core:
        """
        Returns the default submodule based on the provided option.

        :param option: The submodule option.
        :return: The corresponding submodule instance.
        :raises NotImplementedError: If the option is not recognized.
        """
        if option == "MIPSolver":
            # logging.info(f"Using the module {option} requires the installation of Microsoft Visual C++.")
            # logging.info("Please make sure you installed the latest version for your respective system.")
            from modules.solvers.mip_solver_bp import MIPSolver  # pylint: disable=C0415
            return MIPSolver()
        else:
            raise NotImplementedError(f"Solver Option {option} not implemented")
