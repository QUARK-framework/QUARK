from typing import TypedDict, Union
import logging

import matplotlib.pyplot as plt
import numpy as np

from src.modules.applications.simulation.backends.aer_simulator import AerSimulator
from src.modules.applications.simulation.backends.backend_input import BackendInput
from src.modules.applications.simulation.backends.backend_result import BackendResult
from src.modules.applications.simulation.free_fermion.free_fermion_helpers import (
    create_circuit,
    exact_values_and_variance,
    computes_score_values,
    extract_simulation_results
)
from src.modules.applications.simulation.simulation import Simulation
from src.utils import start_time_measurement, end_time_measurement

logger = logging.getLogger()


class FreeFermion(Simulation):

    def __init__(self):
        """
        Initializes the FreeFermion class.
        """
        super().__init__("FreeFermion")
        self.submodule_options = ["AerSimulator"]

    @staticmethod
    def get_requirements() -> list:
        """
        Returns a list of requirements for the FreeFermion application.

        :returns: A list of dictionaries containing the name and version of required packages
        """
        return [
            {"name": "qiskit", "version": "1.3.0"},
            {"name": "numpy", "version": "1.26.4"},
            {"name": "matplotlib", "version": "3.9.3"},
            {"name": "qiskit_aer", "version": "0.15.1"},
            {"name": "scipy", "version": "1.12.0"},
        ]

    def get_default_submodule(self, option: str) -> Union[AerSimulator]:
        """
        Given an option string by the user, this returns a submodule.

        :param option: String with the chosen submodule
        :return: Module of type Core
        :raises NotImplementedError: If the option is not recognized
        """
        if option == "AerSimulator":
            return AerSimulator()
        else:
            raise NotImplementedError(f"Submodule Option {option} not implemented")

    def get_parameter_options(self):
        """
        Returns the parameter options for the application.
        """
        return {
            "Lx": {
                "values": [2, 4, 6],
                "description": "What lattice width Lx to use for the simulation? Must be even integer",
                "custom_input": True,
                "allow_ranges": False,
                "postproc": int
            },
            "Ly": {
                "values": [2, 4, 6],
                "description": "What lattice height Ly to use for the simulation? Must be even integer",
                "custom_input": True,
                "allow_ranges": False,
                "postproc": int
            },
            "trotter_dt": {
                "values": [0.2],
                "description": "Which time step size?",
                "custom_input": True,
                "allow_ranges": False,
                "postproc": float
            },
            "trotter_n_step": {
                "values": ["2*Ly"],
                "description": "Number of time steps (default is twice Ly value)? Provide total number of steps as integer if using custom value",
                "custom_input": True,
                "allow_ranges": False,
            },
        }

    class Config(TypedDict):
        """
        A configuration dictionary for the application.
        """
        Lx: int
        Ly: int
        trotter_dt: float
        trotter_n_step: str | int

    def preprocess(self, input_data: any, conf: Config, **kwargs) -> tuple[BackendInput, float]:
        """
        Generate data that gets passed to the next submodule.

        :param input_data: The input data for preprocessing
        :param conf: The configuration parameters
        :return: A tuple containing the preprocessed output and the time taken for preprocessing
        """
        start = start_time_measurement()
        lx = conf['Lx']
        ly = conf['Ly']
        if lx % 2 == 1:
            raise ValueError(f"Lx must be even. Provided Lx: {lx}")
        if ly % 2 == 1:
            raise ValueError(f"Ly must be even. Provided Ly: {ly}")
        trotter_n_step = conf['trotter_n_step']
        if isinstance(trotter_n_step, str):
            trotter_n_step = 2*ly
        trotter_dt = conf['trotter_dt']
        n_qubits = ly * lx * 3 // 2
        logger.info(
            f"Starting free fermion simulation benchmark on a {lx}x{ly} lattice ({n_qubits} qubits)")
        logger.info(f"Using a trotter step size of {trotter_dt} and up to {trotter_n_step} trotter steps")
        circuits = [create_circuit(lx, ly, trotter_dt, n) for n in range(trotter_n_step)]
        return BackendInput(circuits), end_time_measurement(start)

    def postprocess(self, input_data: BackendResult, conf: Config, **kwargs) -> tuple[any, float]:
        """
        Processes data passed to this module from the submodule.

        :param input_data: The input data for postprocessing
        :param conf: The configuration parameters
        :returns: A tuple containing the processed solution quality and the time taken for evaluation
        """

        start = start_time_measurement()
        counts_per_circuit, n_shots = input_data.counts, input_data.n_shots
        lx, ly, trotter_dt = conf['Lx'], conf['Ly'], conf['trotter_dt']
        trotter_n_step = conf['trotter_n_step']
        if isinstance(trotter_n_step, str):
            trotter_n_step = 2*ly

        simulation_results = np.array(extract_simulation_results(trotter_dt, lx, ly, n_shots, counts_per_circuit))
        exact_results = np.real(np.array(exact_values_and_variance(trotter_n_step, trotter_dt, lx, ly)))
        score_gate, score_shot, score_runtime = computes_score_values(exact_results[:, 1] - simulation_results[:, 1], simulation_results[:, 2],
                                                      exact_results[:, 2], lx * ly)
        logger.info(f"Benchmark score (number of gates): {score_gate}")
        logger.info(f"Benchmark score (number of shots): {score_shot}")
        logger.info(f"Benchmark score (number of trotter steps): {score_runtime}")
        self.metrics.add_metric_batch({
            "application_score_value": score_gate,
            "application_score_value_gates": score_gate,
            "application_score_value_shots": score_shot,
            "application_score_value_trotter_steps": score_runtime,
            "application_score_unit": "N_gates",
            "application_score_type": "int"
        })
        self.create_and_store_plot(trotter_n_step, trotter_dt, simulation_results, exact_results, score_gate, kwargs["store_dir"])
        return computes_score_values, end_time_measurement(start)

    @staticmethod
    def create_and_store_plot(n_trot: int, dt: float, simulation_results, exact_results, score_gates: int, store_dir) -> None:
        plt.plot(np.array(list(range(n_trot)))*dt, exact_results[:, 1], color="black", label="exact")
        plt.errorbar(simulation_results[:, 0], simulation_results[:, 1],
                     yerr=simulation_results[:, 2], label="simulated")
        plt.title("SCORE = " + str(score_gates) + " gates")
        plt.xlabel("Time")
        plt.ylabel("Imbalance")
        plt.legend()
        plt.savefig(f"{store_dir}/simulation_plot.pdf")
        plt.close()

    def save(self, path, iter_count) -> None:
        """
        This method is required to implement the application, but at the moment it does nothing
        """
        pass
