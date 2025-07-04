from typing import TypedDict, Union

import matplotlib.pyplot as plt
import numpy as np

from src.modules.applications.simulation.backends.aer_simulator import AerSimulator
from src.modules.applications.simulation.backends.backend_input import BackendInput
from src.modules.applications.simulation.backends.backend_result import BackendResult
from src.modules.applications.simulation.free_fermion.free_fermion_helpers import create_circuit, exact_values, \
    score_minimal_mean
from src.modules.applications.simulation.simulation import Simulation
from src.utils import start_time_measurement, end_time_measurement


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
            "L": {
                "values": [2, 4, 6],
                "description": "What lattice size (L x L) to use for the simulation?",
                "allow_ranges": False,
                "postproc": int
            },
            "trotter_dt": {
                "values": [0.2, 0.4, 0.6],
                "description": "Which time step size?",
                "allow_ranges": False,
                "postproc": float
            },
            "trotter_n_step": {
                "values": ["1L", "2L"],
                "description": "Number of time steps (in multiples of L value)?",
                "allow_ranges": False,
                "postproc": str
            },
        }


    class Config(TypedDict):
        """
        A configuration dictionary for the application.
        """
        L: int
        trotter_dt: float
        trotter_n_step: str


    def preprocess(self, input_data: any, conf: Config, **kwargs) -> tuple[BackendInput, float]:
        """
        Generate data that gets passed to the next submodule.

        :param input_data: The input data for preprocessing
        :param conf: The configuration parameters
        :return: A tuple containing the preprocessed output and the time taken for preprocessing
        """
        start = start_time_measurement()
        lattice_size = conf['L']
        trotter_n_step = int(conf['trotter_n_step'][:-1])*lattice_size
        circuits = [create_circuit(lattice_size, lattice_size, conf['trotter_dt'], n) for n in range(trotter_n_step)]
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
        lx, ly, trotter_dt  = conf['L'], conf['L'], conf['trotter_dt']
        trotter_n_step = int(conf['trotter_n_step'][:-1])*lx
        l_tot = lx * ly

        to_plot: list = []
        for n in range(trotter_n_step):
            res: float = 0
            var: float = 0
            counts = counts_per_circuit[n]
            for s in counts:
                a: float = 0
                for j in range(l_tot // 2):
                    if s[l_tot * 3 // 2 - 1 - j] == '1':
                        a += -1 / l_tot
                    else:
                        a += 1 / l_tot
                    if s[l_tot * 3 // 2 - 1 - j - l_tot // 2] == '1':
                        a += 1 / l_tot
                    else:
                        a += -1 / l_tot
                res += a * counts[s]
                var += a ** 2 * counts[s]
            res = res / n_shots
            var = var / n_shots
            to_plot.append([n, res, np.sqrt(var - res ** 2) / np.sqrt(n_shots)])

        exact_list_array = np.real(np.array(exact_values(trotter_n_step, trotter_dt, lx, ly)))
        to_plot_array: np.array = np.array(to_plot)
        score, score_variance = score_minimal_mean(exact_list_array[:, 1] - to_plot_array[:, 1], to_plot_array[:, 2], lx * ly)
        self.metrics.add_metric_batch({ "application_score_value": score, "application_score_variance": score_variance, "application_score_unit": "score",
                                       "application_score_type": "float"})
        plt.plot(np.array(list(range(len(exact_list_array[:,1])))), exact_list_array[:, 1], color="black", label="exact")
        plt.errorbar(to_plot_array[:, 0], to_plot_array[:, 1], yerr=to_plot_array[:, 2], label="simulated")
        plt.title("score=10^" + str(score) + " gates")
        plt.xlabel("Trotter step")
        plt.ylabel("Imbalance")
        plt.legend()
        store_dir = kwargs["store_dir"]
        plt.savefig(f"{store_dir}/simulation_plot.pdf")
        plt.close()

        return score, end_time_measurement(start)


    def save(self, path, iter_count) -> None:
        """
        Saves the application state.

        :param path: The path where the application state should be saved
        :param iter_count: The iteration count
        :returns:None
        """
        pass
