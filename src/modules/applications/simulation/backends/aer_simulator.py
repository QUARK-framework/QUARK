from typing import TypedDict

from qiskit_aer import AerSimulator as QiskitAS

from src.modules.applications.simulation.backends.backend_input import BackendInput
from src.modules.core import Core
from src.modules.applications.simulation.backends.backend_result import BackendResult
from src.utils import start_time_measurement, end_time_measurement


class AerSimulator(Core):

    def __init__(self):
        """
        Initializes the AerSimulator class.
        """
        super().__init__("AerSimulator")
        self.submodule_options = []

    @staticmethod
    def get_requirements() -> list:
        """
        Returns a list of requirements for the FreeFermion application.

        :returns: A list of dictionaries containing the name and version of required packages
        """
        return [
            {"name": "qiskit", "version": "1.3.0"},
        ]

    def get_default_submodule(self, option: str) -> None:
        """
        Given an option string by the user, this returns a submodule.

        :param option: String with the chosen submodule
        :return: Module of type Core
        :raises NotImplementedError: If the option is not recognized
        """
        raise NotImplementedError(f"Submodule Option {option} not implemented")

    def get_parameter_options(self):
        """
        Returns the parameter options for the application.
        """
        return {
            "n_shots": {
                "values": [100, 200, 400, 800, 1600],
                "description": "Number of shots?",
                "allow_ranges": False,
                "postproc": int
            }
        }

    class AerSimConfig(TypedDict):
        """
        """
        n_shots: int

    def postprocess(self, input_data: BackendInput, config: AerSimConfig, **kwargs) -> tuple[any, float]:
        """
        Processes data passed to this module from the submodule.

        :param input_data: The input data for postprocessing
        :param config: The configuration dictionary
        :param **kwargs: Additional keyword arguments
        :returns: A tuple containing the processed solution quality and the time taken for evaluation
        """
        start = start_time_measurement()
        backend = QiskitAS()
        circuits = input_data.circuits
        counts = [backend.run(circuit, shots=config['n_shots']).result().get_counts(circuit) for circuit in circuits]
        results = BackendResult(
            counts=counts,
            n_shots=config['n_shots']
        )
        return results, end_time_measurement(start)
