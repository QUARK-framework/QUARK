from typing import TypedDict
import logging

from qiskit_aer import AerSimulator as QiskitAS
from qiskit import QuantumCircuit

from src.modules.applications.simulation.backends.backend_input import BackendInput
from src.modules.core import Core
from src.modules.applications.simulation.backends.backend_result import BackendResult
from src.utils import start_time_measurement, end_time_measurement

logger = logging.getLogger()
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
        self.warn_on_large_circuits(circuits)

        counts_per_circuit = []
        for n, circuit in enumerate(circuits):
            logger.info(f"Running circuit for {n} Trotter steps on AerSimulator")
            counts_per_circuit.append(backend.run(circuit, shots=config['n_shots']).result().get_counts(circuit))

        results = BackendResult(
            counts=counts_per_circuit,
            n_shots=config['n_shots']
        )
        return results, end_time_measurement(start)


    @staticmethod
    def warn_on_large_circuits(circuits: list[QuantumCircuit]) -> None:
        warning_n_qubits = 30
        max_n_qubit = max([circuit.num_qubits for circuit in circuits])
        if max_n_qubit > warning_n_qubits:
            logger.warning(f"Simulating circuits with over {warning_n_qubits} qubits. The high memory"
                           f" requirements can lead to memory errors on some systems.")

