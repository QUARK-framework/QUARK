import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from qiskit_optimization import QuadraticProgram

from docplex.mp.model import Model
from modules.applications.optimization.bp.bp import BP
from modules.applications.optimization.bp.mappings.mip import MIP
from modules.applications.optimization.bp.mappings.ising import Ising
from modules.applications.optimization.bp.mappings.qubo import QUBO


class TestBP(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.bp_instance = BP()

    def setUp(self):
        self.bp_instance.generate_problem({
            "number_of_objects": 5,
            "instance_creating_mode": "linear weights without incompatibilities"
        })

    @classmethod
    def tearDownClass(cls):
        del cls.bp_instance

    def test_initialization(self):
        self.assertEqual(self.bp_instance.name, "BinPacking")
        self.assertEqual(self.bp_instance.submodule_options, ["MIP", "Ising", "QUBO"])

    def test_get_requirements(self):
        requirements = self.bp_instance.get_requirements()
        expected_requirements = [
            {"name": "numpy", "version": "1.26.4"},
            {"name": "qiskit_optimization", "version": "0.6.1"},
            {"name": "docplex", "version": "2.25.236"}
        ]
        self.assertEqual(requirements, expected_requirements)

    def test_get_solution_quality_unit(self):
        self.assertEqual(self.bp_instance.get_solution_quality_unit(), "number_of_bins")

    def test_get_parameter_options(self):
        params = self.bp_instance.get_parameter_options()
        self.assertIn("number_of_objects", params)
        self.assertIn("instance_creating_mode", params)
        self.assertIsInstance(params["number_of_objects"]["values"], list)
        self.assertIsInstance(params["instance_creating_mode"]["values"], list)

    def test_create_bin_packing_instance(self):
        object_weights, bin_capacity, incompatible_objects = self.bp_instance.create_bin_packing_instance(
            number_of_objects=5, mode="linear weights without incompatibilities"
        )

        self.assertEqual(bin_capacity, 5)
        self.assertEqual(len(object_weights), 5)
        self.assertEqual(incompatible_objects, [])

    def test_generate_problem(self):
        config = {
            "number_of_objects": 5,
            "instance_creating_mode": "linear weights without incompatibilities"
        }
        object_weights, bin_capacity, incompatible_objects = self.bp_instance.generate_problem(config)

        self.assertEqual(len(object_weights), 5)
        self.assertEqual(bin_capacity, 5)
        self.assertEqual(incompatible_objects, [])

    def test_validate_invalid_solution(self):
        config_summary = {"mapping": "MIP"}
        solution = None  # Invalid solution case

        validity, _ = self.bp_instance.validate(solution, configuration_summary=config_summary)
        self.assertFalse(validity)

    def test_validate_valid_solution(self):
        config_summary = {"mapping": "MIP"}
        solution = {f"x_{i}": 1 for i in range(30)}

        validity, _ = self.bp_instance.validate(solution, configuration_summary=config_summary)
        self.assertIsInstance(validity, bool)

    def test_evaluate_solution(self):
        config_summary = {"mapping": "MIP"}
        solution = {f"x_{i}": 1 for i in range(5)}

        self.bp_instance.mip_qiskit = MagicMock()
        self.bp_instance.mip_qiskit.objective.evaluate.return_value = 10.0

        obj_value, _ = self.bp_instance.evaluate(solution, configuration_summary=config_summary)
        self.assertIsInstance(obj_value, (float, int))

    def test_create_MIP(self):
        problem = ([2, 4, 6], 10, [])
        model = MIP.create_MIP(self,problem)

        self.assertIsInstance(model, Model)
        self.assertTrue(model.get_objective_expr() is not None)

    def test_transform_docplex_mip_to_qubo(self):
        model = Model()
        model.binary_var(name="x1")
        model.binary_var(name="x2")

        with patch("modules.applications.optimization.bp.bp.from_docplex_mp", return_value=QuadraticProgram()):
            qubo_operator, qubo = QUBO.transform_docplex_mip_to_qubo(model, penalty_factor=1.0)

        self.assertIsInstance(qubo_operator, dict)
        self.assertIsInstance(qubo, QuadraticProgram)

    def test_transform_docplex_mip_to_ising(self):
        model = Model()
        model.binary_var(name="x1")
        model.binary_var(name="x2")

        with patch("modules.applications.optimization.bp.bp.from_docplex_mp", return_value=QuadraticProgram()):
            ising_matrix, ising_vector, ising_offset, qubo = Ising.transform_docplex_mip_to_ising(model, penalty_factor=1.0)
        ising_offset = float(ising_offset)

        self.assertIsInstance(ising_matrix, np.ndarray)
        self.assertIsInstance(ising_vector, np.ndarray)
        self.assertIsInstance(ising_offset, float)
        self.assertIsInstance(qubo, QuadraticProgram)
