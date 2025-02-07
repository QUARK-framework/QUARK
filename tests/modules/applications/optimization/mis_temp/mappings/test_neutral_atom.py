import unittest
import pickle

from src.modules.applications.optimization.mis.mappings.neutral_atom import NeutralAtom


class TestNeutralAtom(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.neutral_atom_instance = NeutralAtom()
        with open("tests/modules/applications/optimization/MIS/mappings/MIS_test_graph.pkl", "rb") as file:
            cls.graph = pickle.load(file)
            cls.config = {}

    def test_get_requirements(self):
        requirements = self.neutral_atom_instance.get_requirements()
        expected_requirements = [{"name": "pulser", "version": "1.1.1"}]
        for req in expected_requirements:
            self.assertIn(req, requirements)

    def test_get_parameter_options(self):
        options = self.neutral_atom_instance.get_parameter_options()
        self.assertEqual(options, {}, "Expected parameter options to be an empty dictionary.")

    def test_map(self):
        neutral_atom_problem, mapping_time = self.neutral_atom_instance.map(self.graph, self.config)

        self.assertIn("graph", neutral_atom_problem)
        self.assertIn("register", neutral_atom_problem)

        self.assertIsNotNone(neutral_atom_problem["register"], "Expected a valid Pulser register.")
        self.assertGreater(mapping_time, 0, "Mapping time should be positive.")

    def test_get_default_submodule(self):
        # Test valid submodule retrieval
        submodule = self.neutral_atom_instance.get_default_submodule("NeutralAtomMIS")
        self.assertIsNotNone(submodule, "Expected 'NeutralAtomMIS' submodule to be returned.")

        # Test invalid submodule option
        with self.assertRaises(NotImplementedError):
            self.neutral_atom_instance.get_default_submodule("InvalidSubmodule")
