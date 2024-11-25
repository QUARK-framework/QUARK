import unittest
import numpy as np
import math

from modules.applications.qml.generative_modeling.metrics.MetricsGeneralization import MetricsGeneralization


class TestMetricsGeneralization(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.n_qubits = 3
        cls.n_states = 2 ** cls.n_qubits
        cls.train_size = 0.5
        cls.train_set = np.array([0, 1, 2])
        cls.solution_set = np.array([3, 4, 5, 6, 7])

        cls.metrics_instance = MetricsGeneralization(
            train_set=cls.train_set,
            train_size=cls.train_size,
            solution_set=cls.solution_set,
            n_qubits=cls.n_qubits
        )

    def test_initialization(self):
        self.assertEqual(self.metrics_instance.train_size, self.train_size)
        self.assertEqual(self.metrics_instance.n_states, self.n_states)
        self.assertEqual(self.metrics_instance.n_shots, 10000)
        np.testing.assert_array_equal(self.metrics_instance.train_set, self.train_set)
        np.testing.assert_array_equal(self.metrics_instance.solution_set, self.solution_set)

    def test_get_masks(self):
        mask_new, mask_sol = self.metrics_instance.get_masks()

        # Verify `mask_new` excludes training indices
        self.assertFalse(mask_new[self.train_set].any())

        # Verify `mask_sol` includes solution indices but excludes training indices
        self.assertTrue(mask_sol[self.solution_set].all())
        self.assertFalse(mask_sol[self.train_set].any())

    def test_get_metrics(self):
        # Simulated generated samples
        generated = np.zeros(self.n_states)
        generated[self.solution_set] = [0.1, 0.2, 0.3, 0.4, 0.5]  # Example values
        generated[self.train_set] = [0.05, 0.05, 0.05]  # Simulated memorized samples

        results = self.metrics_instance.get_metrics(generated)

        # Verify all metrics are calculated
        self.assertIn("fidelity", results)
        self.assertIn("exploration", results)
        self.assertIn("coverage", results)
        self.assertIn("normalized_rate", results)
        self.assertIn("precision", results)

        # Example assertions (adjust based on expected values for the given input)
        self.assertGreaterEqual(results["fidelity"], 0)
        self.assertGreaterEqual(results["exploration"], 0)
        self.assertGreaterEqual(results["coverage"], 0)
        self.assertGreaterEqual(results["normalized_rate"], 0)
        self.assertGreaterEqual(results["precision"], 0)

    def test_fidelity(self):
        fidelity = self.metrics_instance.fidelity(g_new=1.0, g_sol=0.8)
        self.assertAlmostEqual(fidelity, 0.8, msg="Fidelity calculation is incorrect")

    def test_coverage(self):
        coverage = self.metrics_instance.coverage(g_sol_unique=2)
        expected_coverage = 2 / (math.ceil(1 - self.train_size) * len(self.solution_set))
        self.assertAlmostEqual(coverage, expected_coverage, msg="Coverage calculation is incorrect")

    def test_normalized_rate(self):
        normalized_rate = self.metrics_instance.normalized_rate(g_sol=2.5)
        expected_rate = 2.5 / ((1 - self.train_size) * self.metrics_instance.n_shots)
        self.assertAlmostEqual(normalized_rate, expected_rate, msg="Normalized rate calculation is incorrect")

    def test_exploration(self):
        exploration = self.metrics_instance.exploration(g_new=2.5)
        expected_exploration = 2.5 / self.metrics_instance.n_shots
        self.assertAlmostEqual(exploration, expected_exploration, msg="Exploration calculation is incorrect")

    def test_precision(self):
        precision = self.metrics_instance.precision(g_sol=2.5, g_train=1.5)
        expected_precision = (2.5 + 1.5) / self.metrics_instance.n_shots
        self.assertAlmostEqual(precision, expected_precision, msg="Precision calculation is incorrect")


if __name__ == "__main__":
    unittest.main()
