import unittest
import os
import numpy as np
from testing.test_simulation import evaluate_simulation


class TestReverseEngineeringSimulation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.orig_path = "data/spike_data.npz"
        cls.sim_path = "results/spike_data.npz"
        cls.report = evaluate_simulation(cls.orig_path, cls.sim_path)

    def test_file_exists(self):
        """Verify the simulated results file was created."""
        self.assertTrue(os.path.exists(self.sim_path), "Simulated .npz file does not exist.")

    def test_structure_matches_original(self):
        """Verify keys, array shapes, and dtypes match the original file."""
        struct = self.report["structure_match"]
        self.assertTrue(struct["keys_identical"], "NPZ keys do not match the original file.")
        self.assertTrue(struct["shapes_identical"], "Array shapes do not match the original file.")
        self.assertTrue(struct["dtypes_identical"], "Array dtypes do not match the original file.")

    def test_statistical_similarity(self):
        """Verify standard deviation and spectral similarity are within reasonable bounds."""
        std_err = self.report["time_domain_metrics"]["std_relative_error"]
        psd_sim = self.report["frequency_domain_metrics"]["psd_cosine_similarity"]
        self.assertLess(std_err, 0.25, f"Standard deviation relative error too high: {std_err:.2f}")
        self.assertGreater(psd_sim, 0.80, f"PSD cosine similarity too low: {psd_sim:.2f}")


if __name__ == "__main__":
    unittest.main()
