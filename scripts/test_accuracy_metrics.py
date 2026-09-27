"""Reference scoring checks that run without ROS."""
import math
import unittest

from accuracy_metrics import evaluate_position, reference_quality_mask


class AccuracyMetricsTests(unittest.TestCase):
    def test_signed_along_cross_and_terminal_coverage(self):
        positions = [(0, 1.0, 2.0, 0.0, "pathgraph"),
                     (100_000_000, 1.0, 2.0, 0.0, "pathgraph"),
                     (200_000_000, 1.0, 2.0, 0.0, "odom_relative")]
        fixes = [(0, 0.0, 0.0, 0.0, 0),
                 (100_000_000, 0.0, 0.0, 0.0, 0)]
        report = evaluate_position(
            positions, fixes, lambda lat, lon, alt: (lon, lat, alt),
            anchor_ns=0, estimated_distance_m=10.0)
        self.assertEqual(report["absolute_output_count"], 2)
        self.assertEqual(report["quality_matched_count"], 2)
        self.assertAlmostEqual(report["quality"]["along_m"]["mean_m"], 1.0)
        self.assertAlmostEqual(report["quality"]["cross_m"]["mean_m"], 2.0)
        self.assertAlmostEqual(report["endpoint"]["xy_error_m"], math.sqrt(5))
        self.assertAlmostEqual(report["endpoint"]["error_pct_estimated_distance"],
                               10 * math.sqrt(5))

    def test_bad_reference_edge_excludes_adjacent_second(self):
        fixes = [(i * 100_000_000, 0.0, 0.0 if i < 30 else 0.01, 0.0, 0)
                 for i in range(60)]
        quality = reference_quality_mask(fixes)
        self.assertTrue(quality[10])
        self.assertFalse(quality[29])
        self.assertFalse(quality[40])
        self.assertTrue(quality[51])

    def test_distant_reference_does_not_claim_endpoint_accuracy(self):
        positions = [(0, 0.0, 0.0, 0.0, "pathgraph"),
                     (3_000_000_000, 0.0, 0.0, 0.0, "pathgraph")]
        fixes = [(0, 0.0, 0.0, 0.0, 0)]
        report = evaluate_position(positions, fixes,
                                   lambda lat, lon, alt: (lon, lat, alt))
        self.assertEqual(report["endpoint"]["unscored_tail_s"], 3.0)
        self.assertIsNone(report["endpoint"]["xy_error_m"])


if __name__ == "__main__":
    unittest.main()
