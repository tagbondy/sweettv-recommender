"""Known-answer and contract tests for the shared local MAP@5 evaluator."""

import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluate import (average_precision_at_5, evaluate_recommendations, load_template_users,
                      read_predictions, validate_prediction, write_predictions)


class EvaluationTests(unittest.TestCase):
    def test_known_ap_example_and_unique_relevance(self):
        self.assertAlmostEqual(average_precision_at_5([1, 3, 3], [1, 2, 3, 4, 5]), 5 / 6)
        self.assertAlmostEqual(average_precision_at_5({1}, [2, 3, 4, 5, 1]), 1 / 5)

    def test_normalization_caps_at_five(self):
        self.assertEqual(average_precision_at_5(range(1, 8), [1, 2, 3, 4, 5]), 1)

    def test_empty_ground_truth_and_no_hits(self):
        self.assertEqual(average_precision_at_5([], [1, 2, 3, 4, 5]), 0)
        self.assertEqual(average_precision_at_5([9], [1, 2, 3, 4, 5]), 0)

    def test_duplicate_predictions_rejected_even_for_empty_ground_truth(self):
        with self.assertRaises(ValueError):
            average_precision_at_5([], [1, 1, 2, 3, 4])

    def test_incorrect_lengths_rejected(self):
        for prediction in ([], [1, 2, 3, 4], [1, 2, 3, 4, 5, 6], "1 2 3 4 5"):
            with self.subTest(prediction=prediction), self.assertRaises(ValueError):
                validate_prediction(prediction)

    def test_nonpositive_and_noninteger_programme_ids_rejected(self):
        for bad in (0, -1, True, 1.5, "1"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_prediction([bad, 2, 3, 4, 5])

    def test_primary_supplementary_and_cold_start_groups(self):
        users = ["001", "1", "cold"]
        predictions = {user: [1, 2, 3, 4, 5] for user in users}
        result = evaluate_recommendations(users, {"001": {1, 3}, "1": set(), "cold": {5}}, predictions, {"001", "1"})
        metrics = result["metrics"]
        self.assertAlmostEqual(metrics["primary_all_users"]["map_at_5"], (5 / 6 + 1 / 5) / 3)
        self.assertAlmostEqual(metrics["supplementary_nonempty_ground_truth"]["map_at_5"], (5 / 6 + 1 / 5) / 2)
        self.assertAlmostEqual(metrics["with_training_history"]["map_at_5"], 5 / 12)
        self.assertAlmostEqual(metrics["without_training_history"]["map_at_5"], 1 / 5)
        self.assertEqual(set(result["per_user"]), {"001", "1", "cold"})

    def test_no_labels_gives_zero_primary_and_undefined_supplementary(self):
        result = evaluate_recommendations(["u"], {}, {"u": [1, 2, 3, 4, 5]}, {"u"})
        self.assertEqual(result["metrics"]["primary_all_users"]["map_at_5"], 0)
        self.assertIsNone(result["metrics"]["supplementary_nonempty_ground_truth"]["map_at_5"])
        self.assertIsNone(result["metrics"]["without_training_history"]["map_at_5"])

    def test_exact_cohort_and_string_user_ids_required(self):
        for users, predictions in (([1], {1: [1, 2, 3, 4, 5]}), (["u", "u"], {"u": [1, 2, 3, 4, 5]}),
                                   (["u", "v"], {"u": [1, 2, 3, 4, 5]})):
            with self.assertRaises(ValueError):
                evaluate_recommendations(users, {}, predictions, set())

    def test_candidate_filter_never_removes_ground_truth(self):
        result = evaluate_recommendations(["u"], {"u": {1, 9}}, {"u": [1, 2, 3, 4, 5]}, set(), frozenset(range(1, 6)))
        self.assertEqual(result["metrics"]["primary_all_users"]["map_at_5"], .5)
        self.assertEqual(result["metrics"]["ground_truth_candidate_coverage"]["relevance_pairs_outside_candidates"], 1)
        with self.assertRaises(ValueError):
            evaluate_recommendations(["u"], {}, {"u": [1, 2, 3, 4, 9]}, set(), frozenset(range(1, 6)))

    def test_prediction_roundtrip_preserves_user_strings_and_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            users = ["0007", "7"]
            predictions = {user: (1, 2, 3, 4, 5) for user in users}
            candidates = frozenset(range(1, 6))
            write_predictions(path, users, predictions, candidates)
            self.assertEqual(read_predictions(path, users, candidates), predictions)
            with self.assertRaises(ValueError):
                read_predictions(path, users[::-1], candidates)

    def test_template_defaults_to_1260_and_preserves_leading_zeros(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.csv"
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerows([["user_id", "tv_show_id"], ["0007", "0 0 0 0 0"], ["7", "0 0 0 0 0"]])
            self.assertEqual(load_template_users(path, expected_users=2), ["0007", "7"])
            with self.assertRaises(ValueError):
                load_template_users(path)


if __name__ == "__main__":
    unittest.main()
