"""Verify distinct-viewer ranking, availability restrictions, and leakage barriers."""

import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from audit_data import SCHEDULE_COLUMNS
from build_interactions import PAIR_COLUMNS
from evaluate import AggregateReader, load_candidates, load_validation_ground_truth
from popularity import fit_popularity, rank_candidates, run_baseline


def write_csv(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def pair(user, programme, events=1, validation=False):
    start = "2020-05-18 00:00:00" if validation else "2020-03-09 00:00:00"
    end = "2020-05-18 01:00:00" if validation else "2020-03-09 01:00:00"
    return [user, programme, events, events * 120, start, end]


def airing(programme, start="18.05.2020 00:00:00", duration=600):
    return ["6", "Channel", start, str(duration), "Title", str(programme), "Category"] + [""] * 6


class PopularityTests(unittest.TestCase):
    def test_distinct_viewers_not_events_and_duplicate_keys_do_not_inflate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.csv"
            write_csv(path, PAIR_COLUMNS, [pair("0007", 1, 20), pair("0007", 1, 20), pair("7", 1), pair("7", 2, 100)])
            model = fit_popularity(path)
            self.assertEqual(dict(model.scores), {1: 2, 2: 1})
            self.assertEqual(model.training_users, {"0007", "7"})
            self.assertEqual(model.training_statistics["duplicate_keys_ignored"], 1)
            with self.assertRaises(TypeError):
                model.scores[1] = 999

    def test_candidate_restrictions_zero_scores_and_numeric_ties(self):
        scores = {999: 100, 10: 2, 2: 2}
        candidates = frozenset([1, 2, 3, 4, 5, 10])
        expected = (2, 10, 1, 3, 4)
        self.assertEqual(rank_candidates(scores, candidates), expected)
        self.assertEqual(rank_candidates(dict(reversed(list(scores.items()))), frozenset(reversed(sorted(candidates)))), expected)
        self.assertNotIn(999, expected)

    def test_insufficient_or_invalid_candidates_fail(self):
        for candidates in (frozenset([1, 2, 3, 4]), frozenset([0, 1, 2, 3, 4, 5])):
            with self.assertRaises(ValueError):
                rank_candidates({}, candidates)

    def test_availability_uses_time_overlap_not_filename_or_watch_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "schedule.csv"
            write_csv(path, SCHEDULE_COLUMNS, [
                airing(0), airing(1), airing(2, duration=60),
                airing(3, "17.05.2020 23:59:00", duration=120),
                airing(4, "17.05.2020 23:59:00", duration=60),
                airing(5, "27.07.2020 00:00:00"),
                airing(6, "26.07.2020 23:59:30", duration=120),
            ])
            candidates, stats = load_candidates(path)
            self.assertEqual(candidates, {1, 2, 3, 6})
            self.assertEqual(stats["zero_id_rows_excluded"], 1)
            self.assertFalse(stats["viewing_labels_used"])

    def test_validation_timestamps_cannot_enter_training(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "looks_like_train.csv"
            write_csv(path, PAIR_COLUMNS, [pair("u", 1, validation=True)])
            with self.assertRaisesRegex(ValueError, "outside train"):
                fit_popularity(path)

    def test_training_timestamps_cannot_be_validation_ground_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "looks_like_validation.csv"
            write_csv(path, PAIR_COLUMNS, [pair("u", 1)])
            with self.assertRaisesRegex(ValueError, "outside validation"):
                load_validation_ground_truth(path, ["u"])

    def test_unsorted_input_is_rejected_instead_of_miscounting_distinct_users(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.csv"
            write_csv(path, PAIR_COLUMNS, [pair("b", 1), pair("a", 1)])
            with self.assertRaisesRegex(ValueError, "not sorted"):
                fit_popularity(path)

    def test_cold_start_and_validation_labels_do_not_change_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training, validation, schedule, template = [root / name for name in ("train.csv", "validation.csv", "schedule.csv", "submission.csv")]
            write_csv(training, PAIR_COLUMNS, [pair("0007", 1), pair("0007", 2), pair("7", 1)])
            write_csv(validation, PAIR_COLUMNS, [pair("0007", 5, validation=True)])
            write_csv(schedule, SCHEDULE_COLUMNS, [airing(item) for item in range(1, 7)])
            write_csv(template, ["user_id", "tv_show_id"], [["0007", "0 0 0 0 0"], ["cold", "0 0 0 0 0"]])
            first, predictions, candidates, _ = run_baseline(training, validation, schedule, template, expected_users=2)
            self.assertEqual(predictions["0007"], predictions["cold"])
            self.assertEqual(predictions["0007"], (1, 2, 3, 4, 5))
            self.assertEqual(first["evaluation"]["metrics"]["without_training_history"]["users"], 1)
            # Change only validation behaviour, including a programme outside availability.
            write_csv(validation, PAIR_COLUMNS, [pair("cold", 6, validation=True), pair("cold", 99, validation=True)])
            second, changed, same_candidates, _ = run_baseline(training, validation, schedule, template, expected_users=2)
            self.assertEqual(changed, predictions)
            self.assertEqual(same_candidates, candidates)
            self.assertEqual(second["model"]["top_five"], first["model"]["top_five"])
            self.assertEqual(second["evaluation"]["metrics"]["ground_truth_candidate_coverage"]["relevance_pairs_outside_candidates"], 1)
            self.assertEqual(fit_popularity(training).scores[1], 2)


if __name__ == "__main__":
    unittest.main()
