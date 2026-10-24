"""Verify final-window adaptation, retrained fallback, strict CSVs, and reuse."""

import csv
import tempfile
import sys
import unittest
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from audit_data import SCHEDULE_COLUMNS
from build_interactions import PAIR_COLUMNS
from create_final_submission import (
    EXPECTED_CONFIGURATION, TEST_WINDOW, build_final_predictions, load_final_training,
    publish_submission, selected_configuration, validate_submission,
)
from evaluate import load_candidates, write_predictions
from popularity import fit_popularity


def write_csv(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def pair(user, programme, later=False, events=1):
    start = "2020-05-18 00:00:00" if later else "2020-03-09 00:00:00"
    end = "2020-05-18 01:00:00" if later else "2020-03-09 01:00:00"
    return [user, programme, events, events * 120, start, end]


def airing(item, start="27.07.2020 00:00:00", duration=600):
    return ["6", "Channel", start, duration, "Availability Only", item, "Unused Category"] + [""] * 6


class FinalTrainingTests(unittest.TestCase):
    def test_selected_hyperparameters_are_loaded_without_retuning(self):
        saved = {"selected_configuration": "confidence_max", "experiments": [
            {"configuration": {**asdict(EXPECTED_CONFIGURATION), "max_items_per_user": 50},
             "evaluation": {"metrics": {"primary_all_users": {"map_at_5": .5}}}}]}
        config, evidence = selected_configuration(saved)
        self.assertEqual(config, EXPECTED_CONFIGURATION)
        self.assertEqual(evidence["previous_validation_map_at_5"], .5)
        saved["experiments"][0]["configuration"]["min_pair_support"] = 20
        with self.assertRaises(ValueError):
            selected_configuration(saved)

    def test_full_final_window_and_new_popularity_counts_include_later_interactions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "final_train.csv"
            rows = [pair("0007", 1), pair("0007", 2, later=True, events=5),
                    pair("0007", 2, later=True, events=5), pair("7", 2, later=True)]
            write_csv(path, PAIR_COLUMNS, rows)
            baskets, popularity, stats = load_final_training(path, ["0007", "7", "cold"])
            self.assertEqual(dict(popularity.scores), {1: 1, 2: 2})
            self.assertEqual(set(baskets), {"0007", "7"})
            self.assertEqual(stats["cold_start_template_users"], 1)
            self.assertEqual(stats["aggregate_schema_and_time_bounds"]["duplicate_keys_ignored"], 1)
            self.assertEqual(stats["aggregate_schema_and_time_bounds"]["template_users_present"], 2)
            with self.assertRaisesRegex(ValueError, "outside train"):
                fit_popularity(path)

    def test_final_popularity_matches_existing_distinct_viewer_definition(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.csv"
            write_csv(path, PAIR_COLUMNS, [pair("0007", 1, events=10), pair("0007", 2), pair("7", 2)])
            _, final_popularity, _ = load_final_training(path, ["0007", "7"])
            old_popularity = fit_popularity(path)
            self.assertEqual(dict(final_popularity.scores), dict(old_popularity.scores))

    def test_popularity_is_counted_before_fifty_item_basket_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "final.csv"
            write_csv(path, PAIR_COLUMNS, [pair("0007", item) for item in range(1, 52)])
            baskets, popularity, _ = load_final_training(path, ["0007"])
            self.assertEqual(len(baskets["0007"]), 50)
            self.assertEqual(len(popularity.scores), 51)
            self.assertEqual(popularity.scores[51], 1)

    def test_future_viewing_and_aggregates_crossing_test_cutoff_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "final.csv"
            for first, last in (("2020-07-27 00:00:00", "2020-07-27 00:02:00"),
                                ("2020-07-26 23:59:00", "2020-07-27 00:01:00")):
                write_csv(path, PAIR_COLUMNS, [["u", 1, 1, 120, first, last]])
                with self.assertRaisesRegex(ValueError, "outside final_train"):
                    load_final_training(path, ["u"])

    def test_test_candidates_use_exact_positive_overlap_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test_schedule.csv"
            write_csv(path, SCHEDULE_COLUMNS, [airing(0), airing(1),
                airing(2, "26.07.2020 23:59:00", 60), airing(3, "26.07.2020 23:59:00", 120),
                airing(4, "19.10.2020 00:00:00"), airing(5, "18.10.2020 23:59:30", 120)])
            candidates, _ = load_candidates(path, TEST_WINDOW)
            self.assertEqual(candidates, {1, 3, 5})


class FinalFileTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.template = self.root / "template.csv"
        self.output = self.root / "submission.csv"
        self.users = ["0007", "7", "cold"]
        self.candidates = frozenset(range(1, 8))
        write_csv(self.template, ["user_id", "tv_show_id"], [[user, "0 0 0 0 0"] for user in self.users])

    def tearDown(self):
        self.temporary.cleanup()

    def test_exact_schema_order_ids_and_five_positive_unique_candidates(self):
        predictions = {user: (1, 2, 3, 4, 5) for user in self.users}
        write_predictions(self.output, self.users, predictions, self.candidates)
        checks = validate_submission(self.output, self.template, self.candidates, expected_users=3)
        self.assertEqual(checks["columns"], ["user_id", "tv_show_id"])
        self.assertEqual(checks["data_rows"], 3)
        self.assertEqual(checks["unique_user_ids"], 3)
        self.assertTrue(checks["exact_template_user_ids_and_row_order"])
        with self.assertRaises(ValueError):
            validate_submission(self.output, self.template, self.candidates)

    def test_invalid_prediction_values_lengths_and_spacing_are_rejected(self):
        for value in ("1 1 2 3 4", "0 1 2 3 4", "1 2 3 4", "1 2 3 4 99", "1  2 3 4 5", "1 2 3 4 5 "):
            write_csv(self.output, ["user_id", "tv_show_id"], [[user, value] for user in self.users])
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_submission(self.output, self.template, self.candidates, expected_users=3)

    def test_reordered_duplicate_and_missing_users_are_rejected(self):
        for users in (self.users[::-1], ["0007", "0007", "cold"], ["0007", "7"]):
            write_csv(self.output, ["user_id", "tv_show_id"], [[user, "1 2 3 4 5"] for user in users])
            with self.assertRaises(ValueError):
                validate_submission(self.output, self.template, self.candidates, expected_users=3)

    def test_incorrect_csv_columns_are_rejected(self):
        write_csv(self.output, ["user_id", "tv_show_id", "extra"], [["0007", "1 2 3 4 5", "bad"]])
        with self.assertRaises(ValueError):
            validate_submission(self.output, self.template, self.candidates, expected_users=3)

    def test_publishing_never_overwrites_an_existing_file(self):
        source = self.root / "validated.csv"
        write_csv(source, ["user_id", "tv_show_id"], [["0007", "1 2 3 4 5"]])
        publish_submission(source, self.output)
        self.assertEqual(source.read_bytes(), self.output.read_bytes())
        with self.assertRaises(FileExistsError):
            publish_submission(source, self.output)

    def test_end_to_end_reuses_final_histories_rules_fallback_and_is_deterministic(self):
        training, schedule = self.root / "final_train.csv", self.root / "test.csv"
        rows = []
        for user in ["0007", "1", "2", "3", "4", "7"]:
            if user != "7":
                rows.append(pair(user, 1))
            rows.extend([pair(user, 6, later=True, events=3), pair(user, 99, later=True, events=2)])
        rows.sort(key=lambda values: (values[0], values[1]))
        write_csv(training, PAIR_COLUMNS, rows)
        write_csv(schedule, SCHEDULE_COLUMNS, [airing(item) for item in range(1, 8)])
        runs = []
        for _ in range(2):
            with tempfile.TemporaryDirectory(dir=self.root) as scratch:
                runs.append(build_final_predictions(training, schedule, self.template, Path(scratch),
                                                    EXPECTED_CONFIGURATION, expected_users=3))
        first, second = runs
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[2]["model_fingerprints"], second[2]["model_fingerprints"])
        self.assertEqual(first[2]["training"]["final_training_users"], 6)
        self.assertEqual(first[2]["final_popularity_fallback"][0]["tv_show_id"], 6)
        self.assertEqual(first[2]["final_popularity_fallback"][0]["distinct_final_training_users"], 6)
        self.assertEqual(first[0]["cold"], (6, 1, 2, 3, 4))
        self.assertEqual(first[2]["training"]["cold_start_user_ids"], ["cold"])
        self.assertTrue(all(99 not in items for items in first[0].values()))
        self.assertTrue(all(len(items) == len(set(items)) == 5 for items in first[0].values()))
        write_predictions(self.output, self.users, first[0], first[1])
        self.assertEqual(validate_submission(self.output, self.template, first[1], 3)["data_rows"], 3)


if __name__ == "__main__":
    unittest.main()
