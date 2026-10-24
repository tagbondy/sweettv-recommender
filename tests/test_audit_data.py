"""Focused checks for audit parsing, interval validation, and duplicate counting."""

import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from audit_data import (  # noqa: E402
    DuplicateIndex, NumericSummary, SCHEDULE_COLUMNS, SCHEDULE_FILES, VIEW_COLUMNS,
    audit_schedule, audit_views, parse_timestamp,
)


class AuditTests(unittest.TestCase):
    def test_timestamp_formats_agree_and_reject_invalid_dates(self):
        self.assertEqual(parse_timestamp("2020-03-09 06:00:00"),
                         parse_timestamp("09.03.2020 06:00:00", schedule=True))
        self.assertIsNone(parse_timestamp("31.02.2020 06:00:00", schedule=True))
        self.assertIsNone(parse_timestamp("2020-3-9 06:00:00"))
        self.assertIsNone(parse_timestamp("2020-03-09 06:00:00Z"))

    def test_exact_quantiles_with_repetitions_and_zero(self):
        summary = NumericSummary()
        for value in (0, 10, 10, 20):
            summary.add(value)
        result = summary.result()
        self.assertEqual(result["median"], 10)
        self.assertEqual(result["p25"], 7.5)
        self.assertEqual(result["p75"], 12.5)
        self.assertEqual(result["zero_count"], 1)
        self.assertEqual(result["mean"], 10)
        self.assertAlmostEqual(result["population_stddev"], 50 ** .5)
        self.assertEqual(NumericSummary().result()["count"], 0)

    def test_disk_duplicates_across_batches_and_field_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            index = DuplicateIndex(Path(directory) / "duplicates.sqlite3", batch_size=2)
            for row in (["ab", "c"], ["a", "bc"], ["ab", "c"], ["ab", None], ["ab", None]):
                index.add(row)
            self.assertEqual(index.close(), 2)

    def test_view_validation_preserves_ids_and_separates_error_types(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "view.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(VIEW_COLUMNS)
                writer.writerows([
                    ["0007", "6", "2020-05-17 23:59:30", "2020-05-18 00:00:30", "60"],
                    ["7", "6", "2020-03-09 00:00:00", "2020-03-09 00:00:10", "5"],
                    ["8", "bad", "2020-03-09 00:01:00", "2020-03-09 00:00:00", "-60"],
                    ["9", "6", "invalid", "2020-03-09 00:00:00", ""],
                    ["10", "6", "2020-03-09 00:00:00", "2020-03-09 00:00:00", "0"],
                ])
            result, channels = audit_views(path, {"0007", "7", "missing"})
            issues = result["quality"]["issue_counts"]
            self.assertEqual(result["viewing_events"], 5)
            self.assertEqual(result["unique_nonempty_users"], 5)
            self.assertEqual(result["test_user_coverage"]["in_any_viewing_history"], 2)
            self.assertEqual(result["invalid_timestamp_intervals"], 3)
            self.assertEqual(issues["duration_elapsed_mismatch"], 1)
            self.assertEqual(result["elapsed_minus_reported_seconds_for_mismatches"], {"5": 1})
            self.assertEqual(issues["negative_duration"], 1)
            self.assertEqual(issues["intervals_crossing_train_validation_boundary"], 1)
            self.assertEqual(channels, {6})

    def test_schedule_quality_and_candidate_airing_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / SCHEDULE_FILES[0]
            valid = ["6", "Channel", "09.03.2020 00:00:00", "60", "Title", "42", "Category"] + [""] * 6
            zero = valid.copy()
            zero[5] = "0"
            negative = valid.copy()
            negative[3], negative[5] = "-1", "43"
            invalid = valid.copy()
            invalid[2], invalid[5] = "31.02.2020 00:00:00", "44"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(SCHEDULE_COLUMNS)
                writer.writerows([valid, valid, zero, negative, invalid])
            result, state = audit_schedule(path, Path(directory))
            self.assertEqual(result["rows"], 5)
            self.assertEqual(result["duplicate_parsed_records_after_first"], 1)
            self.assertEqual(result["unique_nonzero_programme_ids"], 3)
            self.assertEqual(result["labeled_programmes_with_valid_positive_airings"], 1)
            self.assertEqual(result["zero_id_percentage"], 20)
            self.assertEqual(result["quality"]["issue_counts"]["negative_duration"], 1)
            self.assertEqual(result["quality"]["issue_counts"]["invalid_start_timestamp"], 1)
            self.assertEqual(result["missing_metadata_labeled_rows"]["actors"], 4)
            self.assertEqual(result["nominal_calendar_window"]["labeled_programme_ids_with_valid_airings_overlapping_window"], 1)
            self.assertEqual(state["programme_ids"], {42, 43, 44})

    def test_nominal_window_excludes_padding_but_keeps_crossing_airings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / SCHEDULE_FILES[0]
            base = ["6", "Channel", "08.03.2020 23:59:30", "60", "Title", "42", "Category"] + [""] * 6
            padding = base.copy()
            padding[2], padding[5] = "18.05.2020 00:00:00", "43"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(SCHEDULE_COLUMNS)
                writer.writerows([base, padding])
            result, state = audit_schedule(path, Path(directory))
            window = result["nominal_calendar_window"]
            self.assertEqual(window["starts_before_window"], 1)
            self.assertEqual(window["starts_at_or_after_window_end"], 1)
            self.assertEqual(window["positive_intervals_crossing_window_start"], 1)
            self.assertEqual(window["positive_intervals_overlapping_window"], 1)
            self.assertEqual(state["window_programme_ids"], {42})


if __name__ == "__main__":
    unittest.main()
