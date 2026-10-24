"""Verify interval matching, chronological isolation, and evidence deduplication."""

import csv
import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from audit_data import SCHEDULE_COLUMNS, SCHEDULE_FILES, VIEW_COLUMNS, VIEW_FILE
from build_interactions import (
    TRAIN_START, VALIDATION_START, TEST_START, WINDOWS, Airing, ScheduleIndex,
    View, Window, build_interactions, clipped_match, timestamp, verify_outputs,
)


class IntervalTests(unittest.TestCase):
    def setUp(self):
        self.window = Window("example", 0, 10000)
        self.view = View("000123", 6, 100, 400, 1)
        self.airing = Airing(6, 42, 0, 500)

    def test_matching_channels(self):
        match = clipped_match(self.view, self.airing, self.window, 120)
        self.assertEqual(match.overlap_seconds, 300)

    def test_nonmatching_channels(self):
        self.assertIsNone(clipped_match(self.view, Airing(7, 42, 0, 500), self.window, 120))

    def test_nonoverlapping_intervals(self):
        self.assertIsNone(clipped_match(self.view, Airing(6, 42, 500, 800), self.window, 1))

    def test_exact_touching_boundaries_have_no_overlap(self):
        for airing in (Airing(6, 42, 0, 100), Airing(6, 42, 400, 500)):
            self.assertIsNone(clipped_match(self.view, airing, self.window, 1))

    def test_partial_overlap(self):
        match = clipped_match(self.view, Airing(6, 42, 250, 600), self.window, 120)
        self.assertEqual((match.overlap_start, match.overlap_end, match.overlap_seconds), (250, 400, 150))

    def test_viewing_across_multiple_broadcasts(self):
        index = ScheduleIndex([Airing(6, 1, 0, 200), Airing(6, 2, 200, 350), Airing(6, 3, 350, 500)])
        matches = [clipped_match(self.view, airing, self.window, 40) for airing in index.overlaps(self.view)]
        self.assertEqual([(m.programme_id, m.overlap_seconds) for m in matches], [(1, 100), (2, 150), (3, 50)])

    def test_zero_programme_id_never_becomes_item(self):
        self.assertIsNone(clipped_match(self.view, Airing(6, 0, 0, 500), self.window, 1))
        self.assertEqual(list(ScheduleIndex([Airing(6, 0, 0, 500)]).overlaps(self.view)), [])

    def test_threshold_equality_and_one_second_below(self):
        self.assertIsNotNone(clipped_match(View("u", 6, 100, 220, 1), self.airing, self.window, 120))
        self.assertIsNone(clipped_match(View("u", 6, 100, 219, 1), self.airing, self.window, 120))

    def test_clips_both_intervals_before_threshold(self):
        cutoff = VALIDATION_START
        view = View("u", 6, cutoff - 150, cutoff + 180, 1)
        airing = Airing(6, 42, cutoff - 300, cutoff + 300)
        train = clipped_match(view, airing, WINDOWS[0], 120)
        validation = clipped_match(view, airing, WINDOWS[1], 120)
        self.assertEqual((train.view_end, train.programme_end, train.overlap_end), (cutoff, cutoff, cutoff))
        self.assertEqual(train.overlap_seconds, 150)
        self.assertEqual((validation.view_start, validation.programme_start, validation.overlap_start), (cutoff, cutoff, cutoff))
        self.assertEqual(validation.overlap_seconds, 180)

    def test_short_split_pieces_do_not_inherit_full_interval_threshold(self):
        cutoff = VALIDATION_START
        view = View("u", 6, cutoff - 60, cutoff + 60, 1)
        airing = Airing(6, 42, cutoff - 600, cutoff + 600)
        self.assertIsNone(clipped_match(view, airing, WINDOWS[0], 120))
        self.assertIsNone(clipped_match(view, airing, WINDOWS[1], 120))
        self.assertEqual(clipped_match(view, airing, WINDOWS[2], 120).overlap_seconds, 120)

    def test_future_viewing_is_excluded_and_crossing_tail_is_clipped(self):
        airing = Airing(6, 42, TEST_START - 600, TEST_START + 600)
        future = View("u", 6, TEST_START, TEST_START + 300, 1)
        crossing = View("u", 6, TEST_START - 120, TEST_START + 300, 2)
        self.assertIsNone(clipped_match(future, airing, WINDOWS[2], 120))
        match = clipped_match(crossing, airing, WINDOWS[2], 120)
        self.assertEqual((match.view_end, match.programme_end, match.overlap_end), (TEST_START,) * 3)
        self.assertEqual(match.overlap_seconds, 120)

    def test_user_id_preserved_as_string(self):
        match = clipped_match(self.view, self.airing, self.window, 120)
        self.assertEqual(match.user_id, "000123")
        self.assertIsInstance(match.user_id, str)

    def test_nested_airings_and_out_of_order_view_queries(self):
        long = Airing(6, 1, 0, 1000)
        short = Airing(6, 2, 200, 250)
        index = ScheduleIndex([short, long])
        self.assertEqual(list(index.overlaps(View("u", 6, 800, 900, 1))), [long])
        self.assertEqual(list(index.overlaps(View("u", 6, 210, 230, 2))), [long, short])

    def test_index_matches_brute_force_for_randomized_overlapping_airings(self):
        rng = random.Random(42)
        airings = []
        for item in range(1, 101):
            start = rng.randrange(1000)
            airings.append(Airing(rng.choice([6, 7]), item, start, start + rng.randrange(1, 500)))
        index = ScheduleIndex(airings)
        for number in range(100):
            start = rng.randrange(1500)
            view = View("u", rng.choice([6, 7, 8]), start, start + rng.randrange(1, 300), number)
            expected = {a for a in airings if a.channel_id == view.channel_id and a.start < view.end and a.end > view.start}
            self.assertEqual(set(index.overlaps(view)), expected)

    def test_invalid_threshold_rejected(self):
        with self.assertRaises(ValueError):
            clipped_match(self.view, self.airing, self.window, 0)


class PipelineTests(unittest.TestCase):
    def write_csv(self, path, columns, rows):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerows(rows)

    def schedule_row(self, programme, start, duration=600, channel=6):
        from datetime import datetime
        formatted = datetime.fromisoformat(timestamp(start)).strftime("%d.%m.%Y %H:%M:%S")
        return [str(channel), "Channel", formatted, str(duration), "Title", str(programme), "Category"] + [""] * 6

    def test_duplicate_views_schedule_padding_and_future_schedule_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cutoff = VALIDATION_START
            airing = self.schedule_row(42, cutoff - 300, 600)
            self.write_csv(root / SCHEDULE_FILES[0], SCHEDULE_COLUMNS, [airing, self.schedule_row(0, TRAIN_START, 600)])
            # An identical boundary airing in both files must not double the evidence.
            self.write_csv(root / SCHEDULE_FILES[1], SCHEDULE_COLUMNS, [airing])
            # Even an early timestamp in the test schedule must not be used for interactions.
            self.write_csv(root / SCHEDULE_FILES[2], SCHEDULE_COLUMNS, [self.schedule_row(999, TRAIN_START, 600)])
            self.write_csv(root / "submission.csv", ["user_id", "tv_show_id"], [["0007", "0 0 0 0 0"], ["7", "0 0 0 0 0"]])
            viewing = ["0007", "6", timestamp(cutoff - 150), timestamp(cutoff + 180), "330"]
            changed_duration_duplicate = viewing.copy()
            changed_duration_duplicate[-1] = "331"
            self.write_csv(root / VIEW_FILE, VIEW_COLUMNS, [
                viewing, viewing, changed_duration_duplicate,
                ["0007", "6", timestamp(TRAIN_START), timestamp(TRAIN_START + 300), "300"],
                ["0007", "6", timestamp(TEST_START), timestamp(TEST_START + 300), "300"],
            ])
            output = root / "output"
            result = build_interactions(root, output, root, 120)
            verification = verify_outputs(output, result)
            self.assertEqual(result["viewing_processing"]["duplicate_viewing_events_removed"], 2)
            self.assertEqual(result["viewing_processing"]["outside_final_training_window"], 1)
            self.assertEqual(result["schedule_loading"]["duplicate_clipped_airings_removed_across_and_within_files"], 1)
            for name, overlap in (("train", 150), ("validation", 180), ("final_train", 330)):
                self.assertEqual(result["splits"][name]["matched_interactions"], 1)
                self.assertEqual(result["splits"][name]["overlap_seconds"], overlap)
                self.assertEqual(result["splits"][name]["cold_start_user_ids"], ["7"])
                self.assertTrue(verification[name]["temporal_bounds_threshold_and_nonzero_ids_verified"])
                with (output / f"{name}_interactions.csv").open(newline="") as handle:
                    row = next(csv.DictReader(handle))
                    self.assertEqual(row["user_id"], "0007")
                    self.assertEqual(row["view_event_row"], "1")
                    self.assertEqual(row["tv_show_id"], "42")

    def test_output_verification_rejects_future_timestamp(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_csv(root / SCHEDULE_FILES[0], SCHEDULE_COLUMNS, [self.schedule_row(42, TRAIN_START, 600)])
            self.write_csv(root / SCHEDULE_FILES[1], SCHEDULE_COLUMNS, [])
            self.write_csv(root / "submission.csv", ["user_id", "tv_show_id"], [["0007", "0 0 0 0 0"]])
            self.write_csv(root / VIEW_FILE, VIEW_COLUMNS, [["0007", "6", timestamp(TRAIN_START), timestamp(TRAIN_START + 300), "300"]])
            output = root / "output"
            result = build_interactions(root, output, root, 120)
            path = output / "train_interactions.csv"
            with path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                columns = reader.fieldnames
                rows = list(reader)
            rows[0]["view_end"] = timestamp(VALIDATION_START + 1)
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, columns)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaises(AssertionError):
                verify_outputs(output, result)


if __name__ == "__main__":
    unittest.main()
