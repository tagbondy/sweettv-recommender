"""Continuous-watch fractions, occurrence counting, causal profiles, and ranking."""

import csv
import math
import sqlite3
import tempfile
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from audit_data import SCHEDULE_COLUMNS, SCHEDULE_FILES, VIEW_COLUMNS, VIEW_FILE
from build_interactions import Airing, View, WINDOWS, timestamp
from continuous_watch import (
    EVIDENCE_WINDOWS, PROFILE_COLUMNS, OccurrenceWriter, create_tables,
    add_profile_scan_index, export_profiles, open_database, qualifies, reconstruct_evidence, session_values,
)
from personalized_repeat import (
    CONFIG_BY_NAME, CONFIGURATIONS, iter_profiles, predict_configurations,
    profile_context, recommend_repeat, select_robust_model,
)


def write_csv(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


class ContinuousWatchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.start = WINDOWS[0].start + 600

    def database(self, name="evidence.sqlite3"):
        path = self.root / name
        connection = open_database(path)
        create_tables(connection)
        self.addCleanup(connection.close)
        return path, connection

    def test_exact_thresholds_and_invalid_durations(self):
        self.assertTrue(qualifies(30, 100, "ge30"))
        self.assertFalse(qualifies(29, 100, "ge30"))
        self.assertFalse(qualifies(50, 100, "gt50"))
        self.assertTrue(qualifies(51, 100, "gt50"))
        self.assertFalse(qualifies(80, 100, "gt80"))
        self.assertTrue(qualifies(81, 100, "gt80"))
        for overlap, duration in ((1, 0), (101, 100), (-1, 100)):
            with self.assertRaises(ValueError):
                qualifies(overlap, duration, "gt80")

    def test_disconnected_intervals_never_become_a_continuous_watch(self):
        path, connection = self.database()
        writer = OccurrenceWriter(connection)
        airing = Airing(6, 42, self.start, self.start + 100)
        writer.add(View("0007", 6, self.start, self.start + 40, 1), airing)
        writer.add(View("0007", 6, self.start + 60, self.start + 100, 2), airing)
        writer.flush()
        row = connection.execute("SELECT full_overlap,source_sessions,full_last50 FROM occurrences").fetchone()
        self.assertEqual(row, (40, 2, 0))
        profile = self.root / "profile.csv"
        export_profiles(path, EVIDENCE_WINDOWS[0], profile)
        record = next(iter_profiles(profile, "full"))
        self.assertEqual(record["frequency_ge30"], 1)
        self.assertEqual(record["frequency_gt50"], 0)
        self.assertEqual(record["frequency_gt80"], 0)

    def test_maximum_source_lineage_survives_batched_and_sqlite_updates(self):
        for limit in (1, 1000):
            _, connection = self.database(f"source_{limit}.sqlite3")
            writer = OccurrenceWriter(connection, limit=limit)
            airing = Airing(6, 42, self.start, self.start + 100)
            writer.add(View("0007", 6, self.start, self.start + 60, 10), airing)
            writer.add(View("0007", 6, self.start, self.start + 81, 20), airing)
            writer.flush()
            self.assertEqual(connection.execute("SELECT full_overlap,full_source_row,source_sessions FROM occurrences").fetchone(), (81, 20, 2))

    def test_later_nonqualifying_session_does_not_update_qualified_recency(self):
        _, connection = self.database()
        writer = OccurrenceWriter(connection, limit=1)
        airing = Airing(6, 42, self.start, self.start + 100)
        writer.add(View("0007", 6, self.start, self.start + 60, 1), airing)
        writer.add(View("0007", 6, self.start + 90, self.start + 100, 2), airing)
        writer.flush()
        self.assertEqual(connection.execute("SELECT full_last50 FROM occurrences").fetchone()[0], self.start + 60)

    def test_denominator_is_full_original_broadcast_at_split_boundary(self):
        boundary = WINDOWS[1].start
        airing = Airing(6, 42, boundary - 60, boundary + 40)
        values = session_values(View("0007", 6, boundary - 60, boundary + 40, 1), airing)
        by_name = {window.name: values[5 * i:5 * i + 5] for i, window in enumerate(EVIDENCE_WINDOWS)}
        self.assertEqual(by_name["full"][0], 100)
        self.assertEqual(by_name["train"][0], 60)
        self.assertEqual(by_name["validation"][0], 40)
        self.assertEqual(by_name["train"][4], 0)
        self.assertEqual(by_name["validation"][4], 0)
        self.assertGreater(by_name["full"][4], 0)

    def test_nonmatching_channels_and_touching_intervals_have_no_watch(self):
        airing = Airing(6, 42, self.start, self.start + 100)
        for view in (View("u", 7, self.start, self.start + 100, 1),
                     View("u", 6, self.start + 100, self.start + 200, 2)):
            self.assertTrue(all(value == 0 for value in session_values(view, airing)))

    def test_repeated_broadcasts_are_distinct_but_sessions_do_not_double_frequency(self):
        path, connection = self.database()
        writer = OccurrenceWriter(connection)
        for occurrence in (self.start, self.start + 1000):
            airing = Airing(6, 42, occurrence, occurrence + 100)
            writer.add(View("0007", 6, occurrence, occurrence + 70, occurrence), airing)
            writer.add(View("0007", 6, occurrence + 10, occurrence + 70, occurrence + 1), airing)
        writer.flush()
        profile = self.root / "profile.csv"
        export_profiles(path, EVIDENCE_WINDOWS[0], profile)
        record = next(iter_profiles(profile, "full"))
        self.assertEqual(record["frequency_gt50"], 2)
        self.assertEqual(record["qualified_seconds_gt50"], 140)
        self.assertEqual(connection.execute("SELECT COUNT(*) FROM occurrences").fetchone()[0], 2)

    def test_recency_half_life_is_anchored_to_history_end(self):
        path, connection = self.database()
        writer = OccurrenceWriter(connection)
        end = WINDOWS[0].end
        for last in (end, end - 14 * 86400):
            airing = Airing(6, 42, last - 100, last)
            writer.add(View("0007", 6, last - 100, last, last), airing)
        writer.flush()
        profile = self.root / "profile.csv"
        export_profiles(path, EVIDENCE_WINDOWS[1], profile)
        record = next(iter_profiles(profile, "train"))
        self.assertEqual(record["recency_gt50_half14"], 1.5)
        self.assertAlmostEqual(record["recency_gt50_half28"], 1 + math.sqrt(.5))

    def test_raw_reconstruction_includes_short_watches_and_ignores_duraton_denominator(self):
        raw = self.root / "raw"
        raw.mkdir()
        write_csv(raw / SCHEDULE_FILES[0], SCHEDULE_COLUMNS, [
            [6, "Channel", "09.03.2020 00:10:00", 100, "Title", 42, "Category"] + [""] * 6,
            [6, "Channel", "09.03.2020 00:20:00", 100, "Unlabeled", 0, "Category"] + [""] * 6])
        write_csv(raw / SCHEDULE_FILES[1], SCHEDULE_COLUMNS, [])
        # A poisoned test schedule must not participate in historical evidence.
        write_csv(raw / SCHEDULE_FILES[2], SCHEDULE_COLUMNS, [[6, "Channel", "09.03.2020 00:10:00", 100, "Future", 999, "Category"] + [""] * 6])
        viewing = ["0007", 6, timestamp(self.start), timestamp(self.start + 81), 999999]
        write_csv(raw / VIEW_FILE, VIEW_COLUMNS, [viewing, viewing, ["7", *viewing[1:]]])
        database = self.root / "reconstructed.sqlite3"
        stats = reconstruct_evidence(raw, database)
        self.assertEqual(stats["viewing"]["duplicate_viewing_intervals_removed"], 1)
        self.assertEqual(stats["viewing"]["positive_overlaps_below_120_seconds"], 2)
        self.assertEqual(stats["windows"]["train"]["qualified_gt80_occurrences"], 2)
        with sqlite3.connect(database) as connection:
            self.assertEqual(connection.execute("SELECT DISTINCT broadcast_duration FROM occurrences").fetchall(), [(100,)])
            self.assertEqual(connection.execute("SELECT DISTINCT programme_id FROM occurrences").fetchall(), [(42,)])
            self.assertEqual({row[0] for row in connection.execute("SELECT user_id FROM occurrences")}, {"0007", "7"})

    def test_validation_only_session_does_not_create_training_features(self):
        path, connection = self.database()
        writer = OccurrenceWriter(connection)
        start = WINDOWS[1].start + 100
        writer.add(View("0007", 6, start, start + 100, 1), Airing(6, 42, start, start + 100))
        writer.flush()
        profile = self.root / "train_profile.csv"
        export_profiles(path, EVIDENCE_WINDOWS[1], profile)
        self.assertEqual(list(iter_profiles(profile, "train")), [])

    def test_covering_index_preserves_watch_evidence_and_exports(self):
        path, connection = self.database()
        writer = OccurrenceWriter(connection)
        airing = Airing(6, 42, self.start, self.start + 100)
        writer.add(View("0007", 6, self.start, self.start + 81, 1), airing)
        writer.flush()
        before = connection.execute("SELECT * FROM occurrences").fetchall()
        details = add_profile_scan_index(path)
        self.assertEqual(before, connection.execute("SELECT * FROM occurrences").fetchall())
        self.assertTrue(any("COVERING INDEX occurrence_profile_scan" in row[3] for row in details["query_plan"]))
        profile = self.root / "indexed_profile.csv"
        export_profiles(path, EVIDENCE_WINDOWS[1], profile)
        self.assertEqual(next(iter_profiles(profile, "train"))["frequency_gt80"], 1)


class RepeatRankingTests(unittest.TestCase):
    def record(self, item, frequency, last, **changes):
        row = {"user_id": "0007", "tv_show_id": item, "frequency_ge30": frequency,
               "frequency_gt50": frequency, "frequency_gt80": frequency,
               "recency_gt50_half14": float(frequency), "recency_gt50_half28": float(frequency),
               "qualified_seconds_gt50": 100 * frequency,
               "last_ge30_seconds": last, "last_gt50_seconds": last, "last_gt80_seconds": last,
               "has_120_second_evidence": 0}
        row.update(changes)
        return row

    def test_personal_frequency_is_not_overridden_by_global_popularity(self):
        history = [self.record(item, 1, 10) for item in (6, 7, 8, 9, 10)]
        prediction, diagnostics = recommend_repeat(history, CONFIGURATIONS[1], frozenset(range(1, 11)), (1, 2, 3, 4, 5))
        self.assertEqual(prediction, (6, 7, 8, 9, 10))
        self.assertEqual(diagnostics["fallback_slots"], 0)

    def test_ties_use_personal_recency_then_programme_id(self):
        history = [self.record(7, 2, 20), self.record(6, 2, 20), self.record(8, 2, 10)]
        first, _ = recommend_repeat(history, CONFIGURATIONS[1], frozenset(range(1, 10)), (1, 2, 3, 4, 5))
        second, _ = recommend_repeat(history[::-1], CONFIGURATIONS[1], frozenset(range(1, 10)), (1, 2, 3, 4, 5))
        self.assertEqual(first, (6, 7, 8, 1, 2))
        self.assertEqual(first, second)

    def test_missing_personal_slots_use_unique_available_fallback(self):
        history = [self.record(99, 100, 100), self.record(1, 1, 10)]
        prediction, diagnostics = recommend_repeat(history, CONFIGURATIONS[1], frozenset(range(1, 7)), (1, 2, 3, 4, 5))
        self.assertEqual(prediction, (1, 2, 3, 4, 5))
        self.assertEqual(diagnostics["personal_slots"], 1)
        empty, _ = recommend_repeat([], CONFIGURATIONS[1], frozenset(range(1, 7)), (5, 4, 3, 2, 1))
        self.assertEqual(empty, (5, 4, 3, 2, 1))

    def test_count_recency_and_duration_are_independent_scoring_variants(self):
        history = [self.record(6, 3, 10, recency_gt50_half14=.5, qualified_seconds_gt50=300),
                   self.record(7, 2, 20, recency_gt50_half14=1.5, qualified_seconds_gt50=1000)]
        args = (frozenset(range(1, 8)), (1, 2, 3, 4, 5))
        frequency, _ = recommend_repeat(history, CONFIG_BY_NAME["B_frequency_gt50"], *args)
        recency, _ = recommend_repeat(history, CONFIG_BY_NAME["D_recency_gt50_half14"], *args)
        duration, _ = recommend_repeat(history, CONFIG_BY_NAME["E_qualified_seconds_gt50"], *args)
        self.assertEqual(frequency[0], 6)
        self.assertEqual(recency[0], 7)
        self.assertEqual(duration[0], 7)

    def test_robust_selection_uses_both_strict_holdouts_not_broad_score(self):
        def experiment(name, primary, earlier):
            return {"configuration": {"name": name}, "folds": {
                fold: {"strict_evaluation": {"metrics": {"primary_all_users": {"map_at_5": score}}}}
                for fold, score in (("primary", primary), ("earlier", earlier))}}
        entries = [experiment("unstable", .9, .2), experiment("robust", .6, .5), experiment("lower", .4, .8)]
        self.assertEqual(select_robust_model(entries)["configuration"]["name"], "robust")

    def test_common_warm_cohort_remains_separate_from_model_profile_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.csv"
            last = WINDOWS[0].start + 100
            record = self.record(6, 1, last)
            write_csv(path, PROFILE_COLUMNS, [[record[column] for column in PROFILE_COLUMNS]])
            context = profile_context(path, "train", ["0007", "7"])
            self.assertEqual(context["common_training_users"], set())
            predictions, coverage = predict_configurations(path, "train", ["0007", "7"], frozenset(range(1, 8)), (1, 2, 3, 4, 5))
            self.assertEqual(predictions["B_frequency_gt50"]["0007"], (6, 1, 2, 3, 4))
            self.assertEqual(predictions["B_frequency_gt50"]["7"], (1, 2, 3, 4, 5))
            self.assertEqual(coverage["B_frequency_gt50"]["users_with_no_available_personal_candidates"], 1)


if __name__ == "__main__":
    unittest.main()
