"""Candidate/feature contracts, temporal label barriers, and simple rerankers."""

import csv
import math
import sqlite3
import sys
import tempfile
import unittest
from array import array
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from association_rules import AssociationCounts, BasketItem, RuleConfig, build_rules
from audit_data import SCHEDULE_COLUMNS
from build_interactions import Window
from rerank_v3 import (
    FEATURE_NAMES, FI, MISSING_RANK, FeatureTable, ModelConfig, association_features,
    candidate_diagnostics, candidate_union, concatenate_labeled, enforce_fit_cutoff,
    fit_linear, fit_scaler, load_table, rank_table, save_table, scaled_matrix,
    schedule_statistics, scores_for,
)


def repeat_record(item, frequency, last):
    return {"tv_show_id": item, "frequency_ge30": frequency, "last_ge30_seconds": last}


def toy_table(name="past", history_end=100, label_end=200):
    table = FeatureTable(name, history_end, label_end, labels=bytearray(), fallback=(1, 2, 3, 4, 5))
    for user in ("0007", "7"):
        start = len(table.rows)
        for item in range(1, 7):
            values = [0.] * len(FEATURE_NAMES)
            values[FI["frequency_ge30"]] = 5. if item in (2, 6) else 0.
            values[FI["repeat_rank"]] = item
            values[FI["popularity_rank"]] = item
            values[FI["source_repeat"]] = 1.
            values[FI["source_popularity"]] = 1.
            table.rows.append((user, item))
            table.x.extend(values)
            table.labels.append(int(item in (2, 6)))
        table.groups.append((user, start, len(table.rows)))
    return table


class CandidateTests(unittest.TestCase):
    def test_union_is_unique_and_restricted_to_availability(self):
        available = frozenset(range(1, 101))
        history = [repeat_record(item, 100 - item, item) for item in range(1, 81)]
        history.append(repeat_record(999, 1000, 1000))
        associations = {item: {"best": 1 / item} for item in range(20, 90)}
        pool, ranks = candidate_union(history, associations, list(range(1, 101)), available)
        self.assertTrue(pool <= available)
        self.assertNotIn(999, pool)
        self.assertLessEqual(len(pool), 120)
        self.assertEqual(len(ranks["sources"]["repeat"]), 50)
        self.assertEqual(len(ranks["sources"]["association"]), 50)
        self.assertEqual(len(ranks["sources"]["popularity"]), 20)
        self.assertEqual(ranks["repeat_rank"][1], 1)

    def test_repeat_ties_keep_user_recency_then_programme_id(self):
        history = [repeat_record(8, 2, 100), repeat_record(7, 2, 200), repeat_record(6, 2, 200)]
        _, ranks = candidate_union(history, {}, list(range(1, 10)), frozenset(range(1, 10)))
        self.assertEqual([ranks["repeat_rank"][item] for item in (6, 7, 8)], [1, 2, 3])

    def test_unknown_history_still_has_five_popularity_candidates(self):
        pool, ranks = candidate_union([], {}, list(range(1, 8)), frozenset(range(1, 8)))
        self.assertEqual(pool, set(range(1, 8)))
        self.assertEqual(ranks["sources"]["repeat"], set())

    def test_association_features_agree_with_counted_confidence_and_lift(self):
        with tempfile.TemporaryDirectory() as directory:
            baskets = {"a": (BasketItem(1, 1.), BasketItem(2, 1.)),
                       "b": (BasketItem(1, 1.), BasketItem(2, 1.)), "c": (BasketItem(2, 1.),)}
            counts = AssociationCounts(baskets, Path(directory) / "pairs.sqlite3")
            try:
                rules, _ = build_rules(counts, RuleConfig("test", min_pair_support=1, min_confidence=0))
                features = association_features((BasketItem(1, 1.),), rules, counts.support, counts.user_count, frozenset([1, 2]))
                self.assertEqual(features[2]["best"], 1.)
                self.assertEqual(features[2]["pair"], 2)
                self.assertEqual(features[2]["n"], 1)
                self.assertAlmostEqual(features[2]["lift"], counts.lift(1, 2))
            finally:
                counts.close()

    def test_availability_features_deduplicate_and_clip_without_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "schedule.csv"
            columns = [6, "Channel", "01.01.2020 00:00:00", 100, "Unused title", 1, "Unused category"] + [""] * 6
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(SCHEDULE_COLUMNS)
                writer.writerows([columns, columns, [*columns[:5], 0, *columns[6:]]])
            stats = schedule_statistics([path], Window("target", 50, 150))
            self.assertEqual(set(stats), {1})
            self.assertEqual(stats[1]["airings"], 1)
            self.assertEqual(stats[1]["duration"], 50)
            self.assertEqual(stats[1]["first"], 50)


class LeakageAndModelTests(unittest.TestCase):
    def test_label_cutoff_blocks_training_on_primary_holdout(self):
        primary = toy_table("primary", history_end=100, label_end=200)
        with self.assertRaisesRegex(ValueError, "prediction origin"):
            enforce_fit_cutoff(primary, 100)
        enforce_fit_cutoff(primary, 200)
        unlabelled = FeatureTable("test", 200, None)
        with self.assertRaises(ValueError):
            enforce_fit_cutoff(unlabelled, 200)

    def test_standardization_fits_feature_values_not_validation_labels(self):
        table = toy_table()
        before = table.x.tobytes()
        first = fit_scaler(table)
        table.labels = bytearray(1 - value for value in table.labels)
        self.assertEqual(first, fit_scaler(table))
        self.assertEqual(before, table.x.tobytes())
        transformed = scaled_matrix(table, first)
        self.assertTrue(all(math.isfinite(value) and abs(value) <= 8 for value in transformed))

    def test_training_is_deterministic_and_learns_known_signal(self):
        table = toy_table()
        config = ModelConfig("toy", "logistic", epochs=15, batch_size=4, learning_rate=.2)
        first, second = fit_linear(table, config, 200), fit_linear(table, config, 200)
        self.assertEqual(first["weights"], second["weights"])
        self.assertEqual(first["bias"], second["bias"])
        scores = scores_for(table, config, first)
        self.assertGreater(scores[1], scores[0])
        predictions = rank_table(table, scores, ["0007", "7"])
        self.assertEqual(set(predictions), {"0007", "7"})
        self.assertEqual(set(predictions["0007"][:2]), {2, 6})

    def test_pairwise_model_uses_only_within_query_past_labels(self):
        table = toy_table()
        config = ModelConfig("toy_rank", "pairwise", epochs=10, batch_size=4, learning_rate=.1)
        model = fit_linear(table, config, 200)
        self.assertGreater(model["pairwise_training_pairs"], 0)
        scores = scores_for(table, config, model)
        self.assertGreater(scores[1], scores[0])
        with self.assertRaises(ValueError):
            fit_linear(table, config, 100)

    def test_rank_fusion_and_cold_start_are_deterministic(self):
        table = toy_table()
        config = ModelConfig("rrf", "fusion", (.8, .15, .05))
        first = rank_table(table, scores_for(table, config), ["0007", "7"])
        second = rank_table(table, scores_for(table, config), ["0007", "7"])
        self.assertEqual(first, second)
        for row in range(6):
            table.x[row * len(FEATURE_NAMES) + FI["cold_start"]] = 1.
        cold = rank_table(table, array("d", [100. * row for row in range(12)]), ["0007", "7"])
        self.assertEqual(cold["0007"], table.fallback)

    def test_candidate_oracle_keeps_unretrieved_ground_truth(self):
        table = toy_table()
        truth = {"0007": {1, 2, 3, 4, 99}, "7": set()}
        result = candidate_diagnostics(table, truth, ["0007", "7"])
        self.assertEqual(result["relevant_user_programme_pairs"], 5)
        self.assertEqual(result["relevant_pairs_recalled_by_generators"], 4)
        self.assertEqual(result["candidate_oracle_map_at_5"], .4)
        self.assertFalse(result["ground_truth_filtered_to_candidates"])

    def test_final_refit_concatenates_only_allowed_historical_targets(self):
        earlier = toy_table("early", 50, 100)
        primary = toy_table("primary", 100, 200)
        joined = concatenate_labeled([earlier, primary], 200)
        self.assertEqual(len(joined.rows), 24)
        self.assertEqual(joined.label_end, 200)
        self.assertEqual(joined.groups[2][1], 12)
        self.assertEqual(joined.groups[2][0], "primary:0007")
        with self.assertRaises(ValueError):
            concatenate_labeled([earlier, primary], 100)

    def test_feature_cache_roundtrip_and_labels_absent_for_final(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data/processed").mkdir(parents=True)
            table = toy_table("early")
            save_table(table, root)
            cached = load_table(root, "early")
            self.assertEqual(cached.x.tobytes(), table.x.tobytes())
            self.assertEqual(cached.rows, table.rows)
            self.assertEqual(cached.labels, table.labels)
            table.name, table.label_end, table.labels = "final", None, None
            save_table(table, root)
            self.assertIsNone(load_table(root, "final").labels)
            with (root / "data/processed/rerank_v3_final_rows.csv").open(newline="") as handle:
                self.assertEqual(next(csv.reader(handle)), ["user_id", "tv_show_id"])


if __name__ == "__main__":
    unittest.main()
