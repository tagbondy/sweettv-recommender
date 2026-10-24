"""Real native fits, query grouping, candidate diagnostics, ablations, and fusion."""

import json
import math
import sys
import unittest
from array import array
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from lightgbm_v4 import (
    LGBConfig, ablation_indices, candidate_metrics, expanded_pool, fuse_ranks,
    inner_group_split, matrix, predict_model, query_weights, train_model, user_rank_maps,
)
from rerank_v3 import FEATURE_NAMES, FI, FeatureTable, candidate_union
from evaluate import evaluate_recommendations


def toy_table():
    table = FeatureTable("past", 100, 200, labels=bytearray(), fallback=(1, 2, 3, 4, 5))
    for query in range(20):
        user = "0007" if query == 0 else "7" if query == 1 else f"user{query:02}"
        start = len(table.rows)
        for item in range(1, 7):
            row = [0.] * len(FEATURE_NAMES)
            row[FI["frequency_ge30"]] = 10. if item in (2, 6) else 0.
            row[FI["user_distinct_programmes"]] = float(query)
            row[FI["popularity_rank"]] = item
            row[FI["source_repeat"]] = float(item in (2, 6))
            row[FI["source_popularity"]] = 1.
            table.x.extend(row)
            table.rows.append((user, item))
            table.labels.append(int(item in (2, 6)))
        table.groups.append((user, start, len(table.rows)))
    return table


class CandidateDiagnosticTests(unittest.TestCase):
    def test_diagnostics_include_misses_sources_and_limited_history(self):
        table = toy_table()
        users = [user for user, _, _ in table.groups]
        truth = {user: {2, 6, 99} for user in users}
        metrics = candidate_metrics(table, truth, users)
        self.assertEqual(metrics["union"]["missing_relevant_items"], 20)
        self.assertAlmostEqual(metrics["union"]["micro_recall"], 2 / 3)
        self.assertAlmostEqual(metrics["union"]["candidate_oracle_map_at_5"], 2 / 3)
        self.assertAlmostEqual(metrics["by_source"]["repeat"]["micro_recall"], 2 / 3)
        self.assertEqual(metrics["history_buckets"]["0"]["users"], 1)
        self.assertEqual(metrics["history_buckets"]["1-5"]["users"], 5)
        self.assertFalse(metrics["ground_truth_filtered_or_injected"])

    def test_controlled_expansion_contains_base_without_gold_injection(self):
        available = frozenset(range(1, 301))
        history = [{"tv_show_id": item, "frequency_ge30": 300 - item, "last_ge30_seconds": item} for item in range(1, 151)]
        associations = {item: {"best": 1 / item} for item in range(100, 251)}
        base, _ = candidate_union(history, associations, list(range(1, 301)), available)
        expanded, ranks = expanded_pool(history, associations, list(range(1, 301)), available)
        self.assertTrue(base <= expanded <= available)
        self.assertEqual(len(ranks["sources"]["repeat"]), 100)
        self.assertEqual(len(ranks["sources"]["association"]), 100)
        self.assertEqual(len(ranks["sources"]["popularity"]), 40)
        self.assertNotIn(299, expanded)

    def test_ablation_masks_remove_groups_and_direct_proxies(self):
        for group in ("personal_history", "association", "future_schedule", "global_popularity"):
            kept = ablation_indices(group)
            self.assertTrue(0 < len(kept) < len(FEATURE_NAMES))
            self.assertEqual(kept, sorted(set(kept)))
        association = {FEATURE_NAMES[index] for index in ablation_indices("association")}
        self.assertNotIn("association_best_confidence", association)
        self.assertNotIn("source_count", association)
        schedule = {FEATURE_NAMES[index] for index in ablation_indices("future_schedule")}
        self.assertNotIn("future_airings", schedule)
        self.assertNotIn("expected_future_qualified_airings", schedule)


class NativeLightGBMTests(unittest.TestCase):
    def test_fitting_side_split_preserves_complete_user_groups(self):
        table = toy_table()
        train, valid, train_sizes, valid_sizes = inner_group_split(table)
        self.assertEqual(len(train) + len(valid), len(table.rows))
        self.assertFalse(set(train) & set(valid))
        self.assertEqual(sum(train_sizes), len(train))
        self.assertEqual(sum(valid_sizes), len(valid))
        train_users = {table.rows[index][0] for index in train}
        valid_users = {table.rows[index][0] for index in valid}
        self.assertFalse(train_users & valid_users)
        self.assertIn("0007", train_users | valid_users)
        self.assertIn("7", train_users | valid_users)

    def test_real_binary_fit_is_finite_and_deterministic(self):
        table = toy_table()
        config = LGBConfig("toy_binary", "binary", 3, .1, 2, 1., max_rounds=8, stopping_rounds=3, num_threads=1)
        first, info = train_model(table, config, 200)
        second, other = train_model(table, config, 200)
        predictions = predict_model(first, table, info["feature_indices"])
        self.assertTrue(np.isfinite(predictions).all())
        self.assertTrue(np.array_equal(predictions, predict_model(second, table, other["feature_indices"])))
        self.assertGreater(predictions[1], predictions[0])
        self.assertLessEqual(info["rounds"], config.max_rounds)
        self.assertEqual(info["native_params"]["deterministic"], True)
        self.assertTrue(any(item["gain"] > 0 for item in info["feature_importance"]))
        json.dumps(info)

    def test_real_lambdarank_fit_and_external_map_are_distinct_from_ndcg(self):
        table = toy_table()
        config = LGBConfig("toy_rank", "lambdarank", 3, .1, 2, 1., max_rounds=5, stopping_rounds=2, num_threads=1)
        booster, info = train_model(table, config, 200)
        predictions = predict_model(booster, table, info["feature_indices"])
        self.assertTrue(np.isfinite(predictions).all())
        self.assertEqual(info["native_params"]["eval_at"], [5])
        self.assertIn("NDCG@5", info["inner_early_stopping"]["metric_role"])
        external = evaluate_recommendations(["u"], {"u": {1, 3}}, {"u": [1, 2, 3, 4, 5]}, set())
        self.assertAlmostEqual(external["metrics"]["primary_all_users"]["map_at_5"], 5 / 6)

    def test_future_labels_blocked_before_native_training(self):
        table = toy_table()
        config = LGBConfig("bad", "binary", 3, .1, 2, 1., max_rounds=2)
        with self.assertRaisesRegex(ValueError, "prediction origin"):
            train_model(table, config, 100)

    def test_fixed_round_final_refit_does_not_need_hidden_labels(self):
        table = toy_table()
        config = LGBConfig("final", "binary", 3, .1, 2, 1., max_rounds=5, num_threads=1)
        model, info = train_model(table, config, 200, rounds=3)
        self.assertIsNone(info["inner_early_stopping"])
        self.assertEqual(info["rounds"], 3)
        self.assertEqual(model.num_trees(), 3)
        self.assertAlmostEqual(float(query_weights(table).sum()), len(table.rows))


class EnsembleTests(unittest.TestCase):
    def test_rank_fusion_aligns_by_user_and_item_not_row_positions(self):
        users = ["0007", "7"]
        maps = [
            {"0007": {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}, "7": {5: 1, 4: 2, 3: 3, 2: 4, 1: 5}},
            {"7": {6: 1, 5: 2, 4: 3, 3: 4, 2: 5}, "0007": {6: 1, 1: 2, 2: 3, 3: 4, 4: 5}},
        ]
        available = frozenset(range(1, 7))
        result = fuse_ranks(maps, (.5, .5), users, available, (1, 2, 3, 4, 5), set())
        self.assertEqual(result["0007"][0], 1)
        self.assertEqual(result["7"][0], 5)
        self.assertTrue(all(len(items) == len(set(items)) == 5 for items in result.values()))
        cold = fuse_ranks(maps, (.5, .5), users, available, (1, 2, 3, 4, 5), {"0007"})
        self.assertEqual(cold["0007"], (1, 2, 3, 4, 5))

    def test_invalid_weights_and_unavailable_votes_rejected(self):
        with self.assertRaises(ValueError):
            fuse_ranks([{"u": {1: 1}}], (.5,), ["u"], frozenset(range(1, 6)), (1, 2, 3, 4, 5), set())
        with self.assertRaises(ValueError):
            fuse_ranks([{"u": {99: 1}}], (1.,), ["u"], frozenset(range(1, 6)), (1, 2, 3, 4, 5), set())

    def test_rank_map_ties_are_deterministic(self):
        table = toy_table()
        users = [user for user, _, _ in table.groups]
        maps = user_rank_maps(table, np.zeros(len(table.rows)), users)
        self.assertEqual(maps["0007"], {item: item for item in range(1, 7)})
        self.assertEqual(matrix(table).dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
