"""Known-answer association statistics, basket bounds, ranking, and leakage tests."""

import csv
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from build_interactions import PAIR_COLUMNS
from evaluate import AggregateRecord
from association_rules import (
    AssociationCounts, BasketItem, RuleConfig, build_rules, generate_predictions,
    load_training_baskets, recommend_basket, select_basket,
)
from diagnose_evaluation import independent_ap


def basket(*items):
    return tuple(BasketItem(item, math.log(2)) for item in items)


def row(item, count=1, duration=120):
    return AggregateRecord("0007", item, count, duration, "2020-03-09 00:00:00", "2020-03-09 01:00:00")


def write_pairs(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(PAIR_COLUMNS)
        writer.writerows(rows)


class AssociationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.counts = AssociationCounts({"a": basket(1, 2, 3), "b": basket(1, 2),
                                         "c": basket(2, 3), "d": basket(4)}, self.root / "pairs.sqlite3", batch_limit=2)

    def tearDown(self):
        self.counts.close()
        self.temporary.cleanup()

    def test_support_counts_and_cooccurrence(self):
        self.assertEqual(dict(self.counts.support), {1: 2, 2: 3, 3: 2, 4: 1})
        self.assertEqual(self.counts.pair_count(1, 2), 2)
        self.assertEqual(self.counts.pair_count(2, 1), 2)
        self.assertEqual(self.counts.pair_count(1, 3), 1)
        self.assertEqual(self.counts.pair_count(2, 3), 2)
        self.assertEqual(self.counts.pair_count(1, 1), 0)
        self.assertEqual(self.counts.pair_count(1, 4), 0)

    def test_directional_confidence_and_lift(self):
        self.assertEqual(self.counts.confidence(1, 2), 1)
        self.assertAlmostEqual(self.counts.confidence(2, 1), 2 / 3)
        self.assertAlmostEqual(self.counts.lift(1, 2), 4 / 3)
        self.assertEqual(self.counts.confidence(999, 2), 0)

    def test_rule_support_confidence_and_neighbor_filters(self):
        rules, stats = build_rules(self.counts, RuleConfig("small", min_pair_support=2, min_confidence=.7, max_neighbors=1))
        self.assertEqual(rules[1], ((2, 1.0),))
        self.assertEqual(rules[3], ((2, 1.0),))
        self.assertNotIn(2, rules)
        self.assertEqual(stats["retained_directional_rules"], 2)

    def test_rule_shrinkage_is_applied_without_popularity_fusion(self):
        rules, _ = build_rules(self.counts, RuleConfig("shrunk", min_pair_support=2, min_confidence=0, shrinkage=2))
        self.assertEqual(dict(rules[1])[2], .5)

    def test_duplicate_items_never_multiply_user_support_or_pair_counts(self):
        duplicate_counts = AssociationCounts({"u": basket(1, 1, 2, 2)}, self.root / "duplicates.sqlite3")
        try:
            self.assertEqual(dict(duplicate_counts.support), {1: 1, 2: 1})
            self.assertEqual(duplicate_counts.pair_count(1, 2), 1)
        finally:
            duplicate_counts.close()

    def test_mining_cap_is_enforced(self):
        with self.assertRaises(ValueError):
            AssociationCounts({"u": basket(1, 2, 3)}, self.root / "too_large.sqlite3", max_items=2)


class BasketAndRankingTests(unittest.TestCase):
    def test_basket_cap_uniqueness_and_deterministic_selection(self):
        rows = [row(3, 2, 500), row(1, 2, 500), row(2, 3, 600), row(2, 3, 600)]
        selected = select_basket(rows, max_items=2)
        self.assertEqual([item.programme_id for item in selected], [1, 2])
        self.assertAlmostEqual(selected[1].log_view_count, math.log(4))
        self.assertEqual(select_basket(reversed(rows), 2), selected)

    def test_personalized_ranking_and_preserved_user_strings(self):
        rules = {1: ((6, .9),), 2: ((7, .9),)}
        users = ["0007", "7", "cold"]
        predictions, _ = generate_predictions(users, {"0007": basket(1), "7": basket(2)}, rules,
                                              frozenset(range(1, 8)), (1, 2, 3, 4, 5), RuleConfig("example"))
        self.assertEqual(predictions["0007"], (6, 1, 2, 3, 4))
        self.assertEqual(predictions["7"], (7, 1, 2, 3, 4))
        self.assertEqual(predictions["cold"], (1, 2, 3, 4, 5))

    def test_missing_rules_and_cold_start_fallback(self):
        for query in ((), basket(999)):
            predictions, stats = recommend_basket(query, {}, frozenset(range(1, 6)), (5, 4, 3, 2, 1), RuleConfig("example"))
            self.assertEqual(predictions, (5, 4, 3, 2, 1))
            self.assertEqual(stats["fallback_slots"], 5)

    def test_candidate_restrictions_numeric_ties_and_no_duplicate_filling(self):
        rules = {1: ((99, 100), (6, .8), (1, .8), (7, .8))}
        prediction, _ = recommend_basket(basket(1, 1), rules, frozenset(range(1, 8)), (1, 2, 3, 4, 5), RuleConfig("example"))
        self.assertEqual(prediction, (1, 6, 7, 2, 3))
        self.assertEqual(len(set(prediction)), 5)
        self.assertNotIn(99, prediction)

    def test_sum_and_max_variants_have_known_different_rankings(self):
        rules = {1: ((3, .6), (4, .8)), 2: ((3, .6),)}
        args = (basket(1, 2), rules, frozenset(range(1, 6)), (1, 2, 3, 4, 5))
        summed, _ = recommend_basket(*args, RuleConfig("sum", "confidence_sum"))
        maximum, _ = recommend_basket(*args, RuleConfig("max", "confidence_max"))
        self.assertEqual(summed[:2], (3, 4))
        self.assertEqual(maximum[:2], (4, 3))

    def test_weighted_variant_and_duplicate_query_items(self):
        query = (BasketItem(1, 1.0), BasketItem(2, 3.0), BasketItem(2, 3.0))
        rules = {1: ((3, 1.0),), 2: ((4, 1.0),)}
        prediction, stats = recommend_basket(query, rules, frozenset(range(1, 6)), (1, 2, 3, 4, 5),
                                             RuleConfig("weighted", "weighted_confidence_sum"))
        self.assertEqual(prediction[:2], (4, 3))
        self.assertEqual(stats["top_five"][0]["association_score"], 3)

    def test_determinism_is_independent_of_query_and_rule_insertion_order(self):
        rules = {1: ((6, .5), (7, .5)), 2: ((6, .5), (7, .5))}
        candidates = frozenset(range(1, 8))
        first, _ = recommend_basket(basket(1, 2), rules, candidates, (1, 2, 3, 4, 5), RuleConfig("example"))
        other = {2: tuple(reversed(rules[2])), 1: tuple(reversed(rules[1]))}
        second, _ = recommend_basket(basket(2, 1), other, candidates, (1, 2, 3, 4, 5), RuleConfig("example"))
        self.assertEqual(first, second)

    def test_training_reader_rejects_validation_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "looks_like_train.csv"
            write_pairs(path, [["u", 1, 1, 120, "2020-05-18 00:00:00", "2020-05-18 00:02:00"]])
            with self.assertRaisesRegex(ValueError, "outside train"):
                load_training_baskets(path)

    def test_changing_validation_file_does_not_change_training_baskets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training, validation = root / "train.csv", root / "validation.csv"
            write_pairs(training, [["0007", 1, 1, 120, "2020-03-09 00:00:00", "2020-03-09 00:02:00"]])
            write_pairs(validation, [["0007", 6, 1, 120, "2020-05-18 00:00:00", "2020-05-18 00:02:00"]])
            first, _ = load_training_baskets(training)
            write_pairs(validation, [["0007", 99, 100, 12000, "2020-05-18 00:00:00", "2020-05-18 04:00:00"]])
            second, _ = load_training_baskets(training)
            self.assertEqual(first, second)

    def test_independent_rational_ap_known_example(self):
        self.assertEqual(str(independent_ap({1, 3}, [1, 2, 3, 4, 5])), "5/6")


if __name__ == "__main__":
    unittest.main()
