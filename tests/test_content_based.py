"""Content feature, cosine, profile, fallback, and temporal-isolation tests."""

import csv
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from audit_data import SCHEDULE_COLUMNS
from build_interactions import PAIR_COLUMNS
from evaluate import AggregateRecord
from content_based import (
    FIELDS, CBFConfig, ContentIndex, build_user_profile, cosine_similarity,
    extract_metadata_features, fit_item_vectors, generate_predictions,
    load_training_metadata, predict_validation, recommend,
)


def write_csv(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def airing(programme, start="09.03.2020 00:00:00", category="Drama", genre="Crime", actors="Actor One", duration=600):
    return ["6", "Channel", start, str(duration), "Programme Title", str(programme), category, genre, "", "", "2010-2019", "Director One", actors]


def history_row(programme, count=1, seconds=120):
    return AggregateRecord("0007", programme, count, seconds, "2020-03-09 00:00:00", "2020-03-09 01:00:00")


class FeatureTests(unittest.TestCase):
    def test_metadata_extraction_normalizes_and_namespaces_fields(self):
        row = dict(zip(SCHEDULE_COLUMNS, airing(1, category="  DRAMA  ", genre="Crime", actors="Actor One; Actor Two|Actor One")))
        row["tv_show_genre_2"] = " crime "
        groups = extract_metadata_features(row)
        self.assertEqual(groups["category"], {"drama"})
        self.assertEqual(groups["genre"], {"crime"})
        self.assertEqual(groups["actor"], {"actor one", "actor two"})
        self.assertEqual(groups["director"], {"director one"})
        self.assertEqual(groups["year_decade"], {"2010"})
        self.assertEqual(groups["title"], set())
        self.assertEqual(extract_metadata_features(row, include_title=True)["title"], {"programme", "title"})

    def test_missing_metadata_does_not_create_fake_features(self):
        row = {key: "" for key in SCHEDULE_COLUMNS}
        row.update({"tv_show_category": "unknown", "actors": "0", "director": "N/A", "year_of_production": "-"})
        self.assertTrue(all(not values for values in extract_metadata_features(row).values()))
        model = fit_item_vectors({1: extract_metadata_features(row)}, CBFConfig())
        self.assertEqual(model.vectors[1], {})
        self.assertEqual(model.statistics["empty_item_vectors"], 1)

    def test_timestamp_filter_excludes_zero_ids_and_future_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train_schedule.csv"
            write_csv(path, SCHEDULE_COLUMNS, [airing(0), airing(1),
                      airing(1, start="18.05.2020 00:00:00", actors="Future Actor"),
                      airing(2, start="08.03.2020 23:59:00", duration=120)])
            metadata, stats = load_training_metadata(path, CBFConfig())
            self.assertEqual(set(metadata), {1, 2})
            self.assertNotIn("future actor", metadata[1]["actor"])
            self.assertEqual(stats["labeled_rows_outside_training_window"], 1)
            self.assertEqual(stats["zero_id_rows_excluded"], 1)

    def test_field_weights_and_item_vectors_are_normalized(self):
        groups = {field: set() for field in FIELDS}
        groups["category"], groups["genre"] = {"drama"}, {"crime"}
        model = fit_item_vectors({1: groups}, CBFConfig())
        vector = model.vectors[1]
        by_name = {model.feature_keys[feature][0]: value for feature, value in vector.items()}
        self.assertAlmostEqual(by_name["category"], 2 / math.sqrt(13))
        self.assertAlmostEqual(by_name["genre"], 3 / math.sqrt(13))
        self.assertAlmostEqual(sum(value * value for value in vector.values()), 1)

    def test_idf_and_feature_order_depend_only_on_training_documents(self):
        first = {field: set() for field in FIELDS}
        first["category"], first["genre"] = {"drama"}, {"rare"}
        second = {field: set() for field in FIELDS}
        second["category"] = {"drama"}
        model = fit_item_vectors({2: second, 1: first}, CBFConfig())
        alternate = fit_item_vectors({1: first, 2: second}, CBFConfig())
        self.assertEqual(model.vectors, alternate.vectors)
        self.assertEqual(model.feature_keys, alternate.feature_keys)
        idf_by_key = {key: model.idf[position] for position, key in enumerate(model.feature_keys)}
        self.assertEqual(idf_by_key[("category", "drama")], 1)
        self.assertAlmostEqual(idf_by_key[("genre", "rare")], math.log(3 / 2) + 1)

    def test_repeated_airing_metadata_is_binary_not_frequency_weighted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "schedule.csv"
            write_csv(path, SCHEDULE_COLUMNS, [airing(1), airing(1), airing(2)])
            metadata, _ = load_training_metadata(path, CBFConfig())
            model = fit_item_vectors(metadata, CBFConfig())
            self.assertEqual(model.vectors[1], model.vectors[2])


class ProfileAndRankingTests(unittest.TestCase):
    def test_log_count_weighted_profile_and_unknown_item_handling(self):
        profile, stats = build_user_profile([history_row(1, 1), history_row(2, 3), history_row(3)],
                                             {1: {0: 1.0}, 2: {1: 1.0}}, CBFConfig())
        self.assertAlmostEqual(profile[0], 1 / math.sqrt(5))
        self.assertAlmostEqual(profile[1], 2 / math.sqrt(5))
        self.assertEqual(stats["programmes_without_usable_metadata"], 1)
        self.assertEqual(stats["programmes_contributing_features"], 2)

    def test_cosine_known_answers_and_zero_vectors(self):
        self.assertAlmostEqual(cosine_similarity({0: 3, 1: 4}, {0: 2}), .6)
        self.assertEqual(cosine_similarity({}, {0: 1}), 0)
        self.assertEqual(cosine_similarity({0: 1}, {1: 1}), 0)
        self.assertAlmostEqual(cosine_similarity({0: 3, 1: 4}, {0: 3, 1: 4}), 1)

    def test_cold_start_is_exact_popularity_fallback(self):
        fallback = (5, 4, 3, 2, 1)
        predictions, diagnostics = recommend({}, ContentIndex({1: {0: 1}}, frozenset(range(1, 7))), fallback)
        self.assertEqual(predictions, fallback)
        self.assertEqual(diagnostics["popularity_fallback_slots"], 5)

    def test_candidate_restrictions_deterministic_ties_and_unique_filling(self):
        candidates = frozenset(range(1, 8))
        vectors = {99: {0: 1.0}, 7: {0: 1.0}, 6: {0: 1.0}, 1: {1: 1.0}}
        first, diagnostics = recommend({0: 1.0}, ContentIndex(vectors, candidates), (1, 2, 3, 4, 5))
        second, _ = recommend({0: 1.0}, ContentIndex(dict(reversed(list(vectors.items()))), candidates), (1, 2, 3, 4, 5))
        self.assertEqual(first, (6, 7, 1, 2, 3))
        self.assertEqual(first, second)
        self.assertEqual(diagnostics["content_slots"], 2)
        self.assertEqual(len(set(first)), 5)
        self.assertNotIn(99, first)

    def test_fallback_does_not_repeat_a_content_choice(self):
        result, _ = recommend({0: 1.0}, ContentIndex({1: {0: 1.0}}, frozenset(range(1, 6))), (1, 2, 3, 4, 5))
        self.assertEqual(result, (1, 2, 3, 4, 5))

    def test_sparse_index_equals_direct_cosine(self):
        vectors = {1: {0: .6, 1: .8}, 2: {1: 1.0}}
        profile = {0: 1.0}
        scores = ContentIndex(vectors, frozenset([1, 2])).score(profile)
        self.assertAlmostEqual(scores[1], cosine_similarity(profile, vectors[1]))
        self.assertEqual(scores.get(2, 0), 0)

    def test_history_without_metadata_preserves_string_ids_and_warm_status(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.csv"
            write_csv(path, PAIR_COLUMNS, [["0007", 99, 1, 120, "2020-03-09 00:00:00", "2020-03-09 00:02:00"],
                                         ["7", 1, 1, 120, "2020-03-09 00:00:00", "2020-03-09 00:02:00"]])
            predictions, training_users, stats = generate_predictions(path, ["0007", "7", "cold"], {1: {0: 1.0}},
                frozenset(range(1, 6)), (1, 2, 3, 4, 5), CBFConfig())
            self.assertEqual(training_users, {"0007", "7"})
            self.assertEqual(predictions["0007"], predictions["cold"])
            self.assertEqual(stats["per_user_profiles_and_ranking"]["0007"]["reason"], "history_without_usable_metadata")
            self.assertEqual(stats["users_without_training_history"], 1)
            self.assertEqual(stats["users_with_history_but_no_usable_metadata"], 1)

    def test_validation_history_cannot_enter_profiles(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "looks_like_train.csv"
            write_csv(path, PAIR_COLUMNS, [["u", 1, 1, 120, "2020-05-18 00:00:00", "2020-05-18 00:02:00"]])
            with self.assertRaisesRegex(ValueError, "outside train"):
                generate_predictions(path, ["u"], {1: {0: 1.0}}, frozenset(range(1, 6)), (1, 2, 3, 4, 5), CBFConfig())

    def test_invalid_feature_weights_and_profile_rule_rejected(self):
        with self.assertRaises(ValueError):
            CBFConfig(genre_weight=-1).weights()
        with self.assertRaises(ValueError):
            CBFConfig(profile_weighting="future_labels").weights()


class LeakageTests(unittest.TestCase):
    def test_validation_metadata_changes_do_not_change_features_or_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training, metadata, schedule, template = [root / name for name in ("train.csv", "metadata.csv", "validation_schedule.csv", "submission.csv")]
            write_csv(training, PAIR_COLUMNS, [["0007", 1, 1, 120, "2020-03-09 00:00:00", "2020-03-09 00:02:00"]])
            write_csv(metadata, SCHEDULE_COLUMNS, [airing(1), airing(6)])
            rows = [airing(item, start="18.05.2020 00:00:00") for item in range(1, 7)]
            write_csv(schedule, SCHEDULE_COLUMNS, rows)
            write_csv(template, ["user_id", "tv_show_id"], [["0007", "0 0 0 0 0"], ["cold", "0 0 0 0 0"]])
            first, predictions, candidates, training_users = predict_validation(training, metadata, schedule, template, CBFConfig(), expected_users=2)
            for row in rows:
                row[4], row[6], row[7], row[11], row[12] = "Future title", "Future category", "Future genre", "Future director", "Future actor"
            write_csv(schedule, SCHEDULE_COLUMNS, rows)
            second, changed, same_candidates, same_users = predict_validation(training, metadata, schedule, template, CBFConfig(), expected_users=2)
            self.assertEqual(changed, predictions)
            self.assertEqual(candidates, same_candidates)
            self.assertEqual(training_users, same_users)
            self.assertEqual(first["item_vectors"], second["item_vectors"])
            self.assertEqual(predictions["cold"], tuple(first["fallback_top_five"]))
            self.assertFalse(first["validation_metadata_used_for_features"])
            self.assertEqual(first["metadata_candidate_coverage"]["candidates_without_training_metadata"], 4)


if __name__ == "__main__":
    unittest.main()
