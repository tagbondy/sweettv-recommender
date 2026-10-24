"""Causal channel/time allocation, context-only unions, and rolling boundaries."""

import copy
import csv
import hashlib
import math
import sqlite3
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from audit_data import VIEW_COLUMNS, VIEW_FILE
from build_interactions import Window, timestamp
from continuous_watch import qualifies
from behavioral_evidence_v5 import (
    CHANNEL_FEATURES, TIME_FEATURES, START, CUT13, NEW_WINDOWS,
    add_original_event, add_union_duration, behavior_features, chunks, create_raw_spool,
    empty_context, export_new_profiles, merge_intervals, read_profile,
)


class DynamicSchemaTests(unittest.TestCase):
    def test_chunked_collaborative_cosine_matches_dense_reference(self):
        import numpy as np
        from scipy import sparse
        from behavioral_v5 import collaborative_neighbors
        dense = np.asarray([[1., 2., 0.], [1., 0., 3.], [0., 2., 3.]], dtype=np.float32)
        binary = sparse.csr_matrix((dense > 0).astype(np.float32))
        x = sparse.csr_matrix(dense)
        actual, budget = collaborative_neighbors(x, binary, neighbors=3, block_size=1)
        norms = np.sqrt((dense * dense).sum(axis=0))
        common = binary.toarray().T @ binary.toarray()
        expected = (dense.T @ dense) / (norms[:, None] * norms[None, :]) * common / (common + 20.)
        np.fill_diagonal(expected, 0.)
        np.testing.assert_allclose(actual.toarray(), expected, rtol=1e-6)
        self.assertLess(budget["estimated_block_and_neighbors_bytes"], budget["preallocation_limit_bytes"])
        np.testing.assert_array_equal(x.toarray(), dense)

    def test_ablation_does_not_add_features_absent_from_parent(self):
        from behavioral_v5 import selected_columns
        for parent in ("base", "channel", "time", "both"):
            self.assertTrue(set(selected_columns("drop_association:" + parent)) <= set(selected_columns(parent)))

    def test_new_matrix_schema_does_not_change_old_stride(self):
        from behavioral_v5 import D, NAMES, selected_columns
        from rerank_v3 import FEATURE_NAMES
        self.assertEqual(len(FEATURE_NAMES), 61)
        self.assertEqual(D, 61 + len(CHANNEL_FEATURES) + len(TIME_FEATURES))
        self.assertEqual(NAMES[:61], FEATURE_NAMES)
        self.assertEqual(len(selected_columns("channel")), 61 + len(CHANNEL_FEATURES))
        self.assertEqual(len(selected_columns("time")), 61 + len(TIME_FEATURES))
        self.assertEqual(selected_columns("both"), list(range(D)))

    def test_disjoint_rolling_targets_end_before_primary_origin(self):
        labels = sorted((window for window in NEW_WINDOWS if window.name.startswith("labels")), key=lambda window: window.start)
        self.assertTrue(all(left.end == right.start for left, right in zip(labels, labels[1:])))
        self.assertLessEqual(labels[-1].end, __import__("build_interactions").WINDOWS[0].end)

    def test_row_alignment_query_weights_and_label_cutoff(self):
        import numpy as np
        from behavioral_v5 import D, save_bundle, load_bundle, fitting_arrays
        from rerank_v3 import FeatureTable
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data/processed").mkdir(parents=True)
            table = FeatureTable("toy", 50, 100, labels=bytearray([1, 0, 1, 0]), fallback=(1, 2, 3, 4, 5))
            table.rows = [("0007", 1), ("0007", 2), ("extra", 1), ("extra", 2)]
            table.groups = [("0007", 0, 2), ("extra", 2, 4)]
            x = np.zeros((4, D), dtype=np.float32)
            x[:, 0] = [1, 2, 3, 4]
            save_bundle(root, "toy", table, x, {})
            loaded, features, _ = load_bundle(root, "toy")
            self.assertEqual(loaded.rows, table.rows)
            self.assertTrue(np.array_equal(features, x))
            _, y, groups, weights, windows = fitting_arrays(root, ["toy"], [0], 100, False, ["0007"])
            self.assertEqual(groups, [2, 2])
            self.assertAlmostEqual(float(weights.sum()), len(y), places=5)
            self.assertAlmostEqual(float(weights[0] / weights[2]), 2.)
            self.assertEqual(windows[0]["extra_users"], 1)
            with self.assertRaises(ValueError):
                fitting_arrays(root, ["toy"], [0], 99, False, ["0007"])


class ContextTests(unittest.TestCase):
    def test_overlap_union_does_not_change_single_session_label(self):
        intervals = [(0, 40), (60, 100)]
        self.assertEqual(sum(end - start for start, end in merge_intervals(intervals)), 80)
        self.assertFalse(qualifies(max(end - start for start, end in intervals), 100, "gt80"))
        self.assertEqual(list(merge_intervals([(0, 60), (30, 90), (90, 100)])), [(0, 100)])

    def test_hour_midnight_and_weekend_boundaries(self):
        # Jan 4, 2020 is Saturday; the allocation crosses into Sunday.
        parts = list(chunks(3 * 86400 + 23 * 3600 + 1800, 4 * 86400 + 1800))
        self.assertEqual([(hour, weekday, last - first) for first, last, hour, weekday in parts], [(23, 5, 1800), (0, 6, 1800)])

    def test_cutoff_clipping_and_future_invariance(self):
        window = Window("prefix", 0, 3600)
        context = empty_context()
        add_original_event(context, 6, 3500, 4000, window)
        add_union_duration(context, 6, 3500, 4000, window)
        before = copy.deepcopy(context)
        add_original_event(context, 6, 4000, 5000, window)
        add_union_duration(context, 6, 4000, 5000, window)
        self.assertEqual(context, before)
        self.assertEqual(context["channels"]["6"][0], 100)
        self.assertEqual(context["channels"]["6"][5], 3600)

    def test_event_counts_and_duration_union_are_separate(self):
        window = Window("prefix", 0, 1000)
        context = empty_context()
        for low, high in ((0, 100), (50, 150)):
            add_original_event(context, 6, low, high, window)
        for low, high in merge_intervals([(0, 100), (50, 150)]):
            add_union_duration(context, 6, low, high, window)
        self.assertEqual(context["events"], 2)
        self.assertEqual(context["raw_session_seconds"], 200)
        self.assertEqual(context["channels"]["6"][0], 150)

    def test_recent_channel_affinity_excludes_old_seconds(self):
        window = Window("prefix", 0, 30 * 86400)
        context = empty_context()
        add_union_duration(context, 6, 0, 100, window)
        add_union_duration(context, 7, window.end - 100, window.end, window)
        self.assertEqual(context["channels"]["6"][3], 0)
        self.assertEqual(context["channels"]["7"][3], 100)
        self.assertGreater(context["channels"]["7"][4], context["channels"]["6"][4])

    def test_behavior_schema_and_personal_channel_alignment(self):
        context = empty_context()
        window = Window("prefix", 0, 1000)
        add_original_event(context, 6, 0, 1000, window)
        add_union_duration(context, 6, 0, 1000, window)
        context["seconds"] = 1000.
        global_ = {"seconds": 1000., "channels": {"6": 1000.}, "hours": [1000.] + [0.] * 23,
                   "weekdays": [0., 0., 1000., 0., 0., 0., 0.]}
        def opportunity(chan):
            return {"n": 1, "channels": Counter({str(chan): 1}), "channel_parts": Counter({(str(chan), 0): 1}),
                    "hours": [100.] + [0.] * 23, "weekdays": [0., 0., 100., 0., 0., 0., 0.],
                    "start_hours": Counter({0: 1}), "evening": 0, "prime": 0, "weekend": 0}
        match = behavior_features(context, global_, opportunity(6), window.end)
        mismatch = behavior_features(context, global_, opportunity(7), window.end)
        self.assertEqual(len(match), len(CHANNEL_FEATURES) + len(TIME_FEATURES))
        self.assertGreater(match[0], mismatch[0])
        with_integer_history = behavior_features(context, global_, opportunity(6), window.end, historical_channels={6})
        self.assertGreater(with_integer_history[15], 0.)
        self.assertEqual(with_integer_history[15], behavior_features(context, global_, opportunity(6), window.end, historical_channels={"6"})[15])
        self.assertTrue(all(math.isfinite(value) for value in match))
        self.assertEqual(match[-1], 1.)


class RollingBoundaryTests(unittest.TestCase):
    def test_new_prefix_does_not_inherit_a_later_maximum(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, processed = root / "data/raw", root / "data/processed"
            raw.mkdir(parents=True)
            processed.mkdir()
            original = processed / "continuous_watch_v2.sqlite3"
            connection = sqlite3.connect(original)
            connection.execute("""CREATE TABLE occurrences(airing_start INTEGER,channel_id INTEGER,programme_id INTEGER,user_id TEXT,
                airing_end INTEGER,broadcast_duration INTEGER,train_overlap INTEGER,train_last30 INTEGER,train_last50 INTEGER,train_last80 INTEGER)""")
            connection.execute("INSERT INTO occurrences VALUES(?,?,?,?,?,?,?,?,?,?)", (CUT13 - 40, 6, 42, "0007", CUT13 + 60, 100, 60, CUT13 + 60, CUT13 + 60, 0))
            connection.commit()
            connection.close()
            original_hash = hashlib.sha256(original.read_bytes()).hexdigest()
            with (raw / VIEW_FILE).open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(VIEW_COLUMNS)
                writer.writerows([["0007", 6, timestamp(CUT13 - 40), timestamp(CUT13), 40],
                                 ["0007", 6, timestamp(CUT13), timestamp(CUT13 + 60), 60]])
            spool, labels = processed / "spool.sqlite3", processed / "origins.sqlite3"
            create_raw_spool(root, spool, labels)
            export_new_profiles(root, labels)
            history_window = next(window for window in NEW_WINDOWS if window.name == "prefix13")
            records = list(read_profile(processed / "behavioral_v5_profile_prefix13.csv", history_window))
            self.assertEqual(records[0]["frequency_ge30"], 1)
            self.assertEqual(records[0]["frequency_gt50"], 0)
            self.assertEqual(records[0]["frequency_gt80"], 0)
            target_window = next(window for window in NEW_WINDOWS if window.name == "labels14_15")
            target = list(read_profile(processed / "behavioral_v5_profile_labels14_15.csv", target_window))
            self.assertEqual(target[0]["frequency_gt50"], 1)
            self.assertEqual(target[0]["frequency_gt80"], 0)
            self.assertEqual(records[0]["user_id"], "0007")
            self.assertEqual(hashlib.sha256(original.read_bytes()).hexdigest(), original_hash)


if __name__ == "__main__":
    unittest.main()
