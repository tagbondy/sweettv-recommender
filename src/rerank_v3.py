"""Leakage-safe candidate generation and standard-library reranking study.

Run: .venv/bin/python src/rerank_v3.py
Resume cached feature generation: --stage features / --stage study.

The user declined package installation. LightGBM experiments are explicitly
skipped, not simulated. Learned models are our standard-library minibatch logistic
and pairwise-logistic baselines. Fit labels end at the prediction origin; primary
validation labels are evaluation-only until final model selection is complete.
"""

from __future__ import annotations

import argparse
import csv
import filecmp
import importlib.util
import itertools
import json
import math
import os
import platform
import random
import sqlite3
import subprocess
import sys
import tempfile
import time
from array import array
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from association_rules import AssociationCounts, RuleConfig, build_rules, select_basket
from audit_data import SCHEDULE_COLUMNS, parse_integer, parse_timestamp, peak_memory
from build_interactions import EVENT_COLUMNS, PAIR_COLUMNS, Window, checked_records, file_fingerprint, seconds, timestamp
from continuous_watch import WINDOW_BY_NAME
from create_final_submission import TEST_WINDOW, project_snapshot, publish_submission, validate_submission
from evaluate import AggregateReader, evaluate_recommendations, load_candidates, load_template_users, read_predictions, write_predictions
from personalized_repeat import CONFIG_BY_NAME, iter_profiles, profile_context, profile_path, recommend_repeat
from popularity import rank_candidates


SEED = 42
REPEAT_DEPTH, ASSOCIATION_DEPTH, POPULARITY_DEPTH = 50, 50, 20
MISSING_RANK = 10000
AR_CONFIG = RuleConfig("confidence_max", "confidence_max", 5, .05, 0., 0., 100)
FOLDS = {
    "early": ("early_train", "early_validation", "export_arh_11-20-final.csv"),
    "primary": ("train", "validation", "export_arh_21-30-final.csv"),
    "final": ("full", "test", "export_arh_31-42-final.csv"),
}
FEATURE_NAMES = (
    "frequency_ge30", "frequency_gt50", "frequency_gt80", "frequency_ge30_per_week", "frequency_gt50_per_week", "frequency_gt80_per_week",
    "recency_weight_half14", "recency_weight_half28", "recency_fraction_half14", "recency_fraction_half28",
    "days_since_ge30", "days_since_gt50", "days_since_gt80", "qualified_minutes_gt50",
    "repeat_rank", "repeat_reciprocal_rank", "repeat_preference_share", "historical_airing_watch_rate", "expected_future_qualified_airings",
    "association_best_confidence", "association_sum_confidence", "association_best_lift", "association_max_pair_support",
    "association_matched_antecedents", "association_target_support", "association_best_source_support", "association_rank", "association_reciprocal_rank",
    "popularity_support", "popularity_normalized", "popularity_rank", "popularity_reciprocal_rank",
    "source_repeat", "source_association", "source_popularity", "source_count",
    "future_airings", "historical_airings", "first_airing_day_since_2020", "last_airing_day_since_2020",
    "days_until_first_airing", "days_until_last_airing", "scheduled_minutes", "airing_channels",
    "prime_time_fraction", "evening_fraction", "weekend_fraction", "airing_day_coverage", "mean_broadcast_minutes",
    "user_matched_events", "user_distinct_programmes", "user_qualified_airings_ge30", "user_qualified_minutes",
    "history_days", "target_days", "heavy_user", "cold_start", "user_overlap_minutes",
    "repeat_popularity_interaction", "repeat_association_interaction", "source_repeat_association_agreement",
)
FI = {name: index for index, name in enumerate(FEATURE_NAMES)}
# Transform only numerical magnitude features, never labels or user/programme IDs.
LOG_FEATURES = frozenset({
    "frequency_ge30", "frequency_gt50", "frequency_gt80", "qualified_minutes_gt50",
    "association_sum_confidence", "association_best_lift", "association_max_pair_support", "association_matched_antecedents",
    "association_target_support", "association_best_source_support", "popularity_support",
    "future_airings", "historical_airings", "scheduled_minutes", "airing_channels", "mean_broadcast_minutes",
    "user_matched_events", "user_distinct_programmes", "user_qualified_airings_ge30", "user_qualified_minutes", "user_overlap_minutes",
    "repeat_rank", "association_rank", "popularity_rank", "expected_future_qualified_airings",
})


@dataclass(frozen=True)
class ModelConfig:
    name: str
    kind: str
    fusion_weights: tuple[float, float, float] = (1., 0., 0.)
    l2: float = .001
    epochs: int = 12
    learning_rate: float = .08
    batch_size: int = 128
    negative_pairs_per_positive: int = 8


MODELS = (
    ModelConfig("repeat_control", "repeat"),
    ModelConfig("rrf_90_10_0", "fusion", (.9, .1, 0.)),
    ModelConfig("rrf_80_15_5", "fusion", (.8, .15, .05)),
    ModelConfig("rrf_70_25_5", "fusion", (.7, .25, .05)),
    ModelConfig("logistic_l2_001", "logistic", l2=.001),
    ModelConfig("logistic_l2_01", "logistic", l2=.01),
    ModelConfig("pairwise_logistic_l2_001", "pairwise", l2=.001, epochs=10, learning_rate=.05),
)
MODEL_BY_NAME = {config.name: config for config in MODELS}


@dataclass
class FeatureTable:
    name: str
    history_end: int
    label_end: int | None
    x: array = field(default_factory=lambda: array("f"))
    rows: list[tuple[str, int]] = field(default_factory=list)
    groups: list[tuple[str, int, int]] = field(default_factory=list)
    labels: bytearray | None = None
    fallback: tuple[int, ...] = ()
    metadata: dict = field(default_factory=dict)

    def value(self, row: int, name: str) -> float:
        return self.x[row * len(FEATURE_NAMES) + FI[name]]


def owned(name: str) -> bool:
    return (name in {"src/rerank_v3.py", "tests/test_rerank_v3.py", "outputs/submission_v3.csv", "reports/rerank_v3.md", "reports/rerank_v3.json"}
            or name.startswith("data/processed/rerank_v3_") or name.startswith("outputs/rerank_v3_")
            or name.startswith("reports/rerank_v3_model_"))


def protected_snapshot(root: Path) -> dict:
    return {name: details for name, details in project_snapshot(root).items() if not owned(name)}


def check_integrity(root: Path, before: dict) -> dict:
    result = {}
    for name, old in before.items():
        current = file_fingerprint(root / name)
        if current != old:
            raise RuntimeError(f"Protected pre-v3 artifact changed: {name}")
        result[name] = {"before": old, "after": current, "unchanged": True}
    return result


def earlier_120_aggregates(root: Path) -> Path:
    """Clip old event-level overlaps; a shorter >=120s piece was never omitted."""
    destination = root / "data/processed/rerank_v3_early_120.csv"
    if destination.exists():
        return destination
    window = WINDOW_BY_NAME["early_train"]
    with tempfile.TemporaryDirectory(prefix="rerank-early-", dir=root / "data/processed") as work:
        connection = sqlite3.connect(Path(work) / "pairs.sqlite3")
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA cache_size=-8192")
        connection.execute("CREATE TABLE pairs(user TEXT,item INTEGER,n INTEGER,seconds INTEGER,first INTEGER,last INTEGER,PRIMARY KEY(user,item)) WITHOUT ROWID")
        pending = []

        def flush():
            connection.executemany("""INSERT INTO pairs VALUES (?,?,?,?,?,?) ON CONFLICT(user,item) DO UPDATE SET
                n=n+excluded.n,seconds=seconds+excluded.seconds,first=MIN(first,excluded.first),last=MAX(last,excluded.last)""", pending)
            pending.clear()
            connection.commit()

        for _, row in checked_records(root / "data/processed/train_interactions.csv", EVENT_COLUMNS):
            low = seconds(parse_timestamp(row["overlap_start"]))
            high = seconds(parse_timestamp(row["overlap_end"]))
            low, high = max(low, window.start), min(high, window.end)
            if high - low >= 120:
                pending.append((row["user_id"], int(row["tv_show_id"]), 1, high - low, low, high))
            if len(pending) >= 5000:
                flush()
        flush()
        with destination.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(PAIR_COLUMNS)
            for user, item, n, duration, first, last in connection.execute("SELECT * FROM pairs ORDER BY user,item"):
                writer.writerow([user, item, n, duration, timestamp(first), timestamp(last)])
        connection.close()
    return destination


def base_histories(path: Path, window: Window, cohort: set[str]) -> tuple[dict, Counter, dict, dict]:
    reader = AggregateReader(path, window)
    baskets, supports, contexts = {}, Counter(), {}
    for user, group in itertools.groupby(reader, key=lambda row: row.user_id):
        history = list(group)
        supports.update(row.programme_id for row in history)
        baskets[user] = select_basket(history, 50)
        if user in cohort:
            contexts[user] = {"user_matched_events": sum(row.matched_events for row in history),
                              "user_distinct_programmes": len(history),
                              "user_overlap_minutes": sum(row.overlap_seconds for row in history) / 60.}
    return baskets, supports, contexts, reader.summary(cohort)


def schedule_statistics(paths: list[Path], window: Window) -> dict:
    stats, seen = {}, set()
    for path in paths:
        for _, row in checked_records(path, SCHEDULE_COLUMNS):
            item = parse_integer(row["tv_show_id"])
            if item == 0:
                continue
            start = parse_timestamp(row["start_time"], schedule=True)
            duration, channel = parse_integer(row["duration"]), parse_integer(row["channel_id"])
            if item is None or item <= 0 or start is None or duration is None or duration <= 0 or channel is None or channel <= 0:
                raise ValueError("Invalid labeled availability interval.")
            original_low = seconds(start)
            low, high = max(original_low, window.start), min(original_low + duration, window.end)
            key = (item, channel, original_low, duration)
            if high <= low or key in seen:
                continue
            seen.add(key)
            entry = stats.setdefault(item, {"airings": 0, "first": low, "last": low, "duration": 0,
                                            "full_duration": 0, "channels": set(), "days": set(),
                                            "prime": 0, "evening": 0, "weekend": 0})
            entry["airings"] += 1
            entry["first"], entry["last"] = min(entry["first"], low), max(entry["last"], low)
            entry["duration"] += high - low
            entry["full_duration"] += duration
            entry["channels"].add(channel)
            entry["days"].add(start.date())
            entry["prime"] += 19 <= start.hour < 22
            entry["evening"] += 18 <= start.hour < 23
            entry["weekend"] += start.weekday() >= 5
    return stats


def association_features(basket, rules: dict, supports: Counter, n_users: int, available: frozenset[int]) -> dict:
    result = {}
    for seed in sorted({item.programme_id for item in basket}):
        for item, confidence in rules.get(seed, ()):
            if item not in available:
                continue
            pair_count = round(confidence * supports[seed])
            if not math.isclose(pair_count, confidence * supports[seed], rel_tol=0, abs_tol=1e-8):
                raise AssertionError("Cannot reconstruct support from the approved unshrunk confidence rule.")
            lift = confidence * n_users / supports[item]
            entry = result.setdefault(item, {"best": 0., "sum": 0., "lift": 0., "pair": 0., "n": 0, "source_support": 0})
            if confidence > entry["best"]:
                entry["best"], entry["source_support"] = confidence, supports[seed]
            entry["sum"] += confidence
            entry["lift"] = max(entry["lift"], lift)
            entry["pair"] = max(entry["pair"], pair_count)
            entry["n"] += 1
    return result


def candidate_union(history: list[dict], associations: dict, pop_order: list[int], available: frozenset[int]) -> tuple[set[int], dict]:
    repeat_order = sorted((row for row in history if row["tv_show_id"] in available and row["frequency_ge30"] > 0),
                          key=lambda row: (-row["frequency_ge30"], -row["last_ge30_seconds"], row["tv_show_id"]))
    ar_order = sorted(associations, key=lambda item: (-associations[item]["best"], item))
    repeat_rank = {row["tv_show_id"]: rank for rank, row in enumerate(repeat_order, 1)}
    ar_rank = {item: rank for rank, item in enumerate(ar_order, 1)}
    sources = {"repeat": set(repeat_rank.keys()) & {row["tv_show_id"] for row in repeat_order[:REPEAT_DEPTH]},
               "association": set(ar_order[:ASSOCIATION_DEPTH]), "popularity": set(pop_order[:POPULARITY_DEPTH])}
    candidates = set().union(*sources.values())
    if not candidates <= available or len(candidates) < 5:
        raise ValueError("Candidate union must supply five unique available programmes.")
    return candidates, {"sources": sources, "repeat_rank": repeat_rank, "association_rank": ar_rank}


def pair_features(item: int, history: list[dict], history_map: dict, associations: dict, ranks: dict,
                  global_support: Counter, capped_support: Counter, pop_rank: dict, user_context: dict,
                  past_schedule: dict, future_schedule: dict, history_window: Window, target_window: Window,
                  n_training_users: int) -> list[float]:
    row = history_map.get(item, {})
    past, future = past_schedule.get(item, {}), future_schedule[item]
    ar = associations.get(item, {})
    history_days, target_days = (history_window.end - history_window.start) / 86400., (target_window.end - target_window.start) / 86400.
    total_repeat = sum(record["frequency_ge30"] for record in history)
    total_qualified_minutes = sum(record["qualified_seconds_gt50"] for record in history) / 60.
    n30, n50, n80 = (row.get(name, 0) for name in ("frequency_ge30", "frequency_gt50", "frequency_gt80"))
    repeat_rank = ranks["repeat_rank"].get(item, MISSING_RANK)
    ar_rank = ranks["association_rank"].get(item, MISSING_RANK)
    source_r, source_a, source_p = (float(item in ranks["sources"][source]) for source in ("repeat", "association", "popularity"))
    historical_airings = past.get("airings", 0)
    watch_rate = min(1., n30 / historical_airings) if historical_airings else 0.
    pop_normalized = global_support[item] / max(1, n_training_users)
    values = {
        "frequency_ge30": n30, "frequency_gt50": n50, "frequency_gt80": n80,
        "frequency_ge30_per_week": n30 * 7 / history_days, "frequency_gt50_per_week": n50 * 7 / history_days, "frequency_gt80_per_week": n80 * 7 / history_days,
        "recency_weight_half14": row.get("recency_gt50_half14", 0.), "recency_weight_half28": row.get("recency_gt50_half28", 0.),
        "recency_fraction_half14": row.get("recency_gt50_half14", 0.) / max(1, n50),
        "recency_fraction_half28": row.get("recency_gt50_half28", 0.) / max(1, n50),
        "days_since_ge30": (history_window.end - row["last_ge30_seconds"]) / 86400. if row.get("last_ge30_seconds") else history_days + 1,
        "days_since_gt50": (history_window.end - row["last_gt50_seconds"]) / 86400. if row.get("last_gt50_seconds") else history_days + 1,
        "days_since_gt80": (history_window.end - row["last_gt80_seconds"]) / 86400. if row.get("last_gt80_seconds") else history_days + 1,
        "qualified_minutes_gt50": row.get("qualified_seconds_gt50", 0) / 60.,
        "repeat_rank": repeat_rank, "repeat_reciprocal_rank": 1 / repeat_rank if repeat_rank < MISSING_RANK else 0.,
        "repeat_preference_share": n30 / max(1, total_repeat), "historical_airing_watch_rate": watch_rate,
        "expected_future_qualified_airings": watch_rate * future["airings"],
        "association_best_confidence": ar.get("best", 0.), "association_sum_confidence": ar.get("sum", 0.),
        "association_best_lift": ar.get("lift", 0.), "association_max_pair_support": ar.get("pair", 0.),
        "association_matched_antecedents": ar.get("n", 0), "association_target_support": capped_support[item],
        "association_best_source_support": ar.get("source_support", 0), "association_rank": ar_rank,
        "association_reciprocal_rank": 1 / ar_rank if ar_rank < MISSING_RANK else 0.,
        "popularity_support": global_support[item], "popularity_normalized": pop_normalized,
        "popularity_rank": pop_rank[item], "popularity_reciprocal_rank": 1 / pop_rank[item],
        "source_repeat": source_r, "source_association": source_a, "source_popularity": source_p, "source_count": source_r + source_a + source_p,
        "future_airings": future["airings"], "historical_airings": historical_airings,
        "first_airing_day_since_2020": future["first"] / 86400., "last_airing_day_since_2020": future["last"] / 86400.,
        "days_until_first_airing": (future["first"] - history_window.end) / 86400.,
        "days_until_last_airing": (future["last"] - history_window.end) / 86400.,
        "scheduled_minutes": future["duration"] / 60., "airing_channels": len(future["channels"]),
        "prime_time_fraction": future["prime"] / future["airings"], "evening_fraction": future["evening"] / future["airings"],
        "weekend_fraction": future["weekend"] / future["airings"], "airing_day_coverage": len(future["days"]) / target_days,
        "mean_broadcast_minutes": future["full_duration"] / future["airings"] / 60.,
        "user_matched_events": user_context.get("user_matched_events", 0), "user_distinct_programmes": user_context.get("user_distinct_programmes", 0),
        "user_qualified_airings_ge30": total_repeat, "user_qualified_minutes": total_qualified_minutes,
        "history_days": history_days, "target_days": target_days,
        "heavy_user": float(user_context.get("user_distinct_programmes", 0) >= 100),
        "cold_start": float(not user_context.get("user_distinct_programmes", 0) and not total_repeat),
        "user_overlap_minutes": user_context.get("user_overlap_minutes", 0.),
        "repeat_popularity_interaction": math.log1p(n30) * pop_normalized,
        "repeat_association_interaction": math.log1p(n30) * ar.get("best", 0.),
        "source_repeat_association_agreement": source_r * source_a,
    }
    result = [float(values[name]) for name in FEATURE_NAMES]
    if any(not math.isfinite(value) for value in result):
        raise ValueError("Nonfinite candidate feature.")
    return result


def build_features(root: Path, fold: str, users: list[str]) -> FeatureTable:
    history_name, target_name, schedule_name = FOLDS[fold]
    history_window = WINDOW_BY_NAME[history_name]
    target_window = TEST_WINDOW if target_name == "test" else WINDOW_BY_NAME[target_name]
    aggregate_path = earlier_120_aggregates(root) if fold == "early" else root / "data/processed" / ("final_train_user_programmes.csv" if fold == "final" else "train_user_programmes.csv")
    started = time.perf_counter()
    baskets, support, contexts, stats = base_histories(aggregate_path, history_window, set(users))
    past = schedule_statistics([root / "data/raw/export_arh_11-20-final.csv", root / "data/raw/export_arh_21-30-final.csv"], history_window)
    future = schedule_statistics([root / "data/raw" / schedule_name], target_window)
    available, availability_stats = load_candidates(root / "data/raw" / schedule_name, target_window)
    if set(future) != available:
        raise AssertionError("Availability feature pool differs from the shared evaluator's pool.")
    pop_order = sorted(available, key=lambda item: (-support[item], item))
    fallback = rank_candidates(support, available)
    pop_rank = {item: rank for rank, item in enumerate(pop_order, 1)}
    table = FeatureTable(fold, history_window.end, None if fold == "final" else target_window.end, fallback=fallback)
    observed_users = set()
    cohort = set(users)
    repeat_reference = {}
    association_reference = {}
    with tempfile.TemporaryDirectory(prefix="rerank-counts-", dir=root / "data/processed") as work:
        counts = AssociationCounts(baskets, Path(work) / "pairs.sqlite3")
        try:
            rules, rule_stats = build_rules(counts, AR_CONFIG)

            def append_user(user, history):
                associations = association_features(baskets.get(user, ()), rules, counts.support, counts.user_count, available)
                pool, ranks = candidate_union(history, associations, pop_order, available)
                mapping = {row["tv_show_id"]: row for row in history}
                start = len(table.rows)
                for item in sorted(pool):
                    table.rows.append((user, item))
                    table.x.extend(pair_features(item, history, mapping, associations, ranks, support, counts.support, pop_rank,
                                                 contexts.get(user, {}), past, future, history_window, target_window, len(baskets)))
                table.groups.append((user, start, len(table.rows)))
                repeat_reference[user] = recommend_repeat(history, CONFIG_BY_NAME["A_frequency_ge30"], available, fallback)[0]
                ar_order = sorted(associations, key=lambda item: (-associations[item]["best"], item))[:5]
                association_reference[user] = tuple(ar_order + [item for item in fallback if item not in ar_order][:5 - len(ar_order)])

            for user, group in itertools.groupby(iter_profiles(profile_path(root, history_name), history_name), key=lambda row: row["user_id"]):
                if user not in cohort:
                    for _ in group:
                        pass
                    continue
                observed_users.add(user)
                append_user(user, list(group))
            for user in users:
                if user not in observed_users:
                    append_user(user, [])
        finally:
            counts.close()
    table.metadata = {"history_name": history_name, "target_name": target_name, "history_start": history_window.start,
                      "history_end": history_window.end, "target_start": target_window.start, "target_end": target_window.end,
                      "base_120_history": stats, "rules": rule_stats, "availability": availability_stats,
                      "candidate_rows": len(table.rows), "feature_count": len(FEATURE_NAMES),
                      "candidate_counts": {"min": min(end - start for _, start, end in table.groups),
                                           "max": max(end - start for _, start, end in table.groups),
                                           "mean": len(table.rows) / len(table.groups)},
                      "common_training_users": sorted(baskets), "available_ids": sorted(available),
                      "repeat_reference": repeat_reference, "association_reference": association_reference,
                      "runtime_seconds": time.perf_counter() - started}
    # Only after every feature/candidate is fixed do we attach this fold's labels.
    if fold != "final":
        truth_context = profile_context(profile_path(root, target_name), target_name, users)
        table.labels = bytearray(int(item in truth_context["strict_truth"][user]) for user, item in table.rows)
    return table


def save_table(table: FeatureTable, root: Path) -> dict:
    prefix = root / "data/processed" / f"rerank_v3_{table.name}"
    matrix, rows_path, meta_path, readable = [Path(str(prefix) + suffix) for suffix in (".f32", "_rows.csv", "_meta.json", "_features.csv")]
    if any(path.exists() for path in (matrix, rows_path, meta_path, readable)):
        raise FileExistsError("Preserving previous candidate-feature artifacts.")
    with matrix.open("xb") as handle:
        table.x.tofile(handle)
    with rows_path.open("x", newline="") as handle, readable.open("x", newline="") as feature_handle:
        writer, feature_writer = csv.writer(handle), csv.writer(feature_handle)
        columns = ["user_id", "tv_show_id"] + (["label"] if table.labels is not None else [])
        writer.writerow(columns)
        feature_writer.writerow(["user_id", "tv_show_id", *FEATURE_NAMES] + (["label"] if table.labels is not None else []))
        for index, (user, item) in enumerate(table.rows):
            label = [table.labels[index]] if table.labels is not None else []
            writer.writerow([user, item, *label])
            offset = index * len(FEATURE_NAMES)
            feature_writer.writerow([user, item, *table.x[offset:offset + len(FEATURE_NAMES)], *label])
    metadata = {"name": table.name, "history_end": table.history_end, "label_end": table.label_end,
                "feature_names": FEATURE_NAMES, "byteorder": sys.byteorder, "array_itemsize": table.x.itemsize,
                "rows": len(table.rows), "groups": table.groups, "fallback": table.fallback, "details": table.metadata,
                "artifacts": {str(path.relative_to(root)): file_fingerprint(path) for path in (matrix, rows_path, readable)}}
    with meta_path.open("x") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    return metadata


def load_table(root: Path, fold: str) -> FeatureTable:
    prefix = root / "data/processed" / f"rerank_v3_{fold}"
    metadata = json.loads(Path(str(prefix) + "_meta.json").read_text())
    if tuple(metadata["feature_names"]) != FEATURE_NAMES or metadata["byteorder"] != sys.byteorder or metadata["array_itemsize"] != 4:
        raise ValueError("Candidate-feature schema or numeric environment changed.")
    for name, fingerprint in metadata["artifacts"].items():
        if file_fingerprint(root / name)["sha256"] != fingerprint["sha256"]:
            raise RuntimeError("Cached candidate features changed.")
    table = FeatureTable(fold, metadata["history_end"], metadata["label_end"], fallback=tuple(metadata["fallback"]), metadata=metadata["details"])
    with Path(str(prefix) + ".f32").open("rb") as handle:
        table.x.fromfile(handle, metadata["rows"] * len(FEATURE_NAMES))
    table.labels = bytearray() if table.label_end is not None else None
    canonical_users = {}
    with Path(str(prefix) + "_rows.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            user = canonical_users.setdefault(row["user_id"], row["user_id"])
            table.rows.append((user, int(row["tv_show_id"])))
            if table.labels is not None:
                table.labels.append(int(row["label"]))
    table.groups = [tuple(group) for group in metadata["groups"]]
    if len(table.rows) != metadata["rows"]:
        raise ValueError("Feature/identifier row counts disagree.")
    return table


def transform(value: float, name: str) -> float:
    return math.log1p(max(0., value)) if name in LOG_FEATURES else value


def fit_scaler(table: FeatureTable) -> dict:
    n, d = len(table.rows), len(FEATURE_NAMES)
    sums, squares = [0.] * d, [0.] * d
    for offset in range(0, len(table.x), d):
        for j, name in enumerate(FEATURE_NAMES):
            value = transform(table.x[offset + j], name)
            sums[j] += value
            squares[j] += value * value
    means = [value / max(1, n) for value in sums]
    scales = [math.sqrt(max(0., squares[j] / max(1, n) - means[j] ** 2)) for j in range(d)]
    return {"means": means, "scales": scales, "clip": 8.0, "fit_rows": n}


def scaled_matrix(table: FeatureTable, scaler: dict) -> array:
    result = array("f")
    d = len(FEATURE_NAMES)
    for offset in range(0, len(table.x), d):
        for j, name in enumerate(FEATURE_NAMES):
            scale = scaler["scales"][j]
            value = (transform(table.x[offset + j], name) - scaler["means"][j]) / scale if scale > 1e-10 else 0.
            result.append(max(-scaler["clip"], min(scaler["clip"], value)))
    return result


def enforce_fit_cutoff(table: FeatureTable, prediction_origin: int) -> None:
    if table.labels is None or table.label_end is None or table.label_end > prediction_origin:
        raise ValueError("Training labels extend beyond the prediction origin or are unavailable.")


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1. / (1. + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1. + exponent)


def softplus(value: float) -> float:
    return max(value, 0.) + math.log1p(math.exp(-abs(value)))


def fit_linear(table: FeatureTable, config: ModelConfig, prediction_origin: int) -> dict:
    enforce_fit_cutoff(table, prediction_origin)
    started = time.perf_counter()
    scaler = fit_scaler(table)
    matrix = scaled_matrix(table, scaler)
    n, d = len(table.rows), len(FEATURE_NAMES)
    weights = [0.] * d
    rate = sum(table.labels) / max(1, n)
    bias = math.log(max(1e-8, rate) / max(1e-8, 1 - rate)) if config.kind == "logistic" else 0.
    rng = random.Random(SEED)
    row_weights = [0.] * n
    for _, start, end in table.groups:
        for i in range(start, end):
            row_weights[i] = n / (len(table.groups) * (end - start))
    pairs = []
    if config.kind == "pairwise":
        for _, start, end in table.groups:
            positive = [i for i in range(start, end) if table.labels[i]]
            negative = [i for i in range(start, end) if not table.labels[i]]
            # Fixed hard negatives from historical repeat/popularity ranks, never test labels.
            negative.sort(key=lambda i: (min(table.value(i, "repeat_rank"), table.value(i, "popularity_rank")), table.rows[i][1]))
            negative = negative[:config.negative_pairs_per_positive]
            if positive and negative:
                weight = 1 / (len(positive) * len(negative))
                pairs.extend((p, q, weight) for p in positive for q in negative)
        if not pairs:
            raise ValueError("No mixed-label query groups for pairwise training.")
        factor = len(pairs) / max(1, len(table.groups))
        pairs = [(p, q, weight * factor) for p, q, weight in pairs]
    order = list(range(len(pairs) if config.kind == "pairwise" else n))
    losses = []
    for epoch in range(config.epochs):
        rng.shuffle(order)
        total_loss = total_weight = 0.
        learning_rate = config.learning_rate / math.sqrt(1 + .5 * epoch)
        for batch_start in range(0, len(order), config.batch_size):
            gradient, bias_gradient, batch_weight = [0.] * d, 0., 0.
            for index in order[batch_start:batch_start + config.batch_size]:
                if config.kind == "pairwise":
                    positive, negative, weight = pairs[index]
                    po, no = positive * d, negative * d
                    margin = sum(weights[j] * (matrix[po + j] - matrix[no + j]) for j in range(d))
                    error = (sigmoid(margin) - 1.) * weight
                    for j in range(d):
                        gradient[j] += error * (matrix[po + j] - matrix[no + j])
                    total_loss += softplus(-margin) * weight
                else:
                    offset = index * d
                    weight = row_weights[index]
                    score = bias + sum(weights[j] * matrix[offset + j] for j in range(d))
                    error = (sigmoid(score) - table.labels[index]) * weight
                    for j in range(d):
                        gradient[j] += error * matrix[offset + j]
                    bias_gradient += error
                    total_loss += (softplus(score) - table.labels[index] * score) * weight
                batch_weight += weight
                total_weight += weight
            for j in range(d):
                weights[j] -= learning_rate * (gradient[j] / max(1e-12, batch_weight) + config.l2 * weights[j])
            if config.kind == "logistic":
                bias -= learning_rate * bias_gradient / max(1e-12, batch_weight)
        loss = total_loss / max(1e-12, total_weight)
        if not math.isfinite(loss) or any(not math.isfinite(weight) for weight in weights):
            raise ArithmeticError("Nonfinite training objective or coefficients.")
        losses.append(loss)
    return {"kind": config.kind, "configuration": asdict(config), "scaler": scaler,
            "weights": weights, "bias": bias, "seed": SEED, "fit_label_end": table.label_end,
            "fit_prediction_origin": prediction_origin, "fit_rows": n, "training_positives": sum(table.labels),
            "pairwise_training_pairs": len(pairs), "online_epoch_loss": losses,
            "optimizer_status": "fixed_epoch_budget_completed; no claim of numerical optimum",
            "runtime_seconds": time.perf_counter() - started}


def scores_for(table: FeatureTable, config: ModelConfig, model: dict | None = None) -> array:
    scores = array("d")
    if config.kind in ("logistic", "pairwise"):
        matrix = scaled_matrix(table, model["scaler"])
        d = len(FEATURE_NAMES)
        for offset in range(0, len(matrix), d):
            scores.append(model["bias"] + sum(weight * matrix[offset + j] for j, weight in enumerate(model["weights"])))
    elif config.kind == "fusion":
        for row in range(len(table.rows)):
            score = 0.
            for weight, source, rank in zip(config.fusion_weights, ("source_repeat", "source_association", "source_popularity"), ("repeat_rank", "association_rank", "popularity_rank")):
                if table.value(row, source):
                    score += weight / (20. + table.value(row, rank))
            scores.append(score)
    else:
        for row in range(len(table.rows)):
            rank = table.value(row, "repeat_rank")
            scores.append(1. / rank if rank < MISSING_RANK else -table.value(row, "popularity_rank"))
    return scores


def rank_table(table: FeatureTable, scores: array, users: list[str]) -> dict:
    if len(scores) != len(table.rows) or any(not math.isfinite(value) for value in scores):
        raise ValueError("Invalid reranking scores.")
    predictions = {}
    for user, start, end in table.groups:
        if table.value(start, "cold_start"):
            predictions[user] = table.fallback
            continue
        indices = sorted(range(start, end), key=lambda index: (-scores[index], table.rows[index][1]))[:5]
        predictions[user] = tuple(table.rows[index][1] for index in indices)
    if set(predictions) != set(users):
        raise ValueError("Reranker did not cover the exact template cohort.")
    return {user: predictions[user] for user in users}


def concatenate_labeled(tables: list[FeatureTable], origin: int) -> FeatureTable:
    for table in tables:
        enforce_fit_cutoff(table, origin)
    joined = FeatureTable("final_meta_training", origin, max(table.label_end for table in tables), labels=bytearray())
    for table in tables:
        enforce_fit_cutoff(table, origin)
        offset = len(joined.rows)
        joined.x.extend(table.x)
        joined.rows.extend(table.rows)
        joined.labels.extend(table.labels)
        joined.groups.extend((f"{table.name}:{user}", start + offset, end + offset) for user, start, end in table.groups)
    return joined


def candidate_diagnostics(table: FeatureTable, truth: dict, users: list[str]) -> dict:
    pools = {user: {table.rows[i][1] for i in range(start, end)} for user, start, end in table.groups}
    relevant = sum(len(truth[user]) for user in users)
    found = sum(len(pools[user] & truth[user]) for user in users)
    oracle = sum(min(5, len(pools[user] & truth[user])) / min(5, len(truth[user])) if truth[user] else 0. for user in users) / len(users)
    return {"rows": len(table.rows), "positive_candidate_labels": sum(table.labels) if table.labels is not None else None,
            "relevant_user_programme_pairs": relevant, "relevant_pairs_recalled_by_generators": found,
            "candidate_relevance_recall": found / relevant if relevant else None, "candidate_oracle_map_at_5": oracle,
            "ground_truth_filtered_to_candidates": False}


def feature_manifest(root: Path, protected: dict, users: list[str]) -> dict:
    destination = root / "data/processed/rerank_v3_features_manifest.json"
    if destination.exists():
        manifest = json.loads(destination.read_text())
        for name, fingerprint in manifest["source_files"].items():
            if file_fingerprint(root / name)["sha256"] != fingerprint["sha256"]:
                raise RuntimeError("A v3 feature input changed.")
        return manifest
    started = time.perf_counter()
    bundles = {}
    for fold in FOLDS:
        print(f"Building {fold} candidate features.", flush=True)
        table = build_features(root, fold, users)
        if fold == "primary":
            available = frozenset(table.metadata["available_ids"])
            for method, predicted in (("repeat_v2_primary_A_frequency_ge30.csv", table.metadata["repeat_reference"]),
                                      ("association_validation.csv", table.metadata["association_reference"])):
                saved = read_predictions(root / "outputs" / method, users, available)
                if {user: tuple(predicted[user]) for user in users} != saved:
                    raise AssertionError(f"Candidate generator does not reproduce its verified Top-5: {method}")
        if fold == "final":
            saved = read_predictions(root / "outputs/submission_v2.csv", users, frozenset(table.metadata["available_ids"]))
            if {user: tuple(table.metadata["repeat_reference"][user]) for user in users} != saved:
                raise AssertionError("Final repeat generator differs from current best submission_v2.csv.")
        bundles[fold] = save_table(table, root)
        print(f"Saved {fold}: {len(table.rows):,} candidates, {len(FEATURE_NAMES)} features.", flush=True)
        del table
    check_integrity(root, protected)
    manifest = {"bundles": bundles, "source_files": protected,
                "source_sha256": file_fingerprint(Path(__file__))["sha256"],
                "runtime_seconds": time.perf_counter() - started, **peak_memory()}
    with destination.open("x") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return manifest


def final_worker(root: Path, model_name: str, model_path: Path, destination: Path) -> None:
    users = load_template_users(root / "data/raw/submission.csv")
    table = load_table(root, "final")
    config = MODEL_BY_NAME[model_name]
    model = json.loads(model_path.read_text()) if config.kind in ("logistic", "pairwise") else None
    predictions = rank_table(table, scores_for(table, config, model), users)
    if destination.exists():
        raise FileExistsError("Preserving an existing replay output.")
    write_predictions(destination, users, predictions, frozenset(table.metadata["available_ids"]))


def render_report(report: dict) -> str:
    lines = ["# SWEET.TV Candidate Generation and Reranking v3", "", f"Generated (UTC): {report['generated_at_utc']}", "",
             "## Current best Kaggle result (preserved)", "", "```json", json.dumps(report["current_best_kaggle_v2"], indent=2), "```", "",
             "These scores were reported as verified by the user. No new Kaggle upload or score is claimed.", "",
             "## Package decision", "", "```json", json.dumps(report["packages_and_skipped_models"], indent=2), "```", "",
             "The user selected standard-library-only execution. No packages were installed. LightGBM was not run and receives no invented score. "
             "Logistic models below are explicitly our own standard-library baselines, not LightGBM substitutes labeled as LightGBM.", "",
             "## Chronological supervision", "",
             "- Meta fitting: features from weeks 11–15; binary strict >80% relevance in weeks 16–20.",
             "- Held-out evaluation: features from weeks 11–20; strict relevance in weeks 21–30. The primary feature/label table is saved, but its labels never fit held-out models.",
             "- After selection: refit the selected learned model on both historical labeled origin tables, with labels ending no later than 2020-07-27. Final features use weeks 11–30 and test schedule availability only.",
             "- All 1,260 template users remain in evaluation. Empty relevance contributes zero; the shared AP@5 formula, repeats, and availability policies are unchanged.",
             "- Candidate generators and features are fixed before label attachment. User IDs and programme IDs are identifiers, not learned numerical features.", "",
             "## Candidate generation", "", "```json", json.dumps(report["candidate_generation"], indent=2), "```", "",
             "Repeat source: positive >=30% occurrence-frequency ranks, latest qualifying watch then ID ties. Association source: approved confidence_max rules, top 50 from positive scores; popularity contributes 20 available items. "
             "Histories remain capped at 50 only for AR mining/query antecedents; repeat ranking uses full qualified histories. Zero-scored sources do not get fake source votes.", "",
             "## Exact held-out local validation results", "",
             "| Configuration | Status | MAP@5 all 1,260 | Nonempty-label MAP@5 | Distinct programmes |",
             "|---|---|---:|---:|---:|"]
    for experiment in report["experiments"]:
        metrics = (experiment.get("evaluation") or {}).get("metrics", {})
        primary = metrics.get("primary_all_users", {}).get("map_at_5")
        nonempty = metrics.get("supplementary_nonempty_ground_truth", {}).get("map_at_5")
        coverage = metrics.get("coverage", {}).get("unique_programmes_recommended")
        lines.append(f"| {experiment['configuration']['name']} | {experiment['status']} | {primary!r} | {nonempty!r} | {coverage!r} |")
    lines += ["", f"Selected final strategy: **{report['selected_configuration']}**.", "",
              f"Comparison to v2: {report['comparison_conclusion']}", "",
              "## Reference methods under the same strict labels", "", "```json", json.dumps(report["reference_metrics"], indent=2), "```", "",
              "## Candidate ceiling and source coverage", "", "```json", json.dumps(report["candidate_diagnostics"], indent=2), "```", "",
              "## Feature schema", "", "```json", json.dumps(report["feature_schema"], indent=2), "```", "",
              "Counts, recency, qualified duration, and exposure rates are historical. AR confidence/lift use the capped-basket supports; popularity uses full-history distinct-user counts. "
              "Availability counts/channels/timing are schedule-only. Evening/prime/weekend fractions use the supplied naive broadcast start clock, not an invented timezone. "
              "Relative-rate features reduce the 5-week/10-week/20-week history-length shift. Identifiers and labels are absent from the feature vector.", "",
              "## Every attempted configuration and fitting result", "", "```json",
              json.dumps([{key: value for key, value in experiment.items() if key != "evaluation"} for experiment in report["experiments"]], indent=2), "```", "",
              "Standardization/log transforms are fitted on meta-training features only; constant features are dropped numerically and standardized values clip to [-8,8]. "
              "The classifier uses query-balanced binary logistic minibatch SGD. The pairwise baseline uses within-query positive-versus-fixed-hard-negative logistic loss. "
              "Both use fixed epoch budgets, L2 regularization, and seed 42; no optimal-solver convergence or probability-calibration claim is made.", "",
              "## Final refit and submission", "", "```json", json.dumps(report["final_build"], indent=2), "```", "",
              "## Runtime, environment, and memory", "", "```json", json.dumps(report["execution"], indent=2), "```", "",
              "Feature generation is separately checkpointed. Process high-water RSS is measured, not a combined parent/child working-set estimate.", "",
              "## Preservation verification", "", "| File | SHA-256 | Unchanged |", "|---|---|---|"]
    for name, details in report["input_integrity"].items():
        lines.append(f"| {name} | `{details['before']['sha256']}` | {details['unchanged']} |")
    lines += ["", "## Tests", "", "```text", report["tests"]["output"].rstrip(), "```", "",
              "## Limitations", "",
              "- The earlier meta-training origin differs from the primary hold-out and final longer history/horizon. The controlled primary selection may still be optimistic; there is no unbiased third labeled test.",
              "- Candidate generation caps recall. Full strict ground truth remains in evaluation, including missed/unavailable relevant programmes; no positive labels are injected into the candidate pool.",
              "- Original 120-second event counts are used for AR and user context. Strict >80% continuous labels remain a local proxy, not a verified copy of Kaggle's hidden preference construction.",
              "- Lightweight linear methods may miss nonlinear relationships. LightGBM experiments were skipped because dependencies were unavailable and installation was declined.",
              "- Repeat-only is an explicit control eligible for final selection, preventing an inferior learned reranker from being represented as a validated improvement.",
              "- Source clocks, sparse programme IDs, missing channels, and the uncorrected duration discrepancies remain inherited uncertainties.",
              "- Existing reports, predictions, submissions, source implementations, and original AI-session exports are preserved. No automatic upload was performed, and Public >0.43 remains a goal, not a measured v3 result.", ""]
    return "\n".join(lines)


def run_study(root: Path, manifest: dict, protected: dict, tests: dict) -> dict:
    started = time.perf_counter()
    outputs, reports = root / "outputs", root / "reports"
    report_paths = [reports / "rerank_v3.json", reports / "rerank_v3.md"]
    if any(path.exists() for path in [*report_paths, outputs / "submission_v3.csv"]):
        raise FileExistsError("Preserving previous v3 study artifacts.")
    users = load_template_users(root / "data/raw/submission.csv")
    early, primary = load_table(root, "early"), load_table(root, "primary")
    truth = profile_context(profile_path(root, "validation"), "validation", users)["strict_truth"]
    available = frozenset(primary.metadata["available_ids"])
    common_training_users = frozenset(primary.metadata["common_training_users"])
    references = {}
    for method, filename in (("Popularity", "popularity_validation.csv"), ("Association Rules", "association_validation.csv"), ("Personalized Repeat v2", "repeat_v2_primary_A_frequency_ge30.csv")):
        predictions = read_predictions(outputs / filename, users, available)
        references[method] = evaluate_recommendations(users, truth, predictions, common_training_users, available)["metrics"]
    experiments = []
    model_paths = {}
    for config in MODELS:
        begin = time.perf_counter()
        experiment = {"configuration": asdict(config), "fit_window": "features weeks 11-15, labels weeks 16-20" if config.kind in ("logistic", "pairwise") else "no supervised fit",
                      "status": "started"}
        try:
            model = fit_linear(early, config, primary.history_end) if config.kind in ("logistic", "pairwise") else None
            if model is not None:
                model_path = reports / f"rerank_v3_model_{config.name}.json"
                with model_path.open("x") as handle:
                    json.dump(model, handle, indent=2)
                model_paths[config.name] = model_path
                experiment["fitting"] = {key: value for key, value in model.items() if key not in ("weights", "scaler")}
            predictions = rank_table(primary, scores_for(primary, config, model), users)
            path = outputs / f"rerank_v3_{config.name}_validation.csv"
            if path.exists():
                raise FileExistsError("Previous v3 configuration predictions already exist.")
            write_predictions(path, users, predictions, available)
            saved = read_predictions(path, users, available)
            if config.kind == "repeat" and saved != read_predictions(outputs / "repeat_v2_primary_A_frequency_ge30.csv", users, available):
                raise AssertionError("Repeat control no longer exactly reproduces v2 validation predictions.")
            experiment.update({"status": "complete", "evaluation": evaluate_recommendations(users, truth, saved, common_training_users, available),
                               "prediction_file": {"path": str(path.relative_to(root)), **file_fingerprint(path)}})
            print(f"{config.name}: held-out strict MAP@5={experiment['evaluation']['metrics']['primary_all_users']['map_at_5']!r}", flush=True)
        except (ArithmeticError, ValueError) as error:
            experiment.update({"status": "failed", "error": repr(error), "evaluation": None})
            print(f"{config.name}: failed: {error!r}", flush=True)
        experiment["runtime_seconds"] = time.perf_counter() - begin
        experiments.append(experiment)
        # Durable per-configuration ledger preserves inferior/failed results too.
        with (reports / "rerank_v3_experiment_ledger.jsonl").open("a") as handle:
            handle.write(json.dumps(experiment) + "\n")
    complete = [entry for entry in experiments if entry["status"] == "complete"]
    selected = max(complete, key=lambda entry: entry["evaluation"]["metrics"]["primary_all_users"]["map_at_5"])
    selected_config = MODEL_BY_NAME[selected["configuration"]["name"]]
    repeat_score = references["Personalized Repeat v2"]["primary_all_users"]["map_at_5"]
    selected_score = selected["evaluation"]["metrics"]["primary_all_users"]["map_at_5"]
    difference = selected_score - repeat_score
    begin = time.perf_counter()
    final = load_table(root, "final")
    final_model_path = reports / "rerank_v3_model_final.json"
    if selected_config.kind in ("logistic", "pairwise"):
        combined = concatenate_labeled([early, primary], final.history_end)
        final_model = fit_linear(combined, selected_config, final.history_end)
        with final_model_path.open("x") as handle:
            json.dump(final_model, handle, indent=2)
        del combined
    else:
        final_model = None
    final_predictions = rank_table(final, scores_for(final, selected_config, final_model), users)
    final_available = frozenset(final.metadata["available_ids"])
    with tempfile.TemporaryDirectory(prefix="rerank-v3-replay-", dir=root / "data/processed") as work:
        first, second = Path(work) / "first.csv", Path(work) / "second.csv"
        write_predictions(first, users, final_predictions, final_available)
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = "161803"
        command = [sys.executable, str(Path(__file__).resolve()), "--worker-model", selected_config.name,
                   "--worker-model-file", str(final_model_path), "--worker-output", str(second)]
        completed = subprocess.run(command, cwd=root, env=environment, capture_output=True, text=True)
        if completed.returncode:
            raise RuntimeError(f"Independent final inference replay failed:\n{completed.stderr}")
        if not filecmp.cmp(first, second, shallow=False):
            raise AssertionError("v3 predictions differ under independent hash-seed replay.")
        validate_submission(first, root / "data/raw/submission.csv", final_available)
        publish_submission(first, outputs / "submission_v3.csv")
    final_checks = validate_submission(outputs / "submission_v3.csv", root / "data/raw/submission.csv", final_available)
    integrity = check_integrity(root, protected)
    package_status = {name: {"available": importlib.util.find_spec(name) is not None, "version": None}
                      for name in ("numpy", "pandas", "scipy", "sklearn", "lightgbm")}
    result = {
        "schema_version": 3, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_best_kaggle_v2": json.loads((reports / "kaggle_best_v2.json").read_text()),
        "packages_and_skipped_models": {"installation_decision": "user declined; standard library only", "packages": package_status,
            "skipped": [{"model": name, "status": "not attempted", "score": None,
                         "reason": "LightGBM unavailable and package installation declined"} for name in ("LightGBM binary classification", "LightGBM ranking objective")]},
        "candidate_generation": {"repeat_top_n": REPEAT_DEPTH, "association_top_n": ASSOCIATION_DEPTH, "popularity_top_n": POPULARITY_DEPTH,
                                 "association_configuration": asdict(AR_CONFIG), "basket_cap": 50,
                                 "future_viewing_labels_used": False, "feature_generation_execution": {"runtime_seconds": manifest["runtime_seconds"],
                                                                                                      "peak_process_rss_mib": manifest.get("peak_process_rss_mib")}},
        "experiments": experiments, "selected_configuration": selected_config.name,
        "comparison_conclusion": f"Selected MAP@5={selected_score!r}; v2 MAP@5={repeat_score!r}; signed change={difference!r}. "
                                 + ("The selected reranker beats v2 locally." if difference > 0 else "No validated improvement over v2; the repeat control is retained on an exact tie."),
        "reference_metrics": references,
        "candidate_diagnostics": {"primary": candidate_diagnostics(primary, truth, users),
                                  "early": candidate_diagnostics(early, profile_context(profile_path(root, "early_validation"), "early_validation", users)["strict_truth"], users)},
        "feature_schema": {"feature_count": len(FEATURE_NAMES), "features": FEATURE_NAMES, "log1p_before_standardization": sorted(LOG_FEATURES),
                           "identifiers_are_features": False, "primary_labels_used_to_fit_held_out_model": False,
                           "supervised_fit_label_end": timestamp(early.label_end), "primary_forecast_origin": timestamp(primary.history_end),
                           "final_fit_label_end": timestamp(primary.label_end), "final_forecast_origin": timestamp(final.history_end)},
        "final_build": {"strategy": selected_config.name, "refit_sources": ["early", "primary"] if final_model else [],
                        "test_window_start": timestamp(TEST_WINDOW.start), "test_window_end_exclusive": timestamp(TEST_WINDOW.end),
                        "validation": final_checks, "submission": {"path": "outputs/submission_v3.csv", **file_fingerprint(outputs / "submission_v3.csv")},
                        "byte_identical_hash_seed_replay": True, "replay_hash_seed": "161803",
                        "final_fit_and_replay_runtime_seconds": time.perf_counter() - begin, "automatic_kaggle_upload": False},
        "input_integrity": integrity, "tests": tests,
        "execution": {"python_version": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
                      "seed": SEED, "third_party_packages_used": [], "study_runtime_seconds_before_report_write": time.perf_counter() - started,
                      **peak_memory(), "source_sha256": {name: file_fingerprint(root / name)["sha256"] for name in ("src/rerank_v3.py", "tests/test_rerank_v3.py")}},
    }
    with report_paths[0].open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    with report_paths[1].open("x") as handle:
        handle.write(render_report(result))
    print(json.dumps({"selected": selected_config.name, "selected_map_at_5": selected_score, "v2_map_at_5": repeat_score,
                      "final_build": result["final_build"], "execution": result["execution"]}, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "features", "study"), default="all")
    parser.add_argument("--worker-model", choices=tuple(MODEL_BY_NAME), help=argparse.SUPPRESS)
    parser.add_argument("--worker-model-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.worker_model:
        final_worker(root, args.worker_model, args.worker_model_file, args.worker_output)
        return
    started = time.perf_counter()
    protected = protected_snapshot(root)
    best = json.loads((root / "reports/kaggle_best_v2.json").read_text())
    if protected["outputs/submission_v2.csv"]["sha256"] != best["submission_sha256"]:
        raise RuntimeError("The current-best v2 file differs from its recorded checksum.")
    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
    completed = subprocess.run(command, cwd=root, capture_output=True, text=True)
    output = completed.stdout + completed.stderr
    print(output, end="", flush=True)
    if completed.returncode:
        raise RuntimeError("Complete test suite failed; v3 study not started.")
    tests = {"command": command, "returncode": completed.returncode, "output": output}
    users = load_template_users(root / "data/raw/submission.csv")
    manifest = feature_manifest(root, protected, users)
    if args.stage != "features":
        run_study(root, manifest, protected, tests)
    print(f"Total invocation runtime: {time.perf_counter() - started:.3f} seconds", flush=True)


if __name__ == "__main__":
    main()
