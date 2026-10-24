"""Real LightGBM classification, LambdaRank, ablations, and rank-fusion v4.

Run with the project's configured .venv:
    .venv/bin/python src/lightgbm_v4.py --stage diagnostics
    .venv/bin/python src/lightgbm_v4.py --stage study

V3 implementations and artifacts are read-only. All learned validation models fit
only labels available by the primary forecast origin. Boosting rounds are chosen
with a fitting-side user hold-out, then refitted on all fitting-side users. Actual
reported MAP@5 always comes from the unchanged external Python evaluator.
"""

from __future__ import annotations

import argparse
import csv
import filecmp
import hashlib
import itertools
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from array import array
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("KMP_BLOCKTIME", "0")

import lightgbm as lgb
import numpy as np
import scipy

import rerank_v3 as v3
from association_rules import AssociationCounts, build_rules
from audit_data import peak_memory
from build_interactions import file_fingerprint, timestamp
from continuous_watch import WINDOW_BY_NAME
from create_final_submission import TEST_WINDOW, project_snapshot, publish_submission, validate_submission
from evaluate import evaluate_recommendations, load_candidates, load_template_users, read_predictions, write_predictions
from personalized_repeat import iter_profiles, profile_context, profile_path
from popularity import rank_candidates


THREADS, SEED = 2, 42
EXPANDED_DEPTHS = (100, 100, 40)
EXPANSION_ORACLE_TRIGGER = .90
RANK_FUSION_DEPTH = 50
ENSEMBLES = (
    ("ensemble_0_50_50", (0., .50, .50)),
    ("ensemble_10_40_50", (.10, .40, .50)),
    ("ensemble_20_30_50", (.20, .30, .50)),
    ("ensemble_10_20_70", (.10, .20, .70)),
    ("ensemble_0_75_25", (0., .75, .25)),
)


@dataclass(frozen=True)
class LGBConfig:
    name: str
    objective: str
    num_leaves: int
    learning_rate: float
    min_data_in_leaf: int
    lambda_l2: float
    max_rounds: int = 400
    stopping_rounds: int = 40
    num_threads: int = THREADS


BINARY_CONFIGS = (
    LGBConfig("binary_A", "binary", 7, .05, 30, 5.),
    LGBConfig("binary_B", "binary", 15, .05, 30, 5.),
    LGBConfig("binary_C", "binary", 31, .03, 50, 10.),
)
RANK_CONFIGS = (
    LGBConfig("lambdarank_7", "lambdarank", 7, .05, 30, 5.),
    LGBConfig("lambdarank_15", "lambdarank", 15, .05, 30, 5.),
)


def owned(name: str) -> bool:
    return (name in {"src/lightgbm_v4.py", "tests/test_lightgbm_v4.py", "outputs/submission_v4.csv"}
            or name.startswith("data/processed/lightgbm_v4_") or name.startswith("outputs/lightgbm_v4_")
            or (name.startswith("reports/lightgbm_v4") and name != "reports/lightgbm_v4_environment.json"))


def protected_snapshot(root: Path) -> dict:
    return {name: details for name, details in project_snapshot(root).items() if not owned(name)}


def check_integrity(root: Path, before: dict) -> dict:
    result = {}
    for name, old in before.items():
        current = file_fingerprint(root / name)
        if current != old:
            raise RuntimeError(f"Pre-v4 artifact changed: {name}")
        result[name] = {"before": old, "after": current, "unchanged": True}
    return result


def matrix(table: v3.FeatureTable) -> np.ndarray:
    values = np.frombuffer(table.x, dtype=np.float32).reshape(len(table.rows), len(v3.FEATURE_NAMES))
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite cached numeric feature.")
    return values


def candidate_metrics(table: v3.FeatureTable, truth: dict, users: list[str]) -> dict:
    data = matrix(table)
    pools, source_pools, limited = {}, {source: {} for source in ("repeat", "association", "popularity")}, {}
    counts = []
    for user, start, end in table.groups:
        pools[user] = {table.rows[i][1] for i in range(start, end)}
        counts.append(end - start)
        history = int(table.value(start, "user_distinct_programmes"))
        limited[user] = "0" if history == 0 else "1-5" if history <= 5 else "6-20" if history <= 20 else "21+"
        for source in source_pools:
            flags = data[start:end, v3.FI[f"source_{source}"]]
            source_pools[source][user] = {table.rows[start + i][1] for i in np.flatnonzero(flags)}

    def summarize(selected, candidate_sets):
        relevant = sum(len(truth[user]) for user in selected)
        found = sum(len(candidate_sets[user] & truth[user]) for user in selected)
        nonempty = [user for user in selected if truth[user]]
        oracle = sum(min(5, len(candidate_sets[user] & truth[user])) / min(5, len(truth[user])) if truth[user] else 0. for user in selected)
        return {"users": len(selected), "users_with_labels": len(nonempty), "relevant_items": relevant,
                "recalled_relevant_items": found, "missing_relevant_items": relevant - found,
                "micro_recall": found / relevant if relevant else None,
                "macro_recall_nonempty": sum(len(candidate_sets[user] & truth[user]) / len(truth[user]) for user in nonempty) / len(nonempty) if nonempty else None,
                "candidate_oracle_map_at_5": oracle / len(selected) if selected else None,
                "users_with_fewer_than_five_candidates": sum(len(candidate_sets[user]) < 5 for user in selected),
                "labeled_users_with_insufficient_relevant_candidates_for_perfect_ap": sum(len(candidate_sets[user] & truth[user]) < min(5, len(truth[user])) for user in nonempty)}
    return {"union": summarize(users, pools), "by_source": {source: summarize(users, sets) for source, sets in source_pools.items()},
            "history_buckets": {bucket: summarize([user for user in users if limited[user] == bucket], pools) for bucket in ("0", "1-5", "6-20", "21+")},
            "candidate_rows": len(table.rows), "candidate_count_min": min(counts), "candidate_count_max": max(counts),
            "candidate_count_mean": float(np.mean(counts)), "ground_truth_filtered_or_injected": False}


def expanded_pool(history: list[dict], associations: dict, pop_order: list[int], available: frozenset[int], depths=EXPANDED_DEPTHS):
    # Reuse V3's full source ranks, changing only the prespecified proposal depths.
    _, ranks = v3.candidate_union(history, associations, pop_order, available)
    repeat = {item for item, rank in ranks["repeat_rank"].items() if rank <= depths[0]}
    association = {item for item, rank in ranks["association_rank"].items() if rank <= depths[1]}
    ranks["sources"] = {"repeat": repeat, "association": association, "popularity": set(pop_order[:depths[2]])}
    pool = set().union(*ranks["sources"].values())
    if len(pool) < 5 or not pool <= available:
        raise ValueError("Invalid expanded available candidate union.")
    return pool, ranks


def build_expanded(root: Path, fold: str, users: list[str], base: v3.FeatureTable) -> v3.FeatureTable:
    started = time.perf_counter()
    history_name, target_name, schedule_name = v3.FOLDS[fold]
    history_window = WINDOW_BY_NAME[history_name]
    target_window = TEST_WINDOW if target_name == "test" else WINDOW_BY_NAME[target_name]
    aggregate = v3.earlier_120_aggregates(root) if fold == "early" else root / "data/processed" / ("final_train_user_programmes.csv" if fold == "final" else "train_user_programmes.csv")
    baskets, support, contexts, stats = v3.base_histories(aggregate, history_window, set(users))
    past = v3.schedule_statistics([root / "data/raw/export_arh_11-20-final.csv", root / "data/raw/export_arh_21-30-final.csv"], history_window)
    future = v3.schedule_statistics([root / "data/raw" / schedule_name], target_window)
    available, availability_stats = load_candidates(root / "data/raw" / schedule_name, target_window)
    if set(future) != available:
        raise AssertionError("Expanded availability differs from shared protocol.")
    pop_order = sorted(available, key=lambda item: (-support[item], item))
    pop_rank = {item: rank for rank, item in enumerate(pop_order, 1)}
    fallback = rank_candidates(support, available)
    table = v3.FeatureTable(f"expanded_{fold}", history_window.end, None if fold == "final" else target_window.end, fallback=fallback)
    base_groups = {user: (start, end) for user, start, end in base.groups}
    processed, cohort = set(), set(users)
    copied = fresh = 0
    d = len(v3.FEATURE_NAMES)
    with tempfile.TemporaryDirectory(prefix="v4-pairs-", dir=root / "data/processed") as temporary:
        counts = AssociationCounts(baskets, Path(temporary) / "pairs.sqlite3")
        try:
            rules, rule_stats = build_rules(counts, v3.AR_CONFIG)

            def append_user(user, history):
                nonlocal copied, fresh
                associations = v3.association_features(baskets.get(user, ()), rules, counts.support, counts.user_count, available)
                pool, ranks = expanded_pool(history, associations, pop_order, available)
                history_map = {row["tv_show_id"]: row for row in history}
                start_old, end_old = base_groups[user]
                existing = {base.rows[i][1]: i for i in range(start_old, end_old)}
                start = len(table.rows)
                for item in sorted(pool):
                    if item in existing:
                        offset = existing[item] * d
                        features = list(base.x[offset:offset + d])
                        # Proposal indicators change with depths; other V3 values stay exact.
                        for source in ("repeat", "association", "popularity"):
                            features[v3.FI[f"source_{source}"]] = float(item in ranks["sources"][source])
                        features[v3.FI["source_count"]] = sum(features[v3.FI[f"source_{source}"]] for source in ("repeat", "association", "popularity"))
                        features[v3.FI["source_repeat_association_agreement"]] = features[v3.FI["source_repeat"]] * features[v3.FI["source_association"]]
                        copied += 1
                    else:
                        features = v3.pair_features(item, history, history_map, associations, ranks, support, counts.support, pop_rank,
                                                   contexts.get(user, {}), past, future, history_window, target_window, len(baskets))
                        fresh += 1
                    table.rows.append((user, item))
                    table.x.extend(features)
                table.groups.append((user, start, len(table.rows)))

            for user, group in itertools.groupby(iter_profiles(profile_path(root, history_name), history_name), key=lambda row: row["user_id"]):
                if user not in cohort:
                    for _ in group:
                        pass
                    continue
                processed.add(user)
                append_user(user, list(group))
            for user in users:
                if user not in processed:
                    append_user(user, [])
        finally:
            counts.close()
    table.metadata = {"history_name": history_name, "target_name": target_name, "base_120_history": stats,
                      "available_ids": sorted(available), "common_training_users": sorted(baskets), "availability": availability_stats,
                      "rules": rule_stats, "depths": EXPANDED_DEPTHS, "copied_v3_feature_rows": copied,
                      "newly_computed_feature_rows": fresh, "runtime_seconds": time.perf_counter() - started}
    if fold != "final":
        truth = profile_context(profile_path(root, target_name), target_name, users)["strict_truth"]
        table.labels = bytearray(int(item in truth[user]) for user, item in table.rows)
    return table


def save_expanded(root: Path, fold: str, table: v3.FeatureTable) -> dict:
    prefix = root / "data/processed" / f"lightgbm_v4_expanded_{fold}"
    numeric, identifiers, metadata = [Path(str(prefix) + suffix) for suffix in (".npy", "_rows.csv", "_meta.json")]
    if any(path.exists() for path in (numeric, identifiers, metadata)):
        raise FileExistsError("Preserving completed v4 expanded features.")
    with numeric.open("xb") as handle:
        np.save(handle, matrix(table), allow_pickle=False)
    with identifiers.open("x", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["user_id", "tv_show_id"] + (["label"] if table.labels is not None else []))
        for index, (user, item) in enumerate(table.rows):
            writer.writerow([user, item] + ([table.labels[index]] if table.labels is not None else []))
    result = {"history_end": table.history_end, "label_end": table.label_end, "groups": table.groups, "fallback": table.fallback,
              "feature_names": v3.FEATURE_NAMES, "details": table.metadata,
              "files": {str(path.relative_to(root)): file_fingerprint(path) for path in (numeric, identifiers)}}
    with metadata.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    return result


def load_expanded(root: Path, fold: str) -> v3.FeatureTable:
    prefix = root / "data/processed" / f"lightgbm_v4_expanded_{fold}"
    metadata = json.loads(Path(str(prefix) + "_meta.json").read_text())
    if tuple(metadata["feature_names"]) != v3.FEATURE_NAMES:
        raise ValueError("Expanded feature schema differs from V3.")
    for name, expected in metadata["files"].items():
        if file_fingerprint(root / name)["sha256"] != expected["sha256"]:
            raise RuntimeError("Expanded numeric/identifier artifact changed.")
    numeric = np.load(Path(str(prefix) + ".npy"), allow_pickle=False)
    table = v3.FeatureTable(f"expanded_{fold}", metadata["history_end"], metadata["label_end"],
                            fallback=tuple(metadata["fallback"]), metadata=metadata["details"])
    table.x.frombytes(np.asarray(numeric, dtype=np.float32).tobytes())
    table.labels = bytearray() if table.label_end is not None else None
    canonical = {}
    with Path(str(prefix) + "_rows.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            user = canonical.setdefault(row["user_id"], row["user_id"])
            table.rows.append((user, int(row["tv_show_id"])))
            if table.labels is not None:
                table.labels.append(int(row["label"]))
    table.groups = [tuple(group) for group in metadata["groups"]]
    matrix(table)
    return table


def table_for(root: Path, pool: str, fold: str) -> v3.FeatureTable:
    return load_expanded(root, fold) if pool == "expanded" else v3.load_table(root, fold)


def run_diagnostics(root: Path, users: list[str], protected: dict) -> dict:
    destination = root / "reports/lightgbm_v4_candidates.json"
    if destination.exists():
        cached = json.loads(destination.read_text())
        for name, old in cached["protected_original_files"].items():
            if file_fingerprint(root / name)["sha256"] != old["sha256"]:
                raise RuntimeError("Original input changed since candidate diagnostics.")
        return cached
    started = time.perf_counter()
    baseline = {}
    for fold, target in (("early", "early_validation"), ("primary", "validation")):
        table = v3.load_table(root, fold)
        truth = profile_context(profile_path(root, target), target, users)["strict_truth"]
        baseline[fold] = candidate_metrics(table, truth, users)
        print(f"Base {fold}: recall={baseline[fold]['union']['micro_recall']!r}, oracle MAP@5={baseline[fold]['union']['candidate_oracle_map_at_5']!r}", flush=True)
    expand = any(item["union"]["candidate_oracle_map_at_5"] < EXPANSION_ORACLE_TRIGGER for item in baseline.values())
    expanded = {}
    if expand:
        for fold in ("early", "primary", "final"):
            prefix = root / "data/processed" / f"lightgbm_v4_expanded_{fold}_meta.json"
            if prefix.exists():
                table = load_expanded(root, fold)
            else:
                base = v3.load_table(root, fold)
                table = build_expanded(root, fold, users, base)
                save_expanded(root, fold, table)
            if fold != "final":
                target = "early_validation" if fold == "early" else "validation"
                truth = profile_context(profile_path(root, target), target, users)["strict_truth"]
                expanded[fold] = candidate_metrics(table, truth, users)
                print(f"Expanded {fold}: recall={expanded[fold]['union']['micro_recall']!r}, oracle MAP@5={expanded[fold]['union']['candidate_oracle_map_at_5']!r}", flush=True)
            del table
    working_pool = "expanded" if expanded and expanded["early"]["union"]["candidate_oracle_map_at_5"] >= baseline["early"]["union"]["candidate_oracle_map_at_5"] + .01 else "base"
    check_integrity(root, protected)
    result = {"baseline_depths": [50, 50, 20], "expanded_depths": EXPANDED_DEPTHS,
              "expansion_trigger_oracle_below": EXPANSION_ORACLE_TRIGGER, "expansion_tested": expand,
              "base": baseline, "expanded": expanded, "working_pool": working_pool,
              "pool_selection_rule": "use expansion if fitting-side early oracle improves by at least 0.01; primary labels do not fit generators or inject candidates",
              "interpretation": "Ranking remains a major bottleneck when oracle is well above the actual V3 score; expansion addresses some additional recall losses.",
              "protected_original_files": protected, "runtime_seconds": time.perf_counter() - started, **peak_memory()}
    with destination.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    return result


def inner_group_split(table: v3.FeatureTable) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    valid_groups = [int.from_bytes(hashlib.sha256(user.encode()).digest()[:4], "big") % 5 == 0 for user, _, _ in table.groups]
    if not any(valid_groups) or all(valid_groups):
        valid_groups = [index % 5 == 0 for index in range(len(table.groups))]
    train, valid, train_sizes, valid_sizes = [], [], [], []
    for is_valid, (_, start, end) in zip(valid_groups, table.groups):
        (valid if is_valid else train).extend(range(start, end))
        (valid_sizes if is_valid else train_sizes).append(end - start)
    if not train or not valid:
        raise ValueError("Fitting-side user split must have nonempty train/validation queries.")
    return np.asarray(train, dtype=np.int32), np.asarray(valid, dtype=np.int32), train_sizes, valid_sizes


def query_weights(table: v3.FeatureTable) -> np.ndarray:
    weights = np.empty(len(table.rows), dtype=np.float32)
    for _, start, end in table.groups:
        weights[start:end] = len(table.rows) / (len(table.groups) * (end - start))
    return weights


def native_params(config: LGBConfig) -> dict:
    params = {"objective": config.objective, "num_leaves": config.num_leaves, "learning_rate": config.learning_rate,
              "min_data_in_leaf": config.min_data_in_leaf, "lambda_l2": config.lambda_l2,
              "num_threads": config.num_threads, "deterministic": True, "force_col_wise": True,
              "seed": SEED, "feature_fraction_seed": SEED, "bagging_seed": SEED, "data_random_seed": SEED,
              "feature_fraction": 1., "bagging_fraction": 1., "max_bin": 63, "verbosity": -1}
    if config.objective == "binary":
        params["metric"] = "binary_logloss"
    else:
        params.update({"metric": "ndcg", "eval_at": [5], "label_gain": [0, 1],
                       "lambdarank_truncation_level": 10, "lambdarank_norm": True})
    return params


def train_model(table: v3.FeatureTable, config: LGBConfig, forecast_origin: int,
                feature_indices: list[int] | None = None, rounds: int | None = None) -> tuple[lgb.Booster, dict]:
    v3.enforce_fit_cutoff(table, forecast_origin)
    started = time.perf_counter()
    features = list(range(len(v3.FEATURE_NAMES))) if feature_indices is None else list(feature_indices)
    x = np.ascontiguousarray(matrix(table)[:, features])
    y = np.frombuffer(table.labels, dtype=np.uint8).astype(np.int32)
    names = [v3.FEATURE_NAMES[index] for index in features]
    params = native_params(config)
    weights = query_weights(table) if config.objective == "binary" else None
    inner = None
    if rounds is None:
        train_indices, valid_indices, train_sizes, valid_sizes = inner_group_split(table)
        fitting = lgb.Dataset(x[train_indices], label=y[train_indices], weight=weights[train_indices] if weights is not None else None,
                             group=train_sizes if config.objective == "lambdarank" else None, feature_name=names)
        validation = lgb.Dataset(x[valid_indices], label=y[valid_indices], weight=weights[valid_indices] if weights is not None else None,
                                group=valid_sizes if config.objective == "lambdarank" else None, reference=fitting, feature_name=names)
        curve = {}
        stopped = lgb.train(params, fitting, num_boost_round=config.max_rounds, valid_sets=[validation], valid_names=["fitting_side_users"],
                            callbacks=[lgb.early_stopping(config.stopping_rounds, first_metric_only=True, verbose=False), lgb.record_evaluation(curve)])
        rounds = stopped.best_iteration or config.max_rounds
        inner = {"train_queries": len(train_sizes), "validation_queries": len(valid_sizes),
                 "train_rows": len(train_indices), "validation_rows": len(valid_indices),
                 "best_iteration": rounds, "best_native_score": stopped.best_score,
                 "metric_role": "binary logloss / NDCG@5 selects rounds on fitting-side users only; neither is reported as MAP@5",
                 "curve": curve, "labels_end": timestamp(table.label_end)}
        del stopped, fitting, validation
    dataset = lgb.Dataset(x, label=y, weight=weights,
                         group=[end - start for _, start, end in table.groups] if config.objective == "lambdarank" else None, feature_name=names)
    booster = lgb.train(params, dataset, num_boost_round=int(rounds))
    booster.free_dataset()
    importance = [{"feature": name, "gain": float(gain), "split": int(split)} for name, gain, split in
                  zip(names, booster.feature_importance("gain"), booster.feature_importance("split"))]
    importance.sort(key=lambda item: (-item["gain"], item["feature"]))
    return booster, {"configuration": asdict(config), "native_params": params, "feature_indices": features,
                     "feature_names": names, "rounds": int(rounds), "trees": booster.num_trees(),
                     "inner_early_stopping": inner, "refit_all_fitting_rows": len(table.rows),
                     "fit_label_end": timestamp(table.label_end), "forecast_origin": timestamp(forecast_origin),
                     "feature_importance": importance, "runtime_seconds": time.perf_counter() - started}


def predict_model(booster: lgb.Booster, table: v3.FeatureTable, indices: list[int]) -> np.ndarray:
    scores = np.asarray(booster.predict(np.ascontiguousarray(matrix(table)[:, indices]), num_threads=THREADS), dtype=np.float64)
    if scores.shape != (len(table.rows),) or not np.isfinite(scores).all():
        raise ValueError("LightGBM produced invalid prediction scores.")
    return scores


def ablation_indices(group: str) -> list[int]:
    schedule = {"future_airings", "first_airing_day_since_2020", "last_airing_day_since_2020", "days_until_first_airing", "days_until_last_airing",
                "scheduled_minutes", "airing_channels", "prime_time_fraction", "evening_fraction", "weekend_fraction", "airing_day_coverage",
                "mean_broadcast_minutes", "expected_future_qualified_airings", "target_days"}
    if group == "future_schedule":
        drop = schedule
    elif group == "association":
        drop = {name for name in v3.FEATURE_NAMES if name.startswith("association_")} | {"source_association", "source_count", "repeat_association_interaction", "source_repeat_association_agreement"}
    elif group == "global_popularity":
        drop = {name for name in v3.FEATURE_NAMES if name.startswith("popularity_")} | {"source_popularity", "source_count", "repeat_popularity_interaction"}
    elif group == "personal_history":
        drop = {name for name in v3.FEATURE_NAMES if name.startswith(("frequency_", "recency_", "days_since_", "repeat_", "user_"))} | {
            "qualified_minutes_gt50", "historical_airing_watch_rate", "expected_future_qualified_airings", "source_repeat", "source_count", "source_repeat_association_agreement", "heavy_user", "cold_start"}
    else:
        raise ValueError("Unknown ablation group.")
    return [index for index, name in enumerate(v3.FEATURE_NAMES) if name not in drop]


def user_rank_maps(table: v3.FeatureTable, scores, users: list[str], depth: int = RANK_FUSION_DEPTH) -> dict:
    maps = {}
    for user, start, end in table.groups:
        if table.value(start, "cold_start"):
            ordered = list(table.fallback)
            ordered += [table.rows[i][1] for i in sorted(range(start, end), key=lambda i: (table.value(i, "popularity_rank"), table.rows[i][1])) if table.rows[i][1] not in ordered]
        else:
            ordered = [table.rows[i][1] for i in sorted(range(start, end), key=lambda i: (-float(scores[i]), table.rows[i][1]))]
        maps[user] = {item: rank for rank, item in enumerate(ordered[:depth], 1)}
    if set(maps) != set(users):
        raise ValueError("Rank map cohort mismatch.")
    return maps


def fuse_ranks(rank_maps: list[dict], weights: tuple[float, ...], users: list[str], available: frozenset[int],
               fallback: tuple[int, ...], cold_users: set[str]) -> dict:
    if len(rank_maps) != len(weights) or any(weight < 0 for weight in weights) or not np.isclose(sum(weights), 1.):
        raise ValueError("Fusion weights must match sources and sum to one.")
    result = {}
    for user in users:
        if user in cold_users:
            result[user] = fallback
            continue
        scores = {}
        for source, weight in zip(rank_maps, weights):
            if not weight:
                continue
            for item, rank in source[user].items():
                if item not in available:
                    raise ValueError("A fusion source proposed an unavailable item.")
                scores[item] = scores.get(item, 0.) + weight / (20. + rank)
        chosen = sorted(scores, key=lambda item: (-scores[item], item))[:5]
        chosen += [item for item in fallback if item not in chosen][:5 - len(chosen)]
        result[user] = tuple(chosen)
    return result


def cold_users(table: v3.FeatureTable) -> set[str]:
    return {user for user, start, _ in table.groups if table.value(start, "cold_start")}


def append_ledger(root: Path, record: dict) -> None:
    with (root / "reports/lightgbm_v4_experiments.jsonl").open("a") as handle:
        handle.write(json.dumps(record) + "\n")


def final_worker(root: Path, specification_path: Path, destination: Path) -> None:
    specification = json.loads(specification_path.read_text())
    users = load_template_users(root / "data/raw/submission.csv")
    base = v3.load_table(root, "final")
    selected = specification["selected_kind"]
    logistic_model = json.loads((root / "reports/rerank_v3_model_final.json").read_text())
    logistic_scores = v3.scores_for(base, v3.MODEL_BY_NAME["logistic_l2_01"], logistic_model)
    if selected == "v3_control":
        predictions = v3.rank_table(base, logistic_scores, users)
    else:
        table = table_for(root, specification["lightgbm"]["pool"], "final")
        booster = lgb.Booster(model_file=str(root / specification["lightgbm"]["final_model_file"]))
        scores = predict_model(booster, table, specification["lightgbm"]["feature_indices"])
        if selected == "lightgbm":
            predictions = v3.rank_table(table, scores, users)
        else:
            maps = [user_rank_maps(base, v3.scores_for(base, v3.MODEL_BY_NAME["repeat_control"]), users),
                    user_rank_maps(base, logistic_scores, users), user_rank_maps(table, scores, users)]
            predictions = fuse_ranks(maps, tuple(specification["weights"]), users, frozenset(base.metadata["available_ids"]), base.fallback, cold_users(base))
    if destination.exists():
        raise FileExistsError("Preserving an existing final replay artifact.")
    write_predictions(destination, users, predictions, frozenset(base.metadata["available_ids"]))


def render_report(report: dict) -> str:
    lines = ["# SWEET.TV V4 LightGBM and Ensemble Study", "", f"Generated (UTC): {report['generated_at_utc']}", "",
             "## Current best manually verified Kaggle result", "", "```json", json.dumps(report["verified_kaggle_history"], indent=2), "```", "",
             "Kaggle scores are user-reported verified returns. V4 has not been uploaded or scored; Public >=0.46 and Private >=0.45 remain ambitions.", "",
             "## Environment and reproducibility", "", "```json", json.dumps(report["environment"], indent=2), "```", "",
             "Fresh Intel macOS venv setup (all installations/relocations remain inside .venv):", "", "```bash",
             ".venv/bin/python -m pip install --no-cache-dir --only-binary=:all: -r requirements-v4.txt",
             ".venv/bin/python src/setup_lightgbm_v4.py", ".venv/bin/python src/setup_lightgbm_v4.py --configure-loader", "```", "",
             "Run on a fresh output/project copy; completed experiments are never overwritten:", "", "```bash",
             ".venv/bin/python src/lightgbm_v4.py --stage diagnostics", ".venv/bin/python src/lightgbm_v4.py --stage study", "```", "",
             "The initially missing @rpath/libomp.dylib was reported before changes. The user authorized a venv-local runtime. "
             "The official pinned Homebrew Intel bottle was SHA-256 verified, extracted into .venv, and only the local LightGBM binary received a local RPATH/ad-hoc signature. "
             "The original wheel binary is backed up there. Normal import and real five-tree training were verified; system Python and system library locations were not changed.", "",
             "## Candidate recall before recommendation-model training", "", "```json",
             json.dumps({key: value for key, value in report["candidate_diagnostics"].items() if key != "protected_original_files"}, indent=2), "```", "",
             "Recall uses every unique strict-relevant item, including items missed by all generators. Source metrics and limited-history buckets are reported for both historical folds. "
             "A single controlled 100/100/40 expansion is triggered when base oracle MAP@5 <0.90; no gold items are injected. "
             "Working-pool selection uses fitting-side early oracle improvement, not hidden-test or Kaggle scores. Binary A is evaluated on both pools.", "",
             "## Chronological validation and early stopping", "",
             "- Fitting features: weeks 11–15; fitting labels: weeks 16–20, ending 2020-05-18.",
             "- Primary prediction features: weeks 11–20; held-out labels: weeks 21–30. No primary label is passed to any held-out model fit.",
             "- A deterministic fitting-side 80/20 user split selects boosting rounds using binary logloss or NDCG@5, depending on objective. Models are then refitted on all fitting-side users at those fixed rounds.",
             "- NDCG@5 is only a native round-selection metric. All recommendation results below use the unchanged external Python MAP@5 over all 1,260 original users, with empty relevance AP=0.",
             "- Classification uses query-balanced row weights. LambdaRank uses complete contiguous user groups, binary gains, truncation level 10, and native query normalization.",
             "- Model/mask/ensemble selection uses the primary hold-out and is therefore explicitly model selection, not an unbiased estimate of hidden-test performance. Public/Private scores do not optimize weights.", "",
             "## Exact validation comparison", "",
             "| Strategy | Status | MAP@5 all 1,260 | MAP@5 nonempty labels | Distinct programmes |",
             "|---|---|---:|---:|---:|"]
    for entry in report["experiments"]:
        metrics = (entry.get("evaluation") or {}).get("metrics", {})
        lines.append(f"| {entry['name']} | {entry['status']} | {metrics.get('primary_all_users', {}).get('map_at_5')!r} | "
                     f"{metrics.get('supplementary_nonempty_ground_truth', {}).get('map_at_5')!r} | {metrics.get('coverage', {}).get('unique_programmes_recommended')!r} |")
    lines += ["", f"Selected final strategy: **{report['selected_strategy']}**.", "", report["selection_conclusion"], "",
              "## Feature ablations and importance", "", "```json", json.dumps(report["ablation_summary"], indent=2), "```", "",
              "Ablations hold the selected candidate universe fixed and remove named feature groups plus their direct interaction/source-count proxies. "
              "They are feature ablations, not generator-removal experiments or proof of causality. Correlated remaining groups can retain related information. "
              "All models preserve full gain/split importances in their experiment records and native model artifacts.", "",
              "## Every tested LightGBM configuration, rounds, importance, and timing", "", "```json",
              json.dumps([{key: value for key, value in entry.items() if key != "evaluation"} for entry in report["experiments"]], indent=2), "```", "",
              "## Final refit and submission", "", "```json", json.dumps(report["final_build"], indent=2), "```", "",
              "After selection, permitted labeled origin tables (early plus primary) are concatenated, with every label ending no later than the test origin. "
              "The selected LightGBM component uses its fitting-side-selected round count; final refitting never early-stops on test labels. "
              "V2 and V3 ensemble components reuse their existing causal final profiles/model, preserving earlier files.", "",
              "## Execution and source fingerprints", "", "```json", json.dumps(report["execution"], indent=2), "```", "",
              "A normal run records the full controlled-study runtime. If report recovery was required, the original whole-study timer/peak are explicitly null; "
              "surviving per-experiment timers and the independently measured recovery/refit are reported instead. Candidate expansion has its own recorded runtime. "
              "Peak RSS is a process high-water measurement with the explicit execution memory_scope, not summed concurrent parent/child memory.", "",
              "## Protected artifacts", "", "| File | SHA-256 | Unchanged |", "|---|---|---|"]
    for name, details in report["input_integrity"].items():
        lines.append(f"| {name} | `{details['before']['sha256']}` | {details['unchanged']} |")
    if report.get("report_recovery"):
        lines += ["", "## Report-writing recovery", "",
                  "All fits, scoring, final publication, and replay completed before an undefined datetime import interrupted initial report construction. "
                  "The import was fixed; all recorded validation CSVs were rescored and checked against the preserved ledger. "
                  "An optional extra verification refit during recovery exceeded the tool limit and is not claimed as a completed fit. "
                  "The preserved final native model's parameters/rounds/importance were inspected instead. "
                  "Two fresh inference processes matched the existing submission byte-for-byte. No experiment, native model, prediction, or submission was overwritten. "
                  "Lost original total timers/peaks remain unavailable rather than fabricated.", "",
                  "```json", json.dumps(report["report_recovery"], indent=2), "```", "",
                  "Recovery command for a completed study with a report-only failure:", "", "```bash",
                  ".venv/bin/python src/lightgbm_v4.py --recover-report", "```"]
    lines += ["", "## Complete relevant automated tests", "", "```text", report["tests"]["output"].rstrip(), "```", "",
              "## Limitations", "",
              "- Only two labeled chronological origins exist; inner user stopping is not an additional temporal hold-out. Primary-based configuration/ablation/ensemble selection can be optimistic.",
              "- Candidate expansion improves a ceiling but cannot recover every future interest. Ground truth is never filtered to candidates; oracle and actual MAP@5 serve different roles.",
              "- Final history/horizon are longer than fitting-side history/horizon. Absolute calendar features and exposure patterns can drift. Group ablations reveal sensitivity but do not guarantee future gains.",
              "- Strict continuous-watch relevance remains a local proxy for hidden Kaggle labels; sparse IDs, source clocks, missing channels, and unattended intervals remain inherited uncertainties.",
              "- All inferior/failed experiments are preserved. Existing predictions, trained models, source implementations, original data, reports, and AI-session exports are not overwritten.",
              "- No Kaggle upload occurred and no V4 Public/Private score or target achievement is claimed.", ""]
    return "\n".join(lines)


def run_study(root: Path, diagnostics: dict, protected: dict, tests: dict) -> dict:
    started = time.perf_counter()
    outputs, reports = root / "outputs", root / "reports"
    destinations = [reports / "lightgbm_v4.json", reports / "lightgbm_v4.md", outputs / "submission_v4.csv"]
    if any(path.exists() for path in destinations):
        raise FileExistsError("Preserving previous v4 experiment/report/submission artifacts.")
    users = load_template_users(root / "data/raw/submission.csv")
    base_early, base_primary = v3.load_table(root, "early"), v3.load_table(root, "primary")
    truth = profile_context(profile_path(root, "validation"), "validation", users)["strict_truth"]
    available = frozenset(base_primary.metadata["available_ids"])
    training_users = frozenset(base_primary.metadata["common_training_users"])
    v3_model = json.loads((reports / "rerank_v3_model_logistic_l2_01.json").read_text())
    v3_scores = np.asarray(v3.scores_for(base_primary, v3.MODEL_BY_NAME["logistic_l2_01"], v3_model), dtype=np.float64)
    v2_scores = np.asarray(v3.scores_for(base_primary, v3.MODEL_BY_NAME["repeat_control"]), dtype=np.float64)
    experiments, fitted = [], {}
    for name, scores, saved_path in (("V2_repeat_control", v2_scores, "repeat_v2_primary_A_frequency_ge30.csv"),
                                     ("V3_logistic_control", v3_scores, "rerank_v3_logistic_l2_01_validation.csv")):
        predictions = v3.rank_table(base_primary, scores, users)
        if predictions != read_predictions(outputs / saved_path, users, available):
            raise AssertionError(f"Unchanged control no longer reproduces saved predictions: {name}")
        experiments.append({"name": name, "status": "complete", "kind": "control",
                            "evaluation": evaluate_recommendations(users, truth, predictions, training_users, available)})
    pool = diagnostics["working_pool"]
    working_early = table_for(root, pool, "early") if pool != "base" else base_early
    working_primary = table_for(root, pool, "primary") if pool != "base" else base_primary

    def evaluate_model(name: str, config: LGBConfig, candidate_pool: str, mask=None, ablation=None):
        entry = {"name": name, "kind": "lightgbm", "status": "started", "pool": candidate_pool,
                 "configuration": asdict(config), "ablation_group": ablation}
        begin = time.perf_counter()
        early = base_early if candidate_pool == "base" else working_early
        primary = base_primary if candidate_pool == "base" else working_primary
        try:
            booster, fit = train_model(early, config, primary.history_end, mask)
            model_path = reports / f"lightgbm_v4_model_{name}.txt"
            if model_path.exists():
                raise FileExistsError("Preserving an existing trained LightGBM model.")
            booster.save_model(str(model_path))
            scores = predict_model(booster, primary, fit["feature_indices"])
            predictions = v3.rank_table(primary, scores, users)
            prediction_path = outputs / f"lightgbm_v4_{name}_validation.csv"
            if prediction_path.exists():
                raise FileExistsError("Preserving an existing v4 validation prediction.")
            write_predictions(prediction_path, users, predictions, available)
            saved = read_predictions(prediction_path, users, available)
            entry.update({"status": "complete", "fit": fit, "model_file": str(model_path.relative_to(root)),
                          "model_sha256": file_fingerprint(model_path)["sha256"],
                          "prediction_file": {"path": str(prediction_path.relative_to(root)), **file_fingerprint(prediction_path)},
                          "evaluation": evaluate_recommendations(users, truth, saved, training_users, available)})
            fitted[name] = {"scores": scores, "entry": entry}
            print(f"{name}: external MAP@5={entry['evaluation']['metrics']['primary_all_users']['map_at_5']!r}; rounds={fit['rounds']}", flush=True)
            del booster
        except (lgb.basic.LightGBMError, ValueError, ArithmeticError) as error:
            entry.update({"status": "failed", "error": repr(error), "evaluation": None})
            print(f"{name}: failed: {error!r}", flush=True)
        entry["runtime_seconds"] = time.perf_counter() - begin
        experiments.append(entry)
        append_ledger(root, entry)

    evaluate_model("binary_A_base", BINARY_CONFIGS[0], "base")
    if pool == "expanded":
        evaluate_model("binary_A_expanded", BINARY_CONFIGS[0], pool)
    for config in (*BINARY_CONFIGS[1:], *RANK_CONFIGS):
        evaluate_model(f"{config.name}_{pool}", config, pool)
    successful = [entry for entry in experiments if entry["kind"] == "lightgbm" and entry["status"] == "complete"]
    if not successful:
        raise RuntimeError("Every real LightGBM fit failed. Completed diagnostics/ledgers are preserved; no replacement model/submission is created.")
    best_full = max(successful, key=lambda entry: entry["evaluation"]["metrics"]["primary_all_users"]["map_at_5"])
    full_score = best_full["evaluation"]["metrics"]["primary_all_users"]["map_at_5"]
    for group in ("personal_history", "association", "future_schedule", "global_popularity"):
        evaluate_model(f"ablate_{group}", LGBConfig(**best_full["configuration"]), best_full["pool"], ablation_indices(group), group)
    successful = [entry for entry in experiments if entry["kind"] == "lightgbm" and entry["status"] == "complete"]
    best_lgb = max(successful, key=lambda entry: entry["evaluation"]["metrics"]["primary_all_users"]["map_at_5"])
    best_table = base_primary if best_lgb["pool"] == "base" else working_primary
    maps = [user_rank_maps(base_primary, v2_scores, users), user_rank_maps(base_primary, v3_scores, users),
            user_rank_maps(best_table, fitted[best_lgb["name"]]["scores"], users)]
    for name, weights in ENSEMBLES:
        begin = time.perf_counter()
        predictions = fuse_ranks(maps, weights, users, available, base_primary.fallback, cold_users(base_primary))
        path = outputs / f"lightgbm_v4_{name}_validation.csv"
        if path.exists():
            raise FileExistsError("Preserving an existing v4 ensemble prediction.")
        write_predictions(path, users, predictions, available)
        saved = read_predictions(path, users, available)
        entry = {"name": name, "kind": "ensemble", "status": "complete", "weights_v2_v3_lightgbm": weights,
                 "rank_fusion_k": 20, "rank_fusion_depth": RANK_FUSION_DEPTH, "lightgbm_component": best_lgb["name"],
                 "evaluation": evaluate_recommendations(users, truth, saved, training_users, available),
                 "prediction_file": {"path": str(path.relative_to(root)), **file_fingerprint(path)},
                 "runtime_seconds": time.perf_counter() - begin}
        experiments.append(entry)
        append_ledger(root, entry)
        print(f"{name}: external MAP@5={entry['evaluation']['metrics']['primary_all_users']['map_at_5']!r}", flush=True)
    # Unchanged V3 is first in an exact tie with any new strategy; V2 cannot displace it on equal score.
    eligible = [entry for entry in experiments if entry["status"] == "complete" and entry["name"] != "V2_repeat_control"]
    selected = max(eligible, key=lambda entry: entry["evaluation"]["metrics"]["primary_all_users"]["map_at_5"])
    final_started = time.perf_counter()
    final_base = v3.load_table(root, "final")
    specification = {"selected_strategy": selected["name"], "selected_kind": "v3_control"}
    final_fit = None
    if selected["kind"] != "control":
        component = best_lgb if selected["kind"] == "ensemble" else selected
        component_pool = component["pool"]
        final_table = table_for(root, component_pool, "final") if component_pool != "base" else final_base
        early = base_early if component_pool == "base" else working_early
        primary = base_primary if component_pool == "base" else working_primary
        combined = v3.concatenate_labeled([early, primary], final_table.history_end)
        booster, final_fit = train_model(combined, LGBConfig(**component["configuration"]), final_table.history_end,
                                         component["fit"]["feature_indices"], rounds=component["fit"]["rounds"])
        final_model = reports / "lightgbm_v4_model_final.txt"
        if final_model.exists():
            raise FileExistsError("Preserving an existing final native model.")
        booster.save_model(str(final_model))
        specification.update({"selected_kind": "ensemble" if selected["kind"] == "ensemble" else "lightgbm",
                              "lightgbm": {"pool": component_pool, "feature_indices": component["fit"]["feature_indices"],
                                           "final_model_file": str(final_model.relative_to(root)), "component_validation_strategy": component["name"]}})
        if selected["kind"] == "ensemble":
            specification["weights"] = selected["weights_v2_v3_lightgbm"]
        del booster, combined
    specification_path = reports / "lightgbm_v4_final_strategy.json"
    with specification_path.open("x") as handle:
        json.dump(specification, handle, indent=2)
        handle.write("\n")
    final_available = frozenset(final_base.metadata["available_ids"])
    with tempfile.TemporaryDirectory(prefix="v4-final-replay-", dir=root / "data/processed") as temporary:
        first, second = Path(temporary) / "first.csv", Path(temporary) / "second.csv"
        final_worker(root, specification_path, first)
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = "173205"
        command = [sys.executable, str(Path(__file__).resolve()), "--worker-specification", str(specification_path), "--worker-output", str(second)]
        completed = subprocess.run(command, cwd=root, env=environment, capture_output=True, text=True)
        if completed.returncode:
            raise RuntimeError(f"Independent native-model replay failed:\n{completed.stderr}")
        if not filecmp.cmp(first, second, shallow=False):
            raise AssertionError("Final v4 CSV differs under independent inference replay.")
        validate_submission(first, root / "data/raw/submission.csv", final_available)
        publish_submission(first, outputs / "submission_v4.csv")
    validation = validate_submission(outputs / "submission_v4.csv", root / "data/raw/submission.csv", final_available)
    integrity = check_integrity(root, protected)
    v3_score = experiments[1]["evaluation"]["metrics"]["primary_all_users"]["map_at_5"]
    selected_score = selected["evaluation"]["metrics"]["primary_all_users"]["map_at_5"]
    ablations = [{"group": entry["ablation_group"], "status": entry["status"],
                  "map_at_5": entry["evaluation"]["metrics"]["primary_all_users"]["map_at_5"] if entry.get("evaluation") else None,
                  "change_from_best_unablated_model": entry["evaluation"]["metrics"]["primary_all_users"]["map_at_5"] - full_score if entry.get("evaluation") else None}
                 for entry in experiments if entry.get("ablation_group")]
    report = {"schema_version": 4, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "verified_kaggle_history": json.loads((reports / "kaggle_best_v3.json").read_text()),
        "environment": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__, "lightgbm": lgb.__version__,
                        "num_threads": THREADS, "seed": SEED,
                        "native_runtime": json.loads((root / ".venv/libomp_install_v4.json").read_text()),
                        "smoke_test": json.loads((reports / "lightgbm_v4_environment.json").read_text())["smoke_test"]},
        "candidate_diagnostics": diagnostics, "experiments": experiments, "selected_strategy": selected["name"],
        "selection_conclusion": f"Selected local MAP@5={selected_score!r}; unchanged V3={v3_score!r}; signed difference={selected_score - v3_score!r}. "
                               + ("New strategy improves the tuned local hold-out; Kaggle improvement is unverified." if selected_score > v3_score else "No new validated improvement; unchanged V3 is retained."),
        "best_unablated_lightgbm": best_full["name"], "strongest_lightgbm_component": best_lgb["name"],
        "ablation_summary": ablations,
        "final_build": {"strategy": specification, "fit": final_fit, "validation": validation,
                        "submission": {"path": "outputs/submission_v4.csv", **file_fingerprint(outputs / "submission_v4.csv")},
                        "byte_identical_independent_replay": True, "replay_hash_seed": "173205",
                        "runtime_seconds": time.perf_counter() - final_started, "kaggle_upload_performed": False},
        "input_integrity": integrity, "tests": tests,
        "execution": {"python_version": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
                      "study_runtime_seconds_before_report_write": time.perf_counter() - started, **peak_memory(),
                      "source_sha256": {name: file_fingerprint(root / name)["sha256"] for name in
                          ("src/lightgbm_v4.py", "tests/test_lightgbm_v4.py", "src/setup_lightgbm_v4.py", "requirements-v4.txt",
                           "src/rerank_v3.py", "src/personalized_repeat.py", "src/association_rules.py", "src/evaluate.py")}}}
    with destinations[0].open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    with destinations[1].open("x") as handle:
        handle.write(render_report(report))
    print(json.dumps({"selected_strategy": report["selected_strategy"], "selected_map_at_5": selected_score,
                      "v3_map_at_5": v3_score, "final_submission": report["final_build"]["submission"],
                      "validation": validation, "execution": report["execution"]}, indent=2), flush=True)
    return report


def recover_completed_report(root: Path, diagnostics: dict, protected: dict, tests: dict) -> dict:
    """Finish a failed report from preserved actual results, without replacing them."""
    started = time.perf_counter()
    print("Report recovery: validating preserved ledger/model/prediction artifacts.", flush=True)
    reports, outputs = root / "reports", root / "outputs"
    destinations = [reports / "lightgbm_v4.json", reports / "lightgbm_v4.md"]
    if any(path.exists() for path in destinations):
        raise FileExistsError("Preserving completed v4 reports.")
    recovery = json.loads((reports / "lightgbm_v4_recovery.json").read_text())
    ledger_path = reports / "lightgbm_v4_experiments.jsonl"
    entries = [json.loads(line) for line in ledger_path.read_text().splitlines() if line.strip()]
    if len(entries) != recovery["completed_noncontrol_ledger_entries"] or len({entry["name"] for entry in entries}) != len(entries):
        raise RuntimeError("Incomplete or duplicate experiment ledger; cannot finalize as successful.")
    users = load_template_users(root / "data/raw/submission.csv")
    primary = v3.load_table(root, "primary")
    available = frozenset(primary.metadata["available_ids"])
    training_users = frozenset(primary.metadata["common_training_users"])
    truth = profile_context(profile_path(root, "validation"), "validation", users)["strict_truth"]
    experiments = []
    for name, filename in (("V2_repeat_control", "repeat_v2_primary_A_frequency_ge30.csv"),
                           ("V3_logistic_control", "rerank_v3_logistic_l2_01_validation.csv")):
        predictions = read_predictions(outputs / filename, users, available)
        experiments.append({"name": name, "kind": "control", "status": "complete",
                            "evaluation": evaluate_recommendations(users, truth, predictions, training_users, available)})
    for entry in entries:
        prediction_file = root / entry["prediction_file"]["path"]
        if file_fingerprint(prediction_file)["sha256"] != entry["prediction_file"]["sha256"]:
            raise RuntimeError("A completed v4 prediction file changed.")
        predictions = read_predictions(prediction_file, users, available)
        actual = evaluate_recommendations(users, truth, predictions, training_users, available)
        if actual != entry["evaluation"]:
            raise AssertionError("Saved v4 experiment does not reproduce its ledger metrics.")
        if entry.get("model_file") and file_fingerprint(root / entry["model_file"])["sha256"] != entry["model_sha256"]:
            raise RuntimeError("A completed native model changed.")
        experiments.append(entry)
    print(f"Report recovery: all {len(entries)} saved experiment metrics independently reproduced.", flush=True)
    complete_lgb = [entry for entry in experiments if entry["kind"] == "lightgbm" and entry["status"] == "complete"]
    best_full = max((entry for entry in complete_lgb if not entry.get("ablation_group")),
                    key=lambda entry: entry["evaluation"]["metrics"]["primary_all_users"]["map_at_5"])
    best_lgb = max(complete_lgb, key=lambda entry: entry["evaluation"]["metrics"]["primary_all_users"]["map_at_5"])
    selected = max((entry for entry in experiments if entry["name"] != "V2_repeat_control"),
                   key=lambda entry: entry["evaluation"]["metrics"]["primary_all_users"]["map_at_5"])
    specification_path = reports / "lightgbm_v4_final_strategy.json"
    specification = json.loads(specification_path.read_text())
    if selected["name"] != specification["selected_strategy"] or selected["name"] != recovery["selected_strategy"]:
        raise AssertionError("Recovered selection differs from the preserved final specification.")
    submission_path = outputs / "submission_v4.csv"
    if file_fingerprint(submission_path)["sha256"] != recovery["existing_submission_sha256"]:
        raise RuntimeError("The already published v4 submission changed.")
    final_base = v3.load_table(root, "final")
    final_available = frozenset(final_base.metadata["available_ids"])
    final_fit = None
    # Inspect the already completed final fit. An optional repeat refit timed out;
    # never represent its missing result/timer as a completed verification claim.
    if specification["selected_kind"] != "v3_control":
        print("Report recovery: inspecting the completed preserved final native model.", flush=True)
        component = next(entry for entry in complete_lgb if entry["name"] == specification["lightgbm"]["component_validation_strategy"])
        pool = component["pool"]
        original_native = lgb.Booster(model_file=str(root / specification["lightgbm"]["final_model_file"]))
        if original_native.current_iteration() != component["fit"]["rounds"] or original_native.num_feature() != len(component["fit"]["feature_indices"]):
            raise AssertionError("Persisted final model rounds/features differ from the selected component.")
        final_fit = {"configuration": component["configuration"], "rounds": original_native.current_iteration(),
            "trees": original_native.num_trees(), "feature_indices": component["fit"]["feature_indices"],
            "native_model_parameters": original_native.params,
            "feature_importance": [{"feature": name, "gain": float(gain), "split": int(split)} for name, gain, split in
                                   zip(original_native.feature_name(), original_native.feature_importance("gain"), original_native.feature_importance("split"))],
            "declared_refit_label_origins": ["early", "primary"], "all_labels_end_no_later_than": timestamp(final_base.history_end),
            "original_final_refit_runtime_seconds": None,
            "measurement_scope": "inspection of completed preserved native artifact; original whole-run/final-fit timers were not persisted",
            "optional_extra_verification_refit_completed": False}
        del original_native
    with tempfile.TemporaryDirectory(prefix="v4-report-recovery-", dir=root / "data/processed") as temporary:
        first, second = Path(temporary) / "first.csv", Path(temporary) / "second.csv"
        for path, seed in ((first, "173205"), (second, "223606")):
            environment = os.environ.copy()
            environment["PYTHONHASHSEED"] = seed
            print(f"Report recovery: fresh inference replay with hash seed {seed}.", flush=True)
            completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker-specification", str(specification_path),
                                        "--worker-output", str(path)], cwd=root, env=environment, capture_output=True, text=True, timeout=180)
            if completed.returncode:
                raise RuntimeError(f"Recovery prediction replay failed:\n{completed.stderr}")
        if not filecmp.cmp(first, second, shallow=False) or not filecmp.cmp(first, submission_path, shallow=False):
            raise AssertionError("Recovered final predictions differ from the preserved v4 CSV.")
    validation = validate_submission(submission_path, root / "data/raw/submission.csv", final_available)
    print("Report recovery: final inference replay and published CSV verified; checking original file hashes.", flush=True)
    integrity = check_integrity(root, protected)
    full_score = best_full["evaluation"]["metrics"]["primary_all_users"]["map_at_5"]
    v3_score = experiments[1]["evaluation"]["metrics"]["primary_all_users"]["map_at_5"]
    selected_score = selected["evaluation"]["metrics"]["primary_all_users"]["map_at_5"]
    ablations = [{"group": entry["ablation_group"], "status": entry["status"],
                  "map_at_5": entry["evaluation"]["metrics"]["primary_all_users"]["map_at_5"],
                  "change_from_best_unablated_model": entry["evaluation"]["metrics"]["primary_all_users"]["map_at_5"] - full_score}
                 for entry in complete_lgb if entry.get("ablation_group")]
    report = {"schema_version": 4, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "verified_kaggle_history": json.loads((reports / "kaggle_best_v3.json").read_text()),
        "environment": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__, "lightgbm": lgb.__version__,
                        "num_threads": THREADS, "seed": SEED,
                        "native_runtime": json.loads((root / ".venv/libomp_install_v4.json").read_text()),
                        "smoke_test": json.loads((reports / "lightgbm_v4_environment.json").read_text())["smoke_test"]},
        "candidate_diagnostics": diagnostics, "experiments": experiments, "selected_strategy": selected["name"],
        "selection_conclusion": f"Selected local MAP@5={selected_score!r}; unchanged V3={v3_score!r}; signed difference={selected_score - v3_score!r}. "
                               + ("New strategy improves the tuned local hold-out; Kaggle improvement is unverified." if selected_score > v3_score else "Unchanged V3 is retained."),
        "best_unablated_lightgbm": best_full["name"], "strongest_lightgbm_component": best_lgb["name"], "ablation_summary": ablations,
        "final_build": {"strategy": specification, "fit": final_fit, "validation": validation,
                        "submission": {"path": "outputs/submission_v4.csv", **file_fingerprint(submission_path)},
                        "byte_identical_independent_replay": True, "replay_hash_seed": "223606",
                        "original_final_build_runtime_seconds": None, "kaggle_upload_performed": False},
        "input_integrity": integrity, "tests": tests,
        "report_recovery": {"initial_failure": recovery, "followup": json.loads((reports / "lightgbm_v4_recovery_followup.json").read_text())},
        "execution": {"python_version": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
                      "study_runtime_seconds_before_report_write": None,
                      "original_whole_study_runtime_and_peak_memory": None,
                      "completed_experiment_runtime_seconds_sum": sum(entry["runtime_seconds"] for entry in entries),
                      "report_recovery_and_inference_verification_runtime_seconds": time.perf_counter() - started,
                      **peak_memory(),
                      "memory_scope": "measured report-recovery parent process, including artifact rescoring/inspection; original study peak unavailable; no completed verification refit claimed",
                      "source_sha256": {name: file_fingerprint(root / name)["sha256"] for name in
                          ("src/lightgbm_v4.py", "tests/test_lightgbm_v4.py", "src/setup_lightgbm_v4.py", "requirements-v4.txt",
                           "src/rerank_v3.py", "src/personalized_repeat.py", "src/association_rules.py", "src/evaluate.py")}}}
    with destinations[0].open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    with destinations[1].open("x") as handle:
        handle.write(render_report(report))
    print(json.dumps({"report_recovered": True, "selected": selected["name"], "map_at_5": selected_score,
                      "v3_map_at_5": v3_score, "submission": report["final_build"]["submission"],
                      "validation": validation, "execution": report["execution"]}, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "diagnostics", "study"), default="all")
    parser.add_argument("--recover-report", action="store_true", help="Finalize a completed preserved study after a report-writing exception; do not retrain experiments or overwrite artifacts.")
    parser.add_argument("--worker-specification", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.worker_specification:
        if args.worker_output is None:
            parser.error("A fresh worker output is required.")
        final_worker(root, args.worker_specification, args.worker_output)
        return
    started = time.perf_counter()
    print("V4: fingerprinting protected preexisting project files.", flush=True)
    protected = protected_snapshot(root)
    print(f"V4: fingerprinted {len(protected)} protected files; running complete tests.", flush=True)
    best = json.loads((root / "reports/kaggle_best_v3.json").read_text())
    if protected["outputs/submission_v3.csv"]["sha256"] != best["submission_sha256"]:
        raise RuntimeError("Current-best v3 file changed.")
    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
    completed = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=120)
    test_output = completed.stdout + completed.stderr
    print(test_output, end="", flush=True)
    if completed.returncode:
        raise RuntimeError("Complete tests failed; actual candidate/model study not started.")
    tests = {"command": command, "returncode": completed.returncode, "output": test_output}
    users = load_template_users(root / "data/raw/submission.csv")
    diagnostics = run_diagnostics(root, users, protected)
    if args.recover_report:
        recover_completed_report(root, diagnostics, protected, tests)
    elif args.stage != "diagnostics":
        run_study(root, diagnostics, protected, tests)
    print(f"Total invocation runtime: {time.perf_counter() - started:.3f} seconds", flush=True)


if __name__ == "__main__":
    main()
