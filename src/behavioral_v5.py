"""Versioned V5 behavioral ranking, chronological examples, and CF pilot.

Run evidence first, then this script with --stage features / --stage study.
New optimization jobs are refused after 2026-10-24 13:00 UTC (16:00 MSK).
Every native fitting job runs in a fresh, timeout-bounded subprocess and saves
its model/predictions/measurements before human report generation.
"""

from __future__ import annotations

import argparse
import csv
import filecmp
import hashlib
import itertools
import json
import os
import subprocess
import sys
import tempfile
import time
from array import array
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("KMP_BLOCKTIME", "0")

import lightgbm as lgb
import numpy as np
import scipy
from scipy import sparse

import rerank_v3 as v3
import lightgbm_v4 as v4
from association_rules import AssociationCounts, build_rules
from audit_data import peak_memory
from behavioral_evidence_v5 import (
    CHANNEL_FEATURES, TIME_FEATURES, HISTORY_WINDOWS, NEW_WINDOWS, ROLLING_ORIGINS,
    behavior_features, load_context, opportunities, open_work_database, read_profile,
)
from build_interactions import EVENT_COLUMNS, PAIR_COLUMNS, checked_records, file_fingerprint, parse_timestamp, seconds, timestamp
from create_final_submission import TEST_WINDOW, project_snapshot, publish_submission, validate_submission
from evaluate import evaluate_recommendations, load_candidates, load_template_users, read_predictions, write_predictions
from personalized_repeat import iter_profiles, profile_context, profile_path
from popularity import rank_candidates


FREEZE = datetime(2026, 10, 24, 13, 0, tzinfo=timezone.utc)
NAMES = tuple(v3.FEATURE_NAMES) + CHANNEL_FEATURES + TIME_FEATURES
D = len(NAMES)
NEW_BY_NAME = {window.name: window for window in NEW_WINDOWS}
ORIGIN_BY_NAME = {name: (history, target) for name, history, target in ROLLING_ORIGINS}
EXTRA_USER_LIMIT = 500
CONFIG = v4.LGBConfig("v5_binary_C", "binary", 31, .03, 50, 10.)


def optimization_allowed() -> bool:
    return datetime.now(timezone.utc) < FREEZE


def own(name: str) -> bool:
    return ("behavioral_v5" in name or name == "outputs/submission_v5.csv")


def protected_snapshot(root: Path) -> dict:
    return {name: value for name, value in project_snapshot(root).items() if not own(name)}


def preserve(root: Path, before: dict) -> dict:
    result = {}
    for name, old in before.items():
        current = file_fingerprint(root / name)
        if current != old:
            raise RuntimeError(f"Protected V1-V4 artifact changed: {name}")
        result[name] = {"before": old, "after": current, "unchanged": True}
    return result


def bundle_paths(root, name):
    prefix = root / "data/processed" / f"behavioral_v5_{name}"
    return Path(str(prefix) + ".npz"), Path(str(prefix) + "_ids.csv"), Path(str(prefix) + "_meta.json")


def save_bundle(root, name, table, x, details):
    numeric, ids, meta = bundle_paths(root, name)
    if any(path.exists() for path in (numeric, ids, meta)):
        raise FileExistsError("Preserving a completed V5 feature bundle.")
    with numeric.open("xb") as handle:
        np.savez_compressed(handle, x=np.asarray(x, dtype=np.float32))
    with ids.open("x", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["user_id", "tv_show_id"] + (["label"] if table.labels is not None else []))
        for index, (user, item) in enumerate(table.rows):
            writer.writerow([user, item] + ([table.labels[index]] if table.labels is not None else []))
    data = {"name": name, "history_end": table.history_end, "label_end": table.label_end, "groups": table.groups,
            "fallback": table.fallback, "metadata": table.metadata, "feature_names": NAMES, "details": details,
            "files": {str(path.relative_to(root)): file_fingerprint(path) for path in (numeric, ids)}}
    with meta.open("x") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")
    return data


def load_bundle(root, name, corrected=True):
    if corrected and bundle_paths(root, name + "_normalized_channels")[2].exists():
        name += "_normalized_channels"
    numeric, ids, meta = bundle_paths(root, name)
    data = json.loads(meta.read_text())
    if tuple(data["feature_names"]) != NAMES:
        raise ValueError("V5 matrix schema changed.")
    for path, expected in data["files"].items():
        if file_fingerprint(root / path)["sha256"] != expected["sha256"]:
            raise RuntimeError("V5 bundle changed.")
    with np.load(numeric, allow_pickle=False) as archive:
        x = np.asarray(archive["x"], dtype=np.float32)
    table = v3.FeatureTable(name, data["history_end"], data["label_end"], fallback=tuple(data["fallback"]), metadata=data["metadata"])
    table.x.frombytes(np.ascontiguousarray(x[:, :len(v3.FEATURE_NAMES)]).tobytes())
    table.groups = [tuple(group) for group in data["groups"]]
    table.labels = bytearray() if table.label_end is not None else None
    canonical = {}
    with ids.open(newline="") as handle:
        for row in csv.DictReader(handle):
            user = canonical.setdefault(row["user_id"], row["user_id"])
            table.rows.append((user, int(row["tv_show_id"])))
            if table.labels is not None:
                table.labels.append(int(row["label"]))
    if x.shape != (len(table.rows), D) or not np.isfinite(x).all():
        raise ValueError("V5 numeric/identifier alignment failure.")
    return table, x, data


def normalize_channel_bundles(root):
    """Version a corrected column; preserve completed caches with integer channel keys."""
    result = {}
    feature_index = NAMES.index("channel_programme_seen_preference")
    for name, context_name in (("early", "early"), ("primary", "primary"), ("final", "final"),
                               ("roll13", "prefix13"), ("roll15", "early"), ("roll17", "prefix17")):
        corrected_name = name + "_normalized_channels"
        if bundle_paths(root, corrected_name)[2].exists():
            result[name] = str(bundle_paths(root, corrected_name)[2].relative_to(root))
            continue
        table, x, original = load_bundle(root, name, corrected=False)
        contexts, global_ = load_context(root, context_name, {user for user, _, _ in table.groups})
        past = historical_channels(root, HISTORY_WINDOWS[context_name])
        global_total = max(1., global_["seconds"])
        changed = 0
        for user, start, end in table.groups:
            context = contexts.get(user, {})
            channels = context.get("channels", {})
            preferred = set(sorted(channels, key=lambda chan: (-channels[chan][0], int(chan)))[:3])
            denominator = context.get("seconds", 0.) + 600.
            affinity = {chan: (channels[chan][0] + 600. * global_["channels"].get(chan, 0.) / global_total) / denominator for chan in preferred}
            for row in range(start, end):
                value = sum(affinity[str(chan)] for chan in past.get(table.rows[row][1], ()) if str(chan) in preferred)
                changed += not np.isclose(x[row, feature_index], value)
                x[row, feature_index] = value
        save_bundle(root, corrected_name, table, x, {**original["details"], "parent_bundle": name,
                    "correction": "normalize historical integer channel IDs to context string keys in channel_programme_seen_preference only",
                    "changed_feature_values": changed})
        result[name] = str(bundle_paths(root, corrected_name)[2].relative_to(root))
        print(f"Versioned channel-key normalization for {name}: {changed:,} changed values; original cache preserved.", flush=True)
    return result


def additional_120(root: Path):
    paths = {name: root / "data/processed" / f"behavioral_v5_120_{name}.csv" for name in ("prefix13", "prefix17")}
    if all(path.exists() for path in paths.values()):
        return paths
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("Partial earlier-origin aggregates are preserved; inspect before resuming.")
    with tempfile.TemporaryDirectory(prefix="v5-120-", dir=root / "data/processed") as temporary:
        connection = open_work_database(Path(temporary) / "aggregates.sqlite3")
        connection.execute("CREATE TABLE pairs(w TEXT,u TEXT,p INTEGER,n INTEGER,seconds INTEGER,first INTEGER,last INTEGER,PRIMARY KEY(w,u,p)) WITHOUT ROWID")
        pending = []
        def flush():
            connection.executemany("""INSERT INTO pairs VALUES(?,?,?,?,?,?,?) ON CONFLICT(w,u,p) DO UPDATE SET
                n=n+excluded.n,seconds=seconds+excluded.seconds,first=MIN(first,excluded.first),last=MAX(last,excluded.last)""", pending)
            pending.clear()
            connection.commit()
        for _, row in checked_records(root / "data/processed/train_interactions.csv", EVENT_COLUMNS):
            low, high = seconds(parse_timestamp(row["overlap_start"])), seconds(parse_timestamp(row["overlap_end"]))
            for name in paths:
                window = HISTORY_WINDOWS[name]
                first, last = max(low, window.start), min(high, window.end)
                if last - first >= 120:
                    pending.append((name, row["user_id"], int(row["tv_show_id"]), 1, last - first, first, last))
            if len(pending) >= 5000:
                flush()
        flush()
        for name, path in paths.items():
            with path.open("x", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(PAIR_COLUMNS)
                for user, item, count, duration, first, last in connection.execute("SELECT u,p,n,seconds,first,last FROM pairs WHERE w=? ORDER BY u,p", (name,)):
                    writer.writerow([user, item, count, duration, timestamp(first), timestamp(last)])
        connection.close()
    return paths


def historical_channels(root, window):
    stats = v3.schedule_statistics([root / "data/raw/export_arh_11-20-final.csv", root / "data/raw/export_arh_21-30-final.csv"], window)
    return {item: entry["channels"] for item, entry in stats.items()}


def append_behavior(root, name, table, context_name, history_window, target_window, target_file, past=None):
    users = {user for user, _, _ in table.groups}
    contexts, global_ = load_context(root, context_name, users)
    future = opportunities([root / "data/raw" / target_file], target_window)
    if set(future) != set(table.metadata["available_ids"]):
        raise AssertionError("Behavioral availability differs from V4 policy.")
    past = historical_channels(root, history_window) if past is None else past
    x = np.empty((len(table.rows), D), dtype=np.float32)
    x[:, :len(v3.FEATURE_NAMES)] = v4.matrix(table)
    started = time.perf_counter()
    for user, start, end in table.groups:
        context = contexts.get(user)
        for row in range(start, end):
            item = table.rows[row][1]
            x[row, len(v3.FEATURE_NAMES):] = behavior_features(context, global_, future[item], history_window.end,
                table.value(row, "frequency_ge30"), table.value(row, "association_best_confidence"), past.get(item, ()))
    return save_bundle(root, name, table, x, {"context_origin": context_name, "target_start": timestamp(target_window.start),
        "target_end_exclusive": timestamp(target_window.end), "behavior_feature_runtime_seconds": time.perf_counter() - started,
        "context_users": len(contexts), "user_channel_context_includes_unlabeled_tuning": True,
        "channel_duration_policy": "same-user/channel interval union; raw event counts retained separately",
        "label_policy": "individual-session >80%, never contextual interval union"})


def rolling_table(root, name, users, aggregate_paths):
    history_name, target_name = ORIGIN_BY_NAME[name]
    history_window, target_window = HISTORY_WINDOWS[history_name], NEW_BY_NAME[target_name]
    aggregate = root / "data/processed/rerank_v3_early_120.csv" if history_name == "early" else aggregate_paths[history_name]
    baskets, supports, contexts, stats = v3.base_histories(aggregate, history_window, set(users))
    past_stats = v3.schedule_statistics([root / "data/raw/export_arh_11-20-final.csv"], history_window)
    future_stats = v3.schedule_statistics([root / "data/raw/export_arh_11-20-final.csv"], target_window)
    available, availability = load_candidates(root / "data/raw/export_arh_11-20-final.csv", target_window)
    pop_order = sorted(available, key=lambda item: (-supports[item], item))
    pop_rank = {item: rank for rank, item in enumerate(pop_order, 1)}
    table = v3.FeatureTable(name, history_window.end, target_window.end, fallback=rank_candidates(supports, available))
    cohort, observed = set(users), set()
    path = profile_path(root, "early_train") if history_name == "early" else root / "data/processed" / f"behavioral_v5_profile_{history_name}.csv"
    records = iter_profiles(path, "early_train") if history_name == "early" else read_profile(path, history_window)
    with tempfile.TemporaryDirectory(prefix="v5-roll-rules-", dir=root / "data/processed") as temporary:
        counts = AssociationCounts(baskets, Path(temporary) / "pairs.sqlite3")
        try:
            rules, rule_stats = build_rules(counts, v3.AR_CONFIG)
            def append_user(user, history):
                association = v3.association_features(baskets.get(user, ()), rules, counts.support, counts.user_count, available)
                pool, ranks = v4.expanded_pool(history, association, pop_order, available)
                mapping = {row["tv_show_id"]: row for row in history}
                start = len(table.rows)
                for item in sorted(pool):
                    table.rows.append((user, item))
                    table.x.extend(v3.pair_features(item, history, mapping, association, ranks, supports, counts.support, pop_rank,
                                                   contexts.get(user, {}), past_stats, future_stats, history_window, target_window, len(baskets)))
                table.groups.append((user, start, len(table.rows)))
            for user, group in itertools.groupby(records, key=lambda row: row["user_id"]):
                if user not in cohort:
                    for _ in group:
                        pass
                    continue
                observed.add(user)
                append_user(user, list(group))
            for user in users:
                if user not in observed:
                    append_user(user, [])
        finally:
            counts.close()
    truth = {user: set() for user in users}
    for row in read_profile(root / "data/processed" / f"behavioral_v5_profile_{target_name}.csv", target_window):
        if row["user_id"] in truth and row["frequency_gt80"]:
            truth[row["user_id"]].add(row["tv_show_id"])
    table.labels = bytearray(int(item in truth[user]) for user, item in table.rows)
    table.metadata = {"available_ids": sorted(available), "common_training_users": sorted(baskets),
        "base_120_history": stats, "availability": availability, "rules": rule_stats,
        "history_start": history_window.start, "target_start": target_window.start,
        "original_evaluation_cohort": users[:1260], "extra_fitting_users": users[1260:],
        "label_target_windows_do_not_overlap_other_roll_targets": True}
    return table, {item: entry["channels"] for item, entry in past_stats.items()}


def prepare_features(root, users):
    manifest_path = root / "data/processed/behavioral_v5_features_manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    if not (root / "data/processed/behavioral_v5_context_checkpoint.json").exists() or not (root / "data/processed/behavioral_v5_profile_checkpoint.json").exists():
        raise RuntimeError("Finish the evidence/context/profile checkpoints first.")
    started = time.perf_counter()
    bundles = {}
    for fold, context_name, history_name, target_name, target_file in (
        ("early", "early", "early_train", "early_validation", "export_arh_11-20-final.csv"),
        ("primary", "primary", "train", "validation", "export_arh_21-30-final.csv"),
        ("final", "final", "full", "test", "export_arh_31-42-final.csv")):
        if bundle_paths(root, fold)[2].exists():
            bundles[fold] = json.loads(bundle_paths(root, fold)[2].read_text())
            continue
        from continuous_watch import WINDOW_BY_NAME
        table = v4.load_expanded(root, fold)
        target = TEST_WINDOW if target_name == "test" else WINDOW_BY_NAME[target_name]
        print(f"Appending personalized context to {fold}: {len(table.rows):,} candidates.", flush=True)
        bundles[fold] = append_behavior(root, fold, table, context_name, HISTORY_WINDOWS[context_name], target, target_file)
        del table
    aggregate_paths = additional_120(root)
    # Enrollment uses only the first new origin's historical activity, never target labels.
    n_history = Counter()
    for row in v3.AggregateReader(aggregate_paths["prefix13"], HISTORY_WINDOWS["prefix13"]):
        n_history[row.user_id] += 1
    extra = sorted((user for user, count in n_history.items() if user not in set(users) and count >= 5),
                   key=lambda user: hashlib.sha256(("v5-extra:" + user).encode()).hexdigest())[:EXTRA_USER_LIMIT]
    fitting_users = users + extra
    for name, history_name, target_name in ROLLING_ORIGINS:
        if bundle_paths(root, name)[2].exists():
            bundles[name] = json.loads(bundle_paths(root, name)[2].read_text())
            continue
        print(f"Building exact chronological origin {name}; {len(fitting_users)} fitting users.", flush=True)
        table, past = rolling_table(root, name, fitting_users, aggregate_paths)
        bundles[name] = append_behavior(root, name, table, history_name, HISTORY_WINDOWS[history_name], NEW_BY_NAME[target_name], "export_arh_11-20-final.csv", past)
        del table
    result = {"bundles": {name: {"rows": sum(end - start for _, start, end in data["groups"]),
                                        "queries": len(data["groups"]), "history_end": timestamp(data["history_end"]),
                                        "label_end": timestamp(data["label_end"]) if data["label_end"] else None}
                           for name, data in bundles.items()},
              "extra_fitting_users": extra, "extra_user_selection": "500 deterministic hashed non-template users with >=5 distinct broad programmes before 2020-03-30",
              "runtime_seconds": time.perf_counter() - started, **peak_memory()}
    with manifest_path.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    return result


def selected_columns(group):
    base = list(range(len(v3.FEATURE_NAMES)))
    channel = list(range(len(base), len(base) + len(CHANNEL_FEATURES)))
    time_ = list(range(len(base) + len(CHANNEL_FEATURES), D))
    if group == "base": return base
    if group == "channel": return base + channel
    if group == "time": return base + time_
    if group == "both": return base + channel + time_
    if group.startswith("drop_"):
        name, _, parent = group[5:].partition(":")
        eligible = selected_columns(parent or "both")
        keep = [i for i in v4.ablation_indices(name) if i in eligible]
        extras = [i for i in channel + time_ if i in eligible]
        if name == "personal_history":
            extras = [i for i in extras if "repeat_interaction" not in NAMES[i]]
        if name == "association":
            extras = [i for i in extras if "association_interaction" not in NAMES[i]]
        return keep + extras
    raise ValueError("Unknown feature block.")


def fitting_arrays(root, names, columns, origin, templates_only, users):
    xs, ys, groups, weights, windows = [], [], [], [], []
    cohort = set(users)
    for name in names:
        table, x, data = load_bundle(root, name)
        v3.enforce_fit_cutoff(table, origin)
        retained = [(user, start, end) for user, start, end in table.groups if not templates_only or user in cohort]
        indices = np.asarray([i for _, start, end in retained for i in range(start, end)], dtype=np.int32)
        origin_mass = sum(1. if user in cohort else .5 for user, _, _ in retained)
        row_weights = []
        for user, start, end in retained:
            row_weights.extend([(1. if user in cohort else .5) / (len(names) * origin_mass * (end - start))] * (end - start))
        xs.append(np.ascontiguousarray(x[indices][:, columns]))
        ys.append(np.frombuffer(table.labels, dtype=np.uint8)[indices].astype(np.int32))
        groups.extend(end - start for _, start, end in retained)
        weights.extend(row_weights)
        windows.append({"origin": name, "history_end": timestamp(table.history_end), "label_end": timestamp(table.label_end),
                        "users": len(retained), "candidate_pairs": len(indices), "positive_labels": int(ys[-1].sum()),
                        "extra_users": sum(user not in cohort for user, _, _ in retained)})
    x, y = np.concatenate(xs), np.concatenate(ys)
    w = np.asarray(weights, dtype=np.float32) * len(y)
    return x, y, groups, w, windows


def fit_worker(root, spec_path):
    spec = json.loads(spec_path.read_text())
    result_path = root / spec["result_path"]
    if result_path.exists():
        raise FileExistsError("Preserving a completed V5 fit.")
    started = time.perf_counter()
    users = load_template_users(root / "data/raw/submission.csv")
    target, target_x, _ = load_bundle(root, spec["target"])
    columns = selected_columns(spec["features"])
    x, y, groups, weights, windows = fitting_arrays(root, spec["fit_origins"], columns, target.history_end, spec["templates_only"], users)
    config = CONFIG if spec["objective"] == "binary" else v4.LGBConfig("v5_lambdarank", "lambdarank", 15, .05, 30, 5.)
    params = v4.native_params(config)
    if spec["objective"] == "lambdarank":
        params["lambdarank_truncation_level"] = 5
    dataset = lgb.Dataset(x, label=y, weight=weights if spec["objective"] == "binary" else None,
                         group=groups if spec["objective"] == "lambdarank" else None, feature_name=[NAMES[i] for i in columns])
    booster = lgb.train(params, dataset, num_boost_round=spec["rounds"])
    booster.free_dataset()
    model_path = root / spec["model_path"]
    if model_path.exists():
        raise FileExistsError("Preserving an existing V5 native model.")
    booster.save_model(str(model_path))
    scores = booster.predict(np.ascontiguousarray(target_x[:, columns]), num_threads=2)
    if not np.isfinite(scores).all():
        raise ArithmeticError("Nonfinite V5 predictions.")
    score_path = root / spec["score_path"]
    with score_path.open("xb") as handle:
        np.save(handle, scores, allow_pickle=False)
    all_target_users = [user for user, _, _ in target.groups]
    predictions = v3.rank_table(target, scores, all_target_users)
    # Rolling diagnostic tables also contain extra fitting users; the scored cohort
    # and saved prediction file remain the original 1,260 template users.
    predictions = {user: predictions[user] for user in users}
    path = root / spec["prediction_path"]
    if path.exists():
        raise FileExistsError("Preserving a prior V5 prediction.")
    write_predictions(path, users, predictions, frozenset(target.metadata["available_ids"]))
    importance = [{"feature": NAMES[index], "gain": float(gain), "split": int(split)} for index, gain, split in
                  zip(columns, booster.feature_importance("gain"), booster.feature_importance("split"))]
    importance.sort(key=lambda row: (-row["gain"], row["feature"]))
    result = {"specification": spec, "training_windows": windows, "rows": len(y), "positives": int(y.sum()),
              "features": [NAMES[i] for i in columns], "columns": columns, "native_params": params,
              "rounds": booster.current_iteration(), "importance": importance,
              "model_sha256": file_fingerprint(model_path)["sha256"], "prediction_sha256": file_fingerprint(path)["sha256"],
              "runtime_seconds": time.perf_counter() - started, **peak_memory(),
              "weighting": "equal origin mass; equal user mass within origin; extra fitting users receive half template-user mass; inverse query candidate count"}
    with result_path.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(f"V5 worker {spec['name']} completed in {result['runtime_seconds']:.2f}s", flush=True)


def run_fit(root, name, features, fit_origins, target="primary", templates_only=True, objective="binary", rounds=269, final=False):
    result_path = root / "reports" / f"behavioral_v5_{name}_fit.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    if not final and not optimization_allowed():
        return {"status": "skipped_freeze", "name": name}
    spec = {"name": name, "features": features, "fit_origins": fit_origins, "target": target,
            "templates_only": templates_only, "objective": objective, "rounds": rounds,
            "model_path": f"reports/behavioral_v5_{name}_model.txt", "score_path": f"data/processed/behavioral_v5_{name}_scores.npy",
            "prediction_path": f"outputs/behavioral_v5_{name}.csv", "result_path": str(result_path.relative_to(root))}
    spec_path = root / "reports" / f"behavioral_v5_{name}_spec.json"
    if not spec_path.exists():
        spec_path.write_text(json.dumps(spec, indent=2) + "\n")
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(spec_path)]
    print(f"Starting isolated V5 fit: {name}", flush=True)
    try:
        remaining = (FREEZE - datetime.now(timezone.utc)).total_seconds()
        limit = 300 if final else max(1, min(300, remaining))
        completed = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=limit)
        if completed.returncode:
            return {"status": "failed", "name": name, "error": completed.stderr, "specification": spec}
        print(completed.stdout, end="", flush=True)
        return json.loads(result_path.read_text())
    except subprocess.TimeoutExpired:
        return {"status": "failed_timeout", "name": name, "limit_seconds": 300, "specification": spec}


def strict_truth(root, target, users):
    if target == "primary":
        return profile_context(profile_path(root, "validation"), "validation", users)["strict_truth"]
    window = NEW_BY_NAME["labels18_20"]
    truth = {user: set() for user in users}
    for row in read_profile(root / "data/processed/behavioral_v5_profile_labels18_20.csv", window):
        if row["user_id"] in truth and row["frequency_gt80"]:
            truth[row["user_id"]].add(row["tv_show_id"])
    return truth


def summarize_groups(table, evaluation):
    sizes = {user: table.value(start, "user_distinct_programmes") for user, start, _ in table.groups}
    result = {}
    for name, predicate in (("cold", lambda n: n == 0), ("low_1_20", lambda n: 0 < n <= 20),
                            ("medium_21_100", lambda n: 20 < n <= 100), ("high_101_plus", lambda n: n > 100)):
        selected = [details["ap_at_5"] for user, details in evaluation["per_user"].items() if predicate(sizes[user])]
        result[name] = {"users": len(selected), "map_at_5": float(np.mean(selected)) if selected else None}
    return result


def evaluate_fit(root, fit, users, target="primary"):
    if fit.get("status", "complete") != "complete":
        return fit
    table, _, _ = load_bundle(root, target)
    truth = strict_truth(root, target, users)
    path = root / fit["specification"]["prediction_path"]
    predictions = read_predictions(path, users, frozenset(table.metadata["available_ids"]))
    evaluated = evaluate_recommendations(users, truth, predictions, frozenset(table.metadata["common_training_users"]), frozenset(table.metadata["available_ids"]))
    return {"name": fit["specification"]["name"], "status": "complete", "fit": fit, "evaluation": evaluated,
            "activity_groups": summarize_groups(table, evaluated)}


def collaborative_neighbors(x, binary, shrinkage=20., neighbors=100, block_size=128):
    """Compute exact shrunk item cosine in bounded blocks, retaining top neighbors."""
    count = x.shape[1]
    budget = min(block_size, count) * count * 4 * 5 + count * min(neighbors, count) * 24
    if budget > 256 * 1024 ** 2:
        raise MemoryError(f"CF block/neighbor estimate exceeds 256 MiB: {budget} bytes")
    norms = np.sqrt(np.asarray(x.multiply(x).sum(axis=0)).ravel())
    rr, cc, vv = [], [], []
    for low in range(0, count, block_size):
        high = min(count, low + block_size)
        scores = (x[:, low:high].T @ x).toarray().astype(np.float32, copy=False)
        scores /= np.maximum(1e-12, norms[low:high, None] * norms[None, :])
        common = (binary[:, low:high].T @ binary).toarray().astype(np.float32, copy=False)
        scores *= common / (common + shrinkage)
        for local, item in enumerate(range(low, high)):
            scores[local, item] = 0.
            keep = np.argpartition(scores[local], -min(neighbors, count))[-min(neighbors, count):]
            keep = keep[scores[local, keep] > 0]
            rr.extend([item] * len(keep)); cc.extend(keep.tolist()); vv.extend(scores[local, keep].tolist())
    neighbor = sparse.csr_matrix((np.asarray(vv, dtype=np.float32), (rr, cc)), shape=(count, count))
    return neighbor, {"block_rows": block_size, "estimated_block_and_neighbors_bytes": budget,
                      "preallocation_limit_bytes": 256 * 1024 ** 2, "retained_neighbor_nnz": neighbor.nnz}


def collaborative_pilot(root, users, base, context_name, history_name, target_window, target_file):
    """Uncapped weighted cosine CF and independent channel/time schedule discovery."""
    from continuous_watch import WINDOW_BY_NAME
    started = time.perf_counter()
    contexts, global_ = load_context(root, context_name, set(users))
    future = opportunities([root / "data/raw" / target_file], target_window)
    available = frozenset(future)
    items = sorted(available)
    channels = sorted({chan for entry in future.values() for chan in entry["channels"]}, key=int)
    ci = {chan: i for i, chan in enumerate(channels)}
    channel_matrix = np.zeros((len(items), len(channels)), dtype=np.float32)
    hour_matrix = np.zeros((len(items), 24), dtype=np.float32)
    airing_log = np.zeros(len(items), dtype=np.float32)
    for row, item in enumerate(items):
        entry = future[item]
        for chan, count in entry["channels"].items():
            channel_matrix[row, ci[chan]] = count / entry["n"]
        hour_matrix[row] = np.asarray(entry["hours"]) / max(1., sum(entry["hours"]))
        airing_log[row] = np.log1p(entry["n"])
    source_pools = {"channel": {}, "channel_time": {}, "collaborative": {}}
    for user in users:
        context = contexts.get(user)
        if not context:
            for source in source_pools:
                source_pools[source][user] = set()
            continue
        affinity = np.asarray([context["channels"].get(chan, [0.])[0] / max(1., context["seconds"]) for chan in channels], dtype=np.float32)
        hour = np.asarray(context["hours"], dtype=np.float32) / max(1., context["seconds"])
        ch_score = channel_matrix @ affinity
        time_score = hour_matrix @ hour
        for source, score in (("channel", ch_score * airing_log), ("channel_time", np.sqrt(ch_score) * time_score * airing_log)):
            top = sorted(np.flatnonzero(score > 0), key=lambda index: (-float(score[index]), items[index]))[:20]
            source_pools[source][user] = {items[index] for index in top}
    # Estimate is explicit before allocating a bounded item-item matrix.
    path = profile_path(root, history_name)
    user_ids, item_ids, rows, cols, values = {}, {}, [], [], []
    for record in iter_profiles(path, history_name):
        if record["frequency_ge30"] > 0:
            u = user_ids.setdefault(record["user_id"], len(user_ids))
            i = item_ids.setdefault(record["tv_show_id"], len(item_ids))
            rows.append(u); cols.append(i); values.append(np.log1p(record["frequency_ge30"]))
    x = sparse.csr_matrix((np.asarray(values, dtype=np.float32), (rows, cols)), shape=(len(user_ids), len(item_ids)))
    binary = x.copy(); binary.data[:] = 1.
    df = np.asarray(binary.sum(axis=0)).ravel()
    x = x @ sparse.diags(np.log((1 + len(user_ids)) / (1 + df)) + 1.)
    x = sparse.diags(1. / np.maximum(1e-12, np.sqrt(np.asarray(x.multiply(x).sum(axis=1)).ravel()))) @ x
    neighbor, memory_budget = collaborative_neighbors(x.tocsr(), binary)
    inverse = {index: item for item, index in item_ids.items()}
    for user in users:
        if user not in user_ids:
            source_pools["collaborative"][user] = set()
            continue
        scores = (x.getrow(user_ids[user]) @ neighbor).toarray().ravel()
        order = sorted((i for i in np.flatnonzero(scores > 0) if inverse[i] in available), key=lambda i: (-float(scores[i]), inverse[i]))[:20]
        source_pools["collaborative"][user] = {inverse[i] for i in order}
    truth = profile_context(profile_path(root, "validation" if context_name == "primary" else "early_validation"), "validation" if context_name == "primary" else "early_validation", users)["strict_truth"]
    original = {user: {base.rows[i][1] for i in range(start, end)} for user, start, end in base.groups}
    def metrics(pools):
        relevant = sum(len(truth[user]) for user in users)
        found = sum(len(pools[user] & truth[user]) for user in users)
        return {"recall": found / relevant, "recalled_relevant": found, "missed_relevant": relevant - found,
                "oracle_map_at_5": sum(min(5, len(pools[user] & truth[user])) / min(5, len(truth[user])) if truth[user] else 0. for user in users) / len(users)}
    baseline = metrics(original)
    results = {}
    for source, pools in source_pools.items():
        union = {user: original[user] | pools.get(user, set()) for user in users}
        results[source] = {"source_only": metrics(pools), "union_with_v4": metrics(union),
                           "new_relevant_pairs": sum(len((pools.get(user, set()) - original[user]) & truth[user]) for user in users)}
    return {"baseline": baseline, "sources": results, "runtime_seconds": time.perf_counter() - started,
            "collaborative_shape": x.shape, "collaborative_nnz": x.nnz,
            "memory_budget": memory_budget,
            "cf_method": "uncapped log-count/IDF, user normalization, item cosine, pair-support shrinkage 20, top100 neighbors; top20 proposals",
            "gold_items_injected": False, **peak_memory()}


def render_report(report):
    lines = ["# SWEET.TV Advanced Personalized Behavioral Ranking V5", "", f"Generated: {report['generated_at_utc']}", "",
             "## Scope and preserved best", "", "```json", json.dumps(report["verified_best_v4"], indent=2), "```", "",
             "No V5 Kaggle score has been measured. Targets 0.50 and 0.52–0.55 are ambitions, not guarantees.", "",
             "## Exact historical validation", "", "| Experiment | Status | Strict MAP@5, all 1260 |", "|---|---|---:|"]
    for entry in report["experiments"]:
        score = entry.get("evaluation", {}).get("metrics", {}).get("primary_all_users", {}).get("map_at_5")
        lines.append(f"| {entry['name']} | {entry['status']} | {score!r} |")
    lines += ["", "## Selected final strategy", "", f"**{report['selected_strategy']}**", "", report["selection_conclusion"], "",
              "## Behavioral definitions", "",
              "All original source clocks are compared as naive wall-clock timestamps, without timezone/DST correction. "
              "Events before each origin provide channel counts and duration distributions. Same-user/channel overlap union supplies contextual duration, active days, "
              "24-hour/7-day habits, recent 14-day habits, four channel×daypart cells, concentration, and smoothed affinities. "
              "Different simultaneous channels are not attributed as unique attention. Unlabelled airtime contributes context but programme ID 0 is never an item. "
              "Context unions never construct strict labels: >80% relevance remains one original interval divided by full original broadcast duration.", "",
              "Candidate-level channel/time features summarize source-clock future schedule opportunities, not actual exposure. "
              "Affinity uses 600-second hierarchical smoothing and normalizes user activity. Counts/fractions, preferred-channel and preferred-slot opportunities, "
              "conditional channel×daypart matches, and repeat/association interactions augment unchanged V4 raw features.", "",
              "## Chronological examples and weighting", "", "```json", json.dumps(report["feature_generation"], indent=2), "```", "",
              "Rolling targets are disjoint W14–15, W16–17, W18–20, with respective histories W11–13, W11–15, W11–17. "
              "The alternative extra-origin arm uses W14–15 plus the original W16–20 target, also disjoint. They are never appended together in a way that duplicates target periods. "
              "All fitting labels end before primary origin 2020-05-18. Extra users are a fixed hashed sample selected by >=5 historical items before 2020-03-30, not future activity. "
              "Query weights equalize origin mass and user mass, with extra users at half template-user weight. Evaluation always uses the same 1260 template strings and full unfiltered strict ground truth.", "",
              "## Feature contribution and subgroup results", "", "```json",
              json.dumps([{key: value for key, value in entry.items() if key not in ("evaluation", "fit")} for entry in report["experiments"]], indent=2), "```", "",
               "Fixed 269-round comparisons isolate initial feature/data effects from stopping changes. Earlier causal diagnostic models fit only rolling origins whose labels end by 2020-04-27. "
              "Saved V4 models are not mislabeled as earlier controls because their fitting labels include that earlier target. Primary validation has already been used for selection; none of these tuned scores is an unbiased Kaggle estimate. "
               "Feature gain and ablations are conditional model diagnostics, not causal proof.", "",
               "## Earlier chronological diagnostic", "", "```json", json.dumps(report["earlier_diagnostics"], indent=2), "```", "",
              "## Independent candidate sources", "", "```json", json.dumps(report["candidate_pilots"], indent=2), "```", "",
              "Pilots estimate incremental recall and oracle before any expensive wider reranker. Existing candidate universes stay fixed for feature comparisons. "
              "Optional enhanced-pool reranking may be deferred to preserve the deadline buffer; candidate-only gains are never reported as accuracy gains.", "",
              "## All fitted model specifications and importances", "", "```json",
              json.dumps([entry.get("fit") for entry in report["experiments"] if entry.get("fit")], indent=2), "```", "",
              "## Final submission and determinism", "", "```json", json.dumps(report["final_build"], indent=2), "```", "",
              "## Execution and versions", "", "```json", json.dumps(report["execution"], indent=2), "```", "",
              "Each fit is isolated, seeded, limited to two threads, checkpointed, and timeout bounded. Measurements refer to their explicit stage/worker scope, not an invented combined peak.", "",
              "## Original artifact integrity", "", "| File | SHA-256 | Unchanged |", "|---|---|---|"]
    for name, data in report["input_integrity"].items():
        lines.append(f"| {name} | `{data['before']['sha256']}` | {data['unchanged']} |")
    lines += ["", "## Tests", "", "```text", report["tests"]["output"], "```", "",
              "## Limitations", "",
              "The strict local target differs from unknown hidden Kaggle preference construction. Candidate misses, unattended viewing, missing schedule channels, "
              "source clock uncertainty, activity/horizon shift, and historical-selection optimism remain. Extra data, CF, and more catalogue coverage are not assumed to help. "
              "All unsuccessful experiments and any deadline skips are recorded. Original V1–V4 submissions and original AI export remain unchanged. "
              "No automatic Kaggle upload was performed.", ""]
    return "\n".join(lines)


def full_fusion_rank_maps(maps, weights, users, fallback, cold):
    result = {}
    for user in users:
        scores = {}
        for source, weight in zip(maps, weights):
            for item, rank in source[user].items():
                scores[item] = scores.get(item, 0.) + weight / (20. + rank)
        order = sorted(scores, key=lambda item: (-scores[item], item))
        if user in cold:
            order = list(fallback) + [item for item in order if item not in fallback]
        result[user] = {item: rank for rank, item in enumerate(order[:50], 1)}
    return result


def v4_components(root, fold, users):
    base = v3.load_table(root, fold)
    expanded = v4.load_expanded(root, fold)
    repeat_scores = v3.scores_for(base, v3.MODEL_BY_NAME["repeat_control"])
    logistic_name = "rerank_v3_model_final.json" if fold == "final" else "rerank_v3_model_logistic_l2_01.json"
    logistic = json.loads((root / "reports" / logistic_name).read_text())
    logistic_scores = v3.scores_for(base, v3.MODEL_BY_NAME["logistic_l2_01"], logistic)
    native_name = "lightgbm_v4_model_final.txt" if fold == "final" else "lightgbm_v4_model_binary_C_expanded.txt"
    native = lgb.Booster(model_file=str(root / "reports" / native_name))
    native_scores = v4.predict_model(native, expanded, list(range(len(v3.FEATURE_NAMES))))
    maps = [v4.user_rank_maps(base, repeat_scores, users), v4.user_rank_maps(base, logistic_scores, users),
            v4.user_rank_maps(expanded, native_scores, users)]
    cold = v4.cold_users(base)
    predictions = v4.fuse_ranks(maps, (.1, .4, .5), users, frozenset(base.metadata["available_ids"]), base.fallback, cold)
    if fold == "primary":
        original = read_predictions(root / "outputs/lightgbm_v4_ensemble_10_40_50_validation.csv", users, frozenset(base.metadata["available_ids"]))
        if predictions != original:
            raise AssertionError("V4 ensemble control replay changed.")
    else:
        original = read_predictions(root / "outputs/submission_v4.csv", users, frozenset(base.metadata["available_ids"]))
        if predictions != original:
            raise AssertionError("V4 final strategy replay changed.")
    return predictions, full_fusion_rank_maps(maps, (.1, .4, .5), users, base.fallback, cold), maps[:2]


def normalized_rank_blend(rank_maps, weights, users, available, fallback, cold):
    result = {}
    for user in users:
        if user in cold:
            result[user] = fallback
            continue
        scores = {}
        for source, weight in zip(rank_maps, weights):
            for item, rank in source[user].items():
                if item not in available:
                    raise ValueError("Unavailable normalized blend candidate.")
                scores[item] = scores.get(item, 0.) + weight * max(0., 1 - (rank - 1) / 50.)
        chosen = sorted(scores, key=lambda item: (-scores[item], item))[:5]
        chosen += [item for item in fallback if item not in chosen][:5 - len(chosen)]
        result[user] = tuple(chosen)
    return result


def final_predict(root, strategy, destination):
    users = load_template_users(root / "data/raw/submission.csv")
    if strategy["kind"] == "v4_control":
        table = v3.load_table(root, "final")
        predictions = read_predictions(root / "outputs/submission_v4.csv", users, frozenset(table.metadata["available_ids"]))
    else:
        table, x, _ = load_bundle(root, "final")
        native = lgb.Booster(model_file=str(root / strategy["fit"]["specification"]["model_path"]))
        scores = native.predict(np.ascontiguousarray(x[:, strategy["fit"]["columns"]]), num_threads=2)
        if strategy["kind"] == "native":
            predictions = v3.rank_table(table, scores, users)
        else:
            _, v4_maps, old_maps = v4_components(root, "final", users)
            new_maps = v4.user_rank_maps(table, scores, users)
            sources = [*old_maps, v4_maps, new_maps] if strategy["ensemble"] == "four_way_rrf" else [v4_maps, new_maps]
            weights = tuple(strategy["weights"])
            if strategy["ensemble"] == "normalized_50_50":
                predictions = normalized_rank_blend(sources, weights, users, frozenset(table.metadata["available_ids"]), table.fallback, v4.cold_users(table))
            else:
                predictions = v4.fuse_ranks(sources, weights, users, frozenset(table.metadata["available_ids"]), table.fallback, v4.cold_users(table))
    if destination.exists():
        raise FileExistsError("Preserving a prior V5 final/replay CSV.")
    write_predictions(destination, users, predictions, frozenset(table.metadata["available_ids"]))
    validate_submission(destination, root / "data/raw/submission.csv", frozenset(table.metadata["available_ids"]))


def study(root, users, before, manifest, tests):
    started = time.perf_counter()
    destination = root / "reports/behavioral_v5.json"
    if destination.exists() or (root / "outputs/submission_v5.csv").exists():
        raise FileExistsError("Preserving completed V5 results.")
    table, _, _ = load_bundle(root, "primary")
    truth = strict_truth(root, "primary", users)
    available = frozenset(table.metadata["available_ids"])
    training_users = frozenset(table.metadata["common_training_users"])
    experiments = []
    def add_entry(entry):
        experiments.append(entry)
        with (root / "reports/behavioral_v5_experiments.jsonl").open("a") as handle:
            handle.write(json.dumps(entry) + "\n")
        if entry.get("evaluation"):
            print(f"{entry['name']}: strict primary MAP@5={entry['evaluation']['metrics']['primary_all_users']['map_at_5']!r}", flush=True)
        else:
            print(f"{entry.get('name')}: {entry.get('status')}", flush=True)
    for name, path in (("Popularity", "popularity_validation.csv"), ("Content_based", "cbf_validation.csv"),
                       ("Association_rules", "association_validation.csv"),
                       ("V2_repeat", "repeat_v2_primary_A_frequency_ge30.csv"), ("V3_logistic", "rerank_v3_logistic_l2_01_validation.csv"),
                       ("V4_ensemble_control", "lightgbm_v4_ensemble_10_40_50_validation.csv"), ("V4_binary_C_control", "lightgbm_v4_binary_C_expanded_validation.csv")):
        predictions = read_predictions(root / "outputs" / path, users, available)
        evaluation = evaluate_recommendations(users, truth, predictions, training_users, available)
        add_entry({"name": name, "kind": "reference", "status": "complete", "evaluation": evaluation,
                   "activity_groups": summarize_groups(table, evaluation)})
    baseline_score = experiments[-1]["evaluation"]["metrics"]["primary_all_users"]["map_at_5"]
    v4_score = experiments[-2]["evaluation"]["metrics"]["primary_all_users"]["map_at_5"]
    # All comparisons use the same frozen V4 expanded candidate pool.
    initial = [("A_base_replay", "base", ["early"], True),
               ("B_channel", "channel", ["early"], True), ("C_time", "time", ["early"], True),
               ("D_both", "both", ["early"], True), ("E_add_origin", "base", ["roll13", "early"], True),
               ("E_disjoint_origins", "base", ["roll13", "roll15", "roll17"], True),
               ("E_extra_users", "base", ["roll13", "roll15", "roll17"], False)]
    for name, features, origins, templates_only in initial:
        add_entry(evaluate_fit(root, run_fit(root, name, features, origins, templates_only=templates_only), users))
    fitted = [entry for entry in experiments if entry.get("fit") and entry["status"] == "complete"]
    if not fitted:
        best = None
    else:
        best = max(fitted, key=lambda entry: entry["evaluation"]["metrics"]["primary_all_users"]["map_at_5"])
        benefit = best["evaluation"]["metrics"]["primary_all_users"]["map_at_5"] - baseline_score
        if benefit >= .003 and optimization_allowed():
            data_entries = [entry for entry in fitted if entry["name"].startswith("E_")]
            data_best = max(data_entries, key=lambda entry: entry["evaluation"]["metrics"]["primary_all_users"]["map_at_5"])
            spec = data_best["fit"]["specification"]
            add_entry(evaluate_fit(root, run_fit(root, "F_combined", "both", spec["fit_origins"], templates_only=spec["templates_only"]), users))
            fitted = [entry for entry in experiments if entry.get("fit") and entry["status"] == "complete"]
            best = max(fitted, key=lambda entry: entry["evaluation"]["metrics"]["primary_all_users"]["map_at_5"])
        else:
            add_entry({"name": "F_combined", "status": "skipped", "reason": "prespecified >=0.003 native gain gate or deadline not satisfied"})
    earlier = {}
    if best and optimization_allowed():
        for name, features in (("earlier_base", "base"), ("earlier_best_features", best["fit"]["specification"]["features"])):
            fit = run_fit(root, name, features, ["roll13", "roll15"], target="roll17", templates_only=True)
            earlier[name] = evaluate_fit(root, fit, users, "roll17")
        if optimization_allowed():
            spec = best["fit"]["specification"]
            add_entry(evaluate_fit(root, run_fit(root, "rank_enriched", spec["features"], spec["fit_origins"],
                                               templates_only=spec["templates_only"], objective="lambdarank", rounds=100), users))
        # Old feature-group ablations are kept small; B/C/D already measure new blocks.
        for group in ("personal_history", "association", "future_schedule", "global_popularity"):
            if not optimization_allowed():
                break
            spec = best["fit"]["specification"]
            add_entry(evaluate_fit(root, run_fit(root, f"ablate_{group}", f"drop_{group}:{spec['features']}", spec["fit_origins"], templates_only=spec["templates_only"]), users))
    pilots = {}
    if optimization_allowed():
        from continuous_watch import WINDOW_BY_NAME
        for fold, context_name, history_name, target, target_file in (
            ("early", "early", "early_train", WINDOW_BY_NAME["early_validation"], "export_arh_11-20-final.csv"),
            ("primary", "primary", "train", WINDOW_BY_NAME["validation"], "export_arh_21-30-final.csv")):
            if not optimization_allowed():
                break
            path = root / "reports" / f"behavioral_v5_candidate_{fold}.json"
            if path.exists():
                pilots[fold] = json.loads(path.read_text())
            else:
                base, _, _ = load_bundle(root, fold)
                pilots[fold] = collaborative_pilot(root, users, base, context_name, history_name, target, target_file)
                with path.open("x") as handle:
                    json.dump(pilots[fold], handle, indent=2)
    # Do not hide optional extensions not completed inside the fixed time budget.
    pilots["enhanced_reranker_status"] = "candidate-only pilot; no recall gain is claimed as a trained reranker gain; wider-pool reranking deferred to protect deliverable buffer"
    fitted = [entry for entry in experiments if entry.get("fit") and entry["status"] == "complete" and not entry["name"].startswith("ablate_")]
    if fitted:
        best = max(fitted, key=lambda entry: entry["evaluation"]["metrics"]["primary_all_users"]["map_at_5"])
    strategy_candidates = [{"name": "V4_ensemble_control", "score": v4_score, "kind": "v4_control"}]
    if best:
        strategy_candidates.append({"name": best["name"], "score": best["evaluation"]["metrics"]["primary_all_users"]["map_at_5"], "kind": "native", "fit_entry": best})
        spec = best["fit"]["specification"]
        scores = np.load(root / spec["score_path"], allow_pickle=False)
        new_maps = v4.user_rank_maps(table, scores, users)
        old_predictions, old_maps, components = v4_components(root, "primary", users)
        for name, weights in (("rrf_50_50", (.5, .5)), ("rrf_30_70", (.3, .7)), ("normalized_50_50", (.5, .5)), ("four_way_rrf", (.1, .2, .2, .5))):
            if not optimization_allowed():
                break
            sources = [*components, old_maps, new_maps] if name == "four_way_rrf" else [old_maps, new_maps]
            predictions = normalized_rank_blend(sources, weights, users, available, table.fallback, v4.cold_users(table)) if name == "normalized_50_50" else v4.fuse_ranks(sources, weights, users, available, table.fallback, v4.cold_users(table))
            path = root / "outputs" / f"behavioral_v5_{name}.csv"
            if path.exists():
                raise FileExistsError("Preserving a prior ensemble experiment.")
            write_predictions(path, users, predictions, available)
            evaluated = evaluate_recommendations(users, truth, predictions, training_users, available)
            add_entry({"name": name, "kind": "ensemble", "status": "complete", "weights": weights, "evaluation": evaluated,
                       "activity_groups": summarize_groups(table, evaluated)})
            strategy_candidates.append({"name": name, "kind": "ensemble", "ensemble": name, "weights": weights,
                                        "score": evaluated["metrics"]["primary_all_users"]["map_at_5"], "fit_entry": best})
    chosen = max(strategy_candidates, key=lambda row: row["score"])
    # Earlier comparison is a causal, newly trained baseline, not an invalid reuse of
    # the saved V4 model (which had seen weeks 18-20 labels during its own fitting).
    earlier_gain = None
    if all(entry.get("evaluation") for entry in earlier.values()) and len(earlier) == 2:
        earlier_gain = earlier["earlier_best_features"]["evaluation"]["metrics"]["primary_all_users"]["map_at_5"] - earlier["earlier_base"]["evaluation"]["metrics"]["primary_all_users"]["map_at_5"]
    if earlier_gain is not None and earlier_gain < -.005:
        chosen = strategy_candidates[0]
        rejection = "V5 earlier diagnostic degraded by more than prespecified 0.005; retain V4 despite primary selection gains."
    else:
        rejection = None
    strategy = {"name": chosen["name"], "kind": chosen["kind"]}
    final_fit = None
    if chosen["kind"] != "v4_control":
        selected_spec = chosen["fit_entry"]["fit"]["specification"]
        origins = list(selected_spec["fit_origins"]) + ["primary"]
        final_fit = run_fit(root, "final", selected_spec["features"], origins, target="final",
                            templates_only=selected_spec["templates_only"], objective=selected_spec["objective"], rounds=selected_spec["rounds"], final=True)
        if final_fit.get("status", "complete") != "complete":
            raise RuntimeError(f"Final selected refit failed; preserve all experiments: {final_fit}")
        strategy["fit"] = final_fit
        if chosen["kind"] == "ensemble":
            strategy.update({"ensemble": chosen["ensemble"], "weights": chosen["weights"]})
    strategy_path = root / "reports/behavioral_v5_final_strategy.json"
    if strategy_path.exists():
        raise FileExistsError("Preserving an existing final V5 strategy.")
    strategy_path.write_text(json.dumps(strategy, indent=2) + "\n")
    with tempfile.TemporaryDirectory(prefix="v5-final-replay-", dir=root / "data/processed") as temporary:
        paths = [Path(temporary) / "first.csv", Path(temporary) / "second.csv"]
        for path, seed in zip(paths, ("0", "314159")):
            environment = os.environ.copy(); environment["PYTHONHASHSEED"] = seed
            completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--final-replay", str(strategy_path), "--output", str(path)],
                                       cwd=root, env=environment, capture_output=True, text=True, timeout=180)
            if completed.returncode:
                raise RuntimeError(f"Final V5 replay failed: {completed.stderr}")
        if not filecmp.cmp(*paths, shallow=False):
            raise AssertionError("V5 final output is not deterministic.")
        publish_submission(paths[0], root / "outputs/submission_v5.csv")
    final_table, _, _ = load_bundle(root, "final")
    final_validation = validate_submission(root / "outputs/submission_v5.csv", root / "data/raw/submission.csv", frozenset(final_table.metadata["available_ids"]))
    integrity = preserve(root, before)
    report = {"schema_version": 5, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "verified_best_v4": json.loads((root / "reports/kaggle_best_v4.json").read_text()),
              "experiments": experiments, "earlier_diagnostics": earlier, "earlier_feature_gain": earlier_gain,
              "selected_strategy": chosen["name"], "selection_conclusion": rejection or f"Selected primary MAP@5={chosen['score']!r}; V4={v4_score!r}; difference={chosen['score'] - v4_score!r}. Earlier component gain={earlier_gain!r}. No hidden-test performance claim.",
               "feature_generation": manifest, "candidate_pilots": pilots,
               "evidence_checkpoints": {stage: json.loads((root / "data/processed" / f"behavioral_v5_{stage}_checkpoint.json").read_text())
                                        for stage in ("raw", "context", "profile")},
              "feature_names": NAMES, "final_build": {"strategy": strategy, "validation": final_validation,
                  "submission": {"path": "outputs/submission_v5.csv", **file_fingerprint(root / "outputs/submission_v5.csv")},
                  "independent_hash_seed_replay": True, "no_test_period_viewing_used": True, "kaggle_score": None,
                  "justified_for_manual_submission": chosen["score"] > v4_score and not rejection},
              "input_integrity": integrity, "tests": tests,
              "execution": {"python_version": sys.version, "numpy": np.__version__, "scipy": scipy.__version__, "lightgbm": lgb.__version__,
                  "seed": 42, "threads": 2, "runtime_seconds_before_report_write": time.perf_counter() - started, **peak_memory(),
                  "source_sha256": {name: file_fingerprint(root / name)["sha256"] for name in ("src/behavioral_v5.py", "src/behavioral_evidence_v5.py", "tests/test_behavioral_v5.py")},
                  "optimization_freeze_utc": FREEZE.isoformat()}}
    # Persist measurements first so a human-report formatting error cannot lose them.
    with destination.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    with (root / "reports/behavioral_v5.md").open("x") as handle:
        handle.write(render_report(report))
    print(json.dumps({"selected": chosen["name"], "map_at_5": chosen["score"], "v4_map_at_5": v4_score,
                      "earlier_gain": earlier_gain, "final_validation": final_validation,
                      "submission": report["final_build"]["submission"], "execution": report["execution"]}, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("features", "study", "all"), default="all")
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--final-replay", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.worker:
        fit_worker(root, args.worker)
        return
    if args.final_replay:
        final_predict(root, json.loads(args.final_replay.read_text()), args.output)
        return
    users = load_template_users(root / "data/raw/submission.csv")
    snapshot_path = root / "data/processed/behavioral_v5_inputs.json"
    before = json.loads(snapshot_path.read_text())
    preserve(root, before)
    completed = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=root, capture_output=True, text=True, timeout=120)
    output = completed.stdout + completed.stderr
    print(output, end="", flush=True)
    if completed.returncode:
        raise RuntimeError("Tests failed; no V5 experiments started.")
    tests = {"returncode": completed.returncode, "output": output}
    manifest = prepare_features(root, users)
    manifest = {**manifest, "channel_key_normalized_bundles": normalize_channel_bundles(root)}
    if args.stage != "features":
        study(root, users, before, manifest, tests)


if __name__ == "__main__":
    main()
