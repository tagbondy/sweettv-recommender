"""Build strict continuous-watch evidence, evaluate repeat models, and create v2.

Run: .venv/bin/python src/personalized_repeat.py
Stages can be resumed with --stage evidence or --stage study. Existing original
experiments and submissions are never overwritten. A completed evidence cache is
reused only after its checksums and reconstruction source fingerprint agree.
"""

from __future__ import annotations

import argparse
import filecmp
import hashlib
import itertools
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from audit_data import peak_memory
from build_interactions import checked_records, file_fingerprint
from continuous_watch import (EVIDENCE_WINDOWS, PROFILE_COLUMNS, WINDOW_BY_NAME, add_profile_scan_index,
                              export_profiles, reconstruct_evidence, reconstruction_algorithm_fingerprint,
                              recover_evidence_statistics)
from create_final_submission import TEST_WINDOW, project_snapshot, publish_submission, validate_submission
from evaluate import evaluate_recommendations, load_candidates, load_template_users, read_predictions, validate_prediction, write_predictions
from popularity import rank_candidates


DATABASE = "continuous_watch_v2.sqlite3"
MANIFEST = "continuous_watch_v2_manifest.json"
OWN_SOURCE_FILES = frozenset({"src/continuous_watch.py", "src/personalized_repeat.py", "tests/test_personalized_repeat.py",
                              "docs/methodology/repeat_viewing_v2.md"})


@dataclass(frozen=True)
class RepeatConfig:
    name: str
    score_column: str
    last_column: str
    qualification: str
    description: str


CONFIGURATIONS = (
    RepeatConfig("A_frequency_ge30", "frequency_ge30", "last_ge30_seconds", ">=30%", "qualified broadcast occurrence frequency"),
    RepeatConfig("B_frequency_gt50", "frequency_gt50", "last_gt50_seconds", ">50%", "qualified broadcast occurrence frequency"),
    RepeatConfig("C_frequency_gt80", "frequency_gt80", "last_gt80_seconds", ">80%", "qualified broadcast occurrence frequency"),
    RepeatConfig("D_recency_gt50_half14", "recency_gt50_half14", "last_gt50_seconds", ">50%", "sum of occurrence weights with a 14-day half-life"),
    RepeatConfig("D_recency_gt50_half28", "recency_gt50_half28", "last_gt50_seconds", ">50%", "sum of occurrence weights with a 28-day half-life"),
    RepeatConfig("E_qualified_seconds_gt50", "qualified_seconds_gt50", "last_gt50_seconds", ">50%", "sum of maximum qualifying continuous overlap seconds per occurrence"),
)
CONFIG_BY_NAME = {config.name: config for config in CONFIGURATIONS}
FOLDS = {
    "primary": ("train", "validation", "export_arh_21-30-final.csv"),
    "earlier": ("early_train", "early_validation", "export_arh_11-20-final.csv"),
}


def profile_path(root: Path, window_name: str) -> Path:
    return root / "data/processed" / f"watch_v2_{window_name}_programmes.csv"


def original_snapshot(root: Path) -> dict:
    snapshot = project_snapshot(root)
    return {name: details for name, details in snapshot.items()
            if name not in OWN_SOURCE_FILES
            and not name.startswith("data/processed/watch_v2_")
            and name not in {f"data/processed/{DATABASE}", f"data/processed/{MANIFEST}",
                             "data/processed/continuous_watch_v2_checkpoint.json", "outputs/submission_v2.csv"}
            and not name.startswith("outputs/repeat_v2_")
            and name not in {"reports/repeat_viewing_v2.md", "reports/repeat_viewing_v2.json"}}


def check_preserved(root: Path, before: dict) -> dict:
    result = {}
    for name, expected in before.items():
        actual = file_fingerprint(root / name)
        if actual != expected:
            raise RuntimeError(f"Protected original artifact changed: {name}")
        result[name] = {"before": expected, "after": actual, "unchanged": True}
    return result


def iter_profiles(path: Path, window_name: str):
    window = WINDOW_BY_NAME[window_name]
    previous = None
    for number, row in checked_records(path, PROFILE_COLUMNS):
        user, programme = row["user_id"], int(row["tv_show_id"])
        values = {key: float(row[key]) if key.startswith("recency_") else int(row[key]) for key in PROFILE_COLUMNS[2:]}
        key = (user, programme)
        if not user or programme <= 0 or (previous is not None and key <= previous):
            raise ValueError(f"Invalid profile key at {path.name}:{number}")
        if not values["frequency_ge30"] >= values["frequency_gt50"] >= values["frequency_gt80"] >= 0:
            raise ValueError("Watch qualifications are not nested.")
        for frequency, last in (("frequency_ge30", "last_ge30_seconds"), ("frequency_gt50", "last_gt50_seconds"), ("frequency_gt80", "last_gt80_seconds")):
            if (values[frequency] == 0 and values[last] != 0) or (values[frequency] > 0 and not window.start < values[last] <= window.end):
                raise ValueError("Qualified recency timestamp is missing or outside the historical window.")
        if (values["has_120_second_evidence"] not in (0, 1)
                or values["qualified_seconds_gt50"] < values["frequency_gt50"]
                or any(not math.isfinite(value) or value < 0 for value in values.values())):
            raise ValueError("Invalid repeat-viewing profile values.")
        previous = key
        yield {"user_id": user, "tv_show_id": programme, **values}


def profile_context(path: Path, window_name: str, users: list[str]) -> dict:
    truth = {user: set() for user in users}
    broad_truth = {user: set() for user in users}
    support, broad_users = Counter(), set()
    rows = 0
    for row in iter_profiles(path, window_name):
        rows += 1
        user, item = row["user_id"], row["tv_show_id"]
        if row["has_120_second_evidence"]:
            support[item] += 1
            broad_users.add(user)
            if user in broad_truth:
                broad_truth[user].add(item)
        if row["frequency_gt80"] > 0 and user in truth:
            truth[user].add(item)
    return {"strict_truth": truth, "broad_truth": broad_truth, "popularity": support,
            "common_training_users": frozenset(broad_users), "profile_rows": rows}


def recommend_repeat(history: list[dict], config: RepeatConfig, candidates: frozenset[int], fallback: tuple[int, ...]) -> tuple[tuple[int, ...], dict]:
    validate_prediction(fallback, candidates)
    personal = [row for row in history if row["tv_show_id"] in candidates and row[config.score_column] > 0]
    ordered = sorted(personal, key=lambda row: (-row[config.score_column], -row[config.last_column], row["tv_show_id"]))
    chosen = [row["tv_show_id"] for row in ordered[:5]]
    personal_slots = len(chosen)
    for item in fallback:
        if len(chosen) == 5:
            break
        if item not in chosen:
            chosen.append(item)
    prediction = tuple(chosen)
    validate_prediction(prediction, candidates)
    return prediction, {"positive_personal_candidates": len(personal), "personal_slots": personal_slots,
                        "fallback_slots": 5 - personal_slots}


def predict_configurations(path: Path, window_name: str, users: list[str], candidates: frozenset[int],
                           fallback: tuple[int, ...], configurations=CONFIGURATIONS) -> tuple[dict, dict]:
    predictions = {config.name: {user: fallback for user in users} for config in configurations}
    diagnostics = {config.name: {user: {"positive_personal_candidates": 0, "personal_slots": 0, "fallback_slots": 5}
                                for user in users} for config in configurations}
    cohort = set(users)
    for user, group in itertools.groupby(iter_profiles(path, window_name), key=lambda row: row["user_id"]):
        if user not in cohort:
            for _ in group:
                pass
            continue
        history = list(group)
        for config in configurations:
            predictions[config.name][user], diagnostics[config.name][user] = recommend_repeat(history, config, candidates, fallback)
    summaries = {}
    for config in configurations:
        details = diagnostics[config.name]
        summaries[config.name] = {
            "users_with_five_personal_recommendations": sum(d["personal_slots"] == 5 for d in details.values()),
            "users_requiring_any_global_fallback": sum(d["fallback_slots"] > 0 for d in details.values()),
            "users_with_no_available_personal_candidates": sum(d["personal_slots"] == 0 for d in details.values()),
            "total_global_fallback_slots": sum(d["fallback_slots"] for d in details.values()),
            "per_user": details,
        }
    return predictions, summaries


def ensure_evidence(root: Path, protected: dict) -> dict:
    processed = root / "data/processed"
    database, manifest_path = processed / DATABASE, processed / MANIFEST
    reconstruction_source = file_fingerprint(root / "src/continuous_watch.py")["sha256"]
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if reconstruction_source != manifest.get("profile_export_source_sha256", manifest["reconstruction_source_sha256"]):
            raise RuntimeError("Evidence reconstruction code changed; preserve this cache and use a new version.")
        for name, fingerprint in manifest["artifacts"].items():
            if file_fingerprint(root / name)["sha256"] != fingerprint["sha256"]:
                raise RuntimeError(f"Continuous-watch evidence artifact changed: {name}")
        check_preserved(root, manifest["protected_original_files"])
        print("Reusing the checksum-verified continuous-watch evidence cache.", flush=True)
        return manifest
    started = time.perf_counter()
    recovery_path = processed / "continuous_watch_v2_interrupted_run.json"
    checkpoint_path = processed / "continuous_watch_v2_checkpoint.json"
    recovery = None
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text())
        if file_fingerprint(database)["sha256"] != checkpoint["database_sha256"]:
            raise RuntimeError("The completed evidence checkpoint's database changed.")
        evidence = checkpoint["evidence"]
        recovery = checkpoint.get("recovery")
    elif database.exists():
        if not recovery_path.exists():
            raise FileExistsError("Unfamiliar evidence database exists without a verified completion/recovery record.")
        recovery = json.loads(recovery_path.read_text())
        if (file_fingerprint(database)["sha256"] != recovery["completed_database_before_index"]["sha256"]
                or reconstruction_algorithm_fingerprint() != recovery["reconstruction_algorithm_ast_sha256"]):
            raise RuntimeError("Interrupted evidence or its raw reconstruction logic changed.")
        print("Recovering the completed occurrence database; raw reconstruction logic fingerprint agrees.", flush=True)
        evidence = recover_evidence_statistics(root / "data/raw", database)
    else:
        if any(profile_path(root, window.name).exists() for window in EVIDENCE_WINDOWS):
            raise FileExistsError("Unfamiliar profile artifacts exist without their evidence database.")
        evidence = reconstruct_evidence(root / "data/raw", database)
    if not checkpoint_path.exists():
        evidence["covering_profile_index"] = add_profile_scan_index(database)
        checkpoint = {"evidence": evidence, "recovery": recovery, "database_sha256": file_fingerprint(database)["sha256"], "profiles": {}}
        checkpoint_path.write_text(json.dumps(checkpoint, indent=2) + "\n", encoding="utf-8")
    if recovery and not checkpoint["profiles"]:
        old_profiles = [profile_path(root, window.name) for window in EVIDENCE_WINDOWS if profile_path(root, window.name).exists()]
        if old_profiles:
            archive = processed / "watch_v2_interrupted_exports"
            archive.mkdir(exist_ok=True)
            attempt = Path(tempfile.mkdtemp(prefix="attempt_", dir=archive))
            for path in old_profiles:
                path.rename(attempt / path.name)
            checkpoint["archived_interrupted_exports"] = str(attempt.relative_to(root))
            checkpoint_path.write_text(json.dumps(checkpoint, indent=2) + "\n", encoding="utf-8")
    profiles = {}
    for window in EVIDENCE_WINDOWS:
        cached = checkpoint["profiles"].get(window.name)
        if cached:
            if file_fingerprint(profile_path(root, window.name))["sha256"] != cached["sha256"]:
                raise RuntimeError("A completed profile checkpoint changed.")
            profiles[window.name] = cached["statistics"]
        else:
            profiles[window.name] = export_profiles(database, window, profile_path(root, window.name))
            checkpoint["profiles"][window.name] = {"statistics": profiles[window.name], "sha256": file_fingerprint(profile_path(root, window.name))["sha256"]}
            checkpoint_path.write_text(json.dumps(checkpoint, indent=2) + "\n", encoding="utf-8")
        print(f"Exported {window.name} profile: {profiles[window.name]['rows']:,} user-programme rows.", flush=True)
    check_preserved(root, protected)
    paths = [database, *(profile_path(root, window.name) for window in EVIDENCE_WINDOWS)]
    manifest = {"schema_version": 2, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "reconstruction_source_sha256": recovery["reconstruction_source_sha256"] if recovery else reconstruction_source,
                "profile_export_source_sha256": reconstruction_source, "interrupted_attempt": recovery,
                "reconstruction_algorithm_ast_sha256": reconstruction_algorithm_fingerprint(), "evidence": evidence, "profiles": profiles,
                "artifacts": {str(path.relative_to(root)): file_fingerprint(path) for path in paths},
                "protected_original_files": protected,
                "execution": {"runtime_seconds_before_manifest_write": time.perf_counter() - started, **peak_memory()}}
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return manifest


def select_robust_model(experiments: list[dict]) -> dict:
    def key(entry):
        scores = [entry["folds"][fold]["strict_evaluation"]["metrics"]["primary_all_users"]["map_at_5"] for fold in FOLDS]
        return min(scores), statistics.mean(scores), scores[0]
    return max(experiments, key=key)


def final_worker(root: Path, config_name: str, destination: Path) -> None:
    config = CONFIG_BY_NAME[config_name]
    users = load_template_users(root / "data/raw/submission.csv")
    context = profile_context(profile_path(root, "full"), "full", users)
    candidates, _ = load_candidates(root / "data/raw/export_arh_31-42-final.csv", TEST_WINDOW)
    fallback = rank_candidates(context["popularity"], candidates)
    predictions, _ = predict_configurations(profile_path(root, "full"), "full", users, candidates, fallback, (config,))
    if destination.exists():
        raise FileExistsError("Preserving an existing worker prediction file.")
    write_predictions(destination, users, predictions[config.name], candidates)
    validate_submission(destination, root / "data/raw/submission.csv", candidates)


def render_report(report: dict) -> str:
    lines = ["# SWEET.TV Personalized Repeat-Viewing v2", "", f"Generated (UTC): {report['generated_at_utc']}", "",
             "## Result and scope", "", f"Selected robust repeat model: **{report['selected_configuration']}**.", "",
             "The original 120-second experiments and original submission are preserved. This is a separately versioned strict-relevance study. "
             "No Kaggle upload was performed, and no new Public/Private score or target achievement is claimed.", "",
             "## Reference and reported Kaggle context", "", "```json", json.dumps(report["reference_and_kaggle_context"], indent=2), "```", "",
             "Our own implementation follows the repeat-viewing idea; it does not copy article code or implement its two-layer architecture. "
             "Full-text access was blocked (HTTP 403); source benchmark details were supplied by the user. See docs/methodology/repeat_viewing_v2.md.", "",
             "## Continuous-watch reconstruction", "",
             "- Match original viewing intervals and broadcasts by channel and positive interval overlap. The denominator is the original schedule.duration, never viewing-log duraton or a split-clipped programme length.",
             "- Keep each user/channel/programme/original-airing occurrence. Qualify it only if one individual viewing interval reaches the threshold. Never sum or merge disconnected sessions.",
             "- Preserve maxima and their source viewing-row IDs independently for each chronological window, plus the latest endpoint of individually qualifying sessions at each threshold.",
             "- >=30%, >50%, and >80% are checked with integer arithmetic. Exactly 50% and exactly 80% fail their respective strict tests.",
             "- Rejoin raw logs: the old event CSVs omit sub-120-second overlaps, which can still qualify for short broadcasts. No clock correction is applied.", "",
             "Evidence-build recovery: an earlier invocation hit its 900-second tool limit after completing the raw occurrence database and while exporting profiles. "
             "The completed database was checksummed, its completion marker verified, and the raw reconstruction algorithm's AST fingerprint checked unchanged. "
             "Interrupted CSV exports were archived before indexed re-export. Lost in-memory counters and detailed original timers are explicitly unavailable, not invented; "
             "the evidence-cache runtime records the successful build/recovery invocation separately from that interrupted attempt.", "",
             "```json", json.dumps(report["evidence_manifest"]["evidence"], indent=2), "```", "",
             "## New evaluation protocol", "", "```json", json.dumps(report["strict_protocol"], indent=2), "```", "",
             "Every configuration uses the same all-1,260-user cohort, future availability, and common 120-second distinct-viewer fallback within a split. "
             "Warm/cold metric groups retain that common history definition; per-model personal/fallback coverage is reported separately. "
             "Labels are unique programmes with an individual continuous overlap >80%; no unavailable relevance is removed.", "",
             "## Previous methods under old and new relevance", "",
             "| Method (unchanged saved predictions) | Original broad MAP@5 | New strict >80% MAP@5 | New nonempty-label MAP@5 |",
             "|---|---:|---:|---:|"]
    for name, comparison in report["previous_method_comparison"].items():
        metrics = comparison["strict_evaluation"]["metrics"]
        lines.append(f"| {name} | {comparison['original_broad_map_at_5']!r} | {metrics['primary_all_users']['map_at_5']!r} | {metrics['supplementary_nonempty_ground_truth']['map_at_5']!r} |")
    lines += ["", "## All prespecified personalized configurations", "",
              "| Configuration | Primary strict MAP@5 | Earlier strict MAP@5 | Minimum of both | Primary old-broad MAP@5 (not selection criterion) |",
              "|---|---:|---:|---:|---:|"]
    for experiment in report["experiments"]:
        primary = experiment["folds"]["primary"]["strict_evaluation"]["metrics"]["primary_all_users"]["map_at_5"]
        earlier = experiment["folds"]["earlier"]["strict_evaluation"]["metrics"]["primary_all_users"]["map_at_5"]
        broad = experiment["folds"]["primary"]["broad_evaluation"]["metrics"]["primary_all_users"]["map_at_5"]
        lines.append(f"| {experiment['configuration']['name']} | {primary!r} | {earlier!r} | {min(primary, earlier)!r} | {broad!r} |")
    lines += ["", "Selection maximizes the minimum primary/earlier strict MAP@5, then their mean, then primary score, then prespecified order. "
              "All tested configurations and prediction files are preserved, including inferior results. Selection on these hold-outs is not an unbiased third-test estimate.", "",
              "## Parameters, chronological folds, and coverage", "", "```json",
              json.dumps({"configurations": [entry["configuration"] for entry in report["experiments"]],
                          "folds": report["fold_statistics"],
                          "coverage": {entry["configuration"]["name"]: {
                              name: {"metrics": fold["strict_evaluation"]["metrics"],
                                     "personalization": {key: value for key, value in fold["personalization"].items() if key != "per_user"}}
                              for name, fold in entry["folds"].items()} for entry in report["experiments"]}}, indent=2), "```", "",
              "Personal scores depend only on the user's own qualified occurrences. Exact score ties use the user's most recent qualifying watch and then programme ID. "
              "The global fallback never boosts positive personal scores or breaks their ties. Duration weighting sums the maximum qualifying continuous seconds per broadcast; recency half-lives are 14 and 28 days.", "",
              "## Final refit and submission_v2.csv", "", "```json", json.dumps(report["final_build"], indent=2), "```", "",
              "Final history is weeks 11–30 only; test weeks 31–42 schedule supplies availability only. "
              "The v2 file is validated against the original template and compared byte-for-byte with a separate hash-seed prediction process.", "",
              "## Execution and source fingerprints", "", "```json", json.dumps(report["execution"], indent=2), "```", "",
              "The evidence cache records its separate measured build runtime/peak memory. Study runtime includes comparisons, all six configurations on both folds, and final refit/replay. "
              "RSS measurements are process high-water marks, not simultaneous parent-plus-child totals.", "",
              "## Preservation verification", "", "| Original artifact | SHA-256 | Unchanged |", "|---|---|---|"]
    for name, details in report["input_integrity"].items():
        lines.append(f"| {name} | `{details['before']['sha256']}` | {details['unchanged']} |")
    lines += ["", "## Complete test suite", "", "```text", report["tests"]["output"].rstrip(), "```", "",
              "## Limitations", "",
              "- Strict >80% is a user-specified reconstruction of the competition description, not proof of Kaggle's hidden label/ranking implementation. The article benchmark and Kaggle target are not our measured scores.",
              "- Source intervals can contain unattended viewing. A continuous log interval is evidence of tuning, not proof of attention or enjoyment. Unlabeled airtime and missing channels remain misses.",
              "- Even adjacent records are not merged. This conservative single-record criterion can miss a truly uninterrupted watch split into separate source records.",
              "- Full broadcast duration remains the denominator at boundaries; a partially visible occurrence may not meet strict relevance. Different episode-length distributions affect duration-weighted ranking.",
              "- Repeat models cannot personalize programmes never previously qualified for that user. Availability filtering can remove favorites, making common fallback necessary.",
              "- Earlier sensitivity is five weeks of history/five weeks of relevance, whereas primary validation uses ten/ten and the final test spans twelve weeks. Distribution drift and small cold-start cohorts remain uncertain.",
              "- Earlier CBF and AR fits were not previously saved; no earlier scores for those methods are invented. Their unchanged primary predictions are evaluated under the new relevance rule.",
              "- No original experiment, original submission, AI-session export, or source implementation was overwritten. No automatic Kaggle upload was made.", ""]
    return "\n".join(lines)


def run_study(root: Path, manifest: dict, protected: dict, tests: dict) -> dict:
    started = time.perf_counter()
    reports, outputs = root / "reports", root / "outputs"
    destinations = [reports / "repeat_viewing_v2.json", reports / "repeat_viewing_v2.md", outputs / "submission_v2.csv"]
    prediction_paths = {(fold, config.name): outputs / f"repeat_v2_{fold}_{config.name}.csv" for fold in FOLDS for config in CONFIGURATIONS}
    if any(path.exists() for path in [*destinations, *prediction_paths.values()]):
        raise FileExistsError("Preserving existing v2 experiment artifacts; do not overwrite a previous run.")
    users = load_template_users(root / "data/raw/submission.csv")
    baseline = json.loads((reports / "popularity_baseline.json").read_text())
    cbf = json.loads((reports / "cbf_validation.json").read_text())
    association = json.loads((reports / "association_validation.json").read_text())
    originals = {
        "Popularity": ("popularity_validation.csv", baseline["baseline"]["evaluation"]),
        "Content-Based Filtering": ("cbf_validation.csv", cbf["cbf"]["evaluation"]),
        "Association Rules": ("association_validation.csv", next(entry for entry in association["experiments"]
                         if entry["configuration"]["name"] == association["selected_configuration"])["evaluation"]),
    }
    del baseline, cbf, association
    experiments = [{"configuration": asdict(config), "folds": {}} for config in CONFIGURATIONS]
    fold_stats, previous_methods = {}, {}
    for fold_name, (train_name, relevance_name, schedule_name) in FOLDS.items():
        begin = time.perf_counter()
        training = profile_context(profile_path(root, train_name), train_name, users)
        candidates, availability = load_candidates(root / "data/raw" / schedule_name, WINDOW_BY_NAME[relevance_name])
        fallback = rank_candidates(training["popularity"], candidates)
        predictions, diagnostics = predict_configurations(profile_path(root, train_name), train_name, users, candidates, fallback)
        # All six predictions for this fold are fixed before reading its relevance.
        relevance = profile_context(profile_path(root, relevance_name), relevance_name, users)
        if fold_name == "primary":
            if len(training["popularity"]) != 2946 or len(training["common_training_users"]) != 4837:
                raise AssertionError("Reconstructed 120-second training evidence disagrees with preserved statistics.")
            for method, (filename, old_evaluation) in originals.items():
                saved = read_predictions(outputs / filename, users, candidates)
                broad_check = evaluate_recommendations(users, relevance["broad_truth"], saved, training["common_training_users"], candidates)
                if broad_check != old_evaluation:
                    raise AssertionError(f"Raw occurrence reconstruction does not reproduce old broad evaluation: {method}")
                previous_methods[method] = {"original_broad_map_at_5": old_evaluation["metrics"]["primary_all_users"]["map_at_5"],
                    "strict_evaluation": evaluate_recommendations(users, relevance["strict_truth"], saved, training["common_training_users"], candidates)}
            if any(prediction != fallback for prediction in read_predictions(outputs / "popularity_validation.csv", users, candidates).values()):
                raise AssertionError("Common fallback differs from the preserved original ranking.")
        for experiment, config in zip(experiments, CONFIGURATIONS):
            path = prediction_paths[(fold_name, config.name)]
            write_predictions(path, users, predictions[config.name], candidates)
            saved = read_predictions(path, users, candidates)
            if saved != predictions[config.name]:
                raise AssertionError("Saved repeat predictions differ from generated predictions.")
            strict = evaluate_recommendations(users, relevance["strict_truth"], saved, training["common_training_users"], candidates)
            broad = evaluate_recommendations(users, relevance["broad_truth"], saved, training["common_training_users"], candidates)
            experiment["folds"][fold_name] = {"strict_evaluation": strict, "broad_evaluation": broad,
                                            "personalization": diagnostics[config.name],
                                            "prediction_file": {"path": str(path.relative_to(root)), **file_fingerprint(path)}}
            print(f"{fold_name} / {config.name}: strict MAP@5={strict['metrics']['primary_all_users']['map_at_5']!r}", flush=True)
        label_sizes = [len(relevance["strict_truth"][user]) for user in users]
        fold_stats[fold_name] = {"train_window": train_name, "relevance_window": relevance_name,
            "train_start_inclusive": manifest["profiles"][train_name]["start_inclusive"],
            "train_end_exclusive": manifest["profiles"][train_name]["end_exclusive"],
            "relevance_start_inclusive": manifest["profiles"][relevance_name]["start_inclusive"],
            "relevance_end_exclusive": manifest["profiles"][relevance_name]["end_exclusive"],
            "availability": availability, "common_fallback_top_five": list(fallback),
            "strict_relevant_programme_count": {"min": min(label_sizes), "median": statistics.median(label_sizes),
                                               "mean": statistics.mean(label_sizes), "max": max(label_sizes)},
            "users_without_strict_labels": sum(not size for size in label_sizes),
            "runtime_seconds": time.perf_counter() - begin}
    selected = select_robust_model(experiments)
    config = CONFIG_BY_NAME[selected["configuration"]["name"]]
    print(f"Robust selection: {config.name}", flush=True)
    begin = time.perf_counter()
    context = profile_context(profile_path(root, "full"), "full", users)
    candidates, availability = load_candidates(root / "data/raw/export_arh_31-42-final.csv", TEST_WINDOW)
    fallback = rank_candidates(context["popularity"], candidates)
    final_predictions, final_diagnostics = predict_configurations(profile_path(root, "full"), "full", users, candidates, fallback, (config,))
    with tempfile.TemporaryDirectory(prefix="repeat-v2-", dir=root / "data/processed") as temporary:
        first, second = Path(temporary) / "first.csv", Path(temporary) / "second.csv"
        write_predictions(first, users, final_predictions[config.name], candidates)
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = "271828"
        command = [sys.executable, str(Path(__file__).resolve()), "--worker-final", config.name, "--worker-output", str(second)]
        completed = subprocess.run(command, cwd=root, env=environment, capture_output=True, text=True)
        if completed.returncode:
            raise RuntimeError(f"Independent v2 prediction replay failed:\n{completed.stderr}")
        if not filecmp.cmp(first, second, shallow=False):
            raise AssertionError("Repeat model output differs under an independent hash-seed prediction replay.")
        final_validation = validate_submission(first, root / "data/raw/submission.csv", candidates)
        publish_submission(first, outputs / "submission_v2.csv")
    validate_submission(outputs / "submission_v2.csv", root / "data/raw/submission.csv", candidates)
    integrity = check_preserved(root, protected)
    final_build = {"selected_configuration": asdict(config), "history_window": "weeks 11-30 only",
        "final_availability": availability, "submission_validation": final_validation,
        "final_popularity_fallback_top_five": list(fallback),
        "personalization": {key: value for key, value in final_diagnostics[config.name].items() if key != "per_user"},
        "submission_file": {"path": "outputs/submission_v2.csv", **file_fingerprint(outputs / "submission_v2.csv")},
        "original_submission_unchanged": file_fingerprint(outputs / "submission.csv") == protected["outputs/submission.csv"],
        "determinism": {"independent_hash_seed": "271828", "byte_identical_prediction_replay": True,
                        "parent_hash_seed": os.environ.get("PYTHONHASHSEED", "unset")},
        "runtime_seconds": time.perf_counter() - begin}
    reference = {
        "methodological_inspiration": "Nikita Melentev (2021), Two-layered Recommender System Methodology: A Prize-Winning Solution",
        "article_url": "https://towardsdatascience.com/two-layered-recommender-system-methodology-a-prize-winning-solution-c4bdb815b156/",
        "source_access": "title/author located via search; full text returned HTTP 403; no article code used",
        "published_repeat_baseline_map_at_5_user_supplied": 0.41296,
        "published_threshold_user_supplied": "50% single-session qualification; separate from strict >80% relevance",
        "existing_kaggle_result_user_reported_verified": {"public": 0.31901, "private": 0.31614},
        "new_kaggle_submission_performed": False, "new_public_score": None,
        "ambitious_public_target": 0.45, "target_achievement_claimed": False,
    }
    report = {"schema_version": 2, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_and_kaggle_context": reference,
        "strict_protocol": {"cohort": "all 1260 original template users", "relevance": "any individual continuous overlap >80% of original broadcast duration",
            "normalization": "unchanged conventional AP@5 divided by min(5, unique relevant programme count)",
            "empty_relevance": "AP@5=0; retained in primary cohort", "past_programmes_allowed": True,
            "fallback": "common historical distinct-user popularity based on original 120-second evidence; only fills missing personal slots",
            "warm_cold_metric_groups": "common original 120-second history definition, with separate per-model profile coverage",
            "selection": "maximize minimum primary/earlier strict MAP@5, then mean, then primary, then prespecified order"},
        "evidence_manifest": manifest, "previous_method_comparison": previous_methods, "experiments": experiments,
        "fold_statistics": fold_stats, "selected_configuration": config.name, "final_build": final_build,
        "input_integrity": integrity, "tests": tests,
        "execution": {"python_version": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
            "third_party_packages_used": [], "study_runtime_seconds_before_report_serialization": time.perf_counter() - started,
            "evidence_build_execution": manifest["execution"], **peak_memory(),
            "source_sha256": {name: file_fingerprint(root / name)["sha256"] for name in sorted(OWN_SOURCE_FILES)}}}
    with destinations[0].open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    with destinations[1].open("x", encoding="utf-8") as handle:
        handle.write(render_report(report))
    print(json.dumps({"selected_configuration": config.name, "final_build": final_build}, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "evidence", "study"), default="all")
    parser.add_argument("--worker-final", choices=tuple(CONFIG_BY_NAME), help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.worker_final:
        if args.worker_output is None:
            parser.error("Internal final worker requires a fresh output path.")
        final_worker(root, args.worker_final, args.worker_output)
        return
    protected = original_snapshot(root)
    if (root / "outputs/submission.csv").exists():
        old_final = json.loads((root / "reports/final_submission_build.json").read_text())
        if protected["outputs/submission.csv"]["sha256"] != old_final["submission_file"]["sha256"]:
            raise RuntimeError("The original submission changed since its final-build report.")
    started = time.perf_counter()
    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
    completed = subprocess.run(command, cwd=root, capture_output=True, text=True)
    output = completed.stdout + completed.stderr
    print(output, end="", flush=True)
    if completed.returncode:
        raise RuntimeError("Complete tests failed; no real-data reconstruction or study started.")
    test_result = {"command": command, "returncode": completed.returncode, "output": output,
                   "runtime_seconds": time.perf_counter() - started}
    if args.stage == "study" and not (root / "data/processed" / MANIFEST).exists():
        raise FileNotFoundError("Build the continuous-watch evidence cache first.")
    manifest = ensure_evidence(root, protected)
    if args.stage != "evidence":
        run_study(root, manifest, protected, test_result)
    else:
        print(json.dumps({"evidence": manifest["evidence"], "profiles": manifest["profiles"], "execution": manifest["execution"]}, indent=2))
    print(f"Total invocation runtime: {time.perf_counter() - started:.3f} seconds", flush=True)


if __name__ == "__main__":
    main()
