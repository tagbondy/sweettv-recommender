"""Fit the selected final Association Rules model and prepare a manual upload CSV.

Create: .venv/bin/python src/create_final_submission.py
Reproduce without overwriting: .venv/bin/python src/create_final_submission.py --verify-only

Existing validation implementations and artifacts are never changed. Final-window
input adaptation reuses AggregateReader, select_basket, AssociationCounts,
build_rules, generate_predictions, rank_candidates, and the shared CSV utilities.
Two independently fitted subprocesses with different hash seeds must agree before
the submission is published. There is no Kaggle/network integration.
"""

from __future__ import annotations

import argparse
import filecmp
import hashlib
import itertools
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType

from association_rules import (
    MAX_ITEMS_PER_USER, AssociationCounts, BasketItem, RuleConfig,
    build_rules, generate_predictions, select_basket, size_summary,
)
from audit_data import peak_memory
from build_interactions import TEST_END, TEST_START, WINDOWS, Window, file_fingerprint, timestamp
from evaluate import (
    K, PREDICTION_COLUMNS, AggregateReader, checked_records, load_candidates,
    load_template_users, read_predictions, write_predictions,
)
from popularity import PopularityModel, rank_candidates


TEST_WINDOW = Window("test", TEST_START, TEST_END)
FINAL_TRAIN_NAME = "final_train_user_programmes.csv"
TEST_SCHEDULE_NAME = "export_arh_31-42-final.csv"
EXPECTED_CONFIGURATION = RuleConfig("confidence_max", "confidence_max", 5, 0.05, 0.0, 0.0, 100)
IGNORED_DIRECTORIES = frozenset({".venv", ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"})


def selected_configuration(report: dict) -> tuple[RuleConfig, dict]:
    """Read the already selected configuration; never select or tune again."""
    if report["selected_configuration"] != "confidence_max":
        raise ValueError("The approved selected strategy must be confidence_max.")
    selected = [entry for entry in report["experiments"] if entry["configuration"]["name"] == "confidence_max"]
    if len(selected) != 1:
        raise ValueError("The saved selected configuration is missing or ambiguous.")
    parameters = dict(selected[0]["configuration"])
    cap = parameters.pop("max_items_per_user")
    config = RuleConfig(**parameters)
    config.validate()
    if cap != MAX_ITEMS_PER_USER or cap != 50 or asdict(config) != asdict(EXPECTED_CONFIGURATION):
        raise ValueError("Saved hyperparameters differ from the approved final configuration.")
    return config, {
        "selected_configuration": report["selected_configuration"],
        "hyperparameters": {**asdict(config), "max_items_per_user": cap},
        "previous_validation_map_at_5": selected[0]["evaluation"]["metrics"]["primary_all_users"]["map_at_5"],
        "validation_score_role": "historical selection evidence only; not a final/test score",
    }


def load_final_training(path: Path, users: list[str]) -> tuple[dict, PopularityModel, dict]:
    """Adapt only input time bounds; use the verified basket/popularity semantics.

    AggregateReader yields each sorted user-programme key once. Counting those
    keys gives full-history distinct viewers, independently of basket truncation.
    The recommendation algorithm and all rule math remain in existing modules.
    """
    reader = AggregateReader(path, WINDOWS[2])
    baskets, canonical_ids = {}, {}
    popularity = Counter()
    original_sizes = []
    for user, group in itertools.groupby(reader, key=lambda row: row.user_id):
        history = list(group)  # Retain one user at a time, not the complete CSV.
        popularity.update(row.programme_id for row in history)
        original_sizes.append(len(history))
        baskets[user] = tuple(BasketItem(canonical_ids.setdefault(item.programme_id, item.programme_id), item.log_view_count)
                              for item in select_basket(history, MAX_ITEMS_PER_USER))
    summary = reader.summary(users)
    retained_sizes = [len(basket) for basket in baskets.values()]
    cold = sorted(set(users) - set(baskets))
    model = PopularityModel(MappingProxyType(dict(popularity)), frozenset(reader.users), summary)
    return baskets, model, {
        "source": str(path), "aggregate_schema_and_time_bounds": summary,
        "final_training_users": len(reader.users), "final_training_programmes": len(reader.programmes),
        "template_users_with_history": len(set(users) & reader.users),
        "cold_start_template_users": len(cold), "cold_start_user_ids": cold,
        "original_history_sizes": size_summary(original_sizes), "retained_basket_sizes": size_summary(retained_sizes),
        "users_with_truncated_histories": sum(size > MAX_ITEMS_PER_USER for size in original_sizes),
        "retained_user_programme_pairs": sum(retained_sizes),
        "popularity_definition": "distinct users per programme over all final-training aggregate keys, before basket truncation",
        "basket_selection_implementation": "association_rules.select_basket, unchanged",
        "time_adapter_reason": "old load_training_baskets/fit_popularity fix weeks 11-20; the shared reader is explicitly given WINDOWS[2] for final training",
    }


def digest_records(records) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(json.dumps(record, separators=(",", ":"), ensure_ascii=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_final_predictions(training_path: Path, schedule_path: Path, template_path: Path,
                            scratch: Path, config: RuleConfig, expected_users: int = 1260) -> tuple[dict, frozenset[int], dict]:
    """Fit only final-training evidence; use the test schedule for availability."""
    config.validate()
    if asdict(config) != asdict(EXPECTED_CONFIGURATION):
        raise ValueError("Only the approved final configuration may be fitted.")
    users = load_template_users(template_path, expected_users)
    stages = {}
    begin = time.perf_counter()
    baskets, popularity, training_stats = load_final_training(training_path, users)
    stages["read_final_training_and_prepare_baskets_and_popularity"] = time.perf_counter() - begin
    begin = time.perf_counter()
    candidates, availability = load_candidates(schedule_path, TEST_WINDOW)
    fallback = rank_candidates(popularity.scores, candidates)
    stages["test_availability_and_final_popularity_ranking"] = time.perf_counter() - begin
    begin = time.perf_counter()
    counts = AssociationCounts(baskets, scratch / "final_pair_counts.sqlite3", max_items=MAX_ITEMS_PER_USER)
    stages["mine_final_training_pairs"] = time.perf_counter() - begin
    try:
        begin = time.perf_counter()
        rules, rule_stats = build_rules(counts, config)
        stages["build_selected_directional_rules"] = time.perf_counter() - begin
        fingerprints = {
            "full_final_popularity_counts_sha256": digest_records(sorted(popularity.scores.items())),
            "capped_training_baskets_sha256": digest_records(
                [user, [(item.programme_id, item.log_view_count) for item in baskets[user]]] for user in sorted(baskets)),
            "pruned_directional_rules_sha256": digest_records([source, rules[source]] for source in sorted(rules)),
        }
        begin = time.perf_counter()
        predictions, ranking = generate_predictions(users, baskets, rules, candidates, fallback, config)
        stages["rank_template_users"] = time.perf_counter() - begin
        pair_stats = dict(counts.statistics)
    finally:
        counts.close()
    if any(predictions[user] != fallback for user in users if user not in popularity.training_users):
        raise AssertionError("Cold-start users did not receive final-training popularity fallback.")
    recommended = set().union(*(set(prediction) for prediction in predictions.values()))
    result = {
        "method": "Association Rules", "configuration": {**asdict(config), "max_items_per_user": MAX_ITEMS_PER_USER},
        "training": training_stats, "candidate_pool": availability,
        "pair_counts": pair_stats, "rules": rule_stats,
        "final_popularity_fallback": [
            {"rank": rank, "tv_show_id": item, "distinct_final_training_users": popularity.scores.get(item, 0)}
            for rank, item in enumerate(fallback, 1)],
        "candidate_programmes_without_final_training_interactions": len(candidates - set(popularity.scores)),
        "ranking": ranking, "distinct_recommended_programmes": len(recommended),
        "user_recommendation_coverage": len(predictions) / len(users),
        "model_fingerprints": fingerprints, "stage_runtime_seconds": stages,
        "data_roles": {
            "fitting_and_user_histories": str(training_path), "popularity_counts": str(training_path),
            "availability_only": str(schedule_path), "template_only": str(template_path),
            "test_period_viewing_behavior_used": False, "future_descriptive_metadata_features_used": False,
        },
    }
    return predictions, candidates, result


def validate_submission(path: Path, template_path: Path, candidates: frozenset[int], expected_users: int = 1260) -> dict:
    """Check the complete file using shared validation plus canonical space format."""
    users = load_template_users(template_path, expected_users)
    predictions = read_predictions(path, users, candidates)
    rows = 0
    for rows, row in checked_records(path, PREDICTION_COLUMNS):
        canonical = " ".join(map(str, predictions[row["user_id"]]))
        if row["tv_show_id"] != canonical:
            raise ValueError("Programme IDs must use canonical positive integer strings separated by single spaces.")
    unique_ids = set().union(*(set(prediction) for prediction in predictions.values()))
    return {
        "columns": PREDICTION_COLUMNS, "data_rows": rows, "unique_user_ids": len(set(predictions)),
        "exact_template_user_ids_and_row_order": list(predictions) == users,
        "user_ids_preserved_as_strings": all(isinstance(user, str) for user in predictions),
        "all_rows_have_exactly_five_unique_positive_ids": True,
        "canonical_space_separated_ids_verified": True,
        "allowed_candidate_pool_size": len(candidates), "all_predictions_in_allowed_candidate_pool": unique_ids <= candidates,
        "unavailable_predicted_programmes": sorted(unique_ids - candidates),
        "distinct_recommended_programmes": len(unique_ids), "prediction_slots": K * rows,
    }


def publish_submission(source: Path, destination: Path) -> None:
    """Publish validated bytes exclusively, never replacing an existing file."""
    with source.open("rb") as incoming, destination.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing, length=64 * 1024)


def project_snapshot(root: Path) -> dict:
    """Fingerprint all existing project files, including any AI-session exports.

    Virtual environments, VCS internals, generated Python/tool caches, and OS
    .DS_Store files are excluded. No original project file is opened for writing.
    """
    result = {}
    for directory, directories, filenames in os.walk(root, followlinks=False):
        directories[:] = sorted(name for name in directories if name not in IGNORED_DIRECTORIES)
        for name in sorted(filenames):
            path = Path(directory) / name
            if name == ".DS_Store":
                continue
            if path.is_file():
                result[str(path.relative_to(root))] = file_fingerprint(path)
    return result


def worker(root: Path, output_path: Path, summary_path: Path) -> None:
    """Independent rebuild entry point, invoked only in fresh temporary storage."""
    for destination in (output_path, summary_path):
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Refusing to replace a worker artifact: {destination}")
        if not destination.parent.is_dir():
            raise ValueError("Worker output parent directory does not exist.")
        if root / "data/raw" == destination.parent.resolve() or root / "data/raw" in destination.resolve().parents:
            raise ValueError("Worker outputs must not be inside original data.")
    started = time.perf_counter()
    with (root / "reports/association_validation.json").open(encoding="utf-8") as handle:
        saved = json.load(handle)
    config, selection = selected_configuration(saved)
    del saved
    with tempfile.TemporaryDirectory(prefix="final-model-", dir=output_path.parent) as temporary:
        predictions, candidates, result = build_final_predictions(
            root / "data/processed" / FINAL_TRAIN_NAME, root / "data/raw" / TEST_SCHEDULE_NAME,
            root / "data/raw/submission.csv", Path(temporary), config)
    write_predictions(output_path, list(predictions), predictions, candidates)
    validation = validate_submission(output_path, root / "data/raw/submission.csv", candidates)
    result.update({"selection": selection, "submission_validation": validation,
                   "submission_file": file_fingerprint(output_path),
                   "worker_execution": {"python_hash_seed": os.environ.get("PYTHONHASHSEED"),
                                        "runtime_seconds_before_summary_write": time.perf_counter() - started,
                                        **peak_memory()}})
    with summary_path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(f"Final fit completed with PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')}; "
          f"submission SHA-256={result['submission_file']['sha256']}", flush=True)


def render_report(report: dict) -> str:
    build = report["final_build"]
    training = build["training"]
    concise = {key: value for key, value in build.items() if key != "ranking"}
    concise["ranking"] = {key: value for key, value in build["ranking"].items() if key != "per_user_ranking"}
    lines = ["# SWEET.TV Final Submission Build", "", f"Generated (UTC): {report['generated_at_utc']}", "",
             "## Submission ready for manual upload", "",
             f"- File: `{report['submission_file']['path']}`",
             f"- Data rows: **{report['submission_validation']['data_rows']}**; unique user IDs: **{report['submission_validation']['unique_user_ids']}**.",
             f"- Final training: **{training['final_training_users']:,} users**, **{training['final_training_programmes']:,} programmes**.",
             f"- Test candidates: **{build['candidate_pool']['unique_candidate_programmes']:,}**.",
             f"- Cold-start template users: **{training['cold_start_template_users']}**.",
             f"- Fallback slots: **{build['ranking']['total_fallback_slots']}** across **{build['ranking']['users_requiring_fallback']} users**.",
             f"- Distinct recommended programmes: **{build['distinct_recommended_programmes']}**.",
             f"- SHA-256: `{report['submission_file']['sha256']}`", "",
             "The CSV has been generated and locally validated. It has not been uploaded or scored by Kaggle.", "",
             "## Reproduction", "", "```bash", ".venv/bin/python src/create_final_submission.py", "```", "",
             "After creation, rerun without overwriting any final artifact:", "", "```bash",
             ".venv/bin/python src/create_final_submission.py --verify-only", "```", "",
             "Both modes run the complete test suite and independently refit in two subprocesses with different PYTHONHASHSEED values. "
             "The second mode compares with the already published CSV and writes no report or submission replacement.", "",
             "## Method and chronological boundaries", "",
             "Association Rules with maximum confidence; cap 50 items per basket; minimum pair support 5; minimum confidence 0.05; "
             "100 neighbors per antecedent; no shrinkage, minimum lift filter, popularity fusion, new model, or hybrid.", "",
             f"- Final training, half-open: [{timestamp(WINDOWS[2].start)}, {timestamp(WINDOWS[2].end)}).",
             f"- Test availability, half-open: [{timestamp(TEST_WINDOW.start)}, {timestamp(TEST_WINDOW.end)}).",
             "- Final-training data includes weeks 21–30 only because model/hyperparameter selection has finished.",
             "- The shared AggregateReader checks every aggregate timestamp and integer field. It rejects evidence extending into the test window rather than attempting to clip already aggregated weights.",
             "- Test schedules supply only positive IDs and positive airing overlap, channel validity, and timing. Descriptive metadata and test viewing labels do not enter fitting.",
             "- Distinct-user popularity is recalculated over the complete final-training histories before basket truncation. Previously watched items remain eligible.",
             "- The unchanged select_basket, AssociationCounts, build_rules, generate_predictions, rank_candidates, and shared CSV validators perform the existing algorithms. Only the time-window input adapter is new.",
             "- Exact association-score ties and popularity-count ties use ascending numeric programme IDs.", "",
             "## Selection evidence (not a test score)", "", "```json", json.dumps(build["selection"], indent=2), "```", "",
             "## Computed final-training, availability, rule, and fallback statistics", "", "```json",
             json.dumps(concise, indent=2), "```", "",
             "The JSON report also preserves per-user ranking provenance. Supports/confidences use capped baskets; final fallback counts use full histories.", "",
             "## Submission validation", "", "```json", json.dumps(report["submission_validation"], indent=2), "```", "",
             "## Independent deterministic rebuild", "", "```json", json.dumps(report["determinism"], indent=2), "```", "",
             "## Execution and measured memory", "", "```json", json.dumps(report["execution"], indent=2), "```", "",
             "The total runtime includes two full final-model fits, complete tests, input hashes, and file validation. "
             "The reported maximum per-process RSS is the maximum of measured parent and worker peaks, not a sum of concurrent parent/child memory.", "",
             "## Protected file integrity", "",
             "All preexisting project files outside virtual environments, VCS internals, generated caches, and .DS_Store were fingerprinted. "
             "This includes original datasets/template, previous predictions/reports, existing implementations, and any project-local AI-session exports. "
             "Before/after SHA-256, size, and modification time must match.", "",
             "| Protected file | SHA-256 | Unchanged |", "|---|---|---|"]
    for name, details in report["input_integrity"].items():
        lines.append(f"| {name} | `{details['before']['sha256']}` | {details['unchanged']} |")
    lines += ["", "## Complete automated test results", "", "```text", report["tests"]["output"].rstrip(), "```", "",
              "## Limitations", "",
              "- The 50-programme basket cap drops long-tail evidence. Rule confidence is not a calibrated enjoyment or next-watch probability.",
              "- Programmes absent from final-training baskets cannot receive association scores; availability alone does not supply preference evidence.",
              "- The inherited 120-second interaction threshold, sparse programme labels, missing channel mappings, and uncorrected clock/duration discrepancies remain documented in the preserved earlier reports.",
              "- Final/test MAP@5 is unknown without actual hidden labels. The earlier validation score is retained only as selection evidence.",
              "- No Kaggle upload, academic PDF, new recommendation method, or hybrid was produced.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-summary", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.worker_output is not None or args.worker_summary is not None:
        if args.worker_output is None or args.worker_summary is None or args.verify_only:
            parser.error("Internal worker arguments must be supplied together.")
        worker(root, args.worker_output, args.worker_summary)
        return
    submission = root / "outputs/submission.csv"
    report_paths = [root / "reports/final_submission_build.json", root / "reports/final_submission_build.md"]
    if args.verify_only:
        if not submission.is_file() or not all(path.is_file() for path in report_paths):
            parser.error("--verify-only requires the published submission and final-build reports.")
    elif any(path.exists() or path.is_symlink() for path in [submission, *report_paths]):
        parser.error("Final artifacts already exist; use --verify-only to reproduce without overwriting.")
    started = time.perf_counter()
    generated = datetime.now(timezone.utc).isoformat()
    before = project_snapshot(root)
    with (root / "reports/association_validation.json").open(encoding="utf-8") as handle:
        association = json.load(handle)
    config, selection = selected_configuration(association)
    for name, fingerprint in association["input_integrity"].items():
        if before[name]["sha256"] != fingerprint["after"]["sha256"]:
            raise RuntimeError(f"Protected input changed since model selection: {name}")
    for name, digest in association["execution"]["source_sha256"].items():
        if before[name]["sha256"] != digest:
            raise RuntimeError(f"Verified implementation changed since selection: {name}")
    for name, fingerprint in association["prediction_files"].items():
        if before[name]["sha256"] != fingerprint["sha256"]:
            raise RuntimeError(f"Previous validation prediction file changed: {name}")
    del association
    with (root / "reports/interaction_build.json").open(encoding="utf-8") as handle:
        interaction = json.load(handle)
    expected_training = interaction["build"]["splits"]["final_train"]
    if before[f"data/processed/{FINAL_TRAIN_NAME}"]["sha256"] != interaction["output_files"][FINAL_TRAIN_NAME]["sha256"]:
        raise RuntimeError("Final-training aggregates differ from the verified interaction build.")
    if interaction["build"]["minimum_overlap_seconds"] != 120:
        raise RuntimeError("Final training must reuse the selected 120-second interaction construction.")
    test_started = time.perf_counter()
    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
    tests = subprocess.run(command, cwd=root, capture_output=True, text=True)
    test_output = tests.stdout + tests.stderr
    print(test_output, end="", flush=True)
    if tests.returncode:
        raise RuntimeError("Automated tests failed; final model fitting was not started.")
    test_result = {"command": command, "returncode": tests.returncode, "output": test_output,
                   "runtime_seconds": time.perf_counter() - test_started}
    rebuilds = []
    with tempfile.TemporaryDirectory(prefix="sweettv-final-build-", dir=root / "data/processed") as temporary:
        work = Path(temporary)
        for index, hash_seed in enumerate(("0", "314159"), 1):
            environment = os.environ.copy()
            environment["PYTHONHASHSEED"] = hash_seed
            output_path, summary_path = work / f"rebuild_{index}.csv", work / f"rebuild_{index}.json"
            command = [sys.executable, str(Path(__file__).resolve()), "--worker-output", str(output_path), "--worker-summary", str(summary_path)]
            worker_started = time.perf_counter()
            completed = subprocess.run(command, cwd=root, env=environment, capture_output=True, text=True)
            print(completed.stdout, end="", flush=True)
            if completed.returncode:
                raise RuntimeError(f"Independent final fit failed:\n{completed.stderr}")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["parent_measured_subprocess_seconds"] = time.perf_counter() - worker_started
            rebuilds.append(summary)
        first, second = rebuilds
        if (not filecmp.cmp(work / "rebuild_1.csv", work / "rebuild_2.csv", shallow=False)
                or first["model_fingerprints"] != second["model_fingerprints"]
                or first["submission_file"]["sha256"] != second["submission_file"]["sha256"]):
            raise AssertionError("Independent hash-seed rebuilds disagree; no final file will be published.")
        stats = first["training"]["aggregate_schema_and_time_bounds"]
        for actual, expected in ((stats["unique_users"], expected_training["unique_users"]),
                                 (stats["unique_programmes"], expected_training["unique_programme_ids"]),
                                 (stats["unique_user_programme_pairs"], expected_training["unique_user_programme_pairs"]),
                                 (stats["matched_events"], expected_training["matched_interactions"]),
                                 (stats["template_users_present"], expected_training["template_users_with_labeled_interactions"])):
            if actual != expected:
                raise AssertionError("Measured final-training statistics differ from the verified interaction build.")
        candidates, candidate_stats = load_candidates(root / "data/raw" / TEST_SCHEDULE_NAME, TEST_WINDOW)
        if candidate_stats != first["candidate_pool"]:
            raise AssertionError("Parent and worker test-availability checks disagree.")
        if args.verify_only:
            if not filecmp.cmp(work / "rebuild_1.csv", submission, shallow=False):
                raise AssertionError("Rebuilt submission differs from the preserved published file.")
        else:
            publish_submission(work / "rebuild_1.csv", submission)
        validation = validate_submission(submission, root / "data/raw/submission.csv", candidates)
    integrity = {}
    for name, details in before.items():
        after = file_fingerprint(root / name)
        integrity[name] = {"before": details, "after": after, "unchanged": details == after}
    if not all(details["unchanged"] for details in integrity.values()):
        raise RuntimeError("A protected project artifact changed; final build report will not claim success.")
    submission_details = {"path": str(submission.relative_to(root)), **file_fingerprint(submission)}
    determinism = {
        "independent_full_model_fits": 2, "python_hash_seeds": ["0", "314159"],
        "csv_bytes_identical": True, "model_fingerprints_identical": True,
        "first_submission_sha256": first["submission_file"]["sha256"],
        "second_submission_sha256": second["submission_file"]["sha256"],
        "published_submission_sha256": submission_details["sha256"],
        "model_fingerprints": first["model_fingerprints"],
        "first_worker_execution": first["worker_execution"],
        "second_worker_execution": second["worker_execution"],
        "parent_measured_worker_seconds": [run["parent_measured_subprocess_seconds"] for run in rebuilds],
    }
    if args.verify_only:
        print(json.dumps({"verification_only": True, "submission": submission_details, "validation": validation,
                          "determinism": determinism, "all_existing_files_unchanged": True}, indent=2))
        return
    parent_memory = peak_memory()
    measured_peaks = [memory["peak_process_rss_bytes"] for memory in
                      (parent_memory, first["worker_execution"], second["worker_execution"])
                      if memory.get("peak_process_rss_bytes") is not None]
    maximum_peak = max(measured_peaks) if measured_peaks else None
    first.pop("submission_file")
    first.pop("parent_measured_subprocess_seconds")
    report = {
        "schema_version": 1, "generated_at_utc": generated, "selection": selection,
        "final_build": first, "submission_validation": validation,
        "submission_file": submission_details, "determinism": determinism,
        "input_integrity": integrity, "tests": test_result,
        "execution": {
            "python_version": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
            "invocation": sys.argv, "third_party_packages_used": [], "randomness": "none in the recommendation model",
            "runtime_seconds_before_report_serialization": time.perf_counter() - started,
            "parent_process_peak_memory": parent_memory,
            "highest_measured_per_process_peak_rss_bytes": maximum_peak,
            "highest_measured_per_process_peak_rss_mib": maximum_peak / 1024 ** 2 if maximum_peak is not None else None,
            "memory_scope": "maximum of parent and two sequential worker process peaks, not a combined parent-plus-child working set",
            "source_sha256": {str(path.relative_to(root)): file_fingerprint(path)["sha256"] for path in
                              (Path(__file__), root / "tests/test_final_submission.py", root / "src/association_rules.py",
                               root / "src/popularity.py", root / "src/evaluate.py")},
        },
        "verification": {
            "all_tests_passed": True, "correct_final_training_window_verified_on_all_aggregate_rows": True,
            "no_test_period_viewing_used": True, "test_schedule_used_only_for_availability": True,
            "final_popularity_recomputed_from_full_final_training_histories": True,
            "old_algorithms_and_validation_rules_unchanged": True,
            "all_preexisting_protected_project_files_unchanged": True,
            "independent_rebuilds_verified_byte_for_byte": True,
            "ready_for_manual_submission": True, "uploaded_to_kaggle": False,
        },
    }
    with report_paths[0].open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    with report_paths[1].open("x", encoding="utf-8") as handle:
        handle.write(render_report(report))
    print(json.dumps({"submission": submission_details, "validation": validation,
                      "final_training_users": first["training"]["final_training_users"],
                      "final_training_programmes": first["training"]["final_training_programmes"],
                      "cold_start_users": first["training"]["cold_start_template_users"],
                      "fallback_slots": first["ranking"]["total_fallback_slots"]}, indent=2))
    print(f"Total runtime including report writes: {time.perf_counter() - started:.3f} seconds")
    print(f"Highest measured per-process peak RSS: {report['execution']['highest_measured_per_process_peak_rss_mib']} MiB")


if __name__ == "__main__":
    main()
