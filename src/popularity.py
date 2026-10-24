"""Run the training-only distinct-viewer popularity baseline on local validation.

Reproduce: .venv/bin/python src/popularity.py

No validation interactions enter fitting or candidate generation. Predictions are
generated before validation ground truth is loaded. Final-training aggregates are
scanned for schema/integrity inspection only; they are never passed to fit().
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType

from audit_data import INPUT_FILES, SCHEDULE_FILES, peak_memory
from build_interactions import WINDOWS, file_fingerprint
from evaluate import (
    K, AggregateReader, evaluate_recommendations, load_candidates,
    load_template_users, load_validation_ground_truth, read_predictions,
    validate_programme_id, write_predictions,
)


@dataclass(frozen=True)
class PopularityModel:
    scores: Mapping[int, int]
    training_users: frozenset[str]
    training_statistics: dict


def fit_popularity(training_path: Path) -> PopularityModel:
    """Count each distinct training user once per programme, irrespective of views."""
    reader = AggregateReader(training_path, WINDOWS[0])
    scores = Counter()
    for row in reader:
        scores[row.programme_id] += 1
    return PopularityModel(MappingProxyType(dict(scores)), frozenset(reader.users), reader.summary())


def rank_candidates(scores: Mapping[int, int], candidates: frozenset[int]) -> tuple[int, ...]:
    """Order by descending distinct viewers, then ascending numeric programme ID."""
    for programme in candidates:
        validate_programme_id(programme)
    if len(candidates) < K:
        raise ValueError("At least five distinct available candidates are required.")
    for programme, score in scores.items():
        validate_programme_id(programme)
        if type(score) is not int or score < 0:
            raise ValueError("Popularity scores must be nonnegative distinct-user counts.")
    return tuple(sorted(candidates, key=lambda item: (-scores.get(item, 0), item))[:K])


def run_baseline(training_path: Path, validation_path: Path, schedule_path: Path,
                 template_path: Path, expected_users: int | None = 1260) -> tuple[dict, dict, frozenset[int], dict]:
    """Keep fitting and prediction generation structurally separate from labels."""
    stages = {}
    users = load_template_users(template_path, expected_users)
    started = time.perf_counter()
    model = fit_popularity(training_path)
    stages["fit_training_popularity"] = time.perf_counter() - started
    started = time.perf_counter()
    candidates, candidate_stats = load_candidates(schedule_path)
    top_five = rank_candidates(model.scores, candidates)
    predictions = {user: top_five for user in users}
    stages["schedule_candidates_and_predictions"] = time.perf_counter() - started
    # The model and every recommendation are fixed before any validation labels are read.
    started = time.perf_counter()
    truth, validation_stats = load_validation_ground_truth(validation_path, users)
    evaluation = evaluate_recommendations(users, truth, predictions, model.training_users, candidates)
    stages["load_validation_labels_and_evaluate"] = time.perf_counter() - started
    training_stats = dict(model.training_statistics)
    training_stats["template_users_present"] = len(set(users) & model.training_users)
    result = {
        "model": {
            "name": "distinct_training_viewer_popularity",
            "fit_input": str(training_path), "fit_window": "weeks 11-20 only",
            "score": "number of distinct training users per programme",
            "k": K, "tie_break": "ascending numeric programme ID",
            "already_watched_programmes_allowed": True,
            "unseen_candidate_score": 0,
            "cold_start_fallback": "the identical global popularity Top-5 used for every user",
            "random_seed": None, "randomness": "none; no stochastic operations",
            "validation_labels_used_for_fit": False,
            "validation_labels_used_for_candidates": False,
            "predictions_created_before_validation_labels_loaded": True,
            "training_programmes_in_candidate_pool": len(set(model.scores) & candidates),
            "candidates_without_training_interactions": sum(item not in model.scores for item in candidates),
            "top_five": [{"rank": rank, "tv_show_id": item, "distinct_training_users": model.scores.get(item, 0)}
                         for rank, item in enumerate(top_five, 1)],
        },
        "data_inspection": {"train": training_stats, "validation": validation_stats},
        "candidate_pool": candidate_stats, "evaluation": evaluation,
        "stage_runtime_seconds": stages,
    }
    return result, predictions, candidates, truth


PROTOCOL = {
    "metric": "MAP@5", "k": K,
    "primary_cohort": "all 1260 original Kaggle template users, in original order",
    "ground_truth": "unique positive programme IDs from validation interactions only",
    "ap_normalization": "sum of precision at relevant ranks divided by min(5, unique relevant programme count)",
    "empty_ground_truth": "AP@5 = 0; user remains in the primary denominator",
    "supplementary_metric": "MAP@5 restricted to nonempty validation ground truth",
    "cold_start_definition": "no qualifying interaction in training weeks 11-20",
    "empty_subgroup_metric": "null (undefined), never NaN",
    "candidate_pool": "positive IDs with positive airing overlap in export_arh_21-30-final.csv within validation timestamps",
    "candidate_and_ground_truth_policy": "candidate filtering applies to predictions only; never remove unreachable ground truth",
    "previously_watched_programmes": "allowed",
    "prediction_validation": "exactly five unique positive integer IDs per cohort user, all in the common candidate pool",
    "interaction_threshold_seconds": 120,
    "time_policy": "half-open ISO-week windows; supplied naive timestamps, no timezone correction",
    "local_convention_disclaimer": "This is our local evaluation convention, not a verified reproduction of Kaggle's hidden evaluator.",
}


def render_report(report: dict) -> str:
    result = report["baseline"]
    metrics = result["evaluation"]["metrics"]
    lines = ["# SWEET.TV Popularity Baseline", "", f"Generated (UTC): {report['generated_at_utc']}", "",
             "## Reproduction", "", "```bash", ".venv/bin/python src/popularity.py", "```", "",
             "The CLI runs the full unit-test suite, fits only weeks 11–20 interactions, creates validation predictions, "
             "then reads weeks 21–30 labels for scoring. No third-party packages are required.", "",
             "## Actual validation results", "",
             "| Cohort | Users | Users with labels | MAP@5 |", "|---|---:|---:|---:|"]
    for name in ("primary_all_users", "supplementary_nonempty_ground_truth", "with_training_history", "without_training_history",
                 "with_training_history_and_validation_labels", "without_training_history_with_validation_labels"):
        group = metrics[name]
        score = f"{group['map_at_5']:.10f}" if group["map_at_5"] is not None else "undefined"
        lines.append(f"| {name.replace('_', ' ')} | {group['users']:,} | {group['users_with_validation_labels']:,} | {score} |")
    lines += ["", "## Validation protocol", "", "```json", json.dumps(report["validation_protocol"], indent=2), "```", "",
              "AP@5(u) = sum(P@rank for relevant ranks 1..5) / min(5, number of unique relevant programmes). "
              "If there are no relevant programmes, AP@5(u) = 0. MAP@5 is the arithmetic mean over the stated cohort.", "",
              "**This is our local evaluation convention, not a verified reproduction of Kaggle's hidden evaluator.**", "",
              "## Verified data schemas and chronological boundaries", "", "```json",
              json.dumps(result["data_inspection"], indent=2), "```", "",
              "The final_train file was inspected only; it was not used for fitting, scoring, or predictions.", "",
              "## Candidate availability", "", "```json", json.dumps(result["candidate_pool"], indent=2), "```", "",
              "Candidates come exclusively from the validation-period broadcast schedule, with timestamp filtering. "
              "A short positive-duration airing can be available even if it cannot meet the 120-second watching threshold.", "",
              "## Model parameters and Top-5", "", "```json", json.dumps(result["model"], indent=2), "```", "",
              "## Coverage and candidate/ground-truth compatibility", "", "```json",
              json.dumps({"coverage": metrics["coverage"], "ground_truth_candidate_coverage": metrics["ground_truth_candidate_coverage"]}, indent=2), "```", "",
              "## Execution and environment", "", "```json", json.dumps(report["execution"], indent=2), "```", "",
              "## Verification", "", "```json", json.dumps(report["verification"], indent=2), "```", "",
              "### Protected input fingerprints", "", "| File | SHA-256 | Unchanged |", "|---|---|---|"]
    for name, details in report["input_integrity"].items():
        lines.append(f"| {name} | `{details['before']['sha256']}` | {details['unchanged']} |")
    lines += ["", "### Saved validation predictions", "", "```json", json.dumps(report["prediction_file"], indent=2), "```", "",
              "### Tests executed", "", "```text", report["tests"]["output"].rstrip(), "```", "",
              "## Remaining risks and limitations", "",
              "- Popularity deliberately produces the same five programmes for all users. Full user coverage does not imply personalized or broad catalogue coverage.",
              "- The 120-second interaction threshold remains provisional. Sparse programme labels, missing channel mappings, and schedule inaccuracies limit observable relevance.",
              "- The previously audited 117 duration mismatches remain unresolved; no timestamps were shifted.",
              "- Users without validation labels contribute zero by policy. Nonempty-label and warm/cold subgroup results are reported separately; small cold-start groups are noisy.",
              "- Candidate availability does not imply user interest. Candidates without training observations receive zero score, not an estimated preference.",
              "- Validation interactions were built from both historical schedule files, whereas this candidate pool uses the designated validation schedule. Any unavailable relevance is retained and explicitly counted.",
              "- This baseline is a required reference and does not count as either of the two required recommendation methods.",
              "- The exam documents disagree on public versus private Kaggle scores. No Kaggle score or hidden-evaluator equivalence is claimed.",
              "- CSV inputs are streamed. Aggregate loading requires canonical sorted keys, which are verified; duplicate keys are ignored rather than inflating distinct-user counts.",
              "- Runtime includes unit tests, integrity checks, data validation, fitting, evaluation, and prediction verification; final report serialization is excluded. Peak RSS is for the main process, excluding the separate test process.",
              "- No CBF, association rules, final-training fit, final Kaggle submission, or Kaggle upload was performed.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    raw, processed, reports, outputs = (root / p for p in ("data/raw", "data/processed", "reports", "outputs"))
    prediction_path = outputs / "popularity_validation.csv"
    for path in (prediction_path, reports / "popularity_baseline.md", reports / "popularity_baseline.json"):
        if path.is_symlink():
            parser.error(f"Refusing to overwrite a symbolic-link output: {path}")
    outputs.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    generated = datetime.now(timezone.utc).isoformat()
    build_report = json.loads((reports / "interaction_build.json").read_text(encoding="utf-8"))
    if build_report["build"]["minimum_overlap_seconds"] != 120:
        raise ValueError("This baseline protocol expects interactions built with a 120-second threshold.")
    protected_paths = [*(raw / name for name in INPUT_FILES),
                       *(processed / name for name in build_report["output_files"]),
                       *(reports / name for name in ("data_audit.md", "data_audit.json", "interaction_build.md", "interaction_build.json"))]
    before = {str(path.relative_to(root)): file_fingerprint(path) for path in protected_paths}
    for name in INPUT_FILES:
        if before[f"data/raw/{name}"]["sha256"] != build_report["protected_files"][f"data/raw/{name}"]["after"]["sha256"]:
            raise RuntimeError(f"Original CSV differs from the interaction build: {name}")
    for name, fingerprint in build_report["output_files"].items():
        if before[f"data/processed/{name}"]["sha256"] != fingerprint["sha256"]:
            raise RuntimeError(f"Processed interactions differ from the recorded build: {name}")
    test_start = time.perf_counter()
    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
    tests = subprocess.run(command, cwd=root, capture_output=True, text=True)
    test_output = tests.stdout + tests.stderr
    print(test_output, end="", flush=True)
    if tests.returncode:
        raise RuntimeError("Tests failed; the actual baseline was not fitted.")
    test_result = {"command": command, "returncode": tests.returncode, "output": test_output,
                   "runtime_seconds": time.perf_counter() - test_start}
    result, predictions, candidates, truth = run_baseline(
        processed / "train_user_programmes.csv", processed / "validation_user_programmes.csv",
        raw / SCHEDULE_FILES[1], raw / "submission.csv")
    users = list(predictions)
    # Inspect this schema only; never pass final-training rows to the model.
    inspection_start = time.perf_counter()
    final_reader = AggregateReader(processed / "final_train_user_programmes.csv", WINDOWS[2])
    for _ in final_reader:
        pass
    result["data_inspection"]["final_train_inspection_only"] = final_reader.summary(users)
    for split, stats in (("train", result["data_inspection"]["train"]), ("validation", result["data_inspection"]["validation"]),
                         ("final_train", result["data_inspection"]["final_train_inspection_only"])):
        expected = build_report["build"]["splits"][split]
        if (stats["unique_users"] != expected["unique_users"] or stats["unique_programmes"] != expected["unique_programme_ids"]
                or stats["unique_user_programme_pairs"] != expected["unique_user_programme_pairs"]
                or stats["matched_events"] != expected["matched_interactions"]):
            raise AssertionError(f"Aggregate statistics do not reconcile with the build report: {split}")
    inspection_seconds = time.perf_counter() - inspection_start
    write_predictions(prediction_path, users, predictions, candidates)
    reloaded = read_predictions(prediction_path, users, candidates)
    if reloaded != predictions:
        raise AssertionError("Saved prediction CSV differs from generated predictions.")
    # Re-evaluate the saved artifact, not only the in-memory recommendation mapping.
    training_users = {u for u, details in result["evaluation"]["per_user"].items() if details["has_training_history"]}
    reevaluated = evaluate_recommendations(users, truth, reloaded, training_users, candidates)
    if reevaluated != result["evaluation"]:
        raise AssertionError("Metrics from the saved CSV differ from the original evaluation.")
    after = {str(path.relative_to(root)): file_fingerprint(path) for path in protected_paths}
    integrity = {name: {"before": value, "after": after[name], "unchanged": value == after[name]} for name, value in before.items()}
    if not all(item["unchanged"] for item in integrity.values()):
        raise RuntimeError("Protected input changed; successful baseline reports were not written.")
    score_state = result["model"]["top_five"]
    report = {
        "schema_version": 1, "generated_at_utc": generated, "validation_protocol": PROTOCOL,
        "baseline": result, "input_integrity": integrity,
        "prediction_file": {"path": str(prediction_path.relative_to(root)), **file_fingerprint(prediction_path)},
        "verification": {
            "all_tests_passed": tests.returncode == 0, "all_protected_files_unchanged": True,
            "all_1260_users_have_exactly_five_unique_positive_available_ids": len(users) == 1260,
            "template_order_and_user_id_strings_preserved": True,
            "saved_prediction_metrics_recomputed_and_equal": True,
            "fit_inputs": ["data/processed/train_user_programmes.csv"],
            "candidate_inputs": [f"data/raw/{SCHEDULE_FILES[1]}"],
            "label_inputs_for_scoring_only": ["data/processed/validation_user_programmes.csv"],
            "final_training_data_used_for_fit": False,
        },
        "tests": test_result,
        "execution": {
            "python_version": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
            "invocation": sys.argv, "third_party_packages_used": [],
            "source_sha256": {str(path.relative_to(root)): file_fingerprint(path)["sha256"] for path in
                              (root / "src/evaluate.py", root / "src/popularity.py", root / "src/build_interactions.py",
                               root / "src/audit_data.py", root / "tests/test_evaluate.py", root / "tests/test_popularity.py")},
            "stage_runtime_seconds": {**result["stage_runtime_seconds"], "final_train_inspection_only": inspection_seconds,
                                      "unit_tests": test_result["runtime_seconds"]},
            "runtime_seconds_before_report_serialization": time.perf_counter() - started, **peak_memory(),
        },
    }
    (reports / "popularity_baseline.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (reports / "popularity_baseline.md").write_text(render_report(report), encoding="utf-8")
    print(json.dumps(result["evaluation"]["metrics"], indent=2), flush=True)
    print(f"Top-5: {[row['tv_show_id'] for row in score_state]}")
    print(f"All protected inputs unchanged. Runtime including report writes: {time.perf_counter() - started:.3f} seconds")
    print(f"Peak main-process RSS: {report['execution'].get('peak_process_rss_mib', 'unavailable')} MiB")


if __name__ == "__main__":
    main()
