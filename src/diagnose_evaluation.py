"""Independently diagnose the saved local Popularity score before new modeling.

Reproduce: .venv/bin/python src/diagnose_evaluation.py

CSV parsing, date bounds, distinct-user popularity, availability, and rational
AP@5 calculations below are independent of the production data/evaluator helpers.
The production evaluator is called only to check agreement with known examples.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path

from build_interactions import file_fingerprint
from evaluate import average_precision_at_5


TRAIN_START = datetime(2020, 3, 9)
VALIDATION_START = datetime(2020, 5, 18)
VALIDATION_END = datetime(2020, 7, 27)
PAIR_COLUMNS = ["user_id", "tv_show_id", "matched_events", "overlap_seconds", "first_overlap", "last_overlap"]


def independent_ap(relevant, prediction) -> Fraction:
    """Exact rational AP@5, without calling any production metric function."""
    if len(prediction) != 5 or len(set(prediction)) != 5 or any(type(item) is not int or item <= 0 for item in prediction):
        raise ValueError("Five unique positive integer predictions are required.")
    relevant = set(relevant)
    if not relevant:
        return Fraction(0)
    indicators = [int(item in relevant) for item in prediction]
    numerator = sum((Fraction(sum(indicators[:rank]), rank) * hit for rank, hit in enumerate(indicators, 1)), Fraction(0))
    return numerator / min(5, len(relevant))


def scan_pairs(path, start, end):
    previous = None
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != PAIR_COLUMNS:
            raise AssertionError("Unexpected aggregate schema.")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise AssertionError("Malformed aggregate row.")
            user, programme = row["user_id"], int(row["tv_show_id"])
            first, last = datetime.fromisoformat(row["first_overlap"]), datetime.fromisoformat(row["last_overlap"])
            key = (user, programme)
            if (not user or not isinstance(user, str) or programme <= 0
                    or not start <= first < last <= end or int(row["matched_events"]) <= 0
                    or int(row["overlap_seconds"]) < 120 * int(row["matched_events"])
                    or (previous is not None and key <= previous)):
                raise AssertionError("Invalid, duplicate, unsorted, or temporally misplaced aggregate evidence.")
            previous = key
            yield user, programme


def distribution(values):
    return {"users": len(values), "min": min(values), "median": statistics.median(values),
            "mean": statistics.mean(values), "max": max(values),
            "zero_label_users": sum(value == 0 for value in values)}


def diagnose(root: Path) -> dict:
    baseline = json.loads((root / "reports/popularity_baseline.json").read_text())
    cbf = json.loads((root / "reports/cbf_validation.json").read_text())
    protected = set(cbf["input_integrity"]) | set(cbf["execution"]["source_sha256"])
    protected.update({"reports/cbf_validation.json", "reports/cbf_validation.md", "outputs/cbf_validation.csv"})
    before = {name: file_fingerprint(root / name) for name in sorted(protected)}
    for name, details in cbf["input_integrity"].items():
        if before[name]["sha256"] != details["after"]["sha256"]:
            raise AssertionError(f"Input differs from the preserved CBF experiment: {name}")
    for name, digest in cbf["execution"]["source_sha256"].items():
        if before[name]["sha256"] != digest:
            raise AssertionError(f"Inspected source differs from the preserved experiment: {name}")
    with (root / "data/raw/submission.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["user_id", "tv_show_id"]:
            raise AssertionError("Unexpected template schema.")
        users = [row["user_id"] for row in reader]
    if len(users) != len(set(users)) or len(users) != 1260:
        raise AssertionError("Template cohort is not exactly 1260 unique original user strings.")
    counts, training_users = Counter(), set()
    for user, programme in scan_pairs(root / "data/processed/train_user_programmes.csv", TRAIN_START, VALIDATION_START):
        counts[programme] += 1
        training_users.add(user)
    candidates = set()
    with (root / "data/raw/export_arh_21-30-final.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            programme = int(row["tv_show_id"])
            if programme == 0:
                continue
            start = datetime.strptime(row["start_time"], "%d.%m.%Y %H:%M:%S")
            duration = int(row["duration"])
            if programme < 0 or int(row["channel_id"]) <= 0 or duration <= 0:
                raise AssertionError("Invalid labeled availability interval.")
            if start < VALIDATION_END and start + timedelta(seconds=duration) > VALIDATION_START:
                candidates.add(programme)
    independently_ranked = tuple(sorted(candidates, key=lambda programme: (-counts[programme], programme))[:5])
    with (root / "outputs/popularity_validation.csv").open(newline="") as handle:
        saved = [(row["user_id"], tuple(map(int, row["tv_show_id"].split()))) for row in csv.DictReader(handle)]
    if [user for user, _ in saved] != users or any(prediction != independently_ranked for _, prediction in saved):
        raise AssertionError("Saved popularity predictions differ from independently calculated training-only ranking.")
    # Validation labels enter only after the training ranking and candidates are fixed.
    truth = {user: set() for user in users}
    validation_top_viewers = {programme: set() for programme in independently_ranked}
    validation_users = set()
    for user, programme in scan_pairs(root / "data/processed/validation_user_programmes.csv", VALIDATION_START, VALIDATION_END):
        validation_users.add(user)
        if user in truth:
            truth[user].add(programme)
        if programme in validation_top_viewers:
            validation_top_viewers[programme].add(user)
    rational_ap = {user: independent_ap(truth[user], independently_ranked) for user in users}
    exact_mean = sum(rational_ap.values(), Fraction(0)) / len(users)
    expected = baseline["baseline"]["evaluation"]
    if float(exact_mean) != expected["metrics"]["primary_all_users"]["map_at_5"]:
        raise AssertionError("Independent rational primary MAP@5 does not match the saved result.")
    for user in users:
        if (not math.isclose(float(rational_ap[user]), expected["per_user"][user]["ap_at_5"], rel_tol=0, abs_tol=1e-15)
                or len(truth[user]) != expected["per_user"][user]["relevant_programme_count"]):
            raise AssertionError("Independent per-user AP or ground truth disagrees with the saved evaluator.")
    examples = [({1, 3}, [1, 2, 3, 4, 5], Fraction(5, 6)),
                ({5}, [1, 2, 3, 4, 5], Fraction(1, 5)),
                (set(range(1, 8)), [1, 2, 3, 4, 5], Fraction(1)),
                (set(), [1, 2, 3, 4, 5], Fraction(0)),
                ({9}, [1, 2, 3, 4, 5], Fraction(0))]
    checks = []
    for relevant, prediction, answer in examples:
        rational, production = independent_ap(relevant, prediction), average_precision_at_5(relevant, prediction)
        if rational != answer or not math.isclose(float(answer), production, rel_tol=0, abs_tol=1e-15):
            raise AssertionError("Conventional AP@5 failed an independently specified known-answer example.")
        checks.append({"relevant": sorted(relevant), "prediction": prediction, "expected_rational_ap": str(answer),
                       "independent_rational_ap": str(rational), "production_ap": production, "passed": True})
    sizes = [len(truth[user]) for user in users]
    hit_counts = Counter(len(truth[user] & set(independently_ranked)) for user in users)
    after = {name: file_fingerprint(root / name) for name in sorted(protected)}
    if before != after:
        raise AssertionError("Protected files changed during diagnosis.")
    return {
        "critical_problem_found": False,
        "conclusion": "No leakage or evaluation error was found in the inspected inputs, code paths, independent ranking, and AP checks. The existing protocol is unchanged.",
        "relevant_programmes_per_user_all_1260": distribution(sizes),
        "relevant_programmes_per_user_nonempty": distribution([size for size in sizes if size]),
        "validation_unique_users_all_interactions": len(validation_users),
        "validation_viewers_of_baseline_top_five": [
            {"rank": rank, "tv_show_id": programme, "distinct_training_viewers_all_training_users": counts[programme],
             "validation_viewers_in_template_cohort": len(validation_top_viewers[programme] & set(users)),
             "validation_viewers_all_validation_users": len(validation_top_viewers[programme])}
            for rank, programme in enumerate(independently_ranked, 1)],
        "baseline_relevant_recommendations_per_user": {str(hits): hit_counts[hits] for hits in range(6)},
        "independent_primary_map_at_5_rational": str(exact_mean),
        "independent_primary_map_at_5": float(exact_mean),
        "saved_primary_map_at_5": expected["metrics"]["primary_all_users"]["map_at_5"],
        "all_1260_independent_per_user_scores_match": True,
        "known_answer_examples": checks,
        "training_users": len(training_users), "training_programmes": len(counts),
        "candidate_programmes_independently_reconstructed": len(candidates),
        "temporal_checks": {"training_start_inclusive": str(TRAIN_START), "training_end_exclusive": str(VALIDATION_START),
                            "validation_start_inclusive": str(VALIDATION_START), "validation_end_exclusive": str(VALIDATION_END),
                            "all_aggregate_dates_and_unique_keys_checked": True},
        "leakage_review": {
            "popularity": "fit_popularity reads only train_user_programmes.csv, validates training bounds, and counts distinct user-programme keys; the independently computed Top-5 agrees.",
            "candidates": "load_candidates and the independent reconstruction use schedule IDs, channels, starts, and durations only, not viewing labels.",
            "cbf": "load_training_metadata filters the weeks 11-20 schedule by training timestamps; predict_validation accepts no validation-interaction file and builds training profiles only.",
            "evaluation": "Validation interactions provide labels after fitting and prediction. Empty-label users remain in the primary cohort; repeat recommendations are permitted.",
            "scope": "This checks the supplied aggregate artifacts and their preserved provenance; it does not independently recreate the entire raw interval join or reveal Kaggle's hidden labels.",
        },
        "why_local_score_can_exceed_hidden_score": [
            "Local relevance is every unique programme with a qualifying overlap in the ten-week hold-out, not an observed or verified list of five strongest preferences.",
            "AP@5 divides by min(5, relevant count). Predicting five common relevant programmes can yield 1 even when a user has many other relevant programmes; that is the specified conventional metric, not an error.",
            "Previously watched programmes may be repeated, and the five globally common programmes remain relevant to many users in this hold-out.",
            "Kaggle's hidden relevance construction, weighting, exact normalization, public/private split, and later weeks 31-42 may differ. Their score is unknown until an actual authorized submission; none is claimed here.",
            "Sparse programme labeling, the provisional 120-second threshold, and schedule matching limitations make local observed relevance an imperfect proxy for preference.",
        ],
        "protected_file_fingerprints": before,
        "all_protected_files_unchanged": True,
    }


def render_diagnostic(report: dict) -> str:
    return "\n".join([
        "# SWEET.TV Evaluation Diagnostic", "", f"Generated (UTC): {report['generated_at_utc']}", "",
        "## Conclusion", "", report["diagnostic"]["conclusion"], "",
        "## Computed evidence", "", "```json", json.dumps(report["diagnostic"], indent=2), "```", "",
        "## Reproduction and environment", "", "```bash", ".venv/bin/python src/diagnose_evaluation.py", "```", "",
        "```json", json.dumps(report["execution"], indent=2), "```", "",
        "The production evaluator and all previous experiment reports were left unchanged. No new recommendation model was fitted by this diagnostic.", "",
    ])


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    destinations = [root / "reports/evaluation_diagnostic.json", root / "reports/evaluation_diagnostic.md"]
    if any(path.exists() for path in destinations):
        raise FileExistsError("Preserving the existing diagnostic reports; use a separate run directory for a new diagnostic.")
    started = time.perf_counter()
    result = diagnose(root)
    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "diagnostic": result,
              "execution": {"python_version": sys.version, "script_sha256": file_fingerprint(Path(__file__))["sha256"],
                            "runtime_seconds_before_report_serialization": time.perf_counter() - started}}
    destinations[0].write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    destinations[1].write_text(render_diagnostic(report), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "protected_file_fingerprints"}, indent=2))
    print(f"Diagnostic completed in {time.perf_counter() - started:.3f} seconds.")


if __name__ == "__main__":
    main()
