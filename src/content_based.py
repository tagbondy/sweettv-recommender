"""Training-only sparse content-based filtering for the existing local protocol.

Reproduce: .venv/bin/python src/content_based.py

Metadata vocabulary, IDF, item vectors, and user profiles use weeks 11-20 only.
Validation schedules supply candidate availability only. Validation viewing labels
are loaded after every prediction is fixed. The standard library is sufficient.
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import json
import math
import platform
import re
import subprocess
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from audit_data import SCHEDULE_COLUMNS, SCHEDULE_FILES, parse_integer, parse_timestamp, peak_memory
from build_interactions import WINDOWS, checked_records, file_fingerprint, seconds, timestamp
from evaluate import (
    K, AggregateReader, evaluate_recommendations, load_candidates, load_template_users,
    load_validation_ground_truth, read_predictions, validate_cohort,
    validate_prediction, write_predictions,
)
from popularity import PROTOCOL, fit_popularity, rank_candidates


FIELDS = ("category", "genre", "year_decade", "director", "actor", "title")
EMPTY_VALUES = frozenset({"", "0", "-", "—", "unknown", "n/a", "nan", "none", "null"})
PERSON_SPLIT = re.compile(r"[,;|]+")
YEAR_PATTERN = re.compile(r"(?<!\d)(?:18|19|20)\d{2}(?!\d)")
WORD_PATTERN = re.compile(r"[^\W\d_]+", re.UNICODE)


@dataclass(frozen=True)
class CBFConfig:
    category_weight: float = 2.0
    genre_weight: float = 3.0
    year_decade_weight: float = 0.5
    director_weight: float = 1.0
    actor_weight: float = 1.0
    title_weight: float = 0.5
    include_title: bool = False
    profile_weighting: str = "log_counts"

    def weights(self) -> dict[str, float]:
        values = {
            "category": self.category_weight, "genre": self.genre_weight,
            "year_decade": self.year_decade_weight, "director": self.director_weight,
            "actor": self.actor_weight, "title": self.title_weight if self.include_title else 0.0,
        }
        if any(not math.isfinite(value) or value < 0 for value in values.values()) or not any(values.values()):
            raise ValueError("Feature weights must be finite, nonnegative, and not all zero.")
        if self.profile_weighting not in ("log_counts", "log_minutes"):
            raise ValueError("Unknown profile weighting rule.")
        return values


def clean(value: str) -> str:
    text = " ".join(unicodedata.normalize("NFKC", value).casefold().split())
    return "" if text in EMPTY_VALUES else text


def extract_metadata_features(row: dict[str, str], include_title: bool = False) -> dict[str, set[str]]:
    """Extract namespaced categorical attributes; never invent missing values."""
    groups = {field: set() for field in FIELDS}
    category = clean(row.get("tv_show_category", ""))
    if category:
        groups["category"].add(category)
    for key in ("tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3"):
        value = clean(row.get(key, ""))
        if value:
            groups["genre"].add(value)
    for field, key in (("director", "director"), ("actor", "actors")):
        value = clean(row.get(key, ""))
        if value:
            groups[field].update(name for part in PERSON_SPLIT.split(value) if (name := clean(part)))
    year = YEAR_PATTERN.search(row.get("year_of_production", ""))
    if year:
        groups["year_decade"].add(str(int(year.group()) // 10 * 10))
    if include_title:
        groups["title"].update(word for word in WORD_PATTERN.findall(clean(row.get("tv_show_title", ""))) if len(word) >= 3)
    return groups


def load_training_metadata(path: Path, config: CBFConfig) -> tuple[dict, dict]:
    """Union metadata only from airings positively overlapping the training window."""
    config.weights()
    metadata = {}
    counts = Counter(dict.fromkeys(("rows_read", "zero_id_rows_excluded", "labeled_rows_outside_training_window",
                                   "labeled_rows_in_training_window"), 0))
    missing = Counter({field: 0 for field in FIELDS})
    for number, row in checked_records(path, SCHEDULE_COLUMNS):
        counts["rows_read"] += 1
        programme = parse_integer(row["tv_show_id"])
        if programme is None or programme < 0:
            raise ValueError(f"Invalid metadata programme ID at data row {number}.")
        if programme == 0:
            counts["zero_id_rows_excluded"] += 1
            continue
        start = parse_timestamp(row["start_time"], schedule=True)
        duration, channel = parse_integer(row["duration"]), parse_integer(row["channel_id"])
        if start is None or duration is None or duration <= 0 or channel is None or channel <= 0:
            raise ValueError(f"Invalid labeled metadata airing at data row {number}.")
        low = seconds(start)
        if low >= WINDOWS[0].end or low + duration <= WINDOWS[0].start:
            counts["labeled_rows_outside_training_window"] += 1
            continue
        counts["labeled_rows_in_training_window"] += 1
        groups = extract_metadata_features(row, config.include_title)
        combined = metadata.setdefault(programme, {field: set() for field in FIELDS})
        for field in FIELDS:
            combined[field].update(groups[field])
            missing[field] += not groups[field]
    return metadata, {
        "file": path.name, "training_start_inclusive": timestamp(WINDOWS[0].start),
        "training_end_exclusive": timestamp(WINDOWS[0].end), **counts,
        "unique_programmes_with_training_schedule_rows": len(metadata),
        "labeled_training_rows_without_extracted_field": dict(missing),
        "title_field_disabled": not config.include_title,
        "metadata_aggregation": "union of distinct extracted values per programme within training timestamps",
    }


def normalize(vector: dict[int, float]) -> dict[int, float]:
    ordered = sorted(vector.items())
    norm = math.sqrt(math.fsum(value * value for _, value in ordered))
    return {feature: value / norm for feature, value in ordered if value != 0} if norm else {}


def cosine_similarity(left: dict[int, float], right: dict[int, float]) -> float:
    """Compute sparse cosine similarity, including the zero-vector convention."""
    norm_left = math.sqrt(math.fsum(value * value for _, value in sorted(left.items())))
    norm_right = math.sqrt(math.fsum(value * value for _, value in sorted(right.items())))
    if not norm_left or not norm_right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    return math.fsum(value * right.get(feature, 0.0) for feature, value in sorted(left.items())) / (norm_left * norm_right)


@dataclass
class ContentModel:
    vectors: dict[int, dict[int, float]]
    feature_keys: tuple[tuple[str, str], ...]
    idf: dict[int, float]
    statistics: dict


def fit_item_vectors(metadata: dict, config: CBFConfig) -> ContentModel:
    """Binary TF-IDF, per-field normalization, field weights, then item L2 norm."""
    weights = config.weights()
    document_frequency = Counter()
    for programme in sorted(metadata):
        for field in FIELDS:
            if weights[field] > 0:
                document_frequency.update((field, value) for value in sorted(metadata[programme][field]))
    feature_keys = tuple(sorted(document_frequency))
    feature_ids = {key: position for position, key in enumerate(feature_keys)}
    n_items = len(metadata)
    idf = {feature_ids[key]: math.log((1 + n_items) / (1 + frequency)) + 1.0
           for key, frequency in sorted(document_frequency.items())}
    vectors = {}
    for programme in sorted(metadata):
        vector = {}
        for field in FIELDS:
            if weights[field] <= 0:
                continue
            entries = [(feature_ids[(field, value)], idf[feature_ids[(field, value)]])
                       for value in sorted(metadata[programme][field])]
            field_norm = math.sqrt(math.fsum(value * value for _, value in entries))
            if field_norm:
                for feature, value in entries:
                    vector[feature] = weights[field] * value / field_norm
        vectors[programme] = normalize(vector)
    return ContentModel(vectors, feature_keys, idf, {
        "idf_training_documents": n_items,
        "vocabulary_features": len(feature_keys),
        "vocabulary_features_by_field": dict(Counter(field for field, _ in feature_keys)),
        "nonempty_item_vectors": sum(bool(vector) for vector in vectors.values()),
        "empty_item_vectors": sum(not vector for vector in vectors.values()),
        "item_vector_nonzero_entries": sum(map(len, vectors.values())),
        "idf_formula": "log((1 + number of training-metadata programmes) / (1 + programme document frequency)) + 1",
        "tf": "binary feature presence per programme, not broadcast frequency",
        "normalization": "L2-normalize each field's TF-IDF block, multiply field weight, then L2-normalize the full item",
        "sparse_representation": "standard-library dictionaries; no dense user-by-item or item-by-feature matrices",
    })


def build_user_profile(history, vectors: dict[int, dict[int, float]], config: CBFConfig) -> tuple[dict, dict]:
    config.weights()
    profile = defaultdict(float)
    stats = Counter({"training_programmes": 0, "programmes_contributing_features": 0,
                     "programmes_without_usable_metadata": 0})
    for row in history:
        stats["training_programmes"] += 1
        vector = vectors.get(row.programme_id)
        if not vector:
            stats["programmes_without_usable_metadata"] += 1
            continue
        stats["programmes_contributing_features"] += 1
        weight = math.log1p(row.matched_events) if config.profile_weighting == "log_counts" else math.log1p(row.overlap_seconds / 60.0)
        for feature, value in sorted(vector.items()):
            profile[feature] += weight * value
    result = normalize(profile)
    return result, {**stats, "profile_nonzero_features": len(result)}


class ContentIndex:
    """Sparse feature postings restricted to the unchanged availability pool."""

    def __init__(self, vectors: dict[int, dict[int, float]], candidates: frozenset[int]) -> None:
        self.candidates = candidates
        self.postings = defaultdict(list)
        for programme in sorted(candidates):
            for feature, value in sorted(vectors.get(programme, {}).items()):
                if value > 0:
                    self.postings[feature].append((programme, value))

    def score(self, profile: dict[int, float]) -> dict[int, float]:
        scores = defaultdict(float)
        for feature, value in sorted(profile.items()):
            for programme, item_value in self.postings.get(feature, ()):
                scores[programme] += value * item_value
        return dict(scores)


def recommend(profile: dict[int, float], index: ContentIndex, fallback: tuple[int, ...]) -> tuple[tuple[int, ...], dict]:
    validate_prediction(fallback, index.candidates)
    scores = index.score(profile) if profile else {}
    positive = {programme: score for programme, score in scores.items() if score > 0}
    content = heapq.nsmallest(K, positive, key=lambda programme: (-positive[programme], programme))
    chosen = list(content)
    for programme in fallback:
        if len(chosen) == K:
            break
        if programme not in chosen:
            chosen.append(programme)
    result = tuple(chosen)
    validate_prediction(result, index.candidates)
    return result, {
        "positive_similarity_candidates": len(positive), "content_slots": len(content),
        "popularity_fallback_slots": K - len(content),
        "top_five": [{"tv_show_id": programme, "source": "content" if rank < len(content) else "popularity_fallback",
                      "cosine_similarity": positive.get(programme, 0.0)} for rank, programme in enumerate(result)],
    }


def generate_predictions(training_path: Path, users: list[str], vectors: dict, candidates: frozenset[int],
                         fallback: tuple[int, ...], config: CBFConfig) -> tuple[dict, frozenset[str], dict]:
    """Build and discard one sparse user profile at a time from training rows."""
    validate_cohort(users)
    cohort = set(users)
    index = ContentIndex(vectors, candidates)
    predictions = {user: fallback for user in users}
    details = {user: {"training_programmes": 0, "programmes_contributing_features": 0,
                      "programmes_without_usable_metadata": 0, "profile_nonzero_features": 0,
                      "positive_similarity_candidates": 0, "content_slots": 0,
                      "popularity_fallback_slots": K, "reason": "no_training_history"} for user in users}
    reader = AggregateReader(training_path, WINDOWS[0])
    for user, group in itertools.groupby(reader, key=lambda row: row.user_id):
        if user not in cohort:
            for _ in group:
                pass
            continue
        profile, profile_stats = build_user_profile(group, vectors, config)
        predictions[user], ranking_stats = recommend(profile, index, fallback)
        details[user] = {**profile_stats, **ranking_stats,
                         "reason": "content_profile" if profile else "history_without_usable_metadata"}
    for user in users:
        validate_prediction(predictions[user], candidates)
    return predictions, frozenset(reader.users), {
        "training_data": reader.summary(users),
        "users_with_nonempty_profiles": sum(d["profile_nonzero_features"] > 0 for d in details.values()),
        "users_without_training_history": sum(d["reason"] == "no_training_history" for d in details.values()),
        "users_with_history_but_no_usable_metadata": sum(d["reason"] == "history_without_usable_metadata" for d in details.values()),
        "users_with_five_content_slots": sum(d["content_slots"] == K for d in details.values()),
        "users_requiring_any_fallback": sum(d["popularity_fallback_slots"] > 0 for d in details.values()),
        "users_with_entirely_fallback_lists": sum(d["popularity_fallback_slots"] == K for d in details.values()),
        "total_fallback_slots": sum(d["popularity_fallback_slots"] for d in details.values()),
        "per_user_profiles_and_ranking": details,
    }


def predict_validation(training_path: Path, metadata_path: Path, validation_schedule_path: Path,
                       template_path: Path, config: CBFConfig, expected_users: int | None = 1260) -> tuple:
    """Fit and predict without accepting any validation-interaction input."""
    users = load_template_users(template_path, expected_users)
    stages = {}
    started = time.perf_counter()
    metadata, metadata_stats = load_training_metadata(metadata_path, config)
    model = fit_item_vectors(metadata, config)
    del metadata
    stages["load_training_metadata_and_fit_vectors"] = time.perf_counter() - started
    started = time.perf_counter()
    popularity = fit_popularity(training_path)
    stages["fit_unchanged_training_popularity_fallback"] = time.perf_counter() - started
    started = time.perf_counter()
    candidates, candidate_stats = load_candidates(validation_schedule_path)
    fallback = rank_candidates(popularity.scores, candidates)
    stages["load_shared_validation_availability"] = time.perf_counter() - started
    started = time.perf_counter()
    predictions, training_users, diagnostics = generate_predictions(training_path, users, model.vectors, candidates, fallback, config)
    stages["stream_training_profiles_and_score_candidates"] = time.perf_counter() - started
    if training_users != popularity.training_users:
        raise AssertionError("CBF history cohort differs from the popularity history cohort.")
    metadata_candidate_ids = set(model.vectors) & candidates
    usable_candidate_ids = {item for item in metadata_candidate_ids if model.vectors[item]}
    result = {
        "configuration": asdict(config), "effective_field_weights": config.weights(),
        "metadata": metadata_stats, "item_vectors": model.statistics, "candidate_pool": candidate_stats,
        "metadata_candidate_coverage": {
            "candidate_programmes": len(candidates),
            "candidates_with_training_metadata": len(metadata_candidate_ids),
            "candidates_with_nonempty_content_vectors": len(usable_candidate_ids),
            "candidates_without_training_metadata": len(candidates - set(model.vectors)),
            "candidates_with_empty_vectors": len(metadata_candidate_ids - usable_candidate_ids),
            "unknown_metadata_policy": "no content vector; zero similarity; eligible only through the unchanged popularity fallback",
        },
        "profiles": diagnostics,
        "fallback_top_five": list(fallback), "stage_runtime_seconds": stages,
        "fit_sources": [str(training_path), str(metadata_path)],
        "validation_metadata_used_for_features": False,
        "validation_viewing_labels_used_for_fit_or_prediction": False,
        "profile_weight_formula": "log1p(matched_events)" if config.profile_weighting == "log_counts" else "log1p(overlap_seconds / 60)",
        "tie_break": "ascending numeric programme ID for exactly equal positive cosine scores",
        "repeat_recommendations_allowed": True,
        "random_seed": None, "randomness": "none; vocabulary, feature traversal, and candidate ordering are deterministic",
        "hyperparameter_selection": "single prespecified configuration; no validation-label tuning or model search",
    }
    return result, predictions, candidates, training_users


GROUPS = ("primary_all_users", "supplementary_nonempty_ground_truth", "with_training_history", "without_training_history")


def compare_metrics(cbf: dict, popularity: dict) -> dict:
    comparison = {}
    for group in GROUPS:
        actual, reference = cbf[group]["map_at_5"], popularity[group]["map_at_5"]
        if cbf[group]["users"] != popularity[group]["users"]:
            raise AssertionError(f"Comparison cohorts differ: {group}")
        change = actual - reference if actual is not None and reference is not None else None
        relative = change / reference if change is not None and reference else None
        comparison[group] = {"users": cbf[group]["users"], "popularity_map_at_5": reference,
                             "cbf_map_at_5": actual, "cbf_minus_popularity": change,
                             "absolute_difference_magnitude": abs(change) if change is not None else None,
                             "relative_change": relative,
                             "relative_change_percent": 100 * relative if relative is not None else None}
    return comparison


def render_report(report: dict) -> str:
    cbf = report["cbf"]
    config = cbf["configuration"]
    command = ".venv/bin/python src/content_based.py"
    if config["include_title"]:
        command += " --include-title"
    if config["profile_weighting"] != "log_counts":
        command += f" --profile-weighting {config['profile_weighting']}"
    lines = ["# SWEET.TV Content-Based Filtering Validation", "", f"Generated (UTC): {report['generated_at_utc']}", "",
             "## Reproduction", "", "```bash", command, "```", "",
             "Uses only the Python standard library. The CLI runs the complete unit-test suite before fitting.", "",
             "## Exact local MAP@5 comparison", "",
             "Values below use the full stored float representations, without fixed-decimal rounding.", "",
             "| Cohort | Users | Popularity MAP@5 | CBF MAP@5 | CBF minus Popularity | Relative change (%) |",
             "|---|---:|---:|---:|---:|---:|"]
    for group, values in report["comparison"].items():
        lines.append(f"| {group.replace('_', ' ')} | {values['users']} | {values['popularity_map_at_5']!r} | "
                     f"{values['cbf_map_at_5']!r} | {values['cbf_minus_popularity']!r} | {values['relative_change_percent']!r} |")
    change = report["comparison"]["primary_all_users"]["cbf_minus_popularity"]
    lines += ["", f"CBF {'underperforms' if change < 0 else 'outperforms' if change > 0 else 'ties'} Popularity on primary local MAP@5.", "",
              "Absolute difference magnitude is abs(CBF - Popularity); relative change is (CBF - Popularity) / Popularity. "
              "The machine-readable report includes both the magnitude and signed changes.", "",
              "## Fixed evaluation protocol", "", "```json", json.dumps(report["validation_protocol"], indent=2), "```", "",
              "This is the unchanged local convention, not a verified reproduction of Kaggle's hidden evaluator. "
              "The same 1,260 users, availability pool, unique validation labels, repeat policy, and popularity fallback are used.", "",
              "## Feature construction and model parameters", "",
              "- Source: the weeks 11–20 schedule only, filtered to positive airing overlap with [2020-03-09, 2020-05-18). Filename padding outside the window is excluded.",
              "- Normalize Unicode with NFKC, case-fold, and collapse whitespace. Blank/common missing markers create no features. Source-language values are not translated.",
              "- Category and the union of the three genre fields are categorical features. Director/actor lists split on comma, semicolon, or vertical bar.",
              "- Production decade comes from the first 1800–2099 four-digit year in the field; ranges are not expanded. Title tokens are optional and disabled by default to avoid episode-label noise.",
              "- Namespace features by field; union distinct values across each programme's training airings. Broadcast repetition does not increase term frequency.",
              "- Fit binary TF-IDF on training-metadata programmes only. Normalize each field before weighting so long cast lists do not dominate solely by length; then normalize the whole item.",
              "- Build a user vector as an interaction-weighted sum of watched training item vectors, then L2-normalize. Already-watched candidates remain eligible.",
              "- Score cosine similarity through a sparse inverted index. Exact ties use numeric programme ID. No popularity score fusion or hybrid is used.",
              "- Rank positive content scores; fill short lists from the baseline's identical Top-5, skipping duplicates. Empty profiles receive exactly that fallback.", "",
              "```json", json.dumps({key: value for key, value in cbf.items() if key not in ("profiles", "evaluation")}, indent=2), "```", "",
              "## User profiles, fallback, and coverage", "", "```json",
              json.dumps({"profiles": {key: value for key, value in cbf["profiles"].items() if key != "per_user_profiles_and_ranking"},
                          "cbf_coverage": cbf["evaluation"]["metrics"]["coverage"],
                          "popularity_coverage": report["popularity_reference_metrics"]["coverage"],
                          "ground_truth_candidate_coverage": cbf["evaluation"]["metrics"]["ground_truth_candidate_coverage"]}, indent=2), "```", "",
              "The JSON report preserves per-user AP@5, profile diagnostics, and ranking provenance without saving full user vectors.", "",
              "## Runtime and environment", "", "```json", json.dumps(report["execution"], indent=2), "```", "",
              "## Verification", "", "```json", json.dumps(report["verification"], indent=2), "```", "",
              "### Protected artifacts", "", "| File | SHA-256 | Unchanged |", "|---|---|---|"]
    for name, details in report["input_integrity"].items():
        lines.append(f"| {name} | `{details['before']['sha256']}` | {details['unchanged']} |")
    lines += ["", "### Saved CBF validation predictions", "", "```json", json.dumps(report["prediction_file"], indent=2), "```", "",
              "### Tests", "", "```text", report["tests"]["output"].rstrip(), "```", "",
              "## Limitations and interpretation", "",
              "- No validation-period or test-period descriptive metadata is used. Candidate IDs lacking training metadata cannot receive content scores, even if they have rich later metadata.",
              "- Per-field normalization reduces cast-length effects, but generic categories/genres and metadata aliases can still produce weak or tied content matches.",
              "- Unioning metadata across training airings can mix episode casts or title variants. Name splitting is heuristic and does not resolve aliases or infer enjoyment.",
              "- The configured field weights and profile rule are prespecified, not validated optima. This run does not claim a tuning search or unperformed improvements.",
              "- Larger catalogue coverage does not prove greater relevance. A worse measured MAP@5 is reported as worse, not relabeled as better discovery.",
              "- The 120-second watching threshold, sparse labels, missing channel mappings, 117 uncorrected duration discrepancies, and candidate/ground-truth mismatch remain inherited limitations.",
              "- Cold-start is defined from usable training interactions, not from validation labels. Warm users with no metadata profile receive the same fallback but remain in the warm evaluation group.",
              "- Protected files include all original data, processed outputs, existing reports, popularity predictions, and the unchanged evaluation/baseline source files.",
              "- Runtime includes tests, checksums, metadata/vector fitting, fallback fitting, profile construction/scoring, validation evaluation, and saved-output verification. Final report serialization is excluded. Peak RSS is for the main process, not the test subprocess.",
              "- No association rules, hybrid, validation-data retraining, final Kaggle submission, or Kaggle upload was performed.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-title", action="store_true")
    parser.add_argument("--profile-weighting", choices=("log_counts", "log_minutes"), default="log_counts")
    args = parser.parse_args()
    config = CBFConfig(include_title=args.include_title, profile_weighting=args.profile_weighting)
    root = Path(__file__).resolve().parents[1]
    raw, processed, reports, outputs = (root / p for p in ("data/raw", "data/processed", "reports", "outputs"))
    output_path = outputs / "cbf_validation.csv"
    for path in (output_path, reports / "cbf_validation.md", reports / "cbf_validation.json"):
        if path.is_symlink():
            parser.error(f"Refusing to overwrite a symbolic-link output: {path}")
    started = time.perf_counter()
    generated = datetime.now(timezone.utc).isoformat()
    baseline = json.loads((reports / "popularity_baseline.json").read_text(encoding="utf-8"))
    if baseline["validation_protocol"] != PROTOCOL:
        raise ValueError("The saved baseline and current evaluator protocols differ.")
    protected_names = set(baseline["input_integrity"]) | set(baseline["execution"]["source_sha256"])
    protected_names.update({"reports/popularity_baseline.md", "reports/popularity_baseline.json", "outputs/popularity_validation.csv"})
    protected_paths = [root / name for name in sorted(protected_names)]
    before = {str(path.relative_to(root)): file_fingerprint(path) for path in protected_paths}
    for name, old in baseline["input_integrity"].items():
        if before[name]["sha256"] != old["after"]["sha256"]:
            raise RuntimeError(f"Protected input differs from the preserved baseline: {name}")
    for name, digest in baseline["execution"]["source_sha256"].items():
        if before[name]["sha256"] != digest:
            raise RuntimeError(f"Evaluation/baseline source changed: {name}")
    if before["outputs/popularity_validation.csv"]["sha256"] != baseline["prediction_file"]["sha256"]:
        raise RuntimeError("The saved popularity predictions changed.")
    baseline_metrics = baseline["baseline"]["evaluation"]["metrics"]
    print("Verified exact saved Popularity MAP@5 values:", flush=True)
    for group in GROUPS:
        print(f"  {group}: {baseline_metrics[group]['map_at_5']!r}", flush=True)
    test_start = time.perf_counter()
    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
    tests = subprocess.run(command, cwd=root, capture_output=True, text=True)
    test_output = tests.stdout + tests.stderr
    print(test_output, end="", flush=True)
    if tests.returncode:
        raise RuntimeError("Tests failed; actual CBF fitting was not started.")
    test_result = {"command": command, "returncode": tests.returncode, "output": test_output,
                   "runtime_seconds": time.perf_counter() - test_start}
    result, predictions, candidates, training_users = predict_validation(
        processed / "train_user_programmes.csv", raw / SCHEDULE_FILES[0], raw / SCHEDULE_FILES[1],
        raw / "submission.csv", config)
    users = list(predictions)
    if result["candidate_pool"] != baseline["baseline"]["candidate_pool"]:
        raise AssertionError("CBF candidate availability differs from the baseline.")
    baseline_predictions = read_predictions(outputs / "popularity_validation.csv", users, candidates)
    fallback = tuple(result["fallback_top_five"])
    if any(value != fallback for value in baseline_predictions.values()):
        raise AssertionError("CBF fallback differs from the saved popularity baseline.")
    if any(predictions[user] != fallback for user in users if user not in training_users):
        raise AssertionError("Cold-start users did not receive the unchanged popularity fallback.")
    # Predictions are now fixed. Only now may validation viewing labels be loaded.
    evaluation_start = time.perf_counter()
    truth, validation_stats = load_validation_ground_truth(processed / "validation_user_programmes.csv", users)
    reference_evaluation = evaluate_recommendations(users, truth, baseline_predictions, training_users, candidates)
    if reference_evaluation != baseline["baseline"]["evaluation"]:
        raise AssertionError("Shared ground truth, user cohorts, or baseline metrics changed.")
    result["evaluation"] = evaluate_recommendations(users, truth, predictions, training_users, candidates)
    result["validation_data_for_scoring_only"] = validation_stats
    result["stage_runtime_seconds"]["validation_labels_and_evaluation"] = time.perf_counter() - evaluation_start
    write_predictions(output_path, users, predictions, candidates)
    saved = read_predictions(output_path, users, candidates)
    if saved != predictions or evaluate_recommendations(users, truth, saved, training_users, candidates) != result["evaluation"]:
        raise AssertionError("Saved CBF predictions or metrics differ from memory.")
    after = {str(path.relative_to(root)): file_fingerprint(path) for path in protected_paths}
    integrity = {name: {"before": value, "after": after[name], "unchanged": value == after[name]} for name, value in before.items()}
    if not all(item["unchanged"] for item in integrity.values()):
        raise RuntimeError("Protected artifact changed; successful CBF reports were not written.")
    report = {
        "schema_version": 1, "generated_at_utc": generated, "validation_protocol": PROTOCOL,
        "cbf": result, "popularity_reference_metrics": baseline_metrics,
        "comparison": compare_metrics(result["evaluation"]["metrics"], baseline_metrics),
        "input_integrity": integrity,
        "prediction_file": {"path": str(output_path.relative_to(root)), **file_fingerprint(output_path)},
        "tests": test_result,
        "verification": {
            "all_tests_passed": tests.returncode == 0, "all_protected_artifacts_unchanged": True,
            "shared_candidate_pool_and_exact_baseline_evaluation_reproduced": True,
            "same_template_order_and_user_id_strings": True,
            "all_users_have_five_unique_positive_available_ids": len(users) == 1260,
            "identical_popularity_fallback_for_all_cold_start_users": True,
            "saved_predictions_and_metrics_recomputed_and_equal": True,
            "feature_source": f"data/raw/{SCHEDULE_FILES[0]} filtered to training timestamps",
            "history_source": "data/processed/train_user_programmes.csv",
            "validation_schedule_role": "availability only, through unchanged evaluate.load_candidates",
            "validation_viewing_role": "evaluation only, after predictions were fixed",
            "test_data_used_for_fitting": False,
        },
        "execution": {
            "python_version": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
            "invocation": sys.argv, "third_party_packages_used": [],
            "source_sha256": {str(path.relative_to(root)): file_fingerprint(path)["sha256"] for path in
                              (root / "src/content_based.py", root / "tests/test_content_based.py", root / "src/evaluate.py", root / "src/popularity.py")},
            "stage_runtime_seconds": {**result["stage_runtime_seconds"], "unit_tests": test_result["runtime_seconds"]},
            "runtime_seconds_before_report_serialization": time.perf_counter() - started, **peak_memory(),
        },
    }
    (reports / "cbf_validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (reports / "cbf_validation.md").write_text(render_report(report), encoding="utf-8")
    print(json.dumps(report["comparison"], indent=2), flush=True)
    print(f"All protected artifacts unchanged. Runtime including report writes: {time.perf_counter() - started:.3f} seconds")
    print(f"Peak main-process RSS: {report['execution'].get('peak_process_rss_mib', 'unavailable')} MiB")


if __name__ == "__main__":
    main()
