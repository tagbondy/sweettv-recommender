"""Shared deterministic local MAP@5 protocol and validated streaming inputs.

This is the project's local evaluation convention, not a verified reproduction of
Kaggle's hidden evaluator. Empty validation ground truth contributes AP@5 = 0 to
the primary cohort mean. Recommendations must contain exactly five unique positive
integer IDs. Ground truth is never filtered to a model's supported items.
"""

from __future__ import annotations

import csv
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from audit_data import SCHEDULE_COLUMNS, parse_integer, parse_timestamp
from build_interactions import PAIR_COLUMNS, WINDOWS, Window, checked_records, seconds, timestamp


K = 5
PREDICTION_COLUMNS = ["user_id", "tv_show_id"]
VERIFIED_PAIR_TYPES = {
    "user_id": "original string, never converted to a number",
    "tv_show_id": "positive integer",
    "matched_events": "positive integer",
    "overlap_seconds": "positive integer seconds",
    "first_overlap": "naive YYYY-MM-DD HH:MM:SS timestamp",
    "last_overlap": "naive YYYY-MM-DD HH:MM:SS timestamp",
}


def validate_user_id(user: str) -> None:
    if not isinstance(user, str) or not user or user != user.strip():
        raise ValueError("User IDs must be nonempty original strings without surrounding whitespace.")


def validate_programme_id(programme: int) -> None:
    if type(programme) is not int or programme <= 0:
        raise ValueError("Programme IDs must be positive integers; zero, booleans, and floats are invalid.")


def validate_prediction(prediction: Sequence[int], candidates: set[int] | frozenset[int] | None = None) -> None:
    if isinstance(prediction, (str, bytes)) or not isinstance(prediction, Sequence) or len(prediction) != K:
        raise ValueError("Every recommendation list must contain exactly five programme IDs.")
    for programme in prediction:
        validate_programme_id(programme)
    if len(set(prediction)) != K:
        raise ValueError("Recommendations must contain five unique programme IDs.")
    if candidates is not None and not set(prediction) <= candidates:
        raise ValueError("A recommendation is outside the shared candidate pool.")


def average_precision_at_5(relevant: Iterable[int], prediction: Sequence[int]) -> float:
    """Return sum(P@rank for relevant ranks) / min(5, unique relevant count)."""
    validate_prediction(prediction)
    relevant = set(relevant)
    for programme in relevant:
        validate_programme_id(programme)
    if not relevant:
        return 0.0
    hits = 0
    precisions = []
    for rank, programme in enumerate(prediction, 1):
        if programme in relevant:
            hits += 1
            precisions.append(hits / rank)
    return math.fsum(precisions) / min(K, len(relevant))


def validate_cohort(users: Sequence[str]) -> None:
    if isinstance(users, (str, bytes)) or not users:
        raise ValueError("The evaluation cohort must be a nonempty sequence of user IDs.")
    for user in users:
        validate_user_id(user)
    if len(set(users)) != len(users):
        raise ValueError("The evaluation cohort contains duplicate user IDs.")


def evaluate_recommendations(
    users: Sequence[str],
    ground_truth: Mapping[str, Iterable[int]],
    predictions: Mapping[str, Sequence[int]],
    training_users: set[str] | frozenset[str],
    candidates: set[int] | frozenset[int] | None = None,
) -> dict:
    """Evaluate every cohort user, including cold-start and empty-label users."""
    validate_cohort(users)
    cohort = set(users)
    if set(predictions) != cohort:
        raise ValueError("Predictions must cover exactly the evaluation cohort, without missing or extra users.")
    if set(ground_truth) - cohort:
        raise ValueError("Ground truth must be restricted to the explicitly selected evaluation cohort.")
    for user in training_users:
        validate_user_id(user)
    per_user, relevant_sets = {}, {}
    for user in users:
        validate_prediction(predictions[user], candidates)
        relevant = set(ground_truth.get(user, ()))
        relevant_sets[user] = relevant
        per_user[user] = {
            "ap_at_5": average_precision_at_5(relevant, predictions[user]),
            "relevant_programme_count": len(relevant),
            "has_training_history": user in training_users,
            "hits_at_5": len(relevant & set(predictions[user])),
        }

    def summarize(selected: list[str]) -> dict:
        n = len(selected)
        return {
            "users": n,
            "users_with_validation_labels": sum(per_user[u]["relevant_programme_count"] > 0 for u in selected),
            "users_with_at_least_one_hit": sum(per_user[u]["hits_at_5"] > 0 for u in selected),
            "map_at_5": math.fsum(per_user[u]["ap_at_5"] for u in selected) / n if n else None,
        }

    warm = [u for u in users if u in training_users]
    cold = [u for u in users if u not in training_users]
    nonempty = [u for u in users if per_user[u]["relevant_programme_count"] > 0]
    recommended = set().union(*(set(predictions[u]) for u in users))
    metrics = {
        "k": K,
        "primary_all_users": summarize(list(users)),
        "supplementary_nonempty_ground_truth": summarize(nonempty),
        "with_training_history": summarize(warm),
        "without_training_history": summarize(cold),
        "with_training_history_and_validation_labels": summarize([u for u in warm if per_user[u]["relevant_programme_count"] > 0]),
        "without_training_history_with_validation_labels": summarize([u for u in cold if per_user[u]["relevant_programme_count"] > 0]),
        "coverage": {
            "users_with_valid_top_5": len(users), "user_recommendation_coverage": 1.0,
            "unique_programmes_recommended": len(recommended),
            "candidate_catalogue_coverage": len(recommended) / len(candidates) if candidates else None,
            "users_with_at_least_one_hit_fraction": sum(per_user[u]["hits_at_5"] > 0 for u in users) / len(users),
        },
    }
    if candidates is not None:
        outside = {u: relevant - candidates for u, relevant in relevant_sets.items()}
        metrics["ground_truth_candidate_coverage"] = {
            "user_programme_relevance_pairs": sum(map(len, relevant_sets.values())),
            "relevance_pairs_outside_candidates": sum(map(len, outside.values())),
            "users_with_relevance_outside_candidates": sum(bool(values) for values in outside.values()),
            "programme_ids_outside_candidates": sorted(set().union(*outside.values())),
            "ground_truth_was_filtered_to_candidates": False,
        }
    return {"metrics": metrics, "per_user": per_user}


def load_template_users(path: Path, expected_users: int | None = 1260) -> list[str]:
    users, seen = [], set()
    for number, row in checked_records(path, PREDICTION_COLUMNS):
        user = row["user_id"]
        validate_user_id(user)
        if user in seen:
            raise ValueError(f"Duplicate template user at data row {number}.")
        tokens = row["tv_show_id"].split()
        if len(tokens) != K or any(parse_integer(token) is None or int(token) < 0 for token in tokens):
            raise ValueError("Invalid submission-template placeholders.")
        users.append(user)
        seen.add(user)
    validate_cohort(users)
    if expected_users is not None and len(users) != expected_users:
        raise ValueError(f"Expected {expected_users} template users, found {len(users)}.")
    return users


@dataclass(frozen=True, slots=True)
class AggregateRecord:
    user_id: str
    programme_id: int
    matched_events: int
    overlap_seconds: int
    first_overlap: str
    last_overlap: str


class AggregateReader:
    """Validate canonical sorted aggregate CSVs without retaining their rows.

    Adjacent duplicate user-programme keys are counted and yielded once. Unsorted
    keys fail explicitly, preventing incorrect distinct-user popularity counts.
    The three actual processed files were verified to have sorted, unique keys.
    """

    def __init__(self, path: Path, window: Window, minimum_overlap: int = 120) -> None:
        self.path, self.window, self.minimum_overlap = path, window, minimum_overlap
        self.users: set[str] = set()
        self.programmes: set[int] = set()
        self.rows = self.unique_pairs = self.duplicates = self.total_events = self.total_seconds = 0
        self.first = self.last = None
        self.exhausted = False

    def __iter__(self):
        previous = None
        for number, row in checked_records(self.path, PAIR_COLUMNS):
            self.rows += 1
            user = row["user_id"]
            validate_user_id(user)
            programme = parse_integer(row["tv_show_id"])
            events = parse_integer(row["matched_events"])
            duration = parse_integer(row["overlap_seconds"])
            first, last = parse_timestamp(row["first_overlap"]), parse_timestamp(row["last_overlap"])
            if (programme is None or programme <= 0 or events is None or events <= 0 or duration is None
                    or duration < self.minimum_overlap * events or first is None or last is None):
                raise ValueError(f"Invalid aggregate values in {self.path.name}, data row {number}.")
            if not self.window.start <= seconds(first) < seconds(last) <= self.window.end:
                raise ValueError(f"Aggregate timestamp outside {self.window.name}: {self.path.name}, data row {number}.")
            key = (user, programme)
            if previous is not None and key < previous:
                raise ValueError(f"Aggregate keys are not sorted: {self.path.name}, data row {number}.")
            if key == previous:
                self.duplicates += 1
                continue
            previous = key
            self.unique_pairs += 1
            self.users.add(user)
            self.programmes.add(programme)
            self.total_events += events
            self.total_seconds += duration
            self.first = row["first_overlap"] if self.first is None else min(self.first, row["first_overlap"])
            self.last = row["last_overlap"] if self.last is None else max(self.last, row["last_overlap"])
            yield AggregateRecord(user, programme, events, duration, row["first_overlap"], row["last_overlap"])
        self.exhausted = True

    def summary(self, cohort: Iterable[str] = ()) -> dict:
        if not self.exhausted:
            raise ValueError("The aggregate file must be fully scanned before reporting statistics.")
        cohort = set(cohort)
        return {
            "file": self.path.name, "columns": PAIR_COLUMNS, "verified_types": VERIFIED_PAIR_TYPES,
            "rows_read": self.rows, "unique_user_programme_pairs": self.unique_pairs,
            "duplicate_keys_ignored": self.duplicates, "sorted_keys_verified": True,
            "unique_users": len(self.users), "unique_programmes": len(self.programmes),
            "template_users_present": len(cohort & self.users),
            "matched_events": self.total_events, "overlap_seconds": self.total_seconds,
            "first_overlap_min": self.first, "last_overlap_max": self.last,
            "window_start_inclusive": timestamp(self.window.start), "window_end_exclusive": timestamp(self.window.end),
        }


def load_validation_ground_truth(path: Path, users: Sequence[str]) -> tuple[dict[str, set[int]], dict]:
    validate_cohort(users)
    truth = {user: set() for user in users}
    reader = AggregateReader(path, WINDOWS[1])
    for row in reader:
        if row.user_id in truth:
            truth[row.user_id].add(row.programme_id)
    return truth, reader.summary(users)


def load_candidates(path: Path, window: Window = WINDOWS[1]) -> tuple[frozenset[int], dict]:
    """Use labeled schedule availability only, never validation viewing labels."""
    candidates = set()
    counts = Counter(dict.fromkeys(("rows_read", "zero_id_rows_excluded", "nonzero_airings_outside_window",
                                   "valid_labeled_airings_overlapping_window"), 0))
    for number, row in checked_records(path, SCHEDULE_COLUMNS):
        counts["rows_read"] += 1
        programme = parse_integer(row["tv_show_id"])
        if programme is None or programme < 0:
            raise ValueError(f"Invalid programme ID in schedule data row {number}.")
        if programme == 0:
            counts["zero_id_rows_excluded"] += 1
            continue
        start = parse_timestamp(row["start_time"], schedule=True)
        duration, channel = parse_integer(row["duration"]), parse_integer(row["channel_id"])
        if start is None or duration is None or duration <= 0 or channel is None or channel <= 0:
            raise ValueError(f"Invalid labeled schedule interval at data row {number}.")
        low = seconds(start)
        if low >= window.end or low + duration <= window.start:
            counts["nonzero_airings_outside_window"] += 1
            continue
        candidates.add(programme)
        counts["valid_labeled_airings_overlapping_window"] += 1
    return frozenset(candidates), {
        "file": path.name, "window_start_inclusive": timestamp(window.start),
        "window_end_exclusive": timestamp(window.end), **counts,
        "unique_candidate_programmes": len(candidates), "viewing_labels_used": False,
        "availability_rule": "positive programme ID and positive airing overlap with the half-open window",
    }


def write_predictions(path: Path, users: Sequence[str], predictions: Mapping[str, Sequence[int]], candidates: frozenset[int]) -> None:
    validate_cohort(users)
    if set(predictions) != set(users):
        raise ValueError("Predictions must cover exactly the cohort.")
    for user in users:
        validate_prediction(predictions[user], candidates)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(PREDICTION_COLUMNS)
        for user in users:
            writer.writerow([user, " ".join(map(str, predictions[user]))])


def read_predictions(path: Path, users: Sequence[str], candidates: frozenset[int]) -> dict[str, tuple[int, ...]]:
    result = {}
    for _, row in checked_records(path, PREDICTION_COLUMNS):
        user = row["user_id"]
        validate_user_id(user)
        if user in result:
            raise ValueError("Duplicate prediction user.")
        tokens = row["tv_show_id"].split()
        prediction = tuple(parse_integer(token) for token in tokens)
        validate_prediction(prediction, candidates)
        result[user] = prediction
    if list(result) != list(users):
        raise ValueError("Prediction CSV must preserve the complete template user order and strings.")
    return result
