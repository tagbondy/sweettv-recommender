"""Run a prespecified, training-only pairwise association-rule validation study.

Reproduce after the independent diagnostic: .venv/bin/python src/association_rules.py

All supports/confidences refer to the same capped user baskets. Counts are exact
within that approximation. SQLite stores pair counts; only bounded pair batches
and pruned neighbor lists are held in memory. No popularity score fusion is used.
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import json
import math
import platform
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from audit_data import SCHEDULE_FILES, peak_memory
from build_interactions import WINDOWS, file_fingerprint
from evaluate import (
    K, AggregateReader, evaluate_recommendations, load_candidates, load_template_users,
    load_validation_ground_truth, read_predictions, validate_cohort,
    validate_prediction, validate_programme_id, write_predictions,
)
from popularity import PROTOCOL, fit_popularity, rank_candidates


MAX_ITEMS_PER_USER = 50
PAIR_BATCH_LIMIT = 50000


@dataclass(frozen=True, slots=True)
class BasketItem:
    programme_id: int
    log_view_count: float


@dataclass(frozen=True)
class RuleConfig:
    name: str
    scoring: str = "confidence_sum"
    min_pair_support: int = 5
    min_confidence: float = 0.05
    min_lift: float = 0.0
    shrinkage: float = 0.0
    max_neighbors: int = 100

    def validate(self) -> None:
        if self.scoring not in ("confidence_sum", "confidence_max", "weighted_confidence_sum", "weighted_lift_sum"):
            raise ValueError("Unknown association scoring variant.")
        if (self.min_pair_support < 1 or self.max_neighbors < 1 or not 0 <= self.min_confidence <= 1
                or not math.isfinite(self.min_lift) or self.min_lift < 0
                or not math.isfinite(self.shrinkage) or self.shrinkage < 0):
            raise ValueError("Invalid rule thresholds or neighbor bound.")


# Fixed before examining any association validation results; all four are recorded.
EXPERIMENTS = (
    RuleConfig("confidence_sum", "confidence_sum"),
    RuleConfig("confidence_max", "confidence_max"),
    RuleConfig("weighted_confidence", "weighted_confidence_sum", min_pair_support=20, shrinkage=10.0),
    RuleConfig("weighted_lift", "weighted_lift_sum", min_pair_support=20, min_lift=1.0, shrinkage=10.0),
)


def select_basket(history, max_items: int = MAX_ITEMS_PER_USER) -> tuple[BasketItem, ...]:
    """Retain unique top items by events, seconds, then ascending programme ID.

    Duplicate aggregate keys use their strongest row, never a sum of duplicates.
    The production AggregateReader also independently deduplicates sorted keys.
    """
    if max_items < 1:
        raise ValueError("The basket cap must be positive.")
    unique = {}
    for row in history:
        validate_programme_id(row.programme_id)
        old = unique.get(row.programme_id)
        if old is None or (row.matched_events, row.overlap_seconds) > (old.matched_events, old.overlap_seconds):
            unique[row.programme_id] = row
    best = heapq.nsmallest(max_items, unique.values(), key=lambda row: (-row.matched_events, -row.overlap_seconds, row.programme_id))
    return tuple(BasketItem(row.programme_id, math.log1p(row.matched_events)) for row in sorted(best, key=lambda row: row.programme_id))


def size_summary(values: list[int]) -> dict:
    return {"users": len(values), "min": min(values) if values else None,
            "median": statistics.median(values) if values else None,
            "mean": statistics.mean(values) if values else None, "max": max(values) if values else None}


def load_training_baskets(path: Path, max_items: int = MAX_ITEMS_PER_USER) -> tuple[dict, dict]:
    reader = AggregateReader(path, WINDOWS[0])
    baskets = {}
    original_sizes = []
    canonical_ids = {}
    for user, group in itertools.groupby(reader, key=lambda row: row.user_id):
        rows = list(group)  # Only one user's unique history, never the whole CSV.
        original_sizes.append(len(rows))
        selected = select_basket(rows, max_items)
        baskets[user] = tuple(BasketItem(canonical_ids.setdefault(item.programme_id, item.programme_id), item.log_view_count)
                              for item in selected)
    retained_sizes = [len(basket) for basket in baskets.values()]
    return baskets, {
        "training_data": reader.summary(), "max_items_per_user": max_items,
        "basket_selection": "descending matched_events, descending overlap_seconds, ascending numeric programme ID",
        "history_weight": "log1p(matched_events) for weighted scoring variants only",
        "original_history_sizes": size_summary(original_sizes), "retained_basket_sizes": size_summary(retained_sizes),
        "users_with_truncated_histories": sum(size > max_items for size in original_sizes),
        "retained_user_programme_pairs": sum(retained_sizes),
        "original_user_programme_pairs": sum(original_sizes),
        "pair_updates_without_truncation": sum(size * (size - 1) // 2 for size in original_sizes),
        "pair_updates_with_truncation": sum(size * (size - 1) // 2 for size in retained_sizes),
        "approximation": "rules and query seeds use only retained items; dropped items cannot contribute association evidence",
    }


class AssociationCounts:
    """Exact distinct-user supports/co-occurrences for bounded unique baskets."""

    def __init__(self, baskets: dict[str, tuple[BasketItem, ...]], database: Path,
                 max_items: int = MAX_ITEMS_PER_USER, batch_limit: int = PAIR_BATCH_LIMIT) -> None:
        if database.exists():
            raise FileExistsError("Refusing to reuse a pair-count database.")
        self.connection = sqlite3.connect(database)
        self.connection.execute("PRAGMA journal_mode=OFF")
        self.connection.execute("PRAGMA synchronous=OFF")
        self.connection.execute("PRAGMA cache_size=-8192")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.execute("CREATE TABLE pairs (left_id INTEGER, right_id INTEGER, users INTEGER, PRIMARY KEY(left_id, right_id)) WITHOUT ROWID")
        self.support = Counter()
        self.user_count = len(baskets)
        batch = Counter()
        pair_updates = flushes = peak_batch = 0

        def flush():
            nonlocal flushes
            self.connection.executemany("""
                INSERT INTO pairs VALUES (?, ?, ?)
                ON CONFLICT(left_id, right_id) DO UPDATE SET users = users + excluded.users
            """, ((left, right, count) for (left, right), count in sorted(batch.items())))
            self.connection.commit()
            batch.clear()
            flushes += 1

        try:
            for user in sorted(baskets):
                ids = sorted({item.programme_id for item in baskets[user]})
                if len(ids) > max_items:
                    raise ValueError("A mining basket exceeds the configured cap.")
                for item in ids:
                    validate_programme_id(item)
                self.support.update(ids)
                batch.update(itertools.combinations(ids, 2))
                pair_updates += len(ids) * (len(ids) - 1) // 2
                peak_batch = max(peak_batch, len(batch))
                if len(batch) >= batch_limit:
                    flush()
            if batch:
                flush()
            self.statistics = {
                "training_user_baskets": self.user_count, "programmes_retained_for_mining": len(self.support),
                "distinct_undirected_pairs": self.connection.execute("SELECT COUNT(*) FROM pairs").fetchone()[0],
                "distinct_user_pair_updates": pair_updates,
                "pair_batch_target_distinct_keys": batch_limit, "peak_pair_batch_distinct_keys": peak_batch,
                "sqlite_flushes": flushes, "sqlite_database_bytes": database.stat().st_size,
                "support_scope": "users whose capped basket contains the programme",
                "cooccurrence_scope": "users whose same capped basket contains both distinct programmes",
                "confidence_formula": "pair_support(a,b) / capped_basket_support(a)",
                "lift_formula": "pair_support(a,b) * number_of_training_baskets / (support(a) * support(b))",
            }
        except BaseException:
            self.connection.close()
            raise

    def pair_count(self, left: int, right: int) -> int:
        if left == right:
            return 0
        a, b = sorted((left, right))
        row = self.connection.execute("SELECT users FROM pairs WHERE left_id=? AND right_id=?", (a, b)).fetchone()
        return row[0] if row else 0

    def confidence(self, source: int, target: int) -> float:
        return self.pair_count(source, target) / self.support[source] if self.support[source] else 0.0

    def lift(self, source: int, target: int) -> float:
        denominator = self.support[source] * self.support[target]
        return self.pair_count(source, target) * self.user_count / denominator if denominator else 0.0

    def close(self) -> None:
        self.connection.close()


def build_rules(counts: AssociationCounts, config: RuleConfig) -> tuple[dict, dict]:
    """Learn/prune rules from training counts without inspecting candidates/labels."""
    config.validate()
    heaps = defaultdict(list)
    eligible_pairs = eligible_directions = 0
    query = "SELECT left_id, right_id, users FROM pairs WHERE users >= ? ORDER BY left_id, right_id"
    for left, right, pair_count in counts.connection.execute(query, (config.min_pair_support,)):
        eligible_pairs += 1
        lift = pair_count * counts.user_count / (counts.support[left] * counts.support[right])
        if lift < config.min_lift:
            continue
        shrink = pair_count / (pair_count + config.shrinkage)
        for source, target in ((left, right), (right, left)):
            confidence = pair_count / counts.support[source]
            if confidence < config.min_confidence:
                continue
            strength = (lift if config.scoring == "weighted_lift_sum" else confidence) * shrink
            eligible_directions += 1
            item = (strength, -target)
            if len(heaps[source]) < config.max_neighbors:
                heapq.heappush(heaps[source], item)
            elif item > heaps[source][0]:
                heapq.heapreplace(heaps[source], item)
    rules = {}
    for source in sorted(list(heaps)):
        rules[source] = tuple(sorted(((-negative_target, strength) for strength, negative_target in heaps.pop(source)),
                                     key=lambda item: (-item[1], item[0])))
    return rules, {
        "undirected_pairs_meeting_support_threshold": eligible_pairs,
        "directional_rules_before_neighbor_pruning": eligible_directions,
        "retained_directional_rules": sum(map(len, rules.values())),
        "antecedents_with_rules": len(rules),
        "neighbor_pruning": "keep strongest rules per antecedent, numeric consequent ID breaks exact ties",
        "candidates_or_validation_labels_used_to_fit_or_prune_rules": False,
    }


def recommend_basket(basket: tuple[BasketItem, ...], rules: dict, candidates: frozenset[int],
                     fallback: tuple[int, ...], config: RuleConfig) -> tuple[tuple[int, ...], dict]:
    config.validate()
    validate_prediction(fallback, candidates)
    scores = defaultdict(float)
    # Defensive deduplication makes duplicate query items incapable of adding votes.
    weights = {}
    for item in basket:
        weights[item.programme_id] = max(weights.get(item.programme_id, 0.0), item.log_view_count)
    for source in sorted(weights):
        weight = weights[source] if config.scoring.startswith("weighted_") else 1.0
        for target, strength in rules.get(source, ()):
            if target not in candidates:
                continue
            contribution = weight * strength
            if config.scoring == "confidence_max":
                scores[target] = max(scores[target], contribution)
            else:
                scores[target] += contribution
    positive = {item: value for item, value in scores.items() if value > 0}
    selected = heapq.nsmallest(K, positive, key=lambda item: (-positive[item], item))
    rule_slots = len(selected)
    for item in fallback:
        if len(selected) == K:
            break
        if item not in selected:
            selected.append(item)
    result = tuple(selected)
    validate_prediction(result, candidates)
    return result, {"rule_scored_candidates": len(positive), "association_slots": rule_slots,
                    "fallback_slots": K - rule_slots,
                    "top_five": [{"tv_show_id": item, "association_score": positive.get(item, 0.0),
                                  "source": "association" if rank < rule_slots else "popularity_fallback"}
                                 for rank, item in enumerate(result)]}


def generate_predictions(users: list[str], baskets: dict, rules: dict, candidates: frozenset[int],
                         fallback: tuple[int, ...], config: RuleConfig) -> tuple[dict, dict]:
    validate_cohort(users)
    predictions, details = {}, {}
    for user in users:
        predictions[user], details[user] = recommend_basket(baskets.get(user, ()), rules, candidates, fallback, config)
    return predictions, {
        "users_with_five_association_slots": sum(d["association_slots"] == K for d in details.values()),
        "users_requiring_fallback": sum(d["fallback_slots"] > 0 for d in details.values()),
        "users_receiving_only_fallback": sum(d["fallback_slots"] == K for d in details.values()),
        "total_fallback_slots": sum(d["fallback_slots"] for d in details.values()),
        "per_user_ranking": details,
    }


def render_report(report: dict) -> str:
    lines = ["# SWEET.TV Association Rules and Evaluation Diagnostic", "", f"Generated (UTC): {report['generated_at_utc']}", "",
             "## Reproduction", "", "```bash", ".venv/bin/python src/diagnose_evaluation.py",
             ".venv/bin/python src/association_rules.py", "```", "",
             "Run the diagnostic once. The association CLI consumes its preserved successful report and verifies its protected fingerprints. "
             "Existing reports/predictions are never overwritten; use a fresh project copy for a repeat experiment.", "",
             "## Diagnostic gate", "", report["diagnostic"]["conclusion"], "",
             "The separate evaluation_diagnostic reports contain the independent CSV/date validation, rational AP examples, and all per-programme counts.", "",
             "```json", json.dumps({key: value for key, value in report["diagnostic"].items() if key != "protected_file_fingerprints"}, indent=2), "```", "",
             "## Prespecified experiments: every tested configuration", "",
             "All pair/rule fitting is training-only. The hold-out selects among these four configurations; it is not used to fit counts or rules. "
             "This selection may be optimistic on the same hold-out; there is no claim of an unbiased second-test score.", "",
             "| Configuration | Primary MAP@5 | Nonempty-label MAP@5 | Cold-start MAP@5 | Distinct recommended programmes |",
             "|---|---:|---:|---:|---:|"]
    for experiment in report["experiments"]:
        metrics = experiment["evaluation"]["metrics"]
        lines.append(f"| {experiment['configuration']['name']} | {metrics['primary_all_users']['map_at_5']!r} | "
                     f"{metrics['supplementary_nonempty_ground_truth']['map_at_5']!r} | "
                     f"{metrics['without_training_history']['map_at_5']!r} | {metrics['coverage']['unique_programmes_recommended']} |")
    lines += ["", f"Selected configuration: **{report['selected_configuration']}**. Highest primary MAP@5 wins; an exact tie keeps the earlier prespecified configuration.", "",
              "## Exact individual-method comparison", "",
              "| Method | Primary MAP@5 | Nonempty-label MAP@5 | Cold-start MAP@5 | Distinct programmes | User coverage |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, item in report["individual_method_comparison"].items():
        metrics = item["metrics"]
        lines.append(f"| {name} | {metrics['primary_all_users']['map_at_5']!r} | "
                     f"{metrics['supplementary_nonempty_ground_truth']['map_at_5']!r} | {metrics['without_training_history']['map_at_5']!r} | "
                     f"{metrics['coverage']['unique_programmes_recommended']} | {metrics['coverage']['user_recommendation_coverage']!r} |")
    lines += ["", f"Best individual method on the shared local hold-out: **{report['best_individual_method']}**.", "",
              "## Model and approximation", "",
              "- One distinct-item basket per training user, from weeks 11–20 only. Each user contributes at most once to an item or unordered pair count.",
              "- Retain at most 50 items, ordered by descending matched_events, then descending overlap_seconds, then ascending numeric programme ID. The same capped histories provide query antecedents.",
              "- Item support and pair support use these same capped baskets, so confidence is pair_support / antecedent_support and lift is pair_support * N / (support_a * support_b). These are not full-history supports.",
              "- Mine only A→B with distinct A and B. Recommending an already watched programme remains allowed through other antecedents; no item is excluded merely because it was watched.",
              "- Sum/max confidence variants use uniform query weights. Weighted variants use log1p(matched_events); shrinkage multiplies a rule by pair_support / (pair_support + shrinkage).",
              "- The weighted-lift variant ranks by shrunk lift, not confidence times lift. All variants still apply the documented minimum confidence/support filters.",
              "- Prune to the strongest 100 rules per antecedent without using validation candidates or labels. Candidate filtering happens only while ranking.",
              "- Sort positive association scores descending and programme IDs ascending on exact ties. Fill incomplete lists using the unchanged popularity Top-5, skipping duplicates.",
              "- There is no additive popularity score, rank fusion, hybrid, self-association rule, or validation-label fitting.",
              "- SQLite pair counts and bounded Counter batches avoid a dense item matrix or retaining millions of Python pair objects. A batch can exceed its target by at most one capped basket's pair count.", "",
              "```json", json.dumps(report["mining"], indent=2), "```", "",
              "## Exact validation protocol", "", "```json", json.dumps(report["validation_protocol"], indent=2), "```", "",
              "## All model parameters and measured per-configuration timing", "", "```json",
              json.dumps([{key: value for key, value in experiment.items() if key not in ("evaluation", "ranking_diagnostics")}
                          for experiment in report["experiments"]], indent=2), "```", "",
              "## Selected model coverage and fallback", "", "```json",
              json.dumps({key: value for key, value in report["selected_ranking_diagnostics"].items() if key != "per_user_ranking"}, indent=2), "```", "",
              "## Execution environment and memory", "", "```json", json.dumps(report["execution"], indent=2), "```", "",
              "Timing includes the full four-configuration study, not just the selected model. Per-configuration RSS readings are cumulative process high-water marks, not isolated model peaks. "
              "Earlier methods' original runtime/memory measurements are retained in the JSON comparison with their original scope.", "",
              "## Verification", "", "```json", json.dumps(report["verification"], indent=2), "```", "",
              "### Protected artifacts", "", "| File | SHA-256 | Unchanged |", "|---|---|---|"]
    for name, details in report["input_integrity"].items():
        lines.append(f"| {name} | `{details['before']['sha256']}` | {details['unchanged']} |")
    lines += ["", "### Saved experiment predictions", "", "```json", json.dumps(report["prediction_files"], indent=2), "```", "",
              "### Complete test suite", "", "```text", report["tests"]["output"].rstrip(), "```", "",
              "## Limitations", "",
              "- The 50-item cap changes supports and drops long-tail items. Selected-item pair counts are exact, but the graph is an approximation to full-history co-viewing.",
              "- Ten-week co-view baskets do not establish viewing order, causality, enjoyment, or true next-programme preference. Weighted query evidence is a ranking score, not a calibrated probability.",
              "- High-confidence rules may mostly reproduce common-programme preferences. Lift can favor rarer items; minimum support, shrinkage, and neighbor pruning do not eliminate noise.",
              "- Co-viewing with only the supplied users includes each query user's own training history. This is allowed training data, not validation leakage; it can reinforce familiar items.",
              "- Items never retained in training baskets cannot receive rule-based scores. Future schedule metadata is not used to invent rules.",
              "- The broad local binary relevance definition, repeat policy, 120-second threshold, sparse labels, clock uncertainties, and candidate/ground-truth mismatch remain unchanged. No equivalence to Kaggle's hidden score is claimed.",
              "- All tested configurations, including inferior ones, and all their prediction files are preserved. No hybrid, validation retraining, final submission, or Kaggle upload was performed.", ""]
    return "\n".join(lines)


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    root = Path(__file__).resolve().parents[1]
    reports, outputs = root / "reports", root / "outputs"
    destinations = [reports / "association_validation.json", reports / "association_validation.md", outputs / "association_validation.csv"]
    experiment_paths = {config.name: outputs / f"association_{config.name}_validation.csv" for config in EXPERIMENTS}
    if any(path.exists() or path.is_symlink() for path in [*destinations, *experiment_paths.values()]):
        raise FileExistsError("Preserving previous association experiment artifacts; use a fresh run directory/project copy.")
    started = time.perf_counter()
    generated = datetime.now(timezone.utc).isoformat()
    diagnostic_path = reports / "evaluation_diagnostic.json"
    diagnostic_report = json.loads(diagnostic_path.read_text())
    diagnostic = diagnostic_report["diagnostic"]
    if diagnostic["critical_problem_found"] or not diagnostic["all_1260_independent_per_user_scores_match"]:
        raise RuntimeError("The independent diagnostic gate failed. No association model will be fitted.")
    protected_names = set(diagnostic["protected_file_fingerprints"])
    protected_names.update({"reports/evaluation_diagnostic.json", "reports/evaluation_diagnostic.md", "src/diagnose_evaluation.py"})
    before = {name: file_fingerprint(root / name) for name in sorted(protected_names)}
    for name, details in diagnostic["protected_file_fingerprints"].items():
        if before[name]["sha256"] != details["sha256"]:
            raise RuntimeError(f"Protected input changed since the diagnostic: {name}")
    if before["src/diagnose_evaluation.py"]["sha256"] != diagnostic_report["execution"]["script_sha256"]:
        raise RuntimeError("Diagnostic implementation changed since the successful diagnostic run.")
    baseline = json.loads((reports / "popularity_baseline.json").read_text())
    cbf = json.loads((reports / "cbf_validation.json").read_text())
    if PROTOCOL != baseline["validation_protocol"] or PROTOCOL != cbf["validation_protocol"]:
        raise RuntimeError("The existing evaluation protocol changed.")
    test_start = time.perf_counter()
    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
    tests = subprocess.run(command, cwd=root, capture_output=True, text=True)
    output = tests.stdout + tests.stderr
    print(output, end="", flush=True)
    if tests.returncode:
        raise RuntimeError("Complete test suite failed; actual association mining was not started.")
    test_result = {"command": command, "returncode": tests.returncode, "output": output,
                   "runtime_seconds": time.perf_counter() - test_start}
    stages = {}
    begin = time.perf_counter()
    users = load_template_users(root / "data/raw/submission.csv")
    baskets, basket_stats = load_training_baskets(root / "data/processed/train_user_programmes.csv")
    training_users = frozenset(baskets)
    stages["load_bounded_training_baskets"] = time.perf_counter() - begin
    begin = time.perf_counter()
    popularity = fit_popularity(root / "data/processed/train_user_programmes.csv")
    if training_users != popularity.training_users:
        raise AssertionError("Association and baseline history cohorts differ.")
    candidates, candidate_stats = load_candidates(root / "data/raw" / SCHEDULE_FILES[1])
    if candidate_stats != baseline["baseline"]["candidate_pool"]:
        raise AssertionError("The shared candidate pool changed.")
    fallback = rank_candidates(popularity.scores, candidates)
    baseline_predictions = read_predictions(outputs / "popularity_validation.csv", users, candidates)
    cbf_predictions = read_predictions(outputs / "cbf_validation.csv", users, candidates)
    if any(value != fallback for value in baseline_predictions.values()):
        raise AssertionError("Association fallback differs from the preserved popularity baseline.")
    stages["training_popularity_and_shared_availability"] = time.perf_counter() - begin
    experiments, all_predictions = [], {}
    with tempfile.TemporaryDirectory(prefix="sweettv-association-", dir=root / "data/processed") as scratch:
        begin = time.perf_counter()
        counts = AssociationCounts(baskets, Path(scratch) / "pairs.sqlite3")
        stages["mine_disk_backed_cooccurrences"] = time.perf_counter() - begin
        try:
            mining_stats = {**basket_stats, "pair_counts": counts.statistics,
                            "support_counts_for_capped_baskets": {str(item): count for item, count in sorted(counts.support.items())}}
            print(f"Mined {counts.statistics['distinct_undirected_pairs']:,} unique pairs from {len(baskets):,} capped training baskets.", flush=True)
            # Fit every rule variant and fix all predictions before loading validation labels.
            for config in EXPERIMENTS:
                begin = time.perf_counter()
                rules, rule_stats = build_rules(counts, config)
                rule_seconds = time.perf_counter() - begin
                begin = time.perf_counter()
                predictions, diagnostics = generate_predictions(users, baskets, rules, candidates, fallback, config)
                prediction_seconds = time.perf_counter() - begin
                del rules
                if any(predictions[user] != fallback for user in users if user not in training_users):
                    raise AssertionError("A cold-start user did not receive the shared fallback.")
                write_predictions(experiment_paths[config.name], users, predictions, candidates)
                saved = read_predictions(experiment_paths[config.name], users, candidates)
                if saved != predictions:
                    raise AssertionError("Experiment CSV differs from generated predictions.")
                all_predictions[config.name] = saved
                experiments.append({"configuration": {**asdict(config), "max_items_per_user": MAX_ITEMS_PER_USER},
                                    "rules": rule_stats, "ranking_diagnostics": diagnostics,
                                    "runtime_seconds": {"build_pruned_rules": rule_seconds, "rank_all_users": prediction_seconds},
                                    "cumulative_main_process_memory_at_end": peak_memory(),
                                    "prediction_file": str(experiment_paths[config.name].relative_to(root))})
                print(f"Fixed predictions for {config.name}: {rule_stats['retained_directional_rules']:,} rules.", flush=True)
        finally:
            counts.close()
    # Label use starts here, after all four fitted graphs and prediction files are fixed.
    begin = time.perf_counter()
    truth, validation_stats = load_validation_ground_truth(root / "data/processed/validation_user_programmes.csv", users)
    if evaluate_recommendations(users, truth, baseline_predictions, training_users, candidates) != baseline["baseline"]["evaluation"]:
        raise AssertionError("Popularity evaluation no longer matches the saved reference.")
    if evaluate_recommendations(users, truth, cbf_predictions, training_users, candidates) != cbf["cbf"]["evaluation"]:
        raise AssertionError("CBF evaluation no longer matches the saved reference.")
    stages["load_labels_and_verify_prior_metrics"] = time.perf_counter() - begin
    for experiment in experiments:
        begin = time.perf_counter()
        name = experiment["configuration"]["name"]
        experiment["evaluation"] = evaluate_recommendations(users, truth, all_predictions[name], training_users, candidates)
        experiment["runtime_seconds"]["evaluate_saved_predictions"] = time.perf_counter() - begin
        print(f"{name}: MAP@5 = {experiment['evaluation']['metrics']['primary_all_users']['map_at_5']!r}", flush=True)
    selected = max(experiments, key=lambda experiment: experiment["evaluation"]["metrics"]["primary_all_users"]["map_at_5"])
    selected_name = selected["configuration"]["name"]
    write_predictions(outputs / "association_validation.csv", users, all_predictions[selected_name], candidates)
    selected_saved = read_predictions(outputs / "association_validation.csv", users, candidates)
    if evaluate_recommendations(users, truth, selected_saved, training_users, candidates) != selected["evaluation"]:
        raise AssertionError("Selected saved predictions do not reproduce selected metrics.")
    after = {name: file_fingerprint(root / name) for name in sorted(protected_names)}
    integrity = {name: {"before": details, "after": after[name], "unchanged": details == after[name]} for name, details in before.items()}
    if not all(details["unchanged"] for details in integrity.values()):
        raise RuntimeError("Protected artifacts changed; successful association report not written.")
    comparison = {
        "Popularity": {"metrics": baseline["baseline"]["evaluation"]["metrics"], "original_execution": baseline["execution"]},
        "Content-Based Filtering": {"metrics": cbf["cbf"]["evaluation"]["metrics"], "original_execution": cbf["execution"]},
        "Association Rules": {"metrics": selected["evaluation"]["metrics"], "selected_configuration": selected_name},
    }
    best = max(comparison, key=lambda name: comparison[name]["metrics"]["primary_all_users"]["map_at_5"])
    prediction_files = {str(path.relative_to(root)): file_fingerprint(path) for path in [*experiment_paths.values(), outputs / "association_validation.csv"]}
    report = {
        "schema_version": 1, "generated_at_utc": generated, "diagnostic": diagnostic,
        "validation_protocol": PROTOCOL, "mining": mining_stats, "candidate_pool": candidate_stats,
        "validation_labels_for_scoring_only": validation_stats, "fallback_top_five": list(fallback),
        "prespecified_configuration_order": [config.name for config in EXPERIMENTS], "experiments": experiments,
        "selected_configuration": selected_name, "selected_ranking_diagnostics": selected["ranking_diagnostics"],
        "individual_method_comparison": comparison, "best_individual_method": best,
        "input_integrity": integrity, "prediction_files": prediction_files, "tests": test_result,
        "verification": {
            "independent_diagnostic_passed_before_new_model_implementation": True,
            "all_tests_passed": tests.returncode == 0, "all_protected_artifacts_unchanged": True,
            "same_1260_users_candidates_ground_truth_and_evaluator": True,
            "prior_popularity_and_cbf_evaluations_exactly_reproduced": True,
            "all_experiment_and_selected_csvs_verified": True,
            "cold_start_fallback_matches_popularity_in_every_experiment": True,
            "all_four_rule_graphs_and_predictions_fixed_before_validation_labels_loaded": True,
            "validation_labels_used_for_model_selection_only_after_scoring": True,
            "all_tested_configurations_and_scores_preserved": len(experiments) == len(EXPERIMENTS),
            "training_input": "data/processed/train_user_programmes.csv",
            "validation_viewing_used_for_supports_rules_or_query_histories": False,
            "no_hybrid_or_final_training_or_kaggle_submission": True,
        },
        "execution": {
            "python_version": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
            "invocation": sys.argv, "third_party_packages_used": [], "random_seed": None, "randomness": "none",
            "source_sha256": {str(path.relative_to(root)): file_fingerprint(path)["sha256"] for path in
                              (Path(__file__), root / "tests/test_association_rules.py", root / "src/diagnose_evaluation.py",
                               root / "src/evaluate.py", root / "src/popularity.py")},
            "shared_stage_runtime_seconds": {**stages, "unit_tests": test_result["runtime_seconds"]},
            "runtime_seconds_before_report_serialization": time.perf_counter() - started, **peak_memory(),
            "memory_scope": "main-process cumulative high-water RSS over all four configurations; excludes test subprocess",
        },
    }
    destinations[0].write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    destinations[1].write_text(render_report(report), encoding="utf-8")
    print(f"Selected {selected_name}. Best individual method: {best}.")
    print(f"All protected files unchanged. Runtime including report writes: {time.perf_counter() - started:.3f} seconds")
    print(f"Peak main-process RSS: {report['execution'].get('peak_process_rss_mib', 'unavailable')} MiB")


if __name__ == "__main__":
    main()
