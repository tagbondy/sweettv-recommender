"""Build temporally clipped SWEET.TV interactions with a streaming interval join.

Usage: .venv/bin/python src/build_interactions.py --min-overlap-seconds 120

Uses only the standard library. Viewing records are streamed, labeled historical
airings are indexed by channel, and exact event deduplication and user-programme
aggregation use a temporary disk-backed SQLite database. Test schedules never
enter the interaction index. This script does not implement recommendation models.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sqlite3
import subprocess
import sys
import tempfile
import time
from array import array
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

from audit_data import (
    INPUT_FILES, SCHEDULE_COLUMNS, SCHEDULE_FILES, USER_PATTERN, VIEW_COLUMNS,
    VIEW_FILE, NumericSummary, fingerprints, parse_integer, parse_timestamp,
    peak_memory, records,
)


ORIGIN = datetime(2020, 1, 1)


def seconds(value: datetime) -> int:
    """Represent naive source time without consulting the machine's timezone."""
    delta = value - ORIGIN
    return delta.days * 86400 + delta.seconds


@lru_cache(maxsize=16384)
def timestamp(value: int) -> str:
    return (ORIGIN + timedelta(seconds=value)).isoformat(sep=" ")


@dataclass(frozen=True, slots=True)
class Window:
    name: str
    start: int
    end: int


TRAIN_START = seconds(datetime.fromisocalendar(2020, 11, 1))
VALIDATION_START = seconds(datetime.fromisocalendar(2020, 21, 1))
TEST_START = seconds(datetime.fromisocalendar(2020, 31, 1))
TEST_END = seconds(datetime.fromisocalendar(2020, 43, 1))
WINDOWS = (
    Window("train", TRAIN_START, VALIDATION_START),
    Window("validation", VALIDATION_START, TEST_START),
    Window("final_train", TRAIN_START, TEST_START),
)


@dataclass(frozen=True, slots=True)
class Airing:
    channel_id: int
    programme_id: int
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class View:
    user_id: str
    channel_id: int
    start: int
    end: int
    source_row: int


@dataclass(frozen=True, slots=True)
class Match:
    split: str
    user_id: str
    programme_id: int
    channel_id: int
    source_row: int
    view_start: int
    view_end: int
    programme_start: int
    programme_end: int
    overlap_start: int
    overlap_end: int

    @property
    def overlap_seconds(self) -> int:
        return self.overlap_end - self.overlap_start


class ScheduleIndex:
    """Find every overlap, including nested airings and out-of-order views.

    Binary searches bound the scan using starts and prefix-maximum ends. This
    does not assume that schedule ends increase with starts or that views arrive
    sorted. Identical channel/programme/start/end records are retained once.
    """

    def __init__(self, airings) -> None:
        grouped = defaultdict(set)
        self.duplicates_removed = 0
        self.invalid_or_zero_id_airings_ignored = 0
        for airing in airings:
            if airing.programme_id <= 0 or airing.channel_id <= 0 or airing.end <= airing.start:
                self.invalid_or_zero_id_airings_ignored += 1
                continue
            group = grouped[airing.channel_id]
            if airing in group:
                self.duplicates_removed += 1
            else:
                group.add(airing)
        self.channels = {}
        for channel in list(grouped):
            ordered = sorted(grouped.pop(channel), key=lambda a: (a.start, a.end, a.programme_id))
            starts = array("q", (a.start for a in ordered))
            prefix_ends = array("q")
            maximum = ordered[0].end
            for airing in ordered:
                maximum = max(maximum, airing.end)
                prefix_ends.append(maximum)
            self.channels[channel] = (ordered, starts, prefix_ends)

    def overlaps(self, view: View):
        group = self.channels.get(view.channel_id)
        if group is None or view.end <= view.start:
            return
        ordered, starts, prefix_ends = group
        lower = bisect_right(prefix_ends, view.start)
        upper = bisect_left(starts, view.end)
        for position in range(lower, upper):
            airing = ordered[position]
            if airing.end > view.start:
                yield airing


def clipped_match(view: View, airing: Airing, window: Window, minimum: int) -> Match | None:
    """Clip both intervals first; equality with a positive threshold qualifies."""
    if minimum <= 0:
        raise ValueError("Minimum overlap must be a positive integer number of seconds.")
    if view.channel_id != airing.channel_id or airing.programme_id <= 0:
        return None
    view_start, view_end = max(view.start, window.start), min(view.end, window.end)
    programme_start, programme_end = max(airing.start, window.start), min(airing.end, window.end)
    low, high = max(view_start, programme_start), min(view_end, programme_end)
    if high - low < minimum:
        return None
    return Match(window.name, view.user_id, airing.programme_id, view.channel_id, view.source_row,
                 view_start, view_end, programme_start, programme_end, low, high)


def checked_records(path: Path, columns: list[str]):
    for number, row in records(path, columns):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"Malformed CSV record: {path.name}, data row {number}")
        yield number, row


def load_history_schedule(data_dir: Path) -> tuple[ScheduleIndex, dict, set[int]]:
    stats = {}
    all_channels = set()

    def airings():
        # Only the two historical schedules are permitted in the matching index.
        for name in SCHEDULE_FILES[:2]:
            counts = Counter(dict.fromkeys((
                "rows_read", "unlabeled_rows_excluded", "invalid_channel_id",
                "invalid_programme_id", "invalid_labeled_airing", "outside_final_training_window",
                "labeled_airings_clipped_to_final_window", "labeled_airings_yielded",
            ), 0))
            stats[name] = counts
            for _, row in checked_records(data_dir / name, SCHEDULE_COLUMNS):
                counts["rows_read"] += 1
                channel = parse_integer(row["channel_id"])
                programme = parse_integer(row["tv_show_id"])
                if channel is None or channel <= 0:
                    counts["invalid_channel_id"] += 1
                    continue
                all_channels.add(channel)
                if programme is None or programme < 0:
                    counts["invalid_programme_id"] += 1
                    continue
                if programme == 0:
                    counts["unlabeled_rows_excluded"] += 1
                    continue
                start = parse_timestamp(row["start_time"], schedule=True)
                duration = parse_integer(row["duration"])
                if start is None or duration is None or duration <= 0:
                    counts["invalid_labeled_airing"] += 1
                    continue
                low = seconds(start)
                high = low + duration
                if high <= TRAIN_START or low >= TEST_START:
                    counts["outside_final_training_window"] += 1
                    continue
                clipped_low, clipped_high = max(low, TRAIN_START), min(high, TEST_START)
                counts["labeled_airings_clipped_to_final_window"] += (low, high) != (clipped_low, clipped_high)
                counts["labeled_airings_yielded"] += 1
                yield Airing(channel, programme, clipped_low, clipped_high)

    index = ScheduleIndex(airings())
    return index, {
        "files": {key: dict(value) for key, value in stats.items()},
        "distinct_labeled_airings_indexed": sum(len(group[0]) for group in index.channels.values()),
        "duplicate_clipped_airings_removed_across_and_within_files": index.duplicates_removed,
        "channels_with_any_historical_schedule_rows": len(all_channels),
        "channels_with_indexed_labeled_airings": len(index.channels),
        "test_schedule_loaded_into_interaction_index": False,
    }, all_channels


class InteractionStore:
    """Disk-backed exact deduplication and aggregation, with bounded batches."""

    def __init__(self, path: Path) -> None:
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=OFF")
        self.connection.execute("PRAGMA synchronous=OFF")
        self.connection.execute("PRAGMA cache_size=-8192")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.executescript("""
            CREATE TABLE seen_views (
                start INTEGER, end INTEGER, channel INTEGER, user_id TEXT,
                PRIMARY KEY (start, end, channel, user_id)
            ) WITHOUT ROWID;
            CREATE TABLE pairs (
                split TEXT, user_id TEXT, programme INTEGER,
                matched_events INTEGER, overlap_seconds INTEGER,
                first_overlap INTEGER, last_overlap INTEGER,
                PRIMARY KEY (split, user_id, programme)
            ) WITHOUT ROWID;
        """)
        self.pending = []
        self.since_commit = 0

    def is_new_view(self, view: View) -> bool:
        cursor = self.connection.execute("INSERT OR IGNORE INTO seen_views VALUES (?, ?, ?, ?)",
                                         (view.start, view.end, view.channel_id, view.user_id))
        self.since_commit += 1
        if self.since_commit >= 10000:
            self.flush()
        return cursor.rowcount == 1

    def add(self, match: Match) -> None:
        self.pending.append((match.split, match.user_id, match.programme_id, 1,
                             match.overlap_seconds, match.overlap_start, match.overlap_end))
        if len(self.pending) >= 5000:
            self.flush()

    def flush(self) -> None:
        self.connection.executemany("""
            INSERT INTO pairs VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (split, user_id, programme) DO UPDATE SET
                matched_events = matched_events + excluded.matched_events,
                overlap_seconds = overlap_seconds + excluded.overlap_seconds,
                first_overlap = MIN(first_overlap, excluded.first_overlap),
                last_overlap = MAX(last_overlap, excluded.last_overlap)
        """, self.pending)
        self.pending.clear()
        self.connection.commit()
        self.since_commit = 0

    def close(self) -> None:
        self.connection.close()


EVENT_COLUMNS = [
    "user_id", "tv_show_id", "channel_id", "view_event_row",
    "view_start", "view_end", "programme_start", "programme_end",
    "overlap_start", "overlap_end", "overlap_seconds",
]
PAIR_COLUMNS = ["user_id", "tv_show_id", "matched_events", "overlap_seconds", "first_overlap", "last_overlap"]


def event_row(match: Match) -> list:
    return [match.user_id, match.programme_id, match.channel_id, match.source_row,
            timestamp(match.view_start), timestamp(match.view_end),
            timestamp(match.programme_start), timestamp(match.programme_end),
            timestamp(match.overlap_start), timestamp(match.overlap_end), match.overlap_seconds]


def build_interactions(data_dir: Path, output_dir: Path, scratch: Path, minimum: int = 120,
                       progress: bool = False) -> dict:
    """Build all three training/validation datasets; no test views are retained."""
    if minimum <= 0:
        raise ValueError("Minimum overlap must be positive.")
    template_users = set()
    for _, row in checked_records(data_dir / "submission.csv", ["user_id", "tv_show_id"]):
        if not USER_PATTERN.fullmatch(row["user_id"]) or row["user_id"] in template_users:
            raise ValueError("Submission template contains an invalid or duplicate user ID.")
        template_users.add(row["user_id"])
    begin = time.perf_counter()
    index, schedule_stats, schedule_channels = load_history_schedule(data_dir)
    stages = {"load_and_index_historical_schedules": time.perf_counter() - begin}
    if progress:
        print(f"Indexed {schedule_stats['distinct_labeled_airings_indexed']:,} historical labeled airings "
              f"in {stages['load_and_index_historical_schedules']:.3f} seconds.", flush=True)
    counters = Counter(dict.fromkeys((
        "source_viewing_rows", "invalid_viewing_rows_skipped", "outside_final_training_window",
        "views_clipped_to_final_training_window", "raw_views_crossing_validation_cutoff",
        "reported_duration_mismatches", "invalid_reported_durations", "duplicate_viewing_events_removed",
        "unique_in_window_viewing_events", "unique_views_crossing_validation_cutoff",
        "events_on_channels_without_historical_schedule", "events_on_channels_without_labeled_airings",
        "events_without_positive_labeled_overlap", "events_with_positive_but_subthreshold_labeled_overlap",
        "events_with_qualifying_final_training_overlap",
    ), 0))
    mismatch_deltas = Counter()
    missing_channels = Counter()
    split_counts = {w.name: Counter({"positive_overlap_pairs": 0, "matched_interactions": 0,
                                  "matched_source_viewing_events": 0, "overlap_seconds": 0}) for w in WINDOWS}
    split_users = {w.name: set() for w in WINDOWS}
    split_programmes = {w.name: set() for w in WINDOWS}
    overlap_stats = {w.name: NumericSummary() for w in WINDOWS}
    store = InteractionStore(scratch / "interaction_work.sqlite3")
    output_dir.mkdir(parents=True, exist_ok=True)
    begin = time.perf_counter()
    try:
        with ExitStack() as stack:
            writers = {}
            for window in WINDOWS:
                handle = stack.enter_context((output_dir / f"{window.name}_interactions.csv").open("w", newline="", encoding="utf-8"))
                writers[window.name] = csv.writer(handle)
                writers[window.name].writerow(EVENT_COLUMNS)
            for number, row in checked_records(data_dir / VIEW_FILE, VIEW_COLUMNS):
                counters["source_viewing_rows"] += 1
                if progress and number % 1000000 == 0:
                    print(f"Processed {number:,} viewing rows in {time.perf_counter() - begin:.1f} seconds; "
                          f"{split_counts['final_train']['matched_interactions']:,} final-training matches so far.", flush=True)
                start = parse_timestamp(row["start_time"])
                end = parse_timestamp(row["stop_time"])
                channel = parse_integer(row["vsetv_id"])
                user = row["user_id"]
                if (start is None or end is None or end <= start or channel is None
                        or channel <= 0 or not USER_PATTERN.fullmatch(user)):
                    counters["invalid_viewing_rows_skipped"] += 1
                    continue
                low, high = seconds(start), seconds(end)
                if high <= TRAIN_START or low >= TEST_START:
                    counters["outside_final_training_window"] += 1
                    continue
                reported = parse_integer(row["duraton"])
                if reported is None or reported < 0:
                    counters["invalid_reported_durations"] += 1
                elif reported != high - low:
                    counters["reported_duration_mismatches"] += 1
                    mismatch_deltas[high - low - reported] += 1
                counters["raw_views_crossing_validation_cutoff"] += low < VALIDATION_START < high
                # Deduplicate the original interval, not its possibly identical clipped tail.
                raw_view = View(user, channel, low, high, number)
                if not store.is_new_view(raw_view):
                    counters["duplicate_viewing_events_removed"] += 1
                    continue
                counters["unique_in_window_viewing_events"] += 1
                counters["unique_views_crossing_validation_cutoff"] += low < VALIDATION_START < high
                clipped_low, clipped_high = max(low, TRAIN_START), min(high, TEST_START)
                counters["views_clipped_to_final_training_window"] += (low, high) != (clipped_low, clipped_high)
                view = View(user, channel, clipped_low, clipped_high, number)
                if channel not in schedule_channels:
                    counters["events_on_channels_without_historical_schedule"] += 1
                    missing_channels[channel] += 1
                    continue
                if channel not in index.channels:
                    counters["events_on_channels_without_labeled_airings"] += 1
                    continue
                has_positive = False
                matched_splits = set()
                for airing in index.overlaps(view):
                    has_positive = True
                    for window in WINDOWS:
                        overlap = min(view.end, airing.end, window.end) - max(view.start, airing.start, window.start)
                        if overlap <= 0:
                            continue
                        split_counts[window.name]["positive_overlap_pairs"] += 1
                        overlap_stats[window.name].add(overlap)
                        if overlap < minimum:
                            continue
                        match = clipped_match(view, airing, window, minimum)
                        if match is None:
                            raise AssertionError("Inconsistent overlap calculation.")
                        writers[window.name].writerow(event_row(match))
                        store.add(match)
                        split_counts[window.name]["matched_interactions"] += 1
                        split_counts[window.name]["overlap_seconds"] += match.overlap_seconds
                        split_users[window.name].add(user)
                        split_programmes[window.name].add(airing.programme_id)
                        matched_splits.add(window.name)
                for split in matched_splits:
                    split_counts[split]["matched_source_viewing_events"] += 1
                if not has_positive:
                    counters["events_without_positive_labeled_overlap"] += 1
                elif "final_train" not in matched_splits:
                    counters["events_with_positive_but_subthreshold_labeled_overlap"] += 1
                else:
                    counters["events_with_qualifying_final_training_overlap"] += 1
        store.flush()
        stages["stream_deduplicate_join_and_write_events"] = time.perf_counter() - begin
        if progress:
            print(f"Viewing-log join finished in {stages['stream_deduplicate_join_and_write_events']:.3f} seconds.", flush=True)
        begin = time.perf_counter()
        splits = {}
        for window in WINDOWS:
            pair_count = 0
            with (output_dir / f"{window.name}_user_programmes.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(PAIR_COLUMNS)
                query = "SELECT user_id, programme, matched_events, overlap_seconds, first_overlap, last_overlap FROM pairs WHERE split=? ORDER BY user_id, programme"
                for user, programme, count, duration, first, last in store.connection.execute(query, (window.name,)):
                    writer.writerow([user, programme, count, duration, timestamp(first), timestamp(last)])
                    pair_count += 1
            cold = template_users - split_users[window.name]
            splits[window.name] = {
                "start_inclusive": timestamp(window.start), "end_exclusive": timestamp(window.end),
                **split_counts[window.name], "unique_user_programme_pairs": pair_count,
                "unique_users": len(split_users[window.name]),
                "unique_programme_ids": len(split_programmes[window.name]),
                "template_users_with_labeled_interactions": len(template_users & split_users[window.name]),
                "template_users_without_labeled_interactions": len(cold),
                "cold_start_user_ids": sorted(cold),
                "positive_overlap_seconds_statistics_before_threshold": overlap_stats[window.name].result(),
            }
        stages["export_user_programme_aggregates"] = time.perf_counter() - begin
        sqlite_size = (scratch / "interaction_work.sqlite3").stat().st_size
    finally:
        store.close()
    return {"minimum_overlap_seconds": minimum, "template_unique_users": len(template_users),
            "schedule_loading": schedule_stats, "viewing_processing": dict(counters),
            "duration_mismatch_delta_seconds": {str(k): v for k, v in sorted(mismatch_deltas.items())},
            "events_by_missing_historical_channel": {str(k): v for k, v in sorted(missing_channels.items())},
            "splits": splits, "stage_runtime_seconds": stages,
            "temporary_sqlite_size_bytes_before_cleanup": sqlite_size}


def verify_outputs(output_dir: Path, result: dict, progress: bool = False) -> dict:
    """Independently reread every generated CSV and check split invariants."""
    verified = {}
    minimum = result["minimum_overlap_seconds"]
    for window in WINDOWS:
        counts, users, programmes = Counter(), set(), set()
        event_file = output_dir / f"{window.name}_interactions.csv"
        for number, row in checked_records(event_file, EVENT_COLUMNS):
            values = {}
            for key in EVENT_COLUMNS[4:-1]:
                parsed = parse_timestamp(row[key])
                if parsed is None:
                    raise AssertionError(f"Invalid output timestamp: {event_file}, row {number}")
                values[key] = seconds(parsed)
                if not window.start <= values[key] <= window.end:
                    raise AssertionError("Output contains time outside its split.")
            low = max(values["view_start"], values["programme_start"])
            high = min(values["view_end"], values["programme_end"])
            overlap = int(row["overlap_seconds"])
            if (not USER_PATTERN.fullmatch(row["user_id"]) or int(row["tv_show_id"]) <= 0
                    or int(row["channel_id"]) <= 0 or int(row["view_event_row"]) <= 0
                    or values["overlap_start"] != low or values["overlap_end"] != high
                    or overlap != high - low or overlap < minimum):
                raise AssertionError(f"Invalid output interaction: {event_file}, row {number}")
            counts["matched_interactions"] += 1
            counts["overlap_seconds"] += overlap
            users.add(row["user_id"])
            programmes.add(int(row["tv_show_id"]))
        expected = result["splits"][window.name]
        if (counts["matched_interactions"] != expected["matched_interactions"]
                or counts["overlap_seconds"] != expected["overlap_seconds"]
                or len(users) != expected["unique_users"] or len(programmes) != expected["unique_programme_ids"]):
            raise AssertionError("Event output totals do not match computed build statistics.")
        pair_file = output_dir / f"{window.name}_user_programmes.csv"
        pairs, events, duration, previous = 0, 0, 0, None
        for number, row in checked_records(pair_file, PAIR_COLUMNS):
            key = (row["user_id"], int(row["tv_show_id"]))
            first, last = parse_timestamp(row["first_overlap"]), parse_timestamp(row["last_overlap"])
            count, elapsed = int(row["matched_events"]), int(row["overlap_seconds"])
            if (not USER_PATTERN.fullmatch(key[0]) or key[1] <= 0 or count <= 0 or elapsed < minimum * count
                    or first is None or last is None or not window.start <= seconds(first) < seconds(last) <= window.end
                    or (previous is not None and previous >= key)):
                raise AssertionError(f"Invalid output aggregate: {pair_file}, row {number}")
            previous = key
            pairs += 1
            events += count
            duration += elapsed
        if (pairs != expected["unique_user_programme_pairs"] or events != counts["matched_interactions"]
                or duration != counts["overlap_seconds"]):
            raise AssertionError("Aggregate output does not reconcile with event output.")
        verified[window.name] = {"event_rows_verified": counts["matched_interactions"],
                                 "aggregate_rows_verified": pairs,
                                 "temporal_bounds_threshold_and_nonzero_ids_verified": True}
        if progress:
            print(f"Verified {window.name}: {counts['matched_interactions']:,} event rows and {pairs:,} aggregate rows.", flush=True)
    return verified


def file_fingerprint(path: Path) -> dict:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"File changed during checksum calculation: {path}")
    return {"size_bytes": after.st_size, "mtime_ns": after.st_mtime_ns, "sha256": digest.hexdigest()}


def render_report(report: dict) -> str:
    result = report["build"]
    lines = ["# SWEET.TV Interaction Build", "", f"Generated (UTC): {report['generated_at_utc']}", "",
             "## Reproduction", "", "```bash", ".venv/bin/python src/build_interactions.py --min-overlap-seconds 120", "```", "",
             "The CLI runs the project unit tests before processing real data. Only the Python standard library is required.", "",
             "## Temporal protocol", "",
             "All intervals are half-open [start, end). Boundaries are derived from ISO weeks in 2020, with Monday at 00:00:00.", "",
             "| Dataset | Start inclusive | End exclusive |", "|---|---|---|"]
    for window in WINDOWS:
        lines.append(f"| {window.name} | {timestamp(window.start)} | {timestamp(window.end)} |")
    lines += [f"| test predictions (availability only) | {timestamp(TEST_START)} | {timestamp(TEST_END)} |", "",
              "Both viewing and airing intervals are clipped to each output window before calculating the overlap and applying the threshold. "
              "Final-training interactions are computed over the whole weeks 11–30 window; they are not concatenated thresholded split outputs.", "",
              "## Matching and duplicate policy", "",
              "- Match exactly equal integer channel IDs; no guessed channel remapping.",
              "- overlap_seconds = max(0, min(view_end, programme_end) - max(view_start, programme_start)).",
              f"- Accept overlap_seconds >= {result['minimum_overlap_seconds']}. The threshold is provisional, configurable, and applied per deduplicated viewing-event/airing pair within each window, not to aggregated history.",
              "- This absolute-seconds rule is not programme-length normalization. It has not been optimized using recommendation MAP@5.",
              "- Use the supplied viewing start/stop timestamps; duraton is diagnostic only. Do not shift clocks or infer a timezone from duration differences.",
              "- Exclude tv_show_id = 0 from the index and every output.",
              "- Preserve original user_id strings, including leading zeros. Never cast user IDs to numbers.",
              "- Deduplicate viewing events by exact (user_id string, channel, original start, original end). Ignore duraton when identifying duplicate interval evidence; retain the first source row.",
              "- Deduplicate airings by (channel, programme ID, start, end) after clipping to the final historical window, including repetitions across schedule files.",
              "- Join every qualifying airing, including overlapping/nested broadcasts. Do not pick only the nearest programme start.",
              "- The test schedule is not loaded into the matching index, even if it contains an unexpectedly early row. It is only checksummed here.", "",
              "## Measured interaction results", "",
              "Matched interactions are qualifying deduplicated viewing-event/airing pairs, not unique user-programme pairs.", "",
              "| Split | Matched interactions | User-programme pairs | Users | Programmes | Template users without labeled interactions |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, stats in result["splits"].items():
        lines.append(f"| {name} | {stats['matched_interactions']:,} | {stats['unique_user_programme_pairs']:,} | "
                     f"{stats['unique_users']:,} | {stats['unique_programme_ids']:,} | {stats['template_users_without_labeled_interactions']:,} |")
    lines += ["", "Cold-start for validation is determined by the train row, not by whether validation contains labels. "
              "Cold-start for final prediction is determined by final_train. The validation row instead describes availability of observed validation labels.", "",
              "## Full computed build diagnostics", "", "```json", json.dumps(result, indent=2), "```", "",
              "## Audit observations reviewed", "", "```json", json.dumps(report["audit_context"], indent=2), "```", "",
              "The file-based audit count of 1,437 unseen test IDs differs from 1,436 within exact calendar windows. "
              "These are schedule availability counts, not counts of programmes observed in training interactions.", "",
              "## Outputs and verification", "", "```json", json.dumps(report["output_verification"], indent=2), "```", "",
              "Event CSV columns: " + ", ".join(f"`{name}`" for name in EVENT_COLUMNS) + ".", "",
              "Programme and view times in event outputs are already clipped. view_event_row is the one-based CSV data-record number of the retained source event.", "",
              "Aggregate CSV columns: " + ", ".join(f"`{name}`" for name in PAIR_COLUMNS) + ".", "",
              "Aggregates contain sums over accepted matches and the first overlap start/last overlap end; they do not bridge or use excluded future evidence.", "",
              "| Output | Bytes | SHA-256 |", "|---|---:|---|"]
    for name, details in report["output_files"].items():
        lines.append(f"| {name} | {details['size_bytes']:,} | `{details['sha256']}` |")
    lines += ["", "## Original-data and audit-report integrity", "",
              "Before/after SHA-256, file size, and modification time were checked for all five raw CSVs and both existing audit reports. "
              "Raw CSV hashes were also compared with the preserved audit's hashes before building.", "",
              "| Protected file | SHA-256 | Unchanged |", "|---|---|---|"]
    for name, details in report["protected_files"].items():
        lines.append(f"| {name} | `{details['before']['sha256']}` | {details['unchanged']} |")
    lines += ["", "## Tests executed before the build", "", "```text", report["tests"]["output"].rstrip(), "```", "",
              "## Runtime and environment", "", "```json", json.dumps(report["execution"], indent=2), "```", "",
              "## Assumptions and limitations", "",
              "- Source clocks are compared as naive wall-clock timestamps with no automatic timezone or daylight-saving correction. The audited 3,600-second discrepancy remains unresolved.",
              "- Schedule filenames are not temporal partitions. Historical schedule padding is clipped, and identical airings across historical files are deduplicated.",
              "- Unlabeled airtime, schedule gaps, and schedule inaccuracies can all cause missing interactions. No-positive-labeled-overlap does not prove the viewer watched nothing.",
              "- Partly overlapping but nonidentical viewing events are not merged. Distinct overlapping airings are retained. Summed overlap_seconds therefore need not equal unique watched wall-clock seconds.",
              "- Aggregated matched_events counts event-airing evidence, not guaranteed unique episodes or subjective enjoyment.",
              "- A crossing event can qualify in final_train but not either split, or qualify once in each split and once in final_train. Counts need not add across outputs.",
              "- The 120-second rule may exclude short programmes or short clipped pieces. Overlap distributions are reported; validation quality and threshold sensitivity remain future work.",
              "- Availability of metadata is not equivalent to observed preference. Novel test candidates and cold-start users require later recommender handling.",
              "- Memory retains compact labeled-airing records, per-channel binary-search arrays, bounded aggregate batches, and small ID/statistic sets. All viewing events and pair aggregates remain disk-backed or streamed.",
              "- Output verification rereads every CSV, checks all timestamp bounds, overlap arithmetic, positive IDs, thresholds, unique sorted aggregate keys, and reconciles aggregate/event totals. Source-channel matching and duplicate/boundary semantics are additionally covered by unit tests.",
              "- Runtime includes tests, input/output checksums, CSV processing, output verification, and temporary-database cleanup; final report serialization is excluded. Peak RSS covers the main process, not the separate unit-test process.",
              "- No recommender was implemented or trained, and nothing was submitted to Kaggle.", ""]
    return "\n".join(lines)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "data/raw")
    parser.add_argument("--output-dir", type=Path, default=root / "data/processed")
    parser.add_argument("--report-dir", type=Path, default=root / "reports")
    parser.add_argument("--min-overlap-seconds", type=int, default=120)
    args = parser.parse_args()
    if args.min_overlap_seconds <= 0:
        parser.error("--min-overlap-seconds must be positive.")
    data_dir, output_dir, report_dir = (p.resolve() for p in (args.data_dir, args.output_dir, args.report_dir))
    for directory in (output_dir, report_dir):
        if directory == data_dir or data_dir in directory.parents:
            parser.error("Outputs and temporary files must be outside the original-data directory.")
    audit_paths = [root / "reports/data_audit.json", root / "reports/data_audit.md"]
    for path in [*(data_dir / name for name in INPUT_FILES), *audit_paths]:
        if not path.is_file():
            parser.error(f"Required input is missing: {path}")
    destinations = [output_dir / f"{w.name}_{suffix}.csv" for w in WINDOWS for suffix in ("interactions", "user_programmes")]
    destinations += [report_dir / "interaction_build.md", report_dir / "interaction_build.json"]
    if any(path.is_symlink() for path in destinations):
        parser.error("Refusing to overwrite a symbolic-link output destination.")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    generated_at = datetime.now(timezone.utc).isoformat()
    audit_before = {str(p.relative_to(root)): file_fingerprint(p) for p in audit_paths}
    audit = json.loads(audit_paths[0].read_text(encoding="utf-8"))
    raw_before = fingerprints(data_dir)
    for name, details in raw_before.items():
        if details["sha256"] != audit["input_integrity"][name]["after"]["sha256"]:
            raise RuntimeError(f"Input differs from the preserved audit: {name}")
    print("Original CSV hashes match the preserved audit; both audit reports fingerprinted.", flush=True)
    test_start = time.perf_counter()
    test_command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
    tests = subprocess.run(test_command, cwd=root, capture_output=True, text=True)
    test_output = tests.stdout + tests.stderr
    print(test_output, end="", flush=True)
    if tests.returncode:
        raise RuntimeError("Unit tests failed; interaction building was not started.")
    test_result = {"command": test_command, "returncode": tests.returncode, "output": test_output,
                   "runtime_seconds": time.perf_counter() - test_start}
    with tempfile.TemporaryDirectory(prefix="sweettv-interactions-", dir=output_dir) as temporary:
        result = build_interactions(data_dir, output_dir, Path(temporary), args.min_overlap_seconds, progress=True)
    print("Interaction CSVs written; rereading all six outputs for verification.", flush=True)
    verify_start = time.perf_counter()
    verification = verify_outputs(output_dir, result, progress=True)
    verification_seconds = time.perf_counter() - verify_start
    output_files = {p.name: file_fingerprint(p) for p in destinations[:6]}
    raw_after = fingerprints(data_dir)
    audit_after = {str(p.relative_to(root)): file_fingerprint(p) for p in audit_paths}
    protected = {f"data/raw/{name}": {"before": raw_before[name], "after": raw_after[name],
                                     "unchanged": raw_before[name] == raw_after[name]} for name in INPUT_FILES}
    protected.update({name: {"before": details, "after": audit_after[name], "unchanged": details == audit_after[name]}
                      for name, details in audit_before.items()})
    if not all(item["unchanged"] for item in protected.values()):
        raise RuntimeError("Protected input integrity check failed; no successful build report written.")
    report = {
        "schema_version": 1, "generated_at_utc": generated_at, "build": result,
        "output_verification": verification, "output_files": output_files, "protected_files": protected,
        "tests": test_result,
        "audit_context": {
            "audit_generated_at_utc": audit["generated_at_utc"],
            "raw_views_crossing_validation_cutoff": audit["viewing_log"]["quality"]["issue_counts"]["intervals_crossing_train_validation_boundary"],
            "duration_mismatch_delta_seconds": audit["viewing_log"]["elapsed_minus_reported_seconds_for_mismatches"],
            "missing_schedule_channels": audit["compatibility"]["channel_comparison"]["viewing_channels_missing_from_all_schedules"],
            "file_based_test_candidates": audit["compatibility"]["test_candidates"],
            "calendar_window_test_candidates": audit["compatibility"]["calendar_window_test_candidates"],
        },
        "execution": {
            "python_version": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
            "invocation": sys.argv, "data_directory": str(data_dir), "output_directory": str(output_dir),
            "script_sha256": file_fingerprint(Path(__file__))["sha256"],
            "audit_dependency_sha256": file_fingerprint(root / "src/audit_data.py")["sha256"],
            "interaction_tests_sha256": file_fingerprint(root / "tests/test_interactions.py")["sha256"],
            "stage_runtime_seconds": {**result["stage_runtime_seconds"], "unit_tests": test_result["runtime_seconds"],
                                      "verify_output_csvs": verification_seconds},
            "runtime_seconds_before_report_serialization": time.perf_counter() - started, **peak_memory(),
        },
    }
    (report_dir / "interaction_build.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (report_dir / "interaction_build.md").write_text(render_report(report), encoding="utf-8")
    print(json.dumps(result["splits"], indent=2), flush=True)
    print(f"All outputs verified; all five original CSVs and both audit reports unchanged.")
    print(f"Runtime including report writes: {time.perf_counter() - started:.3f} seconds")
    print(f"Peak main-process RSS: {report['execution'].get('peak_process_rss_mib', 'unavailable')} MiB")


if __name__ == "__main__":
    main()
