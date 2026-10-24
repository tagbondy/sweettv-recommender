"""Audit original SWEET.TV CSV files without fitting models or joining intervals.

Run from any directory with Python 3.13:
    /path/to/project/.venv/bin/python /path/to/project/src/audit_data.py

Only the standard library is used. CSV records are streamed; distinct identifiers,
duration frequency tables, and selected metadata variants are retained in memory.
Schedule duplicate fingerprints are indexed in a temporary, disk-backed SQLite DB.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import sqlite3
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path


VIEW_FILE = "dataset11-30.csv"
SCHEDULE_FILES = (
    "export_arh_11-20-final.csv",
    "export_arh_21-30-final.csv",
    "export_arh_31-42-final.csv",
)
INPUT_FILES = ("submission.csv", VIEW_FILE, *SCHEDULE_FILES)
SCHEDULE_WINDOWS = {
    SCHEDULE_FILES[0]: (datetime.fromisocalendar(2020, 11, 1), datetime.fromisocalendar(2020, 21, 1)),
    SCHEDULE_FILES[1]: (datetime.fromisocalendar(2020, 21, 1), datetime.fromisocalendar(2020, 31, 1)),
    SCHEDULE_FILES[2]: (datetime.fromisocalendar(2020, 31, 1), datetime.fromisocalendar(2020, 43, 1)),
}
VIEW_COLUMNS = ["user_id", "vsetv_id", "start_time", "stop_time", "duraton"]
SCHEDULE_COLUMNS = [
    "channel_id", "channel_title", "start_time", "duration", "tv_show_title",
    "tv_show_id", "tv_show_category", "tv_show_genre_1", "tv_show_genre_2",
    "tv_show_genre_3", "year_of_production", "director", "actors",
]
METADATA_COLUMNS = SCHEDULE_COLUMNS[4:5] + SCHEDULE_COLUMNS[6:]
VIEW_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", re.ASCII)
SCHEDULE_PATTERN = re.compile(r"\d{2}\.\d{2}\.\d{4} \d{2}:\d{2}:\d{2}", re.ASCII)
INTEGER_PATTERN = re.compile(r"-?\d+", re.ASCII)
USER_PATTERN = re.compile(r"\d+", re.ASCII)
SAMPLE_LIMIT = 5


def parse_timestamp(value: str, schedule: bool = False) -> datetime | None:
    """Parse and validate the documented second-resolution timestamp formats."""
    pattern = SCHEDULE_PATTERN if schedule else VIEW_PATTERN
    if not pattern.fullmatch(value):
        return None
    if schedule:
        value = f"{value[6:10]}-{value[3:5]}-{value[:2]}{value[10:]}"
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def parse_integer(value: str) -> int | None:
    if not INTEGER_PATTERN.fullmatch(value):
        return None
    return int(value)


def percentage(count: int, total: int) -> float:
    return 100.0 * count / total if total else 0.0


def iso(value: datetime | None) -> str | None:
    return value.isoformat(sep=" ") if value is not None else None


def normalized_text(value: str) -> str:
    return " ".join(value.split()).casefold()


class NumericSummary:
    """Exact integer-duration quantiles using frequencies rather than all rows."""

    def __init__(self) -> None:
        self.frequencies: Counter[int] = Counter()
        self.count = 0
        self.total = 0
        self.total_squared = 0

    def add(self, value: int) -> None:
        self.frequencies[value] += 1
        self.count += 1
        self.total += value
        self.total_squared += value * value

    def result(self) -> dict:
        if not self.count:
            return {"count": 0, "unit": "seconds"}
        probabilities = {"p01": .01, "p25": .25, "median": .5, "p75": .75,
                         "p90": .9, "p95": .95, "p99": .99}
        positions = {name: (self.count - 1) * p for name, p in probabilities.items()}
        needed = {index for pos in positions.values()
                  for index in (math.floor(pos), math.ceil(pos))}
        values = {}
        offset = 0
        for value, frequency in sorted(self.frequencies.items()):
            for index in needed:
                if offset <= index < offset + frequency:
                    values[index] = value
            offset += frequency
        quantiles = {}
        for name, pos in positions.items():
            lower, upper = math.floor(pos), math.ceil(pos)
            quantiles[name] = values[lower] + (values[upper] - values[lower]) * (pos - lower)
        mean = self.total / self.count
        return {
            "count": self.count, "unit": "seconds",
            "min": min(self.frequencies), "max": max(self.frequencies),
            "mean": mean,
            "population_stddev": math.sqrt(max(0.0, self.total_squared / self.count - mean ** 2)),
            **quantiles,
            "zero_count": self.frequencies[0],
            "over_24_hours_count": sum(n for value, n in self.frequencies.items() if value > 86400),
            "distinct_duration_values": len(self.frequencies),
        }


class Quality:
    def __init__(self, columns: list[str]) -> None:
        self.missing = Counter({column: 0 for column in columns})
        self.whitespace = Counter({column: 0 for column in columns})
        self.issues: Counter[str] = Counter()
        self.examples: dict[str, list[dict]] = defaultdict(list)

    def issue(self, name: str, row_number: int, value: str | None = None) -> None:
        self.issues[name] += 1
        if len(self.examples[name]) < SAMPLE_LIMIT:
            example = {"data_row": row_number}
            if value is not None:
                example["value"] = value[:200]
            self.examples[name].append(example)

    def inspect(self, row: dict, number: int) -> None:
        if None in row:
            self.issue("rows_with_extra_fields", number)
        if any(value is None for key, value in row.items() if key is not None):
            self.issue("rows_with_missing_fields", number)
        for key in self.missing:
            value = row.get(key) or ""
            self.missing[key] += not value.strip()
            self.whitespace[key] += value != value.strip()

    def result(self) -> dict:
        return {"missing_or_blank_by_column": dict(self.missing),
                "surrounding_whitespace_by_column": dict(self.whitespace),
                "issue_counts": dict(sorted(self.issues.items())),
                "issue_examples": dict(self.examples)}


def records(path: Path, expected_columns: list[str]):
    """Yield parsed records; reject unexpected schemas rather than guessing."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_columns:
            raise ValueError(f"Unexpected columns in {path.name}: {reader.fieldnames!r}")
        for number, row in enumerate(reader, 1):
            yield number, row


def fingerprints(data_dir: Path) -> dict:
    result = {}
    for name in INPUT_FILES:
        path = data_dir / name
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"Input changed while being checksummed: {name}")
        result[name] = {"size_bytes": after.st_size, "mtime_ns": after.st_mtime_ns,
                        "sha256": digest.hexdigest()}
    return result


def audit_submission(path: Path) -> tuple[dict, set[str]]:
    quality = Quality(["user_id", "tv_show_id"])
    users: set[str] = set()
    count = 0
    slot_counts: Counter[int] = Counter()
    for count, row in records(path, ["user_id", "tv_show_id"]):
        quality.inspect(row, count)
        user = row["user_id"] or ""
        if not USER_PATTERN.fullmatch(user):
            quality.issue("invalid_user_id", count, user)
        if user in users:
            quality.issue("duplicate_user_id", count, user)
        if user:
            users.add(user)
        tokens = (row["tv_show_id"] or "").split()
        slot_counts[len(tokens)] += 1
        if len(tokens) != 5:
            quality.issue("prediction_count_not_five", count)
        if any(parse_integer(token) is None or int(token) < 0 for token in tokens):
            quality.issue("invalid_placeholder_id", count)
    return {"file": path.name, "columns": ["user_id", "tv_show_id"],
            "data_rows": count, "unique_user_ids": len(users),
            "matches_expected_1260_unique_users": count == len(users) == 1260,
            "prediction_slot_counts": dict(sorted(slot_counts.items())),
            "quality": quality.result()}, users


def audit_views(path: Path, test_users: set[str]) -> tuple[dict, set[int]]:
    quality = Quality(VIEW_COLUMNS)
    quality.issues.update(dict.fromkeys((
        "rows_with_extra_fields", "rows_with_missing_fields", "invalid_user_id",
        "invalid_channel_id", "invalid_duration_value", "negative_duration",
        "invalid_start_timestamp", "invalid_stop_timestamp",
        "intervals_with_unparseable_timestamps", "intervals_with_nonpositive_elapsed_time",
        "duration_elapsed_mismatch", "intervals_crossing_midnight", "intervals_crossing_iso_week",
        "intervals_crossing_train_validation_boundary",
    ), 0))
    users: set[str] = set()
    channels: Counter[int] = Counter()
    durations, elapsed_durations = NumericSummary(), NumericSummary()
    duration_deltas: Counter[int] = Counter()
    timestamps = {"start_min": None, "start_max": None, "stop_min": None, "stop_max": None}
    by_week: Counter[str] = Counter()
    target_windows = {"weeks_11_20": set(), "weeks_21_30": set()}
    earliest_start, latest_start = {}, {}
    count = valid_intervals = 0
    for count, row in records(path, VIEW_COLUMNS):
        quality.inspect(row, count)
        user = row["user_id"] or ""
        if user:
            users.add(user)
        if not USER_PATTERN.fullmatch(user):
            quality.issue("invalid_user_id", count, user)
        channel = parse_integer(row["vsetv_id"] or "")
        if channel is None or channel <= 0:
            quality.issue("invalid_channel_id", count, row["vsetv_id"])
        else:
            channels[channel] += 1
        duration = parse_integer(row["duraton"] or "")
        if duration is None:
            quality.issue("invalid_duration_value", count, row["duraton"])
        elif duration < 0:
            quality.issue("negative_duration", count, row["duraton"])
        else:
            durations.add(duration)
        start = parse_timestamp(row["start_time"] or "")
        stop = parse_timestamp(row["stop_time"] or "")
        for label, value in (("start", start), ("stop", stop)):
            if value is None:
                quality.issue(f"invalid_{label}_timestamp", count, row[f"{label}_time"])
            else:
                low, high = f"{label}_min", f"{label}_max"
                timestamps[low] = value if timestamps[low] is None else min(timestamps[low], value)
                timestamps[high] = value if timestamps[high] is None else max(timestamps[high], value)
        if start is None or stop is None:
            quality.issue("intervals_with_unparseable_timestamps", count)
        elif stop <= start:
            quality.issue("intervals_with_nonpositive_elapsed_time", count)
        else:
            valid_intervals += 1
            elapsed = int((stop - start).total_seconds())
            elapsed_durations.add(elapsed)
            if duration is not None and duration != elapsed:
                duration_deltas[elapsed - duration] += 1
                quality.issue("duration_elapsed_mismatch", count,
                              f"start={row['start_time']}; stop={row['stop_time']}; "
                              f"reported={duration}; elapsed={elapsed}; delta={elapsed - duration}")
            if start.date() != stop.date():
                quality.issue("intervals_crossing_midnight", count)
            if start.isocalendar()[:2] != stop.isocalendar()[:2]:
                quality.issue("intervals_crossing_iso_week", count)
            if start < datetime(2020, 5, 18) < stop:
                quality.issue("intervals_crossing_train_validation_boundary", count)
        if start is not None:
            year, week, _ = start.isocalendar()
            by_week[f"{year}-W{week:02}"] += 1
            if channel is not None and channel > 0:
                earliest_start[channel] = min(earliest_start.get(channel, start), start)
                latest_start[channel] = max(latest_start.get(channel, start), start)
            if user in test_users and year == 2020:
                if 11 <= week <= 20:
                    target_windows["weeks_11_20"].add(user)
                if 21 <= week <= 30:
                    target_windows["weeks_21_30"].add(user)
    issues = quality.issues
    return {
        "file": path.name, "columns": VIEW_COLUMNS, "viewing_events": count,
        "unique_nonempty_users": len(users), "unique_valid_channels": len(channels),
        "channel_ids": sorted(channels),
        "events_by_channel": {str(key): value for key, value in sorted(channels.items())},
        "timestamp_format": "YYYY-MM-DD HH:MM:SS", "timestamps": {k: iso(v) for k, v in timestamps.items()},
        "events_by_start_iso_week": dict(sorted(by_week.items())),
        "channel_start_ranges": {str(key): {"min": iso(earliest_start[key]), "max": iso(latest_start[key])}
                                 for key in sorted(earliest_start)},
        "reported_duration_statistics": durations.result(),
        "positive_elapsed_duration_statistics": elapsed_durations.result(),
        "elapsed_minus_reported_seconds_for_mismatches": {
            str(key): value for key, value in sorted(duration_deltas.items())},
        "valid_positive_timestamp_intervals": valid_intervals,
        "invalid_timestamp_intervals": issues["intervals_with_unparseable_timestamps"]
                                       + issues["intervals_with_nonpositive_elapsed_time"],
        "test_user_coverage": {
            "template_unique_users": len(test_users), "in_any_viewing_history": len(test_users & users),
            "absent_from_viewing_history": len(test_users - users),
            "absent_user_examples": sorted(test_users - users)[:SAMPLE_LIMIT],
            "with_event_start_in_2020_weeks_11_20": len(target_windows["weeks_11_20"]),
            "with_event_start_in_2020_weeks_21_30": len(target_windows["weeks_21_30"]),
        },
        "quality": quality.result(),
    }, set(channels)


class DuplicateIndex:
    """Count duplicate parsed records using disk-backed SHA-256 fingerprints."""

    def __init__(self, path: Path, batch_size: int = 5000) -> None:
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=OFF")
        self.connection.execute("PRAGMA synchronous=OFF")
        self.connection.execute("PRAGMA cache_size=-8192")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.execute("CREATE TABLE records (digest BLOB PRIMARY KEY) WITHOUT ROWID")
        self.batch_size = batch_size
        self.pending: list[tuple[bytes]] = []
        self.duplicates = 0

    def add(self, values: list) -> None:
        encoded = json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.pending.append((hashlib.sha256(encoded).digest(),))
        if len(self.pending) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        before = self.connection.total_changes
        self.connection.executemany("INSERT OR IGNORE INTO records VALUES (?)", self.pending)
        self.duplicates += len(self.pending) - (self.connection.total_changes - before)
        self.connection.commit()
        self.pending.clear()

    def close(self) -> int:
        self.flush()
        self.connection.close()
        return self.duplicates


def metadata_summary(metadata: dict) -> dict:
    result = {}
    for field in ("tv_show_title", "tv_show_category"):
        varying = sorted(key for key, fields in metadata.items() if len(fields[field]) > 1)
        result[field] = {
            "programme_ids_with_multiple_nonblank_normalized_values": len(varying),
            "examples": [{"tv_show_id": key, "values": sorted(metadata[key][field])[:5]}
                         for key in varying[:SAMPLE_LIMIT]],
        }
    return result


def audit_schedule(path: Path, scratch: Path) -> tuple[dict, dict]:
    quality = Quality(SCHEDULE_COLUMNS)
    quality.issues.update(dict.fromkeys((
        "rows_with_extra_fields", "rows_with_missing_fields", "invalid_channel_id",
        "invalid_programme_id", "invalid_duration_value", "negative_duration", "zero_duration",
        "invalid_start_timestamp", "broadcast_end_overflow",
    ), 0))
    channels: set[int] = set()
    programme_ids: set[int] = set()
    valid_airing_ids: set[int] = set()
    metadata = defaultdict(lambda: {"tv_show_title": set(), "tv_show_category": set()})
    missing_labeled = Counter({key: 0 for key in METADATA_COLUMNS})
    durations = NumericSummary()
    by_week: Counter[str] = Counter()
    ranges = {"start_min": None, "start_max": None, "end_min": None, "end_max": None}
    labeled_rows = zero_rows = valid_intervals = 0
    channel_ranges = {}
    window = SCHEDULE_WINDOWS.get(path.name)
    window_counts: Counter[str] = Counter({
        "starts_before_window": 0, "starts_at_or_after_window_end": 0,
        "positive_intervals_overlapping_window": 0,
        "positive_intervals_crossing_window_start": 0,
        "positive_intervals_crossing_window_end": 0,
    })
    window_ids: set[int] = set()
    duplicates = DuplicateIndex(scratch / f"{path.stem}.sqlite3")
    count = 0
    try:
        for count, row in records(path, SCHEDULE_COLUMNS):
            quality.inspect(row, count)
            duplicates.add([row[key] for key in SCHEDULE_COLUMNS] + [row.get(None)])
            channel = parse_integer(row["channel_id"] or "")
            if channel is None or channel <= 0:
                quality.issue("invalid_channel_id", count, row["channel_id"])
            else:
                channels.add(channel)
            programme = parse_integer(row["tv_show_id"] or "")
            if programme is None or programme < 0:
                quality.issue("invalid_programme_id", count, row["tv_show_id"])
            elif programme == 0:
                zero_rows += 1
            else:
                labeled_rows += 1
                programme_ids.add(programme)
                for key in METADATA_COLUMNS:
                    missing_labeled[key] += not (row[key] or "").strip()
                for key in ("tv_show_title", "tv_show_category"):
                    text = normalized_text(row[key] or "")
                    if text:
                        metadata[programme][key].add(text)
            duration = parse_integer(row["duration"] or "")
            if duration is None:
                quality.issue("invalid_duration_value", count, row["duration"])
            elif duration < 0:
                quality.issue("negative_duration", count, row["duration"])
            else:
                durations.add(duration)
                if duration == 0:
                    quality.issue("zero_duration", count)
            start = parse_timestamp(row["start_time"] or "", schedule=True)
            end = None
            if start is None:
                quality.issue("invalid_start_timestamp", count, row["start_time"])
            else:
                year, week, _ = start.isocalendar()
                by_week[f"{year}-W{week:02}"] += 1
                ranges["start_min"] = min(ranges["start_min"] or start, start)
                ranges["start_max"] = max(ranges["start_max"] or start, start)
                if window is not None:
                    window_counts["starts_before_window"] += start < window[0]
                    window_counts["starts_at_or_after_window_end"] += start >= window[1]
                if duration is not None and duration >= 0:
                    try:
                        end = start + timedelta(seconds=duration)
                    except OverflowError:
                        quality.issue("broadcast_end_overflow", count)
            if end is not None:
                ranges["end_min"] = min(ranges["end_min"] or end, end)
                ranges["end_max"] = max(ranges["end_max"] or end, end)
                if duration > 0:
                    valid_intervals += 1
                    if window is not None:
                        window_counts["positive_intervals_crossing_window_start"] += start < window[0] < end
                        window_counts["positive_intervals_crossing_window_end"] += start < window[1] < end
                        if start < window[1] and end > window[0]:
                            window_counts["positive_intervals_overlapping_window"] += 1
                            if programme is not None and programme > 0 and channel is not None and channel > 0:
                                window_ids.add(programme)
                    if programme is not None and programme > 0 and channel is not None and channel > 0:
                        valid_airing_ids.add(programme)
                    if channel is not None and channel > 0:
                        low, high = channel_ranges.get(channel, (start, end))
                        channel_ranges[channel] = (min(low, start), max(high, end))
    finally:
        duplicate_rows = duplicates.close()
    return {
        "file": path.name, "columns": SCHEDULE_COLUMNS, "rows": count,
        "unique_valid_channels": len(channels), "channel_ids": sorted(channels),
        "unique_nonzero_programme_ids": len(programme_ids), "labeled_rows": labeled_rows,
        "zero_id_rows": zero_rows, "zero_id_percentage": percentage(zero_rows, count),
        "labeled_programmes_with_valid_positive_airings": len(valid_airing_ids),
        "valid_positive_timestamp_intervals": valid_intervals,
        "timestamp_format": "DD.MM.YYYY HH:MM:SS", "timestamps": {k: iso(v) for k, v in ranges.items()},
        "broadcasts_by_start_iso_week": dict(sorted(by_week.items())),
        "nominal_calendar_window": {
            "start_inclusive": iso(window[0]), "end_exclusive": iso(window[1]),
            **window_counts,
            "labeled_programme_ids_with_valid_airings_overlapping_window": len(window_ids),
            "file_programme_ids_without_valid_airings_in_window": len(programme_ids - window_ids),
        } if window is not None else None,
        "duration_statistics": durations.result(),
        "missing_metadata_labeled_rows": dict(missing_labeled),
        "duplicate_parsed_records_after_first": duplicate_rows,
        "distinct_parsed_records": count - duplicate_rows,
        "programme_metadata_variants": metadata_summary(metadata),
        "quality": quality.result(),
    }, {"programme_ids": programme_ids, "channels": channels, "metadata": metadata,
        "channel_ranges": channel_ranges, "window_programme_ids": window_ids}


def compatibility(views: dict, view_channels: set[int], schedules: list[dict], states: list[dict]) -> dict:
    combined_channels = set().union(*(state["channels"] for state in states))
    earlier_ids = states[0]["programme_ids"] | states[1]["programme_ids"]
    test_ids = states[2]["programme_ids"]
    known = test_ids & earlier_ids
    new = test_ids - earlier_ids
    earlier_window_ids = states[0]["window_programme_ids"] | states[1]["window_programme_ids"]
    test_window_ids = states[2]["window_programme_ids"]
    pairs = []
    for left in range(len(states)):
        for right in range(left + 1, len(states)):
            a, b = states[left]["programme_ids"], states[right]["programme_ids"]
            pairs.append({"left": schedules[left]["file"], "right": schedules[right]["file"],
                          "shared_programme_ids": len(a & b), "only_left": len(a - b), "only_right": len(b - a)})
    all_metadata = defaultdict(lambda: {"tv_show_title": set(), "tv_show_category": set()})
    for state in states:
        for programme, fields in state["metadata"].items():
            for key, values in fields.items():
                all_metadata[programme][key].update(values)
    by_schedule = []
    for schedule, state in zip(schedules, states):
        missing = view_channels - state["channels"]
        by_schedule.append({
            "file": schedule["file"],
            "shared_viewing_channel_count": len(view_channels & state["channels"]),
            "viewing_channels_missing_from_schedule": sorted(missing),
            "schedule_channels_absent_from_view_log": sorted(state["channels"] - view_channels),
        })
    # This is deliberately a coarse envelope check, not an interval join.
    training_ranges = {}
    for state in states[:2]:
        for channel, (low, high) in state["channel_ranges"].items():
            old_low, old_high = training_ranges.get(channel, (low, high))
            training_ranges[channel] = (min(old_low, low), max(old_high, high))
    coarse_outside = []
    for channel, timestamps in views["channel_start_ranges"].items():
        bounds = training_ranges.get(int(channel))
        if bounds is None:
            continue
        low, high = bounds
        if datetime.fromisoformat(timestamps["min"]) < low or datetime.fromisoformat(timestamps["max"]) >= high:
            coarse_outside.append(int(channel))
    return {
        "channel_comparison": {
            "viewing_channels": len(view_channels), "schedule_channels_union": len(combined_channels),
            "shared_channels": len(view_channels & combined_channels),
            "viewing_channels_missing_from_all_schedules": sorted(view_channels - combined_channels),
            "view_events_on_channels_missing_from_all_schedules": sum(
                views["events_by_channel"][str(key)] for key in view_channels - combined_channels),
            "by_schedule": by_schedule,
        },
        "time_compatibility": {
            "formats_differ_but_parse_to_same_naive_datetime_type": True,
            "timezone_supplied_in_source": False,
            "viewing_channel_start_extrema_outside_training_schedule_envelope": sorted(coarse_outside),
            "note": "Envelope checks cannot establish actual schedule coverage or programme matches.",
        },
        "programme_id_overlap": pairs,
        "test_candidates": {
            "total_labeled_programme_ids": len(test_ids),
            "seen_in_either_earlier_schedule": len(known),
            "seen_in_either_earlier_schedule_percentage": percentage(len(known), len(test_ids)),
            "seen_in_weeks_11_20_schedule": len(test_ids & states[0]["programme_ids"]),
            "seen_in_weeks_21_30_schedule": len(test_ids & states[1]["programme_ids"]),
            "new_relative_to_both_earlier_schedules": len(new),
            "new_programme_id_examples": sorted(new)[:SAMPLE_LIMIT],
        },
        "calendar_window_test_candidates": {
            "total_labeled_programme_ids_with_valid_test_window_airings": len(test_window_ids),
            "also_in_valid_airings_in_earlier_nominal_windows": len(test_window_ids & earlier_window_ids),
            "new_relative_to_earlier_nominal_windows": len(test_window_ids - earlier_window_ids),
            "note": "Uses half-open nominal ISO-week windows, not just schedule filenames.",
        },
        "programme_metadata_variants_across_all_schedules": metadata_summary(all_metadata),
    }


def peak_memory() -> dict:
    try:
        import resource
        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        size = raw if sys.platform == "darwin" else raw * 1024
        return {"peak_process_rss_bytes": size, "peak_process_rss_mib": size / 1024 ** 2,
                "method": "resource.getrusage(RUSAGE_SELF).ru_maxrss; bytes on macOS, KiB on Linux"}
    except ImportError:
        return {"peak_process_rss_bytes": None, "method": "resource module unavailable"}


def build_findings(report: dict) -> list[str]:
    views, schedules = report["viewing_log"], report["schedules"]
    findings = [
        "Zero programme IDs denote unlabeled airtime; never use them as recommendable items.",
        "History coverage is not labeled-interaction coverage. Cold-start after the interval join is not yet known.",
        "Candidate presence in earlier schedules does not imply that any user watched the programme.",
        "Missing metadata may weaken content profiles; multiple titles/categories per ID are diagnostic variants, not proven ID errors.",
        "No timezone is supplied. Matching requires an explicit common-timezone assumption, not an invented conversion.",
        "Split or clip intervals at temporal cutoffs before constructing train/validation interactions; do not assign a crossing interval wholly to training.",
        "Restrict future schedules to permitted availability information. No future viewing behaviour was used.",
    ]
    for schedule in schedules:
        findings.append(f"{schedule['file']}: {schedule['zero_id_percentage']:.4f}% unlabeled rows; "
                        f"{schedule['duplicate_parsed_records_after_first']:,} duplicate parsed records after first occurrences.")
        window = schedule["nominal_calendar_window"]
        if window is not None:
            findings.append(f"{schedule['file']}: {window['starts_before_window']:,} broadcasts start before "
                            f"the nominal window and {window['starts_at_or_after_window_end']:,} start at/after its end. "
                            "Filter by timestamps; filenames are not exact temporal partitions.")
    issues = views["quality"]["issue_counts"]
    findings.append(f"Viewing intervals: {views['invalid_timestamp_intervals']:,} invalid; "
                    f"{issues.get('duration_elapsed_mismatch', 0):,} reported-duration mismatches; "
                    f"{issues.get('intervals_crossing_train_validation_boundary', 0):,} cross the validation cutoff.")
    findings.append("Elapsed minus reported duration for mismatched views (seconds: count): "
                    f"{views['elapsed_minus_reported_seconds_for_mismatches']}. "
                    "A systematic offset needs investigation; it does not by itself establish a timezone or justify a correction.")
    missing = report["compatibility"]["channel_comparison"]["viewing_channels_missing_from_all_schedules"]
    findings.append(f"Viewing channel IDs missing from every schedule: {missing}.")
    candidates = report["compatibility"]["test_candidates"]
    findings.append(f"{candidates['new_relative_to_both_earlier_schedules']:,} test-period programme IDs are absent from both earlier schedules.")
    return findings


def markdown(report: dict) -> str:
    """Render complete measured results, with detailed JSON blocks where useful."""
    views = report["viewing_log"]
    lines = ["# SWEET.TV Data Audit", "", f"Generated (UTC): {report['generated_at_utc']}", "",
             "## Reproduction and scope", "", "```bash", ".venv/bin/python src/audit_data.py", "```", "",
             "Standard-library streaming CSV audit. No interval join, recommendation fitting, or Kaggle submission was performed.", "",
             "## Environment and measured runtime", "", "```json",
             json.dumps(report["execution"], indent=2), "```", "",
             "## Input integrity", "",
             "SHA-256, size, and modification time were checked before and after processing. All inputs were opened read-only.", "",
             "| File | Bytes | SHA-256 | Unchanged |", "|---|---:|---|---|"]
    for name, details in report["input_integrity"].items():
        lines.append(f"| {name} | {details['before']['size_bytes']:,} | `{details['before']['sha256']}` | {details['unchanged']} |")
    lines += ["", "## Submission template", "", "```json",
              json.dumps(report["submission_template"], indent=2), "```", "",
              "## Viewing log", "",
              f"- Viewing events: **{views['viewing_events']:,}**",
              f"- Unique nonempty user IDs: **{views['unique_nonempty_users']:,}**",
              f"- Unique valid channels: **{views['unique_valid_channels']:,}**",
              f"- Invalid timestamp intervals: **{views['invalid_timestamp_intervals']:,}**", ""]
    for key in ("timestamps", "reported_duration_statistics", "positive_elapsed_duration_statistics",
                "elapsed_minus_reported_seconds_for_mismatches",
                "test_user_coverage", "events_by_start_iso_week", "quality"):
        lines += [f"### {key.replace('_', ' ').capitalize()}", "", "```json", json.dumps(views[key], indent=2), "```", ""]
    lines += ["## Broadcast schedules", "",
              "| File | Rows | Channels | Nonzero programmes | Zero ID rows (%) | Duplicate rows |",
              "|---|---:|---:|---:|---:|---:|"]
    for schedule in report["schedules"]:
        lines.append(f"| {schedule['file']} | {schedule['rows']:,} | {schedule['unique_valid_channels']:,} | "
                     f"{schedule['unique_nonzero_programme_ids']:,} | {schedule['zero_id_percentage']:.4f} | "
                     f"{schedule['duplicate_parsed_records_after_first']:,} |")
    for schedule in report["schedules"]:
        details = {key: value for key, value in schedule.items() if key not in ("columns", "channel_ids")}
        lines += ["", f"### {schedule['file']}", "", "```json", json.dumps(details, indent=2, ensure_ascii=False), "```"]
    lines += ["", "## Data compatibility", "", "```json",
              json.dumps(report["compatibility"], indent=2, ensure_ascii=False), "```", "",
              "## Data quality findings and technical risks", ""]
    lines += [f"- {finding}" for finding in report["findings_and_risks"]]
    lines += ["", "## Assumptions and limitations", ""]
    lines += [f"- {item}" for item in report["assumptions_and_limitations"]]
    lines += ["", "The companion `data_audit.json` includes complete channel lists and per-channel viewing ranges.", ""]
    return "\n".join(lines)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "data" / "raw")
    parser.add_argument("--report-dir", type=Path, default=root / "reports")
    args = parser.parse_args()
    data_dir, report_dir = args.data_dir.resolve(), args.report_dir.resolve()
    if report_dir == data_dir or data_dir in report_dir.parents:
        parser.error("Reports and temporary files must be outside the original data directory.")
    for name in INPUT_FILES:
        if not (data_dir / name).is_file():
            parser.error(f"Required input is missing: {data_dir / name}")
    report_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    generated = datetime.now(timezone.utc).isoformat()
    stages = {}

    def timed(label, function, *arguments):
        begin = time.perf_counter()
        result = function(*arguments)
        stages[label] = time.perf_counter() - begin
        print(f"{label}: {stages[label]:.3f} seconds", flush=True)
        return result

    before = timed("input_checksums_before", fingerprints, data_dir)
    submission, users = timed("submission_template", audit_submission, data_dir / "submission.csv")
    views, channels = timed("viewing_log", audit_views, data_dir / VIEW_FILE, users)
    schedules, states = [], []
    with tempfile.TemporaryDirectory(prefix="sweettv-audit-", dir=report_dir) as temporary:
        for name in SCHEDULE_FILES:
            result, state = timed(name, audit_schedule, data_dir / name, Path(temporary))
            schedules.append(result)
            states.append(state)
    compatible = timed("compatibility", compatibility, views, channels, schedules, states)
    after = timed("input_checksums_after", fingerprints, data_dir)
    integrity = {name: {"before": before[name], "after": after[name], "unchanged": before[name] == after[name]}
                 for name in INPUT_FILES}
    if not all(item["unchanged"] for item in integrity.values()):
        raise RuntimeError("Original input integrity check failed; successful audit reports were not written.")
    report = {
        "audit_schema_version": 1, "generated_at_utc": generated,
        "execution": {
            "python_version": sys.version, "python_executable": sys.executable,
            "python_implementation": platform.python_implementation(), "platform": platform.platform(),
            "cpu_count": os.cpu_count(), "data_directory": str(data_dir), "report_directory": str(report_dir),
            "invocation": sys.argv, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "stage_runtime_seconds": stages,
            "runtime_seconds_before_report_serialization": time.perf_counter() - started,
            **peak_memory(),
        },
        "all_expected_input_files_processed": len(schedules) == 3 and set(integrity) == set(INPUT_FILES),
        "input_integrity": integrity, "submission_template": submission,
        "viewing_log": views, "schedules": schedules, "compatibility": compatible,
        "assumptions_and_limitations": [
            "CSV data rows exclude the header and blank physical lines skipped by csv.DictReader; quoted multiline fields remain one record.",
            "Missing means an absent field, empty string, or whitespace-only string. Nonblank placeholders in metadata are not classified as missing.",
            "user_id values are kept as their original strings; numeric parsing is used only for channel/programme IDs and durations.",
            "Timestamp parsing is strict. Invalid intervals include unparseable timestamps or stop <= start; duration-field errors are reported separately.",
            "Duration statistics include all syntactically valid nonnegative integer durations, including zero; negative and malformed values are excluded.",
            "Quantiles are exact linearly interpolated order statistics computed from integer frequency tables; standard deviation is population-based.",
            "Weeks mean ISO weeks in 2020 for template-user coverage. The train/validation cutoff checked is 2020-05-18 00:00:00; no timezone is asserted.",
            "Each schedule's candidate count is the number of distinct positive IDs in that file; valid-airing counts additionally require a positive duration, valid time, and channel.",
            "Nominal calendar-window availability additionally requires positive interval overlap with the half-open ISO-week window; schedule padding is counted separately. No user-programme interval join is performed.",
            "Schedule duplicates compare all parsed field values, including extra/missing-field information, using SHA-256 fingerprints in disk-backed SQLite. Hash collisions are theoretically possible; raw quoting differences are not distinct records.",
            "Duplicate counts mean repetitions after the first occurrence, not the number of duplicate groups. Viewing-log duplicates were not checked.",
            "ID consistency checks compare numeric ID overlap and normalized nonblank title/category variants; they cannot establish semantic identity or detect every mislabeled programme.",
            "Memory grows with unique IDs, metadata variants, duration values, and a 5,000-record fingerprint batch, not with all CSV rows. SQLite uses an approximately 8 MiB page cache.",
            "Only channel-set and coarse timestamp-envelope compatibility are checked. Full schedule gaps, overlapping airings, watch thresholds, and labeled user interactions require a later interval join.",
            "Runtime includes both full input checksum passes, CSV analysis, temporary duplicate-index cleanup, and compatibility checks; final report serialization is excluded.",
            "Programme titles in diagnostic examples are original source values normalized for comparison; report labels and explanations are English.",
        ],
    }
    report["findings_and_risks"] = build_findings(report)
    (report_dir / "data_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (report_dir / "data_audit.md").write_text(markdown(report), encoding="utf-8")
    print(f"All five inputs processed; all input checksums unchanged. Reports: {report_dir}")
    print(f"Measured runtime before report serialization: {report['execution']['runtime_seconds_before_report_serialization']:.3f} seconds")
    print(f"Total runtime including report writes: {time.perf_counter() - started:.3f} seconds")


if __name__ == "__main__":
    main()
