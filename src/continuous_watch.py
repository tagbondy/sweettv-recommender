"""Reconstruct occurrence-level, single-interval viewing fractions from raw data.

Original broadcast durations and boundaries are retained. Existing 120-second
event CSVs are deliberately not used as the reconstruction input: they omit short
qualifying sessions and clip some programme boundaries. No disconnected viewing
intervals are joined together. Only the maximum individual overlap per user,
broadcast, and chronological window can qualify that occurrence.
"""

from __future__ import annotations

import csv
import ast
import hashlib
import math
import sqlite3
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from audit_data import SCHEDULE_COLUMNS, SCHEDULE_FILES, USER_PATTERN, VIEW_COLUMNS, VIEW_FILE, parse_integer, parse_timestamp
from build_interactions import Airing, ScheduleIndex, View, Window, WINDOWS, checked_records, seconds, timestamp


EARLY_CUTOFF = seconds(datetime.fromisocalendar(2020, 16, 1))
EVIDENCE_WINDOWS = (
    Window("full", WINDOWS[2].start, WINDOWS[2].end),
    Window("train", WINDOWS[0].start, WINDOWS[0].end),
    Window("validation", WINDOWS[1].start, WINDOWS[1].end),
    Window("early_train", WINDOWS[0].start, EARLY_CUTOFF),
    Window("early_validation", EARLY_CUTOFF, WINDOWS[0].end),
)
WINDOW_BY_NAME = {window.name: window for window in EVIDENCE_WINDOWS}
WINDOW_FIELDS = ("overlap", "source_row", "last30", "last50", "last80")
PROFILE_COLUMNS = [
    "user_id", "tv_show_id", "frequency_ge30", "frequency_gt50", "frequency_gt80",
    "recency_gt50_half14", "recency_gt50_half28", "qualified_seconds_gt50",
    "last_ge30_seconds", "last_gt50_seconds", "last_gt80_seconds", "has_120_second_evidence",
]


def qualifies(overlap: int, duration: int, threshold: str) -> bool:
    """Use integer comparisons so exactly 30%, 50%, and 80% are unambiguous."""
    if duration <= 0 or not 0 <= overlap <= duration:
        raise ValueError("A positive original broadcast duration and bounded overlap are required.")
    if threshold == "ge30":
        return overlap > 0 and 10 * overlap >= 3 * duration
    if threshold == "gt50":
        return 2 * overlap > duration
    if threshold == "gt80":
        return 5 * overlap > 4 * duration
    raise ValueError("Unknown watch-fraction threshold.")


def session_values(view: View, airing: Airing, windows=EVIDENCE_WINDOWS) -> list[int]:
    """One session contributes maxima and qualified endpoints, never a sum."""
    duration = airing.end - airing.start  # End was constructed from raw schedule.duration.
    values = []
    for window in windows:
        low = max(view.start, airing.start, window.start)
        high = min(view.end, airing.end, window.end)
        overlap = max(0, high - low) if view.channel_id == airing.channel_id else 0
        values.extend((overlap, view.source_row if overlap else 0,
                       high if qualifies(overlap, duration, "ge30") else 0,
                       high if qualifies(overlap, duration, "gt50") else 0,
                       high if qualifies(overlap, duration, "gt80") else 0))
    return values


def open_database(path: Path, readonly: bool = False) -> sqlite3.Connection:
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) if readonly else sqlite3.connect(path)
    connection.execute("PRAGMA cache_size=-16384")
    connection.execute("PRAGMA temp_store=FILE")
    if not readonly:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
    return connection


def create_tables(connection: sqlite3.Connection) -> None:
    fields = []
    checks = []
    for window in EVIDENCE_WINDOWS:
        for field in WINDOW_FIELDS:
            fields.append(f"{window.name}_{field} INTEGER NOT NULL")
        checks.append(f"CHECK ({window.name}_overlap >= 0 AND {window.name}_overlap <= broadcast_duration)")
    connection.execute("""CREATE TABLE occurrences (
        airing_start INTEGER, channel_id INTEGER, programme_id INTEGER, user_id TEXT,
        airing_end INTEGER, broadcast_duration INTEGER NOT NULL CHECK(broadcast_duration > 0),
        source_sessions INTEGER NOT NULL,
    """ + ", ".join(fields + checks) + """,
        PRIMARY KEY (airing_start, channel_id, programme_id, user_id, airing_end),
        CHECK (airing_end - airing_start = broadcast_duration), CHECK(programme_id > 0)
    ) WITHOUT ROWID""")
    connection.execute("""CREATE TABLE seen_views (
        start INTEGER, end INTEGER, channel_id INTEGER, user_id TEXT,
        PRIMARY KEY(start, end, channel_id, user_id)
    ) WITHOUT ROWID""")


class OccurrenceWriter:
    """Bounded batch of maximum single-session evidence, with source-row lineage."""

    def __init__(self, connection: sqlite3.Connection, limit: int = 5000):
        self.connection, self.limit = connection, limit
        self.pending = {}
        updates = ["source_sessions = source_sessions + excluded.source_sessions"]
        for window in EVIDENCE_WINDOWS:
            prefix = window.name
            updates += [
                f"{prefix}_source_row = CASE WHEN excluded.{prefix}_overlap > occurrences.{prefix}_overlap THEN excluded.{prefix}_source_row ELSE occurrences.{prefix}_source_row END",
                f"{prefix}_overlap = MAX({prefix}_overlap, excluded.{prefix}_overlap)",
                *(f"{prefix}_{field} = MAX({prefix}_{field}, excluded.{prefix}_{field})" for field in ("last30", "last50", "last80")),
            ]
        slots = ",".join("?" for _ in range(7 + 5 * len(EVIDENCE_WINDOWS)))
        self.sql = f"INSERT INTO occurrences VALUES ({slots}) ON CONFLICT(airing_start,channel_id,programme_id,user_id,airing_end) DO UPDATE SET " + ",".join(updates)

    def add(self, view: View, airing: Airing) -> None:
        key = (airing.start, airing.channel_id, airing.programme_id, view.user_id, airing.end)
        evidence = session_values(view, airing)
        if not evidence[0]:
            return
        previous = self.pending.get(key)
        if previous is None:
            self.pending[key] = [airing.end - airing.start, 1, *evidence]
        else:
            previous[1] += 1
            for index in range(len(EVIDENCE_WINDOWS)):
                offset = 2 + 5 * index
                incoming = 5 * index
                if evidence[incoming] > previous[offset]:
                    previous[offset:offset + 2] = evidence[incoming:incoming + 2]
                for field in (2, 3, 4):
                    previous[offset + field] = max(previous[offset + field], evidence[incoming + field])
        if len(self.pending) >= self.limit:
            self.flush()

    def flush(self) -> None:
        self.connection.executemany(self.sql, (key + tuple(values) for key, values in sorted(self.pending.items())))
        self.pending.clear()
        self.connection.commit()


def load_original_airings(raw_dir: Path) -> tuple[ScheduleIndex, dict]:
    counts = Counter()

    def airings():
        for name in SCHEDULE_FILES[:2]:
            for _, row in checked_records(raw_dir / name, SCHEDULE_COLUMNS):
                counts[f"{name}:rows_read"] += 1
                programme, channel = parse_integer(row["tv_show_id"]), parse_integer(row["channel_id"])
                if programme == 0:
                    counts["unlabeled_rows_excluded"] += 1
                    continue
                start, duration = parse_timestamp(row["start_time"], schedule=True), parse_integer(row["duration"])
                if programme is None or programme < 0 or channel is None or channel <= 0 or start is None or duration is None or duration <= 0:
                    raise ValueError("Invalid labeled original broadcast.")
                low = seconds(start)
                if low < WINDOWS[2].end and low + duration > WINDOWS[2].start:
                    yield Airing(channel, programme, low, low + duration)
    index = ScheduleIndex(airings())
    return index, {**counts, "distinct_original_labeled_airings": sum(len(group[0]) for group in index.channels.values()),
                   "duplicate_original_airings_removed": index.duplicates_removed,
                   "broadcast_intervals_clipped_in_index": False, "denominator_source": "original schedule.duration"}


def reconstruct_evidence(raw_dir: Path, destination: Path) -> dict:
    if destination.exists():
        raise FileExistsError("Preserving an existing continuous-watch database.")
    started = time.perf_counter()
    index, schedule_stats = load_original_airings(raw_dir)
    stages = {"load_unclipped_historical_schedules": time.perf_counter() - started}
    connection = open_database(destination)
    create_tables(connection)
    writer = OccurrenceWriter(connection)
    counts = Counter({"viewing_rows_read": 0, "duplicate_viewing_intervals_removed": 0,
                      "invalid_viewing_rows": 0, "outside_historical_window": 0,
                      "positive_session_airing_overlaps": 0, "positive_overlaps_below_120_seconds": 0,
                      "reported_duration_mismatches": 0, "crossing_primary_cutoff_views": 0})
    begin = time.perf_counter()
    try:
        for number, row in checked_records(raw_dir / VIEW_FILE, VIEW_COLUMNS):
            counts["viewing_rows_read"] += 1
            if number % 1000000 == 0:
                print(f"Continuous evidence: {number:,} source viewing rows; {time.perf_counter() - begin:.1f} seconds.", flush=True)
            start, end = parse_timestamp(row["start_time"]), parse_timestamp(row["stop_time"])
            channel = parse_integer(row["vsetv_id"])
            user = row["user_id"]
            if start is None or end is None or end <= start or channel is None or channel <= 0 or not USER_PATTERN.fullmatch(user):
                counts["invalid_viewing_rows"] += 1
                continue
            low, high = seconds(start), seconds(end)
            if high <= WINDOWS[2].start or low >= WINDOWS[2].end:
                counts["outside_historical_window"] += 1
                continue
            reported = parse_integer(row["duraton"])
            counts["reported_duration_mismatches"] += reported != high - low
            counts["crossing_primary_cutoff_views"] += low < WINDOWS[1].start < high
            cursor = connection.execute("INSERT OR IGNORE INTO seen_views VALUES (?,?,?,?)", (low, high, channel, user))
            if cursor.rowcount != 1:
                counts["duplicate_viewing_intervals_removed"] += 1
                continue
            view = View(user, channel, max(low, WINDOWS[2].start), min(high, WINDOWS[2].end), number)
            for airing in index.overlaps(view):
                overlap = min(view.end, airing.end) - max(view.start, airing.start)
                counts["positive_session_airing_overlaps"] += 1
                counts["positive_overlaps_below_120_seconds"] += overlap < 120
                writer.add(view, airing)
            if number % 10000 == 0:
                writer.flush()
        writer.flush()
        stages["stream_deduplicate_and_reconstruct_single_interval_evidence"] = time.perf_counter() - begin
        begin = time.perf_counter()
        occurrence_count = connection.execute("SELECT COUNT(*) FROM occurrences").fetchone()[0]
        window_stats = {}
        for window in EVIDENCE_WINDOWS:
            col = f"{window.name}_overlap"
            row = connection.execute(f"""SELECT COUNT(*), SUM(10*{col}>=3*broadcast_duration),
                SUM(2*{col}>broadcast_duration), SUM(5*{col}>4*broadcast_duration),
                SUM({col}<120 AND 5*{col}>4*broadcast_duration)
                FROM occurrences WHERE {col}>0""").fetchone()
            window_stats[window.name] = {"positive_occurrences": row[0], "qualified_ge30_occurrences": row[1] or 0,
                                       "qualified_gt50_occurrences": row[2] or 0, "qualified_gt80_occurrences": row[3] or 0,
                                       "strict_qualified_occurrences_with_overlap_below_120": row[4] or 0,
                                       "start_inclusive": timestamp(window.start), "end_exclusive": timestamp(window.end)}
        connection.execute("DROP TABLE seen_views")
        connection.commit()
        connection.execute("VACUUM")
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError("Reconstructed evidence database failed SQLite integrity_check.")
        connection.execute("PRAGMA user_version=2")
        connection.commit()
        stages["summarize_compact_and_verify_database"] = time.perf_counter() - begin
    finally:
        connection.close()
    return {"schedule": schedule_stats, "viewing": dict(counts), "distinct_user_broadcast_occurrences": occurrence_count,
            "windows": window_stats, "stage_runtime_seconds": stages, "runtime_seconds": time.perf_counter() - started,
            "single_session_policy": "maximum overlap of one original viewing interval per user/channel/programme/original-airing occurrence; sessions are never summed or merged",
            "recency_policy": "last endpoint among individually qualifying sessions, tracked separately at each threshold",
            "timestamp_corrections_applied": False, "sqlite_integrity_check": integrity}


def export_profiles(database: Path, window: Window, destination: Path) -> dict:
    if destination.exists():
        raise FileExistsError("Preserving an existing repeat-viewing profile file.")
    connection = open_database(database, readonly=True)
    connection.create_function("decay14", 1, lambda last: math.exp2(-(window.end - last) / (14 * 86400)), deterministic=True)
    connection.create_function("decay28", 1, lambda last: math.exp2(-(window.end - last) / (28 * 86400)), deterministic=True)
    col = f"{window.name}_overlap"
    indexes = {row[1] for row in connection.execute("PRAGMA index_list(occurrences)")}
    index_hint = " INDEXED BY occurrence_profile_scan" if "occurrence_profile_scan" in indexes else ""
    query = f"""SELECT user_id, programme_id,
        SUM(10*{col}>=3*broadcast_duration), SUM(2*{col}>broadcast_duration), SUM(5*{col}>4*broadcast_duration),
        SUM(CASE WHEN 2*{col}>broadcast_duration THEN decay14({window.name}_last50) ELSE 0 END),
        SUM(CASE WHEN 2*{col}>broadcast_duration THEN decay28({window.name}_last50) ELSE 0 END),
        SUM(CASE WHEN 2*{col}>broadcast_duration THEN {col} ELSE 0 END),
        MAX({window.name}_last30), MAX({window.name}_last50), MAX({window.name}_last80), MAX({col}>=120)
        FROM occurrences{index_hint} WHERE {col}>0 GROUP BY user_id,programme_id ORDER BY user_id,programme_id"""
    started = time.perf_counter()
    rows = 0
    users, programmes = set(), set()
    try:
        with destination.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(PROFILE_COLUMNS)
            for row in connection.execute(query):
                writer.writerow(row)
                rows += 1
                users.add(row[0])
                programmes.add(row[1])
    finally:
        connection.close()
    return {"file": destination.name, "rows": rows, "users_with_any_positive_overlap": len(users),
            "programmes_with_any_positive_overlap": len(programmes), "start_inclusive": timestamp(window.start),
            "end_exclusive": timestamp(window.end), "runtime_seconds": time.perf_counter() - started,
            "frequency_unit": "distinct qualified broadcast occurrences, not disconnected viewing records",
            "time_unit_for_last_fields": "naive wall-clock seconds since 2020-01-01, using original source clocks"}


def reconstruction_algorithm_fingerprint() -> str:
    """Identify the raw reconstruction logic independently of export optimizations."""
    names = {"qualifies", "session_values", "create_tables", "OccurrenceWriter", "load_original_airings", "reconstruct_evidence"}
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    serialized = ast.dump(ast.Module(body=nodes, type_ignores=[]), include_attributes=False)
    return hashlib.sha256(serialized.encode()).hexdigest()


def add_profile_scan_index(database: Path) -> dict:
    """Add a covering index for streaming GROUP BY without sorting each export."""
    started = time.perf_counter()
    connection = open_database(database)
    columns = ["user_id", "programme_id", "broadcast_duration"]
    columns += [f"{window.name}_{field}" for window in EVIDENCE_WINDOWS for field in ("overlap", "last30", "last50", "last80")]
    try:
        connection.execute(f"CREATE INDEX IF NOT EXISTS occurrence_profile_scan ON occurrences ({','.join(columns)})")
        connection.commit()
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError("Evidence/index integrity check failed.")
        plan = connection.execute("EXPLAIN QUERY PLAN SELECT user_id,programme_id,SUM(full_overlap) FROM occurrences INDEXED BY occurrence_profile_scan WHERE full_overlap>0 GROUP BY user_id,programme_id ORDER BY user_id,programme_id").fetchall()
    finally:
        connection.close()
    return {"runtime_seconds": time.perf_counter() - started, "sqlite_integrity_check": integrity,
            "query_plan": plan, "note": "index changes storage/query planning only; occurrence evidence is not updated"}


def recover_evidence_statistics(raw_dir: Path, database: Path) -> dict:
    """Recompute recoverable diagnostics without pretending lost timers survived."""
    started = time.perf_counter()
    connection = open_database(database, readonly=True)
    try:
        if connection.execute("PRAGMA user_version").fetchone()[0] != 2:
            raise RuntimeError("Cannot resume an incomplete raw reconstruction.")
        occurrence_count, source_pairs = connection.execute("SELECT COUNT(*),SUM(source_sessions) FROM occurrences").fetchone()
        window_stats = {}
        for window in EVIDENCE_WINDOWS:
            col = f"{window.name}_overlap"
            row = connection.execute(f"SELECT COUNT(*),SUM(10*{col}>=3*broadcast_duration),SUM(2*{col}>broadcast_duration),SUM(5*{col}>4*broadcast_duration),SUM({col}<120 AND 5*{col}>4*broadcast_duration) FROM occurrences WHERE {col}>0").fetchone()
            window_stats[window.name] = {"positive_occurrences": row[0], "qualified_ge30_occurrences": row[1] or 0,
                "qualified_gt50_occurrences": row[2] or 0, "qualified_gt80_occurrences": row[3] or 0,
                "strict_qualified_occurrences_with_overlap_below_120": row[4] or 0,
                "start_inclusive": timestamp(window.start), "end_exclusive": timestamp(window.end)}
    finally:
        connection.close()
    with (raw_dir / VIEW_FILE).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        if next(reader) != VIEW_COLUMNS:
            raise ValueError("Unexpected viewing schema during independent source recount.")
        source_rows = sum(1 for _ in reader)
    return {"distinct_user_broadcast_occurrences": occurrence_count,
        "viewing": {"independently_recounted_source_rows": source_rows,
                    "positive_session_airing_overlaps_recovered_from_occurrence_session_counts": source_pairs,
                    "lost_raw_loop_counters": None}, "windows": window_stats,
        "recovery_diagnostics_runtime_seconds": time.perf_counter() - started,
        "original_reconstruction_stage_timings": None,
        "single_session_policy": "maximum individual overlap, never summed disconnected sessions",
        "denominator_source": "original schedule.duration; original broadcast endpoints retained",
        "timestamp_corrections_applied": False,
        "recovery_note": "completed evidence resumed after tool timeout; absent in-memory counters/timers are explicitly unavailable"}
