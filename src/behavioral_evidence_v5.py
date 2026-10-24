"""Causal all-session channel/time contexts and exact extra-origin evidence.

Raw intervals are deduplicated once in a disk-backed chronological user/channel
spool. Overlap union is used ONLY for contextual duration. Strict programme labels
continue to use one individual interval; disconnected sessions are never combined.
Existing occurrence maxima are reused only where the airing's visible interval
is wholly inside a requested window. New-boundary cases use raw sessions.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import sqlite3
import time
import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path

from audit_data import VIEW_COLUMNS, VIEW_FILE, parse_integer, parse_timestamp, peak_memory
from build_interactions import Airing, ScheduleIndex, View, Window, WINDOWS, checked_records, seconds, timestamp
from continuous_watch import PROFILE_COLUMNS, session_values


START = WINDOWS[0].start
CUT13 = seconds(datetime(2020, 3, 30))
CUT17 = seconds(datetime(2020, 4, 27))
CUT15 = seconds(datetime(2020, 4, 13))
HISTORY_WINDOWS = {
    "prefix13": Window("prefix13", START, CUT13), "early": Window("early", START, CUT15),
    "prefix17": Window("prefix17", START, CUT17), "primary": Window("primary", START, WINDOWS[0].end),
    "final": Window("final", START, WINDOWS[2].end),
}
NEW_WINDOWS = (
    HISTORY_WINDOWS["prefix13"], HISTORY_WINDOWS["prefix17"],
    Window("labels14_15", CUT13, CUT15), Window("labels16_17", CUT15, CUT17),
    Window("labels18_20", CUT17, WINDOWS[0].end),
)
ROLLING_ORIGINS = (
    ("roll13", "prefix13", "labels14_15"), ("roll15", "early", "labels16_17"),
    ("roll17", "prefix17", "labels18_20"),
)
DAY, HOUR = 86400, 3600
ALPHA_SECONDS = 600.


def daypart(hour: int) -> int:
    return 0 if hour < 6 else 1 if hour < 12 else 2 if hour < 18 else 3


def chunks(start: int, end: int):
    while start < end:
        stop = min(end, (start // HOUR + 1) * HOUR)
        yield start, stop, start // HOUR % 24, (start // DAY + 2) % 7
        start = stop


def empty_context() -> dict:
    return {"channels": {}, "hours": [0.] * 24, "weekdays": [0.] * 7, "recent_hours": [0.] * 24,
            "events": 0, "raw_session_seconds": 0., "raw_session_squared_seconds": 0., "long_sessions": 0}


def channel(context: dict, item: int) -> list:
    # seconds, original events, distinct-day bitset, recent seconds, decayed seconds,
    # last endpoint, and four daypart durations.
    return context["channels"].setdefault(str(item), [0., 0, 0, 0., 0., 0, 0., 0., 0., 0.])


def add_original_event(context: dict, item: int, start: int, end: int, window: Window) -> None:
    low, high = max(start, window.start), min(end, window.end)
    if high <= low:
        return
    duration = high - low
    context["events"] += 1
    context["raw_session_seconds"] += duration
    context["raw_session_squared_seconds"] += duration * duration
    context["long_sessions"] += duration > 7200
    record = channel(context, item)
    record[1] += 1
    record[5] = max(record[5], high)


def add_union_duration(context: dict, item: int, start: int, end: int, window: Window) -> None:
    low, high = max(start, window.start), min(end, window.end)
    if high <= low:
        return
    record = channel(context, item)
    record[0] += high - low
    record[3] += max(0, high - max(low, window.end - 14 * DAY))
    tau = 14 * DAY / math.log(2)
    record[4] += tau * (math.exp(-(window.end - high) / tau) - math.exp(-(window.end - low) / tau))
    for first, last, hour, weekday in chunks(low, high):
        duration = last - first
        context["hours"][hour] += duration
        context["weekdays"][weekday] += duration
        context["recent_hours"][hour] += max(0, last - max(first, window.end - 14 * DAY))
        record[2] |= 1 << (first // DAY)
        record[6 + daypart(hour)] += duration


def merge_intervals(intervals):
    """Context-only union for sorted, same-user/channel intervals."""
    first = last = None
    for start, end in intervals:
        if first is None:
            first, last = start, end
        elif start <= last:
            last = max(last, end)
        else:
            yield first, last
            first, last = start, end
    if first is not None:
        yield first, last


def open_work_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, uri=True)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA cache_size=-16384")
    connection.execute("PRAGMA temp_store=FILE")
    return connection


def boundary_index(original: Path) -> ScheduleIndex:
    connection = sqlite3.connect(original.resolve().as_uri() + "?mode=ro", uri=True)
    airings = set()
    try:
        for cutoff in (CUT13, CUT15, CUT17):
            for start, chan, item, end in connection.execute("""SELECT DISTINCT airing_start,channel_id,programme_id,airing_end
                FROM occurrences WHERE airing_start>=? AND airing_start<? AND airing_end>?""", (cutoff - DAY, cutoff, cutoff)):
                airings.add(Airing(chan, item, start, end))
    finally:
        connection.close()
    return ScheduleIndex(airings)


def create_raw_spool(root: Path, destination: Path, origin_database: Path) -> dict:
    if destination.exists() or origin_database.exists():
        raise FileExistsError("Preserving an existing/incomplete V5 raw spool.")
    started = time.perf_counter()
    connection = open_work_database(destination)
    connection.execute("CREATE TABLE views(user TEXT,channel INTEGER,start INTEGER,end INTEGER,source_row INTEGER,PRIMARY KEY(user,channel,start,end)) WITHOUT ROWID")
    labels = open_work_database(origin_database)
    labels.execute("""CREATE TABLE evidence(w TEXT,user TEXT,item INTEGER,channel INTEGER,start INTEGER,end INTEGER,duration INTEGER,
        overlap INTEGER,last30 INTEGER,last50 INTEGER,last80 INTEGER,PRIMARY KEY(w,user,item,start,channel,end)) WITHOUT ROWID""")
    index = boundary_index(root / "data/processed/continuous_watch_v2.sqlite3")
    pending = []
    counts = Counter()

    def flush():
        connection.commit()
        labels.executemany("""INSERT INTO evidence VALUES(?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(w,user,item,start,channel,end) DO UPDATE SET
            overlap=MAX(overlap,excluded.overlap),last30=MAX(last30,excluded.last30),last50=MAX(last50,excluded.last50),last80=MAX(last80,excluded.last80)""", pending)
        pending.clear()
        labels.commit()

    try:
        for number, row in checked_records(root / "data/raw" / VIEW_FILE, VIEW_COLUMNS):
            counts["source_rows"] += 1
            if number % 1000000 == 0:
                print(f"V5 context spool: {number:,} rows, {time.perf_counter() - started:.1f}s", flush=True)
            start, end = parse_timestamp(row["start_time"]), parse_timestamp(row["stop_time"])
            chan = parse_integer(row["vsetv_id"])
            if start is None or end is None or end <= start or chan is None or chan <= 0 or not row["user_id"]:
                raise ValueError("Invalid original viewing interval.")
            low, high = seconds(start), seconds(end)
            if low >= WINDOWS[2].end or high <= START:
                counts["outside_historical_window"] += 1
                continue
            cursor = connection.execute("INSERT OR IGNORE INTO views VALUES(?,?,?,?,?)", (row["user_id"], chan, low, high, number))
            if cursor.rowcount != 1:
                counts["duplicates_removed"] += 1
                continue
            view = View(row["user_id"], chan, max(low, START), min(high, WINDOWS[2].end), number)
            for airing in index.overlaps(view):
                values = session_values(view, airing, NEW_WINDOWS)
                for i, window in enumerate(NEW_WINDOWS):
                    overlap, _, last30, last50, last80 = values[5 * i:5 * i + 5]
                    if overlap:
                        pending.append((window.name, view.user_id, airing.programme_id, airing.channel_id, airing.start, airing.end,
                                        airing.end - airing.start, overlap, last30, last50, last80))
            if number % 10000 == 0 or len(pending) >= 10000:
                flush()
        flush()
        counts["deduplicated_views"] = connection.execute("SELECT COUNT(*) FROM views").fetchone()[0]
        connection.execute("PRAGMA user_version=5")
        connection.commit()
        labels.execute("PRAGMA user_version=5")
        labels.commit()
    finally:
        connection.close()
        labels.close()
    return {**counts, "boundary_airings": sum(len(group[0]) for group in index.channels.values()),
            "runtime_seconds": time.perf_counter() - started, **peak_memory()}


def export_contexts(spool: Path, output_dir: Path) -> dict:
    paths = {name: output_dir / f"behavioral_v5_context_{name}.jsonl" for name in HISTORY_WINDOWS}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("Preserving V5 context exports.")
    handles = {name: path.open("x") for name, path in paths.items()}
    globals_ = {name: {"channels": Counter(), "hours": [0.] * 24, "weekdays": [0.] * 7, "users": 0, "seconds": 0.} for name in HISTORY_WINDOWS}
    connection = sqlite3.connect(spool.resolve().as_uri() + "?mode=ro", uri=True)
    started = time.perf_counter()
    try:
        records = connection.execute("SELECT user,channel,start,end FROM views ORDER BY user,channel,start,end")
        for user, user_records in itertools.groupby(records, key=lambda row: row[0]):
            contexts = {name: empty_context() for name in HISTORY_WINDOWS}
            for chan, channel_records in itertools.groupby(user_records, key=lambda row: row[1]):
                first = last = None
                for _, _, start, end in channel_records:
                    for name, window in HISTORY_WINDOWS.items():
                        add_original_event(contexts[name], chan, start, end, window)
                    if first is None:
                        first, last = start, end
                    elif start <= last:
                        last = max(last, end)
                    else:
                        for name, window in HISTORY_WINDOWS.items():
                            add_union_duration(contexts[name], chan, first, last, window)
                        first, last = start, end
                if first is not None:
                    for name, window in HISTORY_WINDOWS.items():
                        add_union_duration(contexts[name], chan, first, last, window)
            for name, context in contexts.items():
                total = sum(record[0] for record in context["channels"].values())
                if not total:
                    continue
                context["user_id"], context["seconds"] = user, total
                handles[name].write(json.dumps(context, separators=(",", ":")) + "\n")
                global_ = globals_[name]
                global_["users"] += 1
                global_["seconds"] += total
                global_["channels"].update({chan: record[0] for chan, record in context["channels"].items()})
                global_["hours"] = [left + right for left, right in zip(global_["hours"], context["hours"])]
                global_["weekdays"] = [left + right for left, right in zip(global_["weekdays"], context["weekdays"])]
    finally:
        for handle in handles.values():
            handle.close()
        connection.close()
    for name, global_ in globals_.items():
        global_["channels"] = dict(global_["channels"])
        path = output_dir / f"behavioral_v5_global_{name}.json"
        with path.open("x") as handle:
            json.dump(global_, handle)
    return {"runtime_seconds": time.perf_counter() - started, "origins": {name: {"users": value["users"], "seconds": value["seconds"]} for name, value in globals_.items()}, **peak_memory()}


def export_new_profiles(root: Path, database: Path) -> dict:
    """Reuse only whole visible occurrences; add raw single-session boundary cases."""
    connection = open_work_database(database)
    source = root / "data/processed/continuous_watch_v2.sqlite3"
    connection.execute("ATTACH DATABASE ? AS original", (source.resolve().as_uri() + "?mode=ro",))
    started = time.perf_counter()
    result = {}
    try:
        for window in NEW_WINDOWS:
            # Endpoints equal to the old training boundary are already clipped in
            # train_overlap; all other new-boundary airings were reconstructed.
            lower = "1" if window.start == START else "airing_start>=?"
            upper = "1" if window.end == WINDOWS[0].end else "airing_end<=?"
            bindings = [window.name]
            if window.start != START:
                bindings.append(window.start)
            if window.end != WINDOWS[0].end:
                bindings.append(window.end)
            bindings.extend((window.end, window.start))
            connection.execute(f"""INSERT OR IGNORE INTO evidence
                SELECT ?,user_id,programme_id,channel_id,airing_start,airing_end,broadcast_duration,
                    train_overlap,train_last30,train_last50,train_last80 FROM original.occurrences
                WHERE train_overlap>0 AND {lower} AND {upper} AND airing_start<? AND airing_end>?""", bindings)
            connection.commit()
            connection.create_function("decay14", 1, lambda value, cutoff=window.end: math.exp2(-(cutoff - value) / (14 * DAY)), deterministic=True)
            connection.create_function("decay28", 1, lambda value, cutoff=window.end: math.exp2(-(cutoff - value) / (28 * DAY)), deterministic=True)
            query = """SELECT user,item,SUM(10*overlap>=3*duration),SUM(2*overlap>duration),SUM(5*overlap>4*duration),
                SUM(CASE WHEN 2*overlap>duration THEN decay14(last50) ELSE 0 END),
                SUM(CASE WHEN 2*overlap>duration THEN decay28(last50) ELSE 0 END),
                SUM(CASE WHEN 2*overlap>duration THEN overlap ELSE 0 END),MAX(last30),MAX(last50),MAX(last80),MAX(overlap>=120)
                FROM evidence WHERE w=? AND overlap>0 GROUP BY user,item ORDER BY user,item"""
            path = root / "data/processed" / f"behavioral_v5_profile_{window.name}.csv"
            if path.exists():
                continue
            n = 0
            with path.open("x", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(PROFILE_COLUMNS)
                for row in connection.execute(query, (window.name,)):
                    writer.writerow(row)
                    n += 1
            result[window.name] = {"rows": n, "start": timestamp(window.start), "end_exclusive": timestamp(window.end)}
    finally:
        connection.close()
    return {"windows": result, "runtime_seconds": time.perf_counter() - started}


def read_profile(path: Path, window: Window):
    previous = None
    for _, row in checked_records(path, PROFILE_COLUMNS):
        user, item = row["user_id"], int(row["tv_show_id"])
        values = {name: float(row[name]) if name.startswith("recency_") else int(row[name]) for name in PROFILE_COLUMNS[2:]}
        key = (user, item)
        if previous is not None and key <= previous:
            raise ValueError("Profile keys not sorted and unique.")
        for freq, last in (("frequency_ge30", "last_ge30_seconds"), ("frequency_gt50", "last_gt50_seconds"), ("frequency_gt80", "last_gt80_seconds")):
            if values[freq] and not window.start < values[last] <= window.end:
                raise ValueError("Noncausal profile endpoint.")
        previous = key
        yield {"user_id": user, "tv_show_id": item, **values}


def load_context(root: Path, name: str, users: set[str]) -> tuple[dict, dict]:
    contexts = {}
    with (root / "data/processed" / f"behavioral_v5_context_{name}.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["user_id"] in users:
                contexts[row["user_id"]] = row
    global_ = json.loads((root / "data/processed" / f"behavioral_v5_global_{name}.json").read_text())
    return contexts, global_


def opportunities(paths: list[Path], window: Window) -> dict:
    from audit_data import SCHEDULE_COLUMNS
    seen, result = set(), {}
    for path in paths:
        for _, row in checked_records(path, SCHEDULE_COLUMNS):
            item = int(row["tv_show_id"])
            if item <= 0:
                continue
            start = seconds(parse_timestamp(row["start_time"], schedule=True))
            duration, chan = int(row["duration"]), int(row["channel_id"])
            low, high = max(start, window.start), min(start + duration, window.end)
            key = (item, chan, start, duration)
            if high <= low or key in seen:
                continue
            seen.add(key)
            entry = result.setdefault(item, {"n": 0, "channels": Counter(), "channel_parts": Counter(),
                                              "hours": [0.] * 24, "weekdays": [0.] * 7, "start_hours": Counter(),
                                              "evening": 0, "prime": 0, "weekend": 0})
            entry["n"] += 1
            hour, weekday = start // HOUR % 24, (start // DAY + 2) % 7
            entry["channels"][str(chan)] += 1
            entry["channel_parts"][(str(chan), daypart(hour))] += 1
            entry["start_hours"][hour] += 1
            entry["evening"] += 18 <= hour < 23
            entry["prime"] += 19 <= hour < 22
            entry["weekend"] += weekday >= 5
            for first, last, h, wd in chunks(low, high):
                entry["hours"][h] += last - first
                entry["weekdays"][wd] += last - first
    return result


CHANNEL_FEATURES = (
    "channel_affinity_max", "channel_affinity_mean", "channel_weighted_airings", "channel_preferred_airing_fraction",
    "channel_preferred_airing_count", "channel_recent_affinity_max", "channel_recent_weighted_airings",
    "channel_decayed_weighted_airings", "channel_distinct_days_max", "channel_events_max", "channel_days_since_last_min",
    "user_channel_concentration", "user_channel_entropy", "user_channel_count", "user_top_channel_share",
    "channel_programme_seen_preference", "channel_repeat_interaction", "channel_association_interaction",
)
TIME_FEATURES = (
    "time_hour_alignment", "time_weekday_alignment", "time_start_hour_alignment", "time_recent_hour_alignment",
    "time_preferred_slot_fraction", "time_preferred_slot_airings", "time_evening_compatibility", "time_weekend_compatibility",
    "time_prime_compatibility", "time_channel_daypart_alignment", "time_channel_daypart_max", "time_opportunity_weight",
    "user_hour_concentration", "user_hour_entropy", "user_evening_share", "user_weekend_share", "user_prime_share",
    "user_recent_hour_shift", "user_session_mean_minutes", "user_session_std_minutes", "user_long_session_fraction",
    "time_repeat_interaction", "time_history_present",
)


def behavior_features(context: dict | None, global_: dict, future: dict, origin: int,
                      repeat_frequency: float = 0., association_confidence: float = 0., historical_channels=None) -> list[float]:
    context = context or empty_context()
    total = context.get("seconds", 0.)
    denominator = total + ALPHA_SECONDS
    raw = context["channels"]
    global_total = max(1., global_["seconds"])
    affinities = {chan: (record[0] + ALPHA_SECONDS * global_["channels"].get(chan, 0.) / global_total) / denominator for chan, record in raw.items()}
    def affinity(chan):
        return affinities.get(chan, ALPHA_SECONDS * global_["channels"].get(chan, 0.) / global_total / denominator)
    n = future["n"]
    chans = list(future["channels"])
    preferred = set(sorted(raw, key=lambda chan: (-raw[chan][0], int(chan)))[:3])
    recent_total = sum(record[3] for record in raw.values())
    decayed_total = sum(record[4] for record in raw.values())
    recent = {chan: record[3] / recent_total if recent_total else 0. for chan, record in raw.items()}
    decayed = {chan: record[4] / decayed_total if decayed_total else 0. for chan, record in raw.items()}
    shares = [record[0] / total for record in raw.values()] if total else []
    max_share = max(shares, default=0.)
    ch_values = [affinity(chan) for chan in chans]
    weighted = sum(count * affinity(chan) for chan, count in future["channels"].items())
    pref_n = sum(count for chan, count in future["channels"].items() if chan in preferred)
    recency_days = [max(0., (origin - raw[chan][5]) / DAY) for chan in chans if chan in raw and raw[chan][5]]
    channel_vector = [max(ch_values, default=0.), sum(ch_values) / max(1, len(ch_values)), weighted, pref_n / n, pref_n,
        max((recent.get(chan, 0.) for chan in chans), default=0.), sum(count * recent.get(chan, 0.) for chan, count in future["channels"].items()),
        sum(count * decayed.get(chan, 0.) for chan, count in future["channels"].items()),
        max((raw[chan][2].bit_count() for chan in chans if chan in raw), default=0),
        max((raw[chan][1] for chan in chans if chan in raw), default=0), min(recency_days, default=999.),
        sum(share * share for share in shares), -sum(share * math.log(max(share, 1e-30)) for share in shares), len(raw), max_share,
        sum(affinity(str(chan)) for chan in (historical_channels or ()) if str(chan) in preferred),
        math.log1p(repeat_frequency) * max(ch_values, default=0.), association_confidence * weighted / n]
    hours = [(context["hours"][h] + ALPHA_SECONDS * global_["hours"][h] / global_total) / denominator for h in range(24)]
    weekdays = [(context["weekdays"][wd] + ALPHA_SECONDS * global_["weekdays"][wd] / global_total) / denominator for wd in range(7)]
    recent_hour_sum = sum(context["recent_hours"])
    recent_hours = [value / recent_hour_sum for value in context["recent_hours"]] if recent_hour_sum else hours
    duration = max(1., sum(future["hours"]))
    hour_alignment = sum(left * right for left, right in zip(hours, future["hours"])) / duration
    weekday_alignment = sum(left * right for left, right in zip(weekdays, future["weekdays"])) / duration
    start_alignment = sum(count * hours[hour] for hour, count in future["start_hours"].items()) / n
    recent_alignment = sum(left * right for left, right in zip(recent_hours, future["hours"])) / duration
    preferred_hours = set(sorted(range(24), key=lambda hour: (-hours[hour], hour))[:3])
    slot_n = sum(count for hour, count in future["start_hours"].items() if hour in preferred_hours)
    parts = [sum(context["hours"][h] for h in range(24) if daypart(h) == part) for part in range(4)]
    joint = []
    for (chan, part), count in future["channel_parts"].items():
        conditional = ((raw[chan][6 + part] if chan in raw else 0.) + ALPHA_SECONDS * affinity(chan)) / (parts[part] + ALPHA_SECONDS)
        part_probability = sum(hours[h] for h in range(24) if daypart(h) == part)
        joint.append((count, conditional * part_probability))
    evening, weekend, prime = sum(hours[18:23]), sum(weekdays[5:]), sum(hours[19:22])
    mean_session = context["raw_session_seconds"] / max(1, context["events"])
    variance = context["raw_session_squared_seconds"] / max(1, context["events"]) - mean_session ** 2
    time_vector = [hour_alignment, weekday_alignment, start_alignment, recent_alignment, slot_n / n, slot_n,
        evening * future["evening"] / n, weekend * future["weekend"] / n, prime * future["prime"] / n,
        sum(count * value for count, value in joint) / n, max((value for _, value in joint), default=0.),
        n * hour_alignment * weighted / n,
        sum(value * value for value in hours), -sum(value * math.log(max(value, 1e-30)) for value in hours),
        evening, weekend, prime, sum(abs(left - right) for left, right in zip(hours, recent_hours)),
        mean_session / 60., math.sqrt(max(0., variance)) / 60., context["long_sessions"] / max(1, context["events"]),
        math.log1p(repeat_frequency) * hour_alignment, float(total > 0)]
    result = list(map(float, channel_vector + time_vector))
    if len(result) != len(CHANNEL_FEATURES) + len(TIME_FEATURES) or any(not math.isfinite(value) for value in result):
        raise ValueError("Invalid behavioral vector.")
    return result


def main() -> None:
    from create_final_submission import project_snapshot
    from build_interactions import file_fingerprint
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "raw", "contexts", "profiles"), default="all")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    processed = root / "data/processed"
    snapshot_path = processed / "behavioral_v5_inputs.json"
    if snapshot_path.exists():
        protected = json.loads(snapshot_path.read_text())
    else:
        protected = {name: value for name, value in project_snapshot(root).items()
                     if "v5" not in name.lower() and not name.startswith("outputs/submission_v5")}
        with snapshot_path.open("x") as handle:
            json.dump(protected, handle, indent=2)
    for name, old in protected.items():
        if file_fingerprint(root / name)["sha256"] != old["sha256"]:
            raise RuntimeError(f"Protected original input changed: {name}")
    spool, origin_db = processed / "behavioral_v5_sessions.sqlite3", processed / "behavioral_v5_origins.sqlite3"
    raw_meta = processed / "behavioral_v5_raw_checkpoint.json"
    if not raw_meta.exists():
        value = create_raw_spool(root, spool, origin_db)
        with raw_meta.open("x") as handle:
            json.dump(value, handle, indent=2)
        print(f"Raw V5 spool complete: {value}", flush=True)
    if args.stage in ("all", "contexts"):
        path = processed / "behavioral_v5_context_checkpoint.json"
        if not path.exists():
            result = export_contexts(spool, processed)
            with path.open("x") as handle:
                json.dump(result, handle, indent=2)
            print(f"All causal context origins exported: {result}", flush=True)
    if args.stage in ("all", "profiles"):
        path = processed / "behavioral_v5_profile_checkpoint.json"
        if not path.exists():
            result = export_new_profiles(root, origin_db)
            with path.open("x") as handle:
                json.dump(result, handle, indent=2)
            print(f"Disjoint rolling-origin profiles exported: {result}", flush=True)
    for name, old in protected.items():
        if file_fingerprint(root / name)["sha256"] != old["sha256"]:
            raise RuntimeError(f"Original artifact changed during evidence building: {name}")
    print("V5 evidence checkpoint complete; protected originals unchanged.", flush=True)


if __name__ == "__main__":
    main()
