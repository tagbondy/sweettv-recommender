"""Freeze and verify the existing V5 delivery; never fit or optimize a model."""

from __future__ import annotations

import argparse
import filecmp
import json
import os
import struct
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from build_interactions import file_fingerprint, seconds, timestamp
from create_final_submission import TEST_WINDOW, validate_submission
from evaluate import load_candidates

EXPECTED_V5 = "18b79a44e38d3a19b0edd75587535161c32b29bcadd19d28ad106a4c7cc651c0"
HISTORY = [
    {"version": "V1", "public_map_at_5": .31901, "private_map_at_5": .31614},
    {"version": "V2", "public_map_at_5": .41082, "private_map_at_5": .39407},
    {"version": "V3", "public_map_at_5": .43568, "private_map_at_5": .43224},
    {"version": "V4", "public_map_at_5": .44087, "private_map_at_5": .43552},
    {"version": "V5", "public_map_at_5": .44671, "private_map_at_5": .43927},
]
SCREENSHOTS = ("reports/evidence/kaggle_v5_scores.png", "reports/evidence/kaggle_all_scores.png")


def write_new(path, data):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def png_evidence(path):
    with path.open("rb") as handle:
        header = handle.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"Not an original readable PNG: {path}")
    width, height = struct.unpack(">II", header[16:24])
    return {**file_fingerprint(path), "width_pixels": width, "height_pixels": height}


def freeze(root):
    destination = root / "reports/finalization_v5_freeze.json"
    if destination.exists():
        return json.loads(destination.read_text())
    if file_fingerprint(root / "outputs/submission_v5.csv")["sha256"] != EXPECTED_V5:
        raise AssertionError("Selected V5 bytes differ from the already verified submission.")
    old = json.loads((root / "data/processed/behavioral_v5_inputs.json").read_text())
    ai = "docs/ai-session/sweettv-opencode-original.json"
    if file_fingerprint(root / ai) != old[ai]:
        raise AssertionError("Original AI export changed before finalization.")
    protected = {}
    # Preserve every existing dataset, CSV, model, archived experiment, and draft.
    # README, ignore rules, and the report generator are explicitly editable delivery documents.
    for folder in ("data", "outputs", "reports", "docs/ai-session", "docs/assignment", "tests", "src"):
        for path in sorted((root / folder).rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or path.name == ".DS_Store":
                continue
            relative = str(path.relative_to(root))
            if relative in ("src/academic_report_v5.py", "src/finalize_exam_v5.py"):
                continue
            current = file_fingerprint(path)
            if relative in old and current != old[relative]:
                raise AssertionError(f"Archived V1–V4 input changed before freeze: {relative}")
            protected[relative] = current
    result = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(), "selected_csv": "outputs/submission_v5.csv",
              "selected_sha256": EXPECTED_V5, "optimization_finished": True, "protected_files": protected,
              "screenshots": {name: png_evidence(root / name) for name in SCREENSHOTS}}
    write_new(destination, result)
    history_path = root / "reports/kaggle_verified_history.json"
    if not history_path.exists():
        write_new(history_path, {"provenance": "User-verified actual Kaggle results; both original screenshots visually inspected during finalization.",
                                "history": HISTORY, "selected_version": "V5", "selected_csv": result["selected_csv"],
                                "submission_sha256": EXPECTED_V5, "public_target": .50, "public_target_achieved": False,
                                "improvement_over_v4": {"public_absolute": .00584, "private_absolute": .00375},
                                "screenshots": result["screenshots"]})
    print(f"Frozen {len(protected)} existing artifacts. Selected V5 SHA-256: {EXPECTED_V5}")
    return result


def verify(root, destination):
    frozen = json.loads((root / "reports/finalization_v5_freeze.json").read_text())
    for name, old in frozen["protected_files"].items():
        if file_fingerprint(root / name) != old:
            raise AssertionError(f"Frozen artifact changed: {name}")
    candidates, availability = load_candidates(root / "data/raw/export_arh_31-42-final.csv", TEST_WINDOW)
    validation = validate_submission(root / "outputs/submission_v5.csv", root / "data/raw/submission.csv", candidates)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="2", OMP_WAIT_POLICY="PASSIVE", KMP_BLOCKTIME="0")
    tests = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                           cwd=root, env=env, capture_output=True, text=True, timeout=120)
    if tests.returncode:
        raise RuntimeError(tests.stdout + tests.stderr)
    print(tests.stdout + tests.stderr, end="")
    replays = []
    with tempfile.TemporaryDirectory(prefix="final-v5-inference-") as directory:
        for seed in ("0", "314159"):
            path = Path(directory) / f"replay-{seed}.csv"
            replay_env = {**env, "PYTHONHASHSEED": seed}
            completed = subprocess.run([sys.executable, "src/behavioral_v5.py", "--final-replay",
                                        "reports/behavioral_v5_final_strategy.json", "--output", str(path)],
                                       cwd=root, env=replay_env, capture_output=True, text=True, timeout=180)
            if completed.returncode:
                raise RuntimeError(completed.stdout + completed.stderr)
            if not filecmp.cmp(path, root / "outputs/submission_v5.csv", shallow=False):
                raise AssertionError("Inference-only replay differs from the frozen selected CSV.")
            replays.append({"python_hash_seed": seed, "sha256": file_fingerprint(path)["sha256"], "byte_identical_to_frozen_v5": True})
    study = json.loads((root / "reports/behavioral_v5.json").read_text())
    from behavioral_v5 import load_bundle
    final, _, _ = load_bundle(root, "final")
    strategy = json.loads((root / "reports/behavioral_v5_final_strategy.json").read_text())
    windows = strategy["fit"]["training_windows"]
    from audit_data import parse_timestamp
    if final.label_end is not None or final.history_end != TEST_WINDOW.start:
        raise AssertionError("Final feature cache has future labels or an invalid history cutoff.")
    if any(seconds(parse_timestamp(w["label_end"])) > TEST_WINDOW.start for w in windows):
        raise AssertionError("Final fitting labels enter the test period.")
    for name, expected in study["execution"]["source_sha256"].items():
        if file_fingerprint(root / name)["sha256"] != expected:
            raise AssertionError(f"Selected V5 source changed: {name}")
    ai_path = root / "docs/ai-session/sweettv-opencode-original.json"
    with ai_path.open() as handle:
        json.load(handle)
    final_export = root / "docs/ai-session/sweettv-opencode-final.json"
    exported = None
    if final_export.exists():
        with final_export.open() as handle:
            json.load(handle)
        exported = {"path": str(final_export.relative_to(root)), "valid_json": True, **file_fingerprint(final_export)}
    result = {"verified_at_utc": datetime.now(timezone.utc).isoformat(), "submission_validation": validation,
              "submission": {"path": "outputs/submission_v5.csv", **file_fingerprint(root / "outputs/submission_v5.csv")},
              "protected_file_count": len(frozen["protected_files"]), "all_frozen_artifacts_unchanged": True,
              "v1_v4_submissions": {name: file_fingerprint(root / "outputs" / name) for name in ("submission.csv", "submission_v2.csv", "submission_v3.csv", "submission_v4.csv")},
              "tests": {"returncode": tests.returncode, "output": tests.stdout + tests.stderr}, "inference_only_replays": replays,
              "temporal_audit": {"history_end": timestamp(final.history_end), "final_cache_has_no_labels": True,
                                 "fitting_windows": windows, "all_labels_end_before_test_start": True,
                                 "selected_algorithm_source_hashes_match_executed_study": True,
                                 "no_test_viewing_claim_scope": "Cutoff/cache/source audit plus existing leakage tests; replay performs inference only.",
                                 "availability": availability},
              "screenshots_unchanged": {name: png_evidence(root / name) for name in SCREENSHOTS},
              "original_ai_export": {"valid_json": True, "unchanged": True, **file_fingerprint(ai_path)},
              "final_ai_export": exported, "new_model_training_performed": False, "kaggle_upload_performed": False}
    for name, old in frozen["protected_files"].items():
        if file_fingerprint(root / name) != old:
            raise AssertionError(f"Frozen artifact changed during verification: {name}")
    write_new(destination, result)
    print(json.dumps({"verification": str(destination), "submission_validation": validation,
                      "sha256": EXPECTED_V5, "protected_files_unchanged": len(frozen["protected_files"]),
                      "inference_replays": replays, "final_ai_export": exported}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("reports/finalization_v5_verification.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.freeze:
        freeze(root)
    if args.verify:
        verify(root, root / args.output)
    if not (args.freeze or args.verify):
        parser.error("Choose --freeze and/or --verify; these commands never optimize or fit.")


if __name__ == "__main__":
    main()
