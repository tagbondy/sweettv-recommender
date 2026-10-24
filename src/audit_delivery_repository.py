"""Review an explicit Git publication allowlist and verify GitHub's actual tree."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import subprocess
from pathlib import Path

from finalize_exam_v5 import EXPECTED_V5, write_new

EXACT = {
    ".gitignore", ".gitattributes", "AGENTS.md", "README.md", "requirements-v4.txt",
    "docs/behavioral_v5_reproduction.md", "docs/final_delivery.md",
    "outputs/submission_v5.csv", "reports/academic_report_v5_final.pdf",
    "reports/academic_report_v5_final_examples.json", "reports/kaggle_verified_history.json",
    "reports/final_v5_summary.json", "reports/final_v5_summary.md", "reports/final_delivery_requirements.md",
    "reports/evidence/kaggle_v5_scores.png", "reports/evidence/kaggle_all_scores.png",
}
PATTERNS = {
    "credential_token": rb"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[0-9A-Z]{16}|sk-(?:proj-)?[A-Za-z0-9_-]{24,})\b",
    "private_key": rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    "credential_in_url": rb"https?://[^\s/:@]+:[^\s/@]+@",
}


def run(root, *command):
    return subprocess.check_output(command, cwd=root)


def allowed(name):
    path = Path(name)
    return name in EXACT or (len(path.parts) == 2 and path.parts[0] in ("src", "tests") and path.suffix in (".py", ".swift"))


def audit(root, remote, output):
    entries = run(root, "git", "ls-files", "--stage", "-z").decode().split("\0")
    files = {}
    for entry in filter(None, entries):
        info, name = entry.split("\t", 1)
        mode, blob, stage = info.split()
        if mode not in ("100644", "100755") or stage != "0" or not allowed(name):
            raise AssertionError(f"Unauthorized staged path or mode: {name}")
        data = run(root, "git", "show", ":" + name)
        if len(data) > 5 * 1024 ** 2:
            raise AssertionError(f"Unexpected large publication artifact: {name}")
        for rule, pattern in PATTERNS.items():
            if re.search(pattern, data):
                # Never print the matched secret, even in a failed review.
                raise AssertionError(f"Potential sensitive content ({rule}) in {name}")
        files[name] = {"git_blob_sha1": blob, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    required = EXACT - {"reports/final_delivery_requirements.md"}
    if required - set(files):
        raise AssertionError(f"Mandatory publication files are missing: {sorted(required - set(files))}")
    if files["outputs/submission_v5.csv"]["sha256"] != EXPECTED_V5:
        raise AssertionError("Staged final CSV differs from the frozen V5.")
    ignored = ("data/raw/submission.csv", "data/raw/dataset11-30.csv", "data/processed/continuous_watch_v2.sqlite3",
               ".venv/bin/python", "docs/ai-session/sweettv-opencode-original.json", "docs/ai-session/sweettv-opencode-final.json", ".env", "kaggle.json")
    for name in ignored:
        completed = subprocess.run(["git", "check-ignore", "-q", name], cwd=root)
        if completed.returncode:
            raise AssertionError(f"Sensitive/large local path is not ignored: {name}")
    result = {"publication_file_count": len(files), "files": files,
              "path_allowlist_passed": True, "size_limit_passed": True,
              "credential_pattern_scan_passed": True, "scan_patterns": list(PATTERNS),
              "ignored_paths_verified": ignored,
              "scope": "Explicit path allowlist, staged-blob size and credential-pattern review; original images and rendered PDF also visually reviewed."}
    if remote:
        metadata = json.loads(run(root, "gh", "repo", "view", "--json", "url,visibility,nameWithOwner,defaultBranchRef"))
        if metadata["visibility"] != "PRIVATE":
            raise AssertionError("Publication must remain private without further permission.")
        commit = run(root, "git", "rev-parse", "HEAD").decode().strip()
        remote_commit = run(root, "git", "ls-remote", "origin", "refs/heads/main").decode().split()[0]
        if remote_commit != commit:
            raise AssertionError("GitHub main differs from local committed HEAD.")
        tree = json.loads(run(root, "gh", "api", f"repos/{metadata['nameWithOwner']}/git/trees/{commit}?recursive=1"))
        if tree.get("truncated"):
            raise AssertionError("Cannot audit a truncated remote tree.")
        remote_files = {row["path"]: row for row in tree["tree"] if row["type"] == "blob"}
        if set(remote_files) != set(files):
            raise AssertionError("Remote file set differs from reviewed local index.")
        for name, entry in files.items():
            if remote_files[name]["sha"] != entry["git_blob_sha1"]:
                raise AssertionError(f"Remote committed blob mismatch: {name}")
        # Read critical bytes back through GitHub, beyond just comparing Git IDs.
        fetched = {}
        for name in ("outputs/submission_v5.csv", "reports/academic_report_v5_final.pdf",
                     "reports/evidence/kaggle_v5_scores.png", "reports/evidence/kaggle_all_scores.png"):
            # Use base64 Git blobs: some gh versions apply text transforms to raw
            # binary contents responses, preventing reliable PDF/PNG downloads.
            blob = json.loads(run(root, "gh", "api", f"repos/{metadata['nameWithOwner']}/git/blobs/{remote_files[name]['sha']}"))
            if blob.get("encoding") != "base64":
                raise AssertionError("Unexpected GitHub blob encoding.")
            data = base64.b64decode(blob["content"], validate=False)
            digest = hashlib.sha256(data).hexdigest()
            if digest != files[name]["sha256"]:
                raise AssertionError(f"Downloaded GitHub content mismatch: {name}")
            fetched[name] = digest
        result.update({"github": metadata, "commit": commit, "remote_main_matches_local": True,
                       "remote_tree_all_blobs_verified": True, "critical_downloaded_sha256": fetched})
    write_new(output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "files"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    audit(root, args.remote, root / args.output)


if __name__ == "__main__":
    main()
