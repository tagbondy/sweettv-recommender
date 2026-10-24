"""Build an English academic report from measured artifacts, without fitting models."""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import lightgbm_v4 as v4
from build_interactions import WINDOWS, file_fingerprint, seconds
from audit_data import parse_timestamp
from evaluate import average_precision_at_5, load_template_users, read_predictions
from personalized_repeat import iter_profiles, profile_context, profile_path


def esc(value):
    return html.escape(str(value))


def table(headers, rows):
    class_ = " class='recommendations'" if headers[0] == "Rank" else ""
    return f"<table{class_}><thead><tr>" + "".join(f"<th>{esc(h)}</th>" for h in headers) + "</tr></thead><tbody>" + "".join(
        "<tr>" + "".join(f"<td>{esc(value)}</td>" for value in row) + "</tr>" for row in rows) + "</tbody></table>"


def build(root, repository_url=None, screenshot=None, history_screenshot=None, final=False):
    report = json.loads((root / "reports/behavioral_v5.json").read_text())
    verified = json.loads((root / "reports/kaggle_verified_history.json").read_text())
    current_sha = file_fingerprint(root / "outputs/submission_v5.csv")["sha256"]
    if current_sha != verified["submission_sha256"]:
        raise AssertionError("The report must designate the original frozen V5 CSV.")
    screenshot = screenshot or root / "reports/evidence/kaggle_v5_scores.png"
    history_screenshot = history_screenshot or root / "reports/evidence/kaggle_all_scores.png"
    users = load_template_users(root / "data/raw/submission.csv")
    base = v4.load_expanded(root, "primary")
    counts = {user: int(base.value(start, "user_distinct_programmes")) for user, start, _ in base.groups}
    heavy = min(users, key=lambda user: (-counts[user], user))
    light = min((u for u in users if 0 < counts[u] <= 20), key=lambda user: (abs(counts[user] - 10), user))
    selected_users = [heavy, light]
    available = frozenset(base.metadata["available_ids"])
    truth = profile_context(profile_path(root, "validation"), "validation", users)["strict_truth"]
    histories = {user: set() for user in selected_users}
    for row in iter_profiles(profile_path(root, "train"), "train"):
        if row["user_id"] in histories and row["frequency_ge30"]:
            histories[row["user_id"]].add(row["tv_show_id"])
    paths = {"Popularity": "popularity_validation.csv", "Content-based filtering": "cbf_validation.csv",
             "Association rules": "association_validation.csv", "V2 personalized repeat": "repeat_v2_primary_A_frequency_ge30.csv",
             "V3 logistic reranking": "rerank_v3_logistic_l2_01_validation.csv",
             "V4 ensemble": "lightgbm_v4_ensemble_10_40_50_validation.csv"}
    selected = report["selected_strategy"]
    paths["V5 selected strategy"] = ("lightgbm_v4_ensemble_10_40_50_validation.csv" if selected == "V4_ensemble_control"
                                      else f"behavioral_v5_{selected}.csv")
    predictions = {name: read_predictions(root / "outputs" / path, users, available) for name, path in paths.items()}
    needed = {item for pred in predictions.values() for user in selected_users for item in pred[user]}
    metadata = {item: Counter() for item in needed}
    with (root / "data/raw/export_arh_21-30-final.csv").open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            item = int(row["tv_show_id"])
            if item not in needed:
                continue
            start = seconds(parse_timestamp(row["start_time"], schedule=True))
            if start < WINDOWS[1].end and start + int(row["duration"]) > WINDOWS[1].start:
                metadata[item][(row["tv_show_title"], row["tv_show_category"])] += 1
    details = []
    for user in selected_users:
        for method, pred in predictions.items():
            for rank, item in enumerate(pred[user], 1):
                title, category = min(metadata[item], key=lambda key: (-metadata[item][key], key)) if metadata[item] else ("Unavailable source title", "Unknown")
                details.append({"user_id": user, "method": method, "rank": rank, "tv_show_id": item,
                                "title": title, "category": category, "strict_validation_hit": item in truth[user],
                                "observed_training_ge30": item in histories[user]})
    pending = []
    if not repository_url:
        pending.append("Insert the actual GitHub repository URL.")
    for image in (screenshot, history_screenshot):
        if not image.is_file():
            pending.append(f"Missing mandatory screenshot: {image}")
        elif image.suffix.lower() != ".png":
            raise ValueError("Final evidence must use the supplied original PNG screenshots.")
    if final and pending:
        raise RuntimeError("Final report is incomplete: " + "; ".join(pending))
    generated = datetime.now(timezone.utc).isoformat()
    parts = ["<!doctype html><html lang='en'><head><meta charset='utf-8'><title>SWEET.TV Recommendation Study</title>",
             "<style>@page{size:A4;margin:17mm}body{font:10pt Arial,sans-serif;line-height:1.4;color:#17212b}h1{font-size:23pt}h2{font-size:15pt;margin-top:22pt}h3{font-size:12pt}h2,h3{break-after:avoid}table{border-collapse:collapse;width:100%;font-size:8pt;margin:10pt 0}th,td{border:1px solid #bbc5cf;padding:5pt;text-align:left;vertical-align:top;overflow-wrap:anywhere}th{background:#eaf0f7}thead{display:table-header-group}tr{break-inside:avoid}.recommendations{table-layout:fixed}.recommendations th:nth-child(1){width:6%}.recommendations th:nth-child(2){width:14%}.recommendations th:nth-child(3){width:35%}.recommendations th:nth-child(4){width:17%}.recommendations th:nth-child(5){width:12%}.recommendations th:nth-child(6){width:16%}code{font-size:8pt;overflow-wrap:anywhere}.pending{padding:12pt;background:#fff0cb;border:1px solid #b97d10}.page{break-before:page}img{max-width:100%;max-height:190mm}p,li{orphans:3;widows:3}</style></head><body>",
             "<style>figure.screenshot{margin:0;break-before:page;break-inside:avoid}.screenshot img{display:block;width:100%;height:auto;max-height:220mm;object-fit:contain}.screenshot figcaption{font-size:10pt;line-height:1.4;margin-top:10pt}p{break-inside:avoid}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:8pt}</style>",
             "<h1>SWEET.TV TV Programme Recommendation</h1><p>Individual university examination project · 24 October 2026<br>Artem Bondarenko · Kaggle: tagbondy</p>",
             f"<p>Evidence generated: {esc(generated)}. Evaluation metric: MAP@5.</p>"]
    if pending:
        parts.append("<div class='pending'><strong>DRAFT — required external evidence pending</strong><ul>" + "".join(f"<li>{esc(p)}</li>" for p in pending) + "</ul></div>")
    if repository_url:
        parts.append(f"<p>Repository: <a href='{esc(repository_url)}'>{esc(repository_url)}</a></p>")
    parts += ["<h2>1. Problem, data, and audit</h2>",
              "<p>The goal is five unique programme IDs per one of 1,260 template users for ISO weeks 31–42 of 2020. The supplied viewing log contains 5,821,987 intervals for 4,914 users; the schedules cover 400–404 channels. Only 10.8% of first-period schedule rows have a positive programme ID. ID 0 is unlabelled airtime and is never recommended. V5 uses those intervals as channel/time context, without turning them into programme labels.</p>",
              "<p>Intervals match only on identical channel IDs and positive temporal overlap. Broad baseline interactions require at least 120 overlapping seconds: a simple, duration-independent threshold that excludes brief channel switching but remains vulnerable to partial/background viewing. The stricter target requires one original continuous session to overlap strictly more than 80% of the full original broadcast duration, making completion relative to programme length. Neither threshold is a verified definition of the hidden Kaggle preferences. Disconnected sessions are not added to satisfy that label. Schedule duplicates and 17 duplicate viewing records are removed in the relevant evidence builds. The misspelled viewing field <code>duraton</code> is not a programme-length denominator.</p>",
              "<p>We retain supplied naive source clocks: 117 exactly-one-hour duration discrepancies are unresolved; no timezone correction is guessed. Padding beyond ISO-week windows is excluded, and crossing intervals are clipped to each window. Raw inputs are read-only and SHA-256 checked. Simultaneous channels are tuning context, not evidence of unique attention or enjoyment.</p>",
              "<h2>2. Temporal evaluation and fairness</h2>",
              "<p>Primary history is [9 March, 18 May 2020), weeks 11–20; validation is [18 May, 27 July), weeks 21–30. Final history extends through week 30; only schedule availability, never test-period viewing, is used for weeks 31–42. Nested fitting targets end before the primary origin. Their disjoint windows are W14–15, W16–17 and W18–20, with expanding earlier histories.</p>",
              "<p>For each user AP@5 is the sum of precision at relevant ranks divided by min(5, number of unique relevant IDs). Empty relevance receives zero and stays in the denominator of 1,260. Candidate filtering never removes ground-truth IDs. All methods permit repeats and share positive schedule availability (6,361 validation IDs; 6,632 final IDs), identical cohort strings/order, and the historical popularity fallback. Users with no qualifying training history receive exactly the five most popular available programmes by distinct historical viewers; short warm-user lists fill from the same order while skipping duplicates. The independently compared content/rule/baseline methods use the same full available catalogue. Rerankers retrieve subsets of that catalogue; within V5 feature/data comparisons, the identical expanded V4 candidate pool is held fixed. Candidate-only extension pilots are separately labelled.</p>",
              "<p>The original ≥120-second target produced dense relevance and an unusually high popularity baseline. It is a separate sensitivity analysis, not directly comparable with the strict target. The table below rescored saved outputs against the same strict target. Neither local target is claimed to reproduce Kaggle's unknown hidden preference construction.</p>",
              "<h2>3. Independently implemented methods</h2>",
              "<p><strong>Popularity baseline.</strong> Rank by distinct historical viewers, numeric ID for exact ties, restricted to future availability. It is required but does not count as one of the two examination methods.</p>",
              "<p><strong>Content-based filtering.</strong> Binary metadata TF-IDF from training schedules only; category/genre/decade/director/actor field weights 2/3/0.5/1/1, with title tokens disabled. Item fields and combined vectors are L2-normalized. The user profile is the normalized log-count-weighted sum of watched item vectors; cosine similarity ranks candidates. Missing metadata reduces coverage; no future metadata is fitted. Exact numeric-ID ties and shared fallback fill five unique slots.</p>",
              "<p><strong>Association rules.</strong> Historical user baskets are capped at 50 selected programmes. Pair support ≥5 and confidence ≥0.05 define directed rules; the chosen confidence-max score is the maximum applicable conditional probability from a user's seeds. Rules are fitted on training baskets only; individual confidence-max, confidence-sum, and weighted variants were evaluated before hybrids. Unlike metadata similarity, these rules capture empirical co-view relationships, but popular items and basket truncation constrain discovery.</p>",
              "<p><strong>Personalized repeat and supervised reranking.</strong> V2 ranks counts of broadcast occurrences covered by at least 30% in one session, with personal recency then ID ties. V3 logistic reranking uses 61 historical, association, popularity and schedule features. V4 combines 10% repeat, 40% logistic and 50% binary LightGBM reciprocal ranks, using 1/(20+rank) and top-50 source ranks. The binary model uses 31 leaves, learning rate 0.03, minimum leaf 50, L2 10 and 269 rounds.</p>",
              "<p><strong>V5 behavioral ranking.</strong> Append 18 channel and 23 time/context features to those 61 features: smoothed and recent channel affinity, preferred-airing fractions, time-of-day/weekday alignment, channel×daypart compatibility and repeat/association interactions. Context duration unions are separate from original event-count/session moments and strict labels. A 600-second prior smooths affinities. Fixed-budget feature/data arms compare channel, time, both, additional origins and 500 deterministic earlier-active non-template users; extra users have half weight. Rank fitting and a small ensemble set follow independent comparisons. A separate uncapped log-count/IDF item-cosine pilot uses support shrinkage 20 and top-100 neighbors, measuring candidate recall rather than claiming an untrained reranker's accuracy.</p>",
              "<h2>4. Measured strict validation results</h2>"]
    rows = []
    for entry in report["experiments"]:
        score = entry.get("evaluation", {}).get("metrics", {}).get("primary_all_users", {}).get("map_at_5")
        rows.append((entry["name"], entry["status"], "—" if score is None else f"{score:.9f}"))
    parts.append(table(["Method / controlled experiment", "Status", "MAP@5, all 1,260"], rows))
    selection_text = report["selection_conclusion"].replace("No hidden-test performance claim.", "This local selection preceded the now-verified Kaggle results.")
    parts += [f"<p><strong>Local selection: {esc(selected)}.</strong> {esc(selection_text)}</p>",
              "<p>The selected four-way fusion is the sum of 0.10/(20+repeat rank), 0.20/(20+V3 rank), 0.20/(20+V4-ensemble rank), and 0.50/(20+V5-binary rank), using each source's top-50 items. The V4-ensemble rank is itself the ranked V4 fusion, not a comparable probability. Missing source items contribute zero; ties use numeric ID, and the identical popularity fallback fills any short list. Weights were selected from the small recorded historical ensemble set after evaluating components separately.</p>",
              "<p>The selected V5 binary component is D_both (102 features, fixed 269 rounds). Extra-origin and extra-user arms did not improve its primary score and were not included in its final refit. Its final training combines 134,359 early fitting pairs and 158,956 primary fitting pairs (293,315 total), with labels ending on 18 May and 27 July respectively, then predicts from full history through week 30.</p>",
              "<p>Primary validation was reused for selection, so tuned gains are optimistic. Earlier-origin diagnostics are newly trained on labels available at that origin; a saved V4 model that already saw those labels would be an invalid earlier control. The prespecified diagnostic guard rejects a degradation greater than 0.005.</p>",
              "<h3>Earlier causal diagnostic</h3>"]
    parts.append(table(["Model", "MAP@5"], [(name, entry.get("evaluation", {}).get("metrics", {}).get("primary_all_users", {}).get("map_at_5", "not completed")) for name, entry in report["earlier_diagnostics"].items()]))
    parts += ["<h3>Independent candidate-source recall</h3>"]
    rows = []
    for fold, data in report["candidate_pilots"].items():
        if not isinstance(data, dict) or "sources" not in data:
            continue
        for source, metrics in data["sources"].items():
            rows.append((fold, source, f"{data['baseline']['recall']:.6f}", f"{metrics['union_with_v4']['recall']:.6f}", metrics["new_relevant_pairs"]))
    parts.append(table(["Origin", "Source", "V4 item recall", "Union item recall", "New relevant pairs"], rows))
    component = next(entry["fit"] for entry in report["experiments"] if entry["name"] == "D_both")
    parts.append("<h3>Selected behavioral component: feature contributions</h3><p>Split-gain importance for the fixed D_both fit is a conditional model diagnostic, not causal evidence. The ablation scores above measure the effect of removing groups and refitting under the same budget.</p>")
    parts.append(table(["Feature", "LightGBM split gain", "Splits"],
                       [(row["feature"], f"{row['gain']:.4f}", row["split"]) for row in component["importance"][:10]]))
    parts += ["<h2 class='page'>5. Same two users across every method</h2>",
              f"<p>Users were selected deterministically using training activity only: heavy viewer <code>{esc(heavy)}</code> has {counts[heavy]} broad historical programmes; light viewer <code>{esc(light)}</code> has {counts[light]}. They illustrate activity contrast rather than population-level evidence. Titles and categories are looked up from validation schedules; original source-language names are preserved. 'Observed before' refers only to ≥30% historical programme evidence on this platform. A strict hit means the programme actually meets the documented validation target, not that it was enjoyed.</p>"]
    for user in selected_users:
        parts.append(f"<h3>User {esc(user)} — {len(truth[user])} strict relevant validation programmes</h3>")
        for method in paths:
            selected_rows = [row for row in details if row["user_id"] == user and row["method"] == method]
            parts.append(f"<h3>{esc(method)}</h3>" + table(["Rank", "Programme ID", "Source title", "Source category", "Strict hit", "Observed before"],
                [(r["rank"], r["tv_show_id"], r["title"], r["category"], "yes" if r["strict_validation_hit"] else "no", "yes" if r["observed_training_ge30"] else "no") for r in selected_rows]))
        observations = []
        for method in paths:
            method_rows = [r for r in details if r["user_id"] == user and r["method"] == method]
            familiar = sum(r["observed_training_ge30"] for r in method_rows)
            new_hits = sum(r["strict_validation_hit"] and not r["observed_training_ge30"] for r in method_rows)
            observations.append((method, sum(r["strict_validation_hit"] for r in method_rows),
                                 f"{average_precision_at_5(truth[user], predictions[method][user]):.6f}", familiar, new_hits))
        parts.append(table(["Method", "Strict hits / 5", "AP@5", "Observed / 5", "New strict hits"], observations))
        common = set(predictions["V4 ensemble"][user]) & set(predictions["V5 selected strategy"][user])
        parts.append(f"<p>V4 and V5 share {len(common)}/5 programme IDs. " +
                     ("For this heavy viewer, content filtering concentrates on films and finds no strict hit. Association/repeat models emphasize frequently co-viewed news and children's programming, but only programme 240081 is a strict hit among these top-five lists. V4 ranks it first and V5 second, so V5 is worse for this example despite its aggregate gain. Broad historical activity (750 programmes) does not imply 750 completed preferences."
                      if user == heavy else
                      "For this light viewer, content filtering emphasizes morning/entertainment metadata without a strict hit. Repeat ranking stays with observed items and misses all five strict labels. Association rules and popularity retrieve programme 240081 outside the observed ≥30% training history, demonstrating one supported discovery rather than novelty alone. V3, V4 and V5 place that hit fifth; V4 and V5 are identical for this user. The examples were selected by historical activity rather than a favorable V5 outcome.") + "</p>")
    parts += ["<h2>6. Platform strategy and limitations</h2>",
              "<p>For this repeat-heavy television setting, personal occurrence frequency provides a strong anchor, while association signals and schedule opportunity improve ordering. Content similarity offers a distinct metadata-based approach but its weak measured performance and incomplete candidate metadata coverage do not justify selecting it merely for apparent novelty. A programme absent from the observed training history is not necessarily unseen elsewhere, and an unfamiliar recommendation is not automatically useful discovery.</p>",
              f"<p>The final selected strategy is <strong>V5 ({esc(selected)})</strong>: actual user-verified Kaggle Public MAP@5 = <strong>0.44671</strong> and Private MAP@5 = <strong>0.43927</strong>. It is the best of all five submitted versions on both metrics, exceeding V4 by 0.00584 Public and 0.00375 Private. These improvements agree in direction with local validation, but their magnitudes differ. The aspirational Public target of <strong>0.50 was not achieved</strong> (shortfall 0.05329). This is a measured offline improvement, not a production effect. Candidate recall gains remain potential headroom rather than proof of an untrained wider reranker's quality.</p>",
              "<p>Only labelled airtime is recommendable, so most tuning cannot be attributed to a programme. Missing channels, ambiguous clocks, background viewing, differing history/target horizons, unseen catalogue items, sparse light-user histories and validation-selection optimism limit generalization. No live A/B test, engagement improvement or revenue effect was measured or claimed. Operationally, precompute profiles and candidate ranks offline, update schedules incrementally, and monitor catalogue coverage, repeat concentration, cold-start behavior and latency before any production claim.</p>",
              "<h2>7. Runtime, environment, and reproducibility</h2>",
              f"<p>Python {esc(report['execution']['python_version'].split()[0])}; NumPy {esc(report['execution']['numpy'])}; SciPy {esc(report['execution']['scipy'])}; LightGBM {esc(report['execution']['lightgbm'])}; Intel macOS; seed 42; two native threads. Full test output and scope-specific peak RSS are stored in the JSON evidence. V5 study runtime before report write: {report['execution']['runtime_seconds_before_report_write']:.2f} seconds. Stage timings do not include an invented combined peak.</p>"]
    parts.append(table(["Fit", "Rows", "Features", "Rounds", "Seconds", "Peak RSS MiB"],
                       [(e["name"], e["fit"]["rows"], len(e["fit"]["features"]), e["fit"]["rounds"], f"{e['fit']['runtime_seconds']:.2f}", f"{e['fit']['peak_process_rss_mib']:.2f}")
                        for e in report["experiments"] if e.get("fit")]))
    parts.append("<h3>Reproduction and final-only verification</h3><pre><code>python3.13 -m venv .venv\n.venv/bin/python -m pip install -r requirements-v4.txt\n.venv/bin/python -m unittest discover -s tests -v\n# With the original local data and archived derived caches:\n.venv/bin/python src/finalize_exam_v5.py --verify\n# Rebuild report content only (no model fitting):\n.venv/bin/python src/academic_report_v5.py --final --repository-url " + esc(repository_url or "REPOSITORY_URL") + " --prefix academic_report_v5_final</code></pre><p>The exam run uses a venv-local OpenMP runtime on Intel macOS; dependency installation is an environment-setup instruction, not an installation performed during finalization. Source, tests, aggregate summaries, selected CSV, and final report are in GitHub. Raw data, large processed caches, full archived experiments and unredacted AI exports remain local. Historical fitting commands in the reproduction guide are for a separate fresh workspace; finalization performs inference and verification only.</p>")
    parts += ["<p>See <code>docs/behavioral_v5_reproduction.md</code>, versioned model specifications and per-fit JSON for exact hyperparameters, gains, failures and skips. Raw inputs, earlier submissions and the original AI export were fingerprinted and preserved. The final CSV was replayed with two Python hash seeds and checked for exact row order, string user IDs, five unique positive available IDs and deterministic bytes.</p>",
              "<h2 class='page'>8. Final verified Kaggle results and original evidence</h2>",
              f"<p>The designated final scored submission is <strong><code>outputs/submission_v5.csv</code></strong>, SHA-256 <code>{esc(current_sha)}</code>. Public MAP@5 = <strong>0.44671</strong>; Private MAP@5 = <strong>0.43927</strong>. The user manually submitted this exact selected artifact and verified its returned results. Both original screenshots below visibly substantiate the reported values; no automatic submission, additional optimization or change to the selected predictions was performed during finalization.</p>",
              "<p>V5 improves both metrics over V4. The Public target of 0.50 was not achieved. The supplied Markdown instructions refer to Public score, while the official slide PDF requests Private-score evidence; both are reported here. The referenced generic homework guidelines were not supplied, so requirements beyond the available documents cannot be independently checked.</p>"]
    parts.append(table(["Version", "Architecture", "Public MAP@5", "Private MAP@5"],
                       [(row["version"], architecture, f"{row['public_map_at_5']:.5f}", f"{row['private_map_at_5']:.5f}")
                        for row, architecture in zip(verified["history"], ("Association Rules", "Personalized repeat", "Logistic reranking", "LightGBM ensemble", "Behavioral V5 ensemble"))]))
    captions = ("Figure 1. Best SWEET.TV Kaggle submission (V5): Public MAP@5 = 0.44671; Private MAP@5 = 0.43927.",
                "Figure 2. Complete Kaggle submission history, V1–V5, demonstrating the progression from Association Rules through personalized ranking and the final V5 ensemble.")
    for image, caption in zip((screenshot, history_screenshot), captions):
        if image.is_file():
            data = base64.b64encode(image.read_bytes()).decode("ascii")
            parts.append(f"<figure class='screenshot'><img src='data:image/png;base64,{data}' alt='{esc(image.name)}'><figcaption>{esc(caption)}</figcaption></figure>")
        else:
            parts.append(f"<p class='pending'>Missing mandatory original screenshot: {esc(image)}</p>")
    parts += ["<h2 class='page'>9. Sources and assistance disclosure</h2>",
              "<p>Sources: supplied <code>docs/assignment/exam_quiz_GSB(1).md</code> and <code>LLM4Rec_Midterm_Quiz(1).pdf</code>; original competition data; the local audit, content, association, V2, V3, V4 and V5 measured reports. No external dataset or pretrained model was used. NumPy, SciPy and LightGBM provide numerical and tree-training functionality. OpenCode AI assistance contributed project analysis, implementation, tests and documentation under user direction. The archived original export is retained unchanged; a complete final-session export must correspond to the submitted work under the course rules.</p>",
              "<p>The course submission requires exactly the final academic PDF and the original unmodified AI-session export. The implementation and selected CSV are in the linked private GitHub repository; the instructor must receive access or the owner must explicitly approve a visibility change. Neither raw datasets nor unredacted AI exports are published. The original archived export is preserved byte-for-byte. Export the current complete OpenCode session into a new final JSON after finalization, validate it, and submit that original export through the course channel.</p></body></html>"]
    return "\n".join(parts), {"generated_at_utc": generated, "pending_requirements": pending,
                               "selected_users": selected_users, "examples": details,
                               "repository_url": repository_url, "designated_submission": "outputs/submission_v5.csv",
                               "submission_sha256": current_sha, "kaggle_results": verified["history"],
                               "screenshots": {str(image.relative_to(root)): file_fingerprint(image) for image in (screenshot, history_screenshot) if image.exists()},
                               "source_report": file_fingerprint(root / "reports/behavioral_v5.json"),
                               "original_ai_export": file_fingerprint(root / "docs/ai-session/sweettv-opencode-original.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-url")
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--history-screenshot", type=Path)
    parser.add_argument("--final", action="store_true", help="Require the real repository URL and both original screenshots.")
    parser.add_argument("--archive-prefix", help="Preserve an existing rendering under this new prefix before rebuilding final report content.")
    parser.add_argument("--prefix", default="academic_report_v5_draft")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if Path(args.prefix).name != args.prefix:
        raise ValueError("Report prefix must be a filename stem.")
    destinations = [root / "reports" / (args.prefix + suffix) for suffix in (".html", "_examples.json")]
    if args.archive_prefix:
        if Path(args.archive_prefix).name != args.archive_prefix or args.archive_prefix == args.prefix:
            raise ValueError("Archive prefix must be a different filename stem.")
        moves = [(root / "reports" / (args.prefix + suffix), root / "reports" / (args.archive_prefix + suffix))
                 for suffix in (".html", "_examples.json", ".pdf")]
        if any(new.exists() for _, new in moves):
            raise FileExistsError("Preserving an existing preflight archive.")
        for old, new in moves:
            if old.exists():
                old.rename(new)
    if any(path.exists() for path in destinations):
        raise FileExistsError("Preserving a completed report draft; select a new prefix.")
    document, evidence = build(root, args.repository_url, args.screenshot, args.history_screenshot, args.final)
    destinations[0].write_text(document)
    destinations[1].write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"html": str(destinations[0]), "pending_requirements": evidence["pending_requirements"]}, indent=2))


if __name__ == "__main__":
    main()
