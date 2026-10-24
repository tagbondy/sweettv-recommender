"""Create a small publication-safe measured summary, without user-level logs."""

import json
from pathlib import Path

from finalize_exam_v5 import write_new


def main():
    root = Path(__file__).resolve().parents[1]
    study = json.loads((root / "reports/behavioral_v5.json").read_text())
    history = json.loads((root / "reports/kaggle_verified_history.json").read_text())
    experiments = []
    for entry in study["experiments"]:
        row = {"name": entry["name"], "status": entry["status"]}
        if entry.get("evaluation"):
            row["strict_map_at_5_all_1260"] = entry["evaluation"]["metrics"]["primary_all_users"]["map_at_5"]
        if entry.get("fit"):
            fit = entry["fit"]
            row.update({"specification": fit["specification"], "training_windows": fit["training_windows"],
                        "rows": fit["rows"], "features": len(fit["features"]), "rounds": fit["rounds"],
                        "runtime_seconds": fit["runtime_seconds"], "peak_process_rss_mib": fit["peak_process_rss_mib"],
                        "native_params": fit["native_params"], "top_feature_gains": fit["importance"][:12]})
        experiments.append(row)
    summary = {"repository_url": "https://github.com/tagbondy/sweettv-recommender", "visibility": "PRIVATE",
               "selected_submission": "outputs/submission_v5.csv", "submission_sha256": history["submission_sha256"],
               "verified_kaggle_history": history["history"], "public_target_achieved": False,
               "selected_strategy": study["selected_strategy"], "experiments": experiments,
               "earlier_diagnostic": {name: entry["evaluation"]["metrics"]["primary_all_users"]["map_at_5"]
                                      for name, entry in study["earlier_diagnostics"].items()},
               "earlier_feature_gain": study["earlier_feature_gain"], "candidate_pilots": study["candidate_pilots"],
               "execution": study["execution"], "feature_generation": {key: value for key, value in study["feature_generation"].items() if key != "extra_fitting_users"},
               "evidence_stage_measurements": study["evidence_checkpoints"],
               "final_csv_validation": study["final_build"]["validation"],
               "historical_deterministic_replay_verified": study["final_build"]["independent_hash_seed_replay"],
               "publication_scope": "Aggregate measured results only; no raw data, full user-level experiment logs, or AI-session exports."}
    write_new(root / "reports/final_v5_summary.json", summary)
    lines = ["# Final verified SWEET.TV V5", "", "Repository: https://github.com/tagbondy/sweettv-recommender (private).", "",
             "V5 is the final verified best: **Public MAP@5 0.44671; Private MAP@5 0.43927**. "
             "The Public target of 0.50 was not achieved. Optimization is finished.", "",
             "Selected CSV: `outputs/submission_v5.csv`.", "",
             f"SHA-256: `{history['submission_sha256']}`.", "",
             "## Actual Kaggle history", "", "| Version | Public MAP@5 | Private MAP@5 |", "|---|---:|---:|"]
    for row in history["history"]:
        lines.append(f"| {row['version']} | {row['public_map_at_5']:.5f} | {row['private_map_at_5']:.5f} |")
    lines += ["", "V5 exceeds V4 by 0.00584 Public and 0.00375 Private. Original screenshot evidence is in `reports/evidence/` and the final PDF.",
              "", "## Fixed strict local comparison", "", "| Experiment | Status | MAP@5, all 1,260 users |", "|---|---|---:|"]
    for row in experiments:
        lines.append(f"| {row['name']} | {row['status']} | {row.get('strict_map_at_5_all_1260', 'not measured')} |")
    lines += ["", "The selected four-way reciprocal-rank ensemble scores **0.594623897707231**, versus V4 **0.5842233245149911**. "
              "Earlier causal feature diagnostic: 0.46850749559082894 → 0.4806177248677249. These are local historical targets, not Kaggle scores.",
              "", "## Interpretation", "", "Channel and time features improve native binary C from 0.5829750881834215 to 0.5927374338624338. "
              "Removing personal-history features reduces it to 0.5668302469135802; association and schedule information also contribute conditionally. "
              "Extra-origin/extra-user arms and LambdaRank were measured but not selected. Feature gain and ablations are conditional diagnostics, not causal proof.",
              "", "Channel candidates add 568 primary relevant pairs, channel/time adds 565, and the collaborative pilot adds zero beyond V4's pool. "
              "These are candidate-only measurements; no wider reranker's accuracy gain is claimed.",
              "", "## Runtime and environment", "", "Python 3.13.7; NumPy 2.2.6; SciPy 1.15.3; LightGBM 4.6.0; seed 42; two threads. "
              "The study took 473.79 seconds before report serialization. The selected D_both fit took 13.02 seconds with 436.05 MiB peak worker RSS; "
              "the final selected refit took 27.25 seconds. Raw/context/profile/feature generation are separate stages. Exact configurations and scope-specific measurements are in the JSON summary.",
              "", "## Delivery", "", "The final PDF is `reports/academic_report_v5_final.pdf`. Same two users across seven recommendation methods, "
              "Top-5 IDs/titles/categories and strict relevance observations are included. Source-language item names and original evidence images are preserved. "
              "Raw data, large caches, full historical reports and original unredacted AI exports remain local. See `docs/final_delivery.md`.", ""]
    with (root / "reports/final_v5_summary.md").open("x") as handle:
        handle.write("\n".join(lines))
    print("Publication-safe aggregate summaries created; no fitting performed.")


if __name__ == "__main__":
    main()
