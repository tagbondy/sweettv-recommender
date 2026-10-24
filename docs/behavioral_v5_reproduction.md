# Reproducing the V5 behavioral-ranking extension

This document supplements the preserved earlier project README and experiment
reports. Real measurements and selection decisions are saved in
`reports/behavioral_v5.json`; the human-readable counterpart is
`reports/behavioral_v5.md`. Those immutable historical reports predate manual
Kaggle scoring. Current verified results supersede their then-unscored status:
V5 Public MAP@5 **0.44671**, Private MAP@5 **0.43927**, best of V1–V5 on both.
See `reports/kaggle_verified_history.json` and `reports/final_v5_summary.md`.
The Public target of 0.50 was not achieved. Optimization is finished.

## Environment

The measured environment uses Python 3.13.7, NumPy 2.2.6, SciPy 1.15.3, and
LightGBM 4.6.0 on Intel macOS. LightGBM's OpenMP dependency is installed locally
inside `.venv`, as recorded in `reports/lightgbm_v4_environment.json`.
No GPU is required. Seeds are 42 and native fitting uses two threads.

```bash
export OMP_NUM_THREADS=2
export OMP_WAIT_POLICY=PASSIVE
export KMP_BLOCKTIME=0
.venv/bin/python -m unittest discover -s tests -v
```

## Required earlier evidence

The extension reuses the original event join, continuous-watch occurrence
database, historical profiles, expanded V4 candidate matrices, and V2–V4
reference predictions/models. Keep `data/raw/` unchanged and obtain the original
data through the competition. These large input/derived files are local
artifacts rather than a prerequisite to downloading the source code.

The earlier build order is documented in each source's CLI and report:

1. `src/audit_data.py` and `src/build_interactions.py`.
2. `src/popularity.py`, `src/content_based.py`, `src/diagnose_evaluation.py`,
   and `src/association_rules.py`.
3. `src/continuous_watch.py` and `src/personalized_repeat.py`.
4. `src/rerank_v3.py` and `src/lightgbm_v4.py`.

Existing experiment outputs are protected. Use a separate fresh workspace for
a complete reproduction rather than replacing successful archived artifacts.
Inspect each CLI's `--help` before executing a historical build.

## V5 stages

```bash
.venv/bin/python -u src/behavioral_evidence_v5.py --stage all
.venv/bin/python -u src/behavioral_v5.py --stage features
.venv/bin/python -u src/behavioral_v5.py --stage study
```

The evidence command performs one streaming, disk-backed raw viewing pass,
exports causal channel/time contexts at five cutoffs, and reconstructs exact
additional rolling-origin programme evidence. Completed raw/context/profile
checkpoints are reused. Contextual interval unions never determine relevance:
strict labels continue to require one original session covering strictly more
than 80% of the full original broadcast duration.

Features append 18 channel and 23 time/context attributes to the existing 61
features. Original `FeatureTable.x` retains its 61-column stride. A separate
102-column matrix is used for V5 native fitting. Versioned
`*_normalized_channels` bundles normalize historical integer channel IDs to
context string keys for the historical-channel preference feature; original
completed feature bundles remain preserved.

Initial comparisons keep the existing expanded candidate pool fixed. Rolling
training targets W14–15, W16–17, and W18–20 are disjoint. Extra fitting users are
selected deterministically from pre-origin activity; evaluation remains the
original 1,260 template users. Query weighting equalizes origin and user mass,
with extra users receiving half the mass of template users. Candidate-only
channel/time/collaborative pilots measure recall and oracle limits separately
from ranking accuracy. Collaborative item cosine is computed in bounded blocks
and keeps the top 100 neighbors per item.

Each native fit runs in a fresh subprocess with a 300-second maximum. Experiment
specifications, models, scores, predictions, timing, and memory measurements are
saved independently. The exam-run optimization cutoff is deliberately fixed at
2026-10-24 13:00 UTC; the historical study refuses new optimization fits after
that timestamp. Final selected refitting and deterministic replay are separate
delivery steps. This deadline guard is part of the recorded exam run.

## Outputs and verification

- `reports/behavioral_v5_experiments.jsonl`: incremental experiment ledger.
- `reports/behavioral_v5_*_fit.json`: exact fitted configuration and importance.
- `reports/behavioral_v5_candidate_*.json`: candidate-source diagnostics.
- `reports/behavioral_v5_final_strategy.json`: frozen inference strategy.
- `reports/behavioral_v5.json` and `.md`: integrated evidence and conclusion.
- `outputs/submission_v5.csv`: exactly five unique positive available IDs for
  each original string user ID, in original template order.

The final build checks independent inference with two different Python hash
seeds and checks archived original-file fingerprints. V1–V4 CSVs remain
available. If no validated V5 strategy improves upon V4, the V5 delivery may
retain V4 predictions; consult the report rather than assuming improvement.

No command uploads to Kaggle. The user manually scored the frozen
`outputs/submission_v5.csv`. Both original screenshots are retained in
`reports/evidence/` and embedded in `reports/academic_report_v5_final.pdf`.
For finalization, use only the inference/verification/report commands in
`docs/final_delivery.md`; do not launch the historical fitting stages above.
