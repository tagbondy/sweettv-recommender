# SWEET.TV TV Programme Recommendation

Individual university examination project: recommend exactly **five unique TV
programme IDs** per test user. Final model: **V5 behavioral-personalization
ensemble**, evaluated with **MAP@5**.

**Repository:** https://github.com/tagbondy/sweettv-recommender (private; instructor
access must be granted by the owner).

## Final verified result

| Version | Main strategy | Public MAP@5 | Private MAP@5 |
|---|---|---:|---:|
| V1 | Association Rules | 0.31901 | 0.31614 |
| V2 | Personalized repeat recommendations | 0.41082 | 0.39407 |
| V3 | Logistic regression reranking | 0.43568 | 0.43224 |
| V4 | LightGBM and rank-fusion ensemble | 0.44087 | 0.43552 |
| **V5** | **Behavioral personalization and rank fusion** | **0.44671** | **0.43927** |

These are actual user-verified Kaggle returned results, supported by the two
original screenshots in `reports/evidence/`. V5 improved over V4 by **0.00584
Public** and **0.00375 Private**. The ambitious Public target of **0.50 was not
achieved**. Optimization has ended; finalization does not retrain models, search
hyperparameters, change the selected predictions, or submit to Kaggle.

- Selected CSV: [`outputs/submission_v5.csv`](outputs/submission_v5.csv).
- SHA-256: `18b79a44e38d3a19b0edd75587535161c32b29bcadd19d28ad106a4c7cc651c0`.
- Final academic PDF: [`reports/academic_report_v5_final.pdf`](reports/academic_report_v5_final.pdf).
- Verified history: [`reports/kaggle_verified_history.json`](reports/kaggle_verified_history.json).
- Aggregate experiments: [`reports/final_v5_summary.md`](reports/final_v5_summary.md)
  and its JSON counterpart.

## Data and limitations

The competition supplies 5,821,987 viewing intervals for 4,914 users, three
broadcast schedules, and a template of 1,260 users. Only **10.8%** of first-period
schedule rows have a positive programme ID. `tv_show_id = 0` is unlabelled
airtime, never a recommendable programme. V5 uses raw tuning intervals, including
unlabelled airtime, as channel/time context only. Viewing is not evidence of
attention, enjoyment, or never having watched elsewhere.

Join on identical `vsetv_id == channel_id` and overlapping half-open intervals.
The broad baseline uses at least 120 overlapping seconds. Strict local relevance
requires **one individual session covering >80% of the full original broadcast
duration**. Disconnected sessions are not combined into strict labels. The raw
viewing column `duraton` is not a programme-length denominator. Source clocks
remain uncorrected; 117 one-hour discrepancies are documented rather than guessed.

Raw data, large derived databases/matrices, virtual environments, credentials,
and unredacted AI-session exports are **not included in Git**. Obtain authorized
data from the [competition](https://www.kaggle.com/competitions/sweettv-tv-program-recommender)
and place the five original CSVs under `data/raw/`. Historical local reports and
V1–V4 artifacts remain unchanged in the working directory; only concise final
evidence is published. The selected submission CSV is included as the assignment
requires.

## Evaluation protocol

- Primary history: weeks 11–20, `[2020-03-09, 2020-05-18)`.
- Validation: weeks 21–30, `[2020-05-18, 2020-07-27)`.
- Final history: weeks 11–30; forecast weeks 31–42,
  `[2020-07-27, 2020-10-19)`.
- No test-period viewing behavior is used. Future schedules describe availability
  and airtime opportunities only.
- AP@5 is the sum of precision at relevant ranks divided by
  `min(5, number of unique relevant programmes)`; empty relevance gives zero.
  MAP@5 averages over all 1,260 original users, preserving exact string IDs/order.
- Predictions contain five unique positive available IDs; repeats are allowed.
  Every method shares the available catalogue and historical-popularity fallback.
  Ground truth is never filtered down to retrieved candidates.
- Independent content/rule/baseline comparisons use the full common catalogue.
  Rerankers retrieve subsets; V5 feature/data comparisons hold the expanded V4
  candidate pool fixed. Candidate-only extension pilots are separately reported.
- Local validation is an explicitly documented preference proxy, not a claimed
  reproduction of Kaggle's hidden target. Tuned validation scores have selection
  optimism.

| Strict local method | MAP@5, all 1,260 users |
|---|---:|
| Popularity baseline | 0.459283510 |
| Content-based filtering | 0.009496032 |
| Association Rules | 0.486997575 |
| V2 repeat | 0.532930556 |
| V3 logistic | 0.580376543 |
| V4 ensemble | 0.584223325 |
| **V5 final ensemble** | **0.594623898** |

## Architecture

1. **Popularity** ranks available programmes by distinct historical viewers.
2. **Content-based filtering** uses field-normalized metadata TF-IDF and a
   log-count-weighted user profile, ranked by cosine similarity.
3. **Association Rules** use capped historical baskets, minimum pair support 5,
   confidence 0.05, and maximum applicable confidence. Content and association
   methods independently satisfy the two-method examination requirement.
4. **V2 repeat** uses individual-session occurrence counts and personal recency.
5. **V3 logistic regression** reranks 61 repeat, rule, popularity and schedule
   features using earlier chronological labels.
6. **V4 LightGBM** uses seeded binary trees and reciprocal-rank fusion.
7. **V5** appends 18 channel and 23 time/context features: smoothed/recent channel
   affinity, preferred airing opportunities, hour/weekday alignment, channel×time
   compatibility and personal-history/rule interactions. Context unions never
   change strict label construction.

The selected V5 binary model has 102 features, 31 leaves, learning rate 0.03,
minimum leaf 50, L2 10, **269 rounds**, seed 42 and two threads. Its final refit
uses 293,315 early/primary historical candidate pairs. Four-way fusion uses
weights **0.10 repeat / 0.20 V3 / 0.20 ranked V4 ensemble / 0.50 V5 binary**,
with contribution `weight / (20 + rank)` over each source's top 50. Cold users
receive the same available historical-popularity Top-5.

Additional origins, extra fitting users, LambdaRank and independent collaborative
candidates were measured rather than assumed to help. They were not selected
for the final V5 binary model. All original predictions remain frozen.

## Environment and tests

Measured environment: Python **3.13.7**, NumPy **2.2.6**, SciPy **1.15.3**,
LightGBM **4.6.0**, Intel macOS. The original environment uses a venv-local
OpenMP runtime. No new package installation is required for the existing setup.

For a separately provisioned environment:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-v4.txt
export OMP_NUM_THREADS=2
export OMP_WAIT_POLICY=PASSIVE
export KMP_BLOCKTIME=0
.venv/bin/python -m unittest discover -s tests -v
```

Final-only verification, **with the original local data and frozen caches**:

```bash
.venv/bin/python src/finalize_exam_v5.py --verify
```

This validates the CSV, frozen inputs, source/cutoff lineage, complete tests and
two independent inference-only replays; it never fits or optimizes. Its audit
file is exclusive: choose a new `--output` path for a repeated verification.

Historical from-scratch build stages and measured-runtime scope are documented
in [`docs/behavioral_v5_reproduction.md`](docs/behavioral_v5_reproduction.md).
Historical commands are for a separate fresh workspace, not the frozen delivery.
The V5 study took **473.79 seconds** before report serialization; component
timings and peak RSS are recorded in the summary. That duration excludes earlier
raw/context/feature generation.

## Academic report and AI export

The final English PDF includes the same two real users across seven methods,
Top-5 IDs/titles/categories, strict hits and AP@5, business analysis, exact final
checksum, this repository URL, the V1–V5 results table, and **both original Kaggle
screenshots as separate figures**. Draft PDFs are preserved locally.

Report-content reproduction (no fitting; requires archived local evidence):

```bash
.venv/bin/python src/academic_report_v5.py --final \
  --repository-url https://github.com/tagbondy/sweettv-recommender \
  --prefix academic_report_v5_final
```

Rendering uses the existing local Chrome installation; see
[`docs/final_delivery.md`](docs/final_delivery.md). Existing final artifacts are
never overwritten.

OpenCode AI assistance contributed analysis, implementation, tests and writing
under user direction. Original unredacted exports stay local and are submitted
through the course channel, not GitHub. The course requires **the final PDF and
the original complete AI-session export**. Exact final-export instructions and
private-repository access requirements are in `docs/final_delivery.md`.
