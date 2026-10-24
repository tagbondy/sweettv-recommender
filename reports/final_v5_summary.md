# Final verified SWEET.TV V5

Repository: https://github.com/tagbondy/sweettv-recommender (private).

V5 is the final verified best: **Public MAP@5 0.44671; Private MAP@5 0.43927**. The Public target of 0.50 was not achieved. Optimization is finished.

Selected CSV: `outputs/submission_v5.csv`.

SHA-256: `18b79a44e38d3a19b0edd75587535161c32b29bcadd19d28ad106a4c7cc651c0`.

## Actual Kaggle history

| Version | Public MAP@5 | Private MAP@5 |
|---|---:|---:|
| V1 | 0.31901 | 0.31614 |
| V2 | 0.41082 | 0.39407 |
| V3 | 0.43568 | 0.43224 |
| V4 | 0.44087 | 0.43552 |
| V5 | 0.44671 | 0.43927 |

V5 exceeds V4 by 0.00584 Public and 0.00375 Private. Original screenshot evidence is in `reports/evidence/` and the final PDF.

## Fixed strict local comparison

| Experiment | Status | MAP@5, all 1,260 users |
|---|---|---:|
| Popularity | complete | 0.45928350970017634 |
| Content_based | complete | 0.009496031746031745 |
| Association_rules | complete | 0.4869975749559083 |
| V2_repeat | complete | 0.5329305555555555 |
| V3_logistic | complete | 0.5803765432098766 |
| V4_ensemble_control | complete | 0.5842233245149911 |
| V4_binary_C_control | complete | 0.5829750881834215 |
| A_base_replay | complete | 0.5829750881834215 |
| B_channel | complete | 0.5915934744268078 |
| C_time | complete | 0.5913500881834215 |
| D_both | complete | 0.5927374338624338 |
| E_add_origin | complete | 0.5810789241622575 |
| E_disjoint_origins | complete | 0.5768300264550265 |
| E_extra_users | complete | 0.5795800264550265 |
| F_combined | complete | 0.5901011904761905 |
| rank_enriched | complete | 0.5901115520282187 |
| ablate_personal_history | complete | 0.5668302469135802 |
| ablate_association | complete | 0.5883917548500882 |
| ablate_future_schedule | complete | 0.5855507054673721 |
| ablate_global_popularity | complete | 0.5914764109347442 |
| rrf_50_50 | complete | 0.5934080687830687 |
| rrf_30_70 | complete | 0.592563492063492 |
| normalized_50_50 | complete | 0.5927440476190475 |
| four_way_rrf | complete | 0.594623897707231 |

The selected four-way reciprocal-rank ensemble scores **0.594623897707231**, versus V4 **0.5842233245149911**. Earlier causal feature diagnostic: 0.46850749559082894 → 0.4806177248677249. These are local historical targets, not Kaggle scores.

## Interpretation

Channel and time features improve native binary C from 0.5829750881834215 to 0.5927374338624338. Removing personal-history features reduces it to 0.5668302469135802; association and schedule information also contribute conditionally. Extra-origin/extra-user arms and LambdaRank were measured but not selected. Feature gain and ablations are conditional diagnostics, not causal proof.

Channel candidates add 568 primary relevant pairs, channel/time adds 565, and the collaborative pilot adds zero beyond V4's pool. These are candidate-only measurements; no wider reranker's accuracy gain is claimed.

## Runtime and environment

Python 3.13.7; NumPy 2.2.6; SciPy 1.15.3; LightGBM 4.6.0; seed 42; two threads. The study took 473.79 seconds before report serialization. The selected D_both fit took 13.02 seconds with 436.05 MiB peak worker RSS; the final selected refit took 27.25 seconds. Raw/context/profile/feature generation are separate stages. Exact configurations and scope-specific measurements are in the JSON summary.

## Delivery

The final PDF is `reports/academic_report_v5_final.pdf`. Same two users across seven recommendation methods, Top-5 IDs/titles/categories and strict relevance observations are included. Source-language item names and original evidence images are preserved. Raw data, large caches, full historical reports and original unredacted AI exports remain local. See `docs/final_delivery.md`.
