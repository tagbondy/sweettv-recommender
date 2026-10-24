# AGENTS.md

Project rules for the SWEET.TV TV Program Recommendation exam project. These rules are mandatory for all work in this repository.

1. The project is an individual university exam on SWEET.TV TV Program Recommendation.
2. The goal is to recommend exactly five unique TV programme IDs per test user.
3. Use MAP@5 for evaluation.
4. Implement at least two methods from content-based filtering, association rules, implicit matrix factorization, and personalized PageRank.
5. A popularity baseline is mandatory and does not count as one of the two methods.
6. Join viewing logs to broadcast schedules by matching channel and overlapping time intervals. Document and validate the matching threshold.
7. `tv_show_id = 0` denotes unlabeled airtime, not a recommendable programme.
8. Respect temporal ordering: training on weeks 11-20, validation on weeks 21-30, final training on weeks 11-30, and predictions for weeks 31-42. Never train on test-period viewing behavior.
9. Use identical candidate pools, evaluation rules, and cold-start fallback across methods.
10. Evaluate methods independently before testing hybrids.
11. The submission template is `data/raw/submission.csv`, with 1260 unique `user_id`s and five space-separated programme IDs per row.
12. Never modify original files in `data/raw`.
13. Preserve `user_id` values exactly as strings.
14. Record real experiment results, hyperparameters, seeds, runtime, and environment. Never fabricate scores or verification claims.
15. Prefer memory-efficient processing because the original data is large.
16. Keep an audit trail for the report and original AI-session export.
17. Before implementing recommendation algorithms, inspect the official exam instructions placed in `docs/assignment` and flag ambiguities.

## Language policy

18. Use English exclusively for all project-related responses, explanations, and technical discussions.
19. Write all Python code comments and docstrings in English.
20. Write all project documentation, including GitHub documentation, in English.
21. Use English labels in all experiment results, tables, and charts.
22. Write the final academic PDF report entirely in English.
23. Write all business and algorithmic analysis in English.
24. Preserve original AI-session logs and exports unchanged. Do not translate, rewrite, or modify them.
