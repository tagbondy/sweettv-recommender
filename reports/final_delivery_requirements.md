# Official examination requirement check

Checked against `AGENTS.md`, `docs/assignment/exam_quiz_GSB(1).md`, and
`docs/assignment/LLM4Rec_Midterm_Quiz(1).pdf` during finalization.

| Requirement | Actual evidence / status |
|---|---|
| Individual project; exactly five programme IDs | Frozen `outputs/submission_v5.csv`; 1,260 original string users, 6,300 slots, exact order |
| MAP@5 and consistent temporal hold-out | Final PDF sections 2 and 4; all 1,260 users, empty relevance kept, ground truth unfiltered |
| Popularity baseline plus at least two listed methods | Independent popularity, content-based filtering, and Association Rules; final PDF sections 3–5 |
| Explain the interval join and threshold | Same channel plus half-open overlap; ≥120 seconds broad evidence and >80% individual-session strict relevance; PDF sections 1–2 |
| Treat ID 0 correctly | Never an item; tuning context only in V5; PDF section 1 |
| Respect time and prevent future viewing leakage | History/label cutoffs audited; final cache has no labels; inference-only replays match frozen bytes; existing leakage tests pass |
| Shared candidates, repeat policy and cold fallback | Common available catalogue, explicit reranker subsets, fixed V5 feature-comparison pool, identical popularity fallback; PDF section 2 |
| Independently evaluate before hybrid selection | Baseline/content/rules/V2/V3/native LightGBM and controlled V5 arms measured before selected fusion; PDF sections 3–4 |
| Two same users across methods | Users `13571199046623737884` and `4478271560731083975`; seven methods, 70 ordered recommendation rows with source title/category, hits and AP@5; PDF section 5 |
| Business/algorithmic strategy and supported limitations | PDF sections 5–6, including a worse V5 user example, discovery qualified by actual strict hits, missing labels/clocks/background viewing |
| Exact final scored CSV in GitHub | `outputs/submission_v5.csv`, SHA-256 `18b79a44e38d3a19b0edd75587535161c32b29bcadd19d28ad106a4c7cc651c0` |
| Actual returned Kaggle results | V5 Public 0.44671, Private 0.43927; complete verified V1–V5 table in PDF section 8 |
| Public and Private evidence ambiguity | Markdown requests Public; official slides request Private. Both values and both original screenshots are included |
| Best-result screenshot embedded | `kaggle_v5_scores.png`: actual embedded image and full caption, final PDF page 12 |
| All-attempts screenshot embedded | `kaggle_all_scores.png`: separate actual embedded image and full caption, final PDF page 13 |
| Actual GitHub URL in PDF | `https://github.com/tagbondy/sweettv-recommender`; private; owner must grant instructor access |
| Reproducibility, parameters, environment, runtimes | README, reproduction guide, aggregate JSON/Markdown and PDF sections 3–4 and 7 |
| External code and AI disclosure | NumPy/SciPy/LightGBM and OpenCode assistance disclosed; no external dataset/pretrained model claimed; PDF section 9 |
| Preserve original AI-session export | Original JSON parsed and fingerprint-verified unchanged; never staged or published |
| Complete final AI-session export | New `docs/ai-session/sweettv-opencode-final.json` can be exported at finalization; export again to a new timestamped file after the last assistant response to include that response |
| Submit exactly PDF + original AI export | User must upload these two files through the course channel; the CSV belongs in GitHub |
| Student identity on Kaggle | Original best-result screenshot visibly shows the student's submitted name and `tagbondy`; no account settings were changed |
| General homework guidelines | Not present among supplied files; any additional course-specific rules require the user's separate check |

Final PDF: `reports/academic_report_v5_final.pdf`, 14 A4 pages. The actual
rendered PDF was read and visually inspected for both original evidence figures,
score tables, same-two-user comparisons, captions and readable content. System
PDFKit separately verified text, original screenshot pixel dimensions, figure
pages and absence of obsolete unscored-V5 claims. Existing drafts were preserved.

No model optimization, selected-prediction changes, automatic Kaggle upload,
raw-data publication, or unredacted AI-export publication is part of finalization.
The Public target of 0.50 was not achieved; both verified metrics improved over V4.
