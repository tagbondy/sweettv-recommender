# Final examination delivery

Repository: https://github.com/tagbondy/sweettv-recommender (private).

## Frozen final submission

- `outputs/submission_v5.csv`: 1,260 original string user IDs, five unique positive
  available programme IDs per row, exact original order.
- SHA-256: `18b79a44e38d3a19b0edd75587535161c32b29bcadd19d28ad106a4c7cc651c0`.
- Actual Kaggle Public MAP@5: **0.44671**; Private MAP@5: **0.43927**.
- V5 is the best of the five user-verified submissions on both metrics.
- The Public target of **0.50 was not achieved**.
- The selected predictions, archived V1–V4 files and original AI export are
  preserved. Finalization does not optimize, fit a new model, or upload to Kaggle.

## Final-only verification

These commands require the original local raw data and archived feature/model
caches. Those files are deliberately not committed to GitHub.

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python src/finalize_exam_v5.py --verify
```

The verifier checks all frozen-file fingerprints, exact CSV schema/order/IDs,
schedule availability, source and chronological-cache lineage, and two independent
inference-only replays against the frozen CSV. Repeated verification must use a
new `--output reports/NEW_NAME.json`, preserving prior audit records.

## Final PDF

Course report: `reports/academic_report_v5_final.pdf`.

The original evidence files must both exist:

- `reports/evidence/kaggle_v5_scores.png` — best V5 result.
- `reports/evidence/kaggle_all_scores.png` — complete V1–V5 history.

Both images are embedded unaltered, full-width on separate PDF pages, with
separate captions. The numerical Public/Private comparison table immediately
precedes the figures. Original screenshot files and existing PDF drafts are
preserved.

Build content only, in a fresh destination if a final file already exists:

```bash
.venv/bin/python src/academic_report_v5.py --final \
  --repository-url https://github.com/tagbondy/sweettv-recommender \
  --prefix academic_report_v5_final
```

The final HTML is self-contained, including original PNG bytes as data URLs.
The recorded local rendering workflow uses existing Google Chrome:

```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless --disable-gpu --no-first-run --no-default-browser-check \
  --no-pdf-header-footer \
  --print-to-pdf="$PWD/reports/academic_report_v5_final.pdf" \
  "file://$PWD/reports/academic_report_v5_final.html"
```

Do not rerun rendering over an existing final PDF; preserve it and use a new
versioned filename. `src/verify_report_pdf.swift` uses local PDFKit to inspect
the actual PDF text, page count, figure captions and embedded-image dimensions.
Actual rendered pages also require visual inspection.

## Original and final AI-session exports

Preserve `docs/ai-session/sweettv-opencode-original.json` unchanged. Never format,
redact, translate, truncate or replace either original export. The `.gitignore`
excludes the entire directory and transcript logs, including in a private repo.

After the final assistant response, export the **current full session** again
through the unmodified OpenCode exporter. The current session ID is
`ses_e96d70653ffe082P1hvYQajt9E`; verify it with `opencode session list`.

If `sweettv-opencode-final.json` does not already exist, this exact command creates
the requested new original file exclusively (the `noclobber` option prevents
overwriting):

```bash
(set -o noclobber; opencode export ses_e96d70653ffe082P1hvYQajt9E \
  > docs/ai-session/sweettv-opencode-final.json)
.venv/bin/python -m json.tool docs/ai-session/sweettv-opencode-final.json > /dev/null
shasum -a 256 docs/ai-session/sweettv-opencode-final.json
```

Do not use `--sanitize`, because that changes the original export. If a final
export already exists, retain it and create a new timestamped original export
with the same command and a new filename. The finalization-time export cannot
include messages written after its creation; the post-response export is needed
for a complete final record. Never commit these unredacted exports.

## Course submission and repository access

Submit exactly the final academic PDF and the original, complete AI-session JSON
through the course channel. The CSV and code are in GitHub. Grant the instructor
access to the private repository before submission, or obtain explicit approval
before changing visibility to public. Do not publish raw datasets or unredacted
AI-session exports.

The supplied Markdown asks for Public-score evidence, while the supplied slides
ask for Private-score evidence. Both are included. The generic homework
guidelines referred to in the instructions were not supplied; check any separate
course-specific requirements and the student's displayed submission identity.
