# Codex Collaboration Rules

This repository may be edited by multiple Codex sessions at the same time.

1. Read `AGENTS.md` and `CODEX_STATUS.md` before starting work.
2. Add or update your row in `CODEX_STATUS.md` before editing files.
3. Claim exact files or directories. Do not edit a path claimed by another active session.
4. Keep browser ownership isolated. Do not claim, navigate, or modify a Kaggle tab owned by another session.
5. Record completed commands, test results, external actions, blockers, and the next action in `CODEX_STATUS.md`.
6. Never overwrite or revert changes made by another session. Stop and record a blocker if work overlaps.
7. Never commit Kaggle credentials, raw competition data, DICOM files, patient identifiers, generated submissions, or model checkpoints.
8. Keep Kaggle notebooks private while testing. Publishing code or making a competition submission requires explicit user approval at action time.
9. Use `submission.csv` only after validating its columns, row count, identifiers, numeric values, finite values, and allowed probability range.
10. When finished, mark your row `handoff` or `complete` and list any files that remain modified but uncommitted.
11. Always test every Kaggle notebook change on Kaggle before handoff. Run a smoke version first, record whether every cell completed, and validate the generated `submission.csv`. A notebook that has not completed this test must be marked `untested` and must not be submitted.
