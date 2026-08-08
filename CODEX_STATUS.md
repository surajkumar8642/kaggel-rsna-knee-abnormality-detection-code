# Codex Shared Status

Last updated: 2026-08-08 Asia/Calcutta

## Active sessions

| Session | Status | Work lane | Claimed paths | Browser ownership | Next action |
|---|---|---|---|---|---|
| model-code-agent | active | Write and review the approved Version 5 improvement specification | `docs/superpowers/specs/2026-08-08-version-5-score-improvement-design.md`, `CODEX_STATUS.md` | No active Kaggle compute; accelerator remains None | Commit the specification, complete written-spec review, then create the implementation plan |

## Current verified evidence

- Kaggle Version 3 (`scriptVersionId=340880172`) completed in 1h32m29s on T4 x2.
- All 4,407 training studies were encoded; 58/58 studies received exactly one OOF prediction.
- OOF macro AUC was `0.6280` across all 12 scorable targets versus the earlier `0.5833` baseline.
- Train and visible-test fallback counts were zero.
- The visible-test submission was exactly 3 x 13, finite, in `[0,1]`, and independently validated.
- Version 4 quick-saved the current source and documentation without another full compute run.
- Kaggle draft session is stopped, accelerator is **None**, internet is off, and competition Submit was not clicked.

## Safety boundaries

- Never commit competition CSVs, reports, DICOMs, identifiers, derived per-study labels, embeddings, checkpoints, generated submissions, or credentials.
- Keep model changes private until smoke and full Kaggle validation pass.
- A GitHub push does not authorize a competition submission.

## Handoff log

`2026-08-08 | model-code-agent | active | Downloaded and sanitized the Version 4 notebook source; updated repository documentation and coordination files | Local privacy, notebook, and pipeline verification pending | Finish verification, commit/push feature branch, open draft PR; then present ranked Version 5 design for approval`

`2026-08-08 | model-code-agent | handoff | Committed verified baseline at 66c580c, pushed feature/rsna-2-5d-baseline, and opened draft PR #1 | Pipeline, notebook schema, syntax, sanitizer idempotence, privacy scan, and diff checks passed; independent audit found test-order and weak-label-gating blockers | Keep PR draft; fix blockers with regression tests before any competition submission`

`2026-08-08 | model-code-agent | active | User approved the staged Version 5 direction; claimed the Version 5 design spec | Three read-only reviewers are checking safety gates, experiment order, and cache-runner boundaries | Write, self-review, commit, and present the specification before implementation planning`
