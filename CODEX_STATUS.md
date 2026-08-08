# Codex Shared Status

Last updated: 2026-08-08 Asia/Calcutta

## Active sessions

| Session | Status | Work lane | Claimed paths | Browser ownership | Next action |
|---|---|---|---|---|---|
| model-code-agent | active | Finalize corrected browser-only Version 5 evidence and Git handoff | `README.md`, `notebooks/`, `CODEX_STATUS.md` | Private V5 Version 3 is successful; all sessions stopped; accelerator None | Verify and independently review the source-only mirrors, then commit, push, and update draft PR #1 |

## Current verified evidence

- Kaggle Version 3 (`scriptVersionId=340880172`) completed in 1h32m29s on T4 x2.
- All 4,407 training studies were encoded; 58/58 studies received exactly one OOF prediction.
- OOF macro AUC was `0.6280` across all 12 scorable targets versus the earlier `0.5833` baseline.
- Train and visible-test fallback counts were zero.
- The visible-test submission was exactly 3 x 13, finite, in `[0,1]`, and independently validated.
- Version 4 quick-saved the current source and documentation without another full compute run.
- A private exact-cache dataset, `surajkumar8642/rsna-knee-v3-private-cached-features`, was created from Version 3 output. The pinned NPZ SHA256 is `811a7619e2f5deb556f1268f1c86ee17ae213910ae446fc996c66d26d549ed9e`.
- Private V5 notebook Version 3 (`scriptVersionId=340945234`) completed successfully in 53 seconds on CPU with every browser contract passing and the execution guard idle.
- V5 H0 reproduced `0.6280362353` exactly. H1 reached `0.6447939998` but failed the common screen: paired study-level bootstrap `P(delta > 0)=0.7485` was below `0.75`, and two targets dropped by more than `0.10`. H2 reached `0.6176430711` and was rejected.
- H3/H4 were not run because their written continuation gates were not met. About ten T4 minutes were used across H0-H2 and the corrected H0/H1 audit rerun.
- All Kaggle sessions are stopped, accelerator is **None**, internet is off, and competition Submit was not clicked.

## Safety boundaries

- Never commit competition CSVs, reports, DICOMs, identifiers, derived per-study labels, embeddings, checkpoints, generated submissions, or credentials.
- Keep model changes private until smoke and full Kaggle validation pass.
- A GitHub push does not authorize a competition submission.

## Handoff log

`2026-08-08 | model-code-agent | active | Downloaded and sanitized the Version 4 notebook source; updated repository documentation and coordination files | Local privacy, notebook, and pipeline verification pending | Finish verification, commit/push feature branch, open draft PR; then present ranked Version 5 design for approval`

`2026-08-08 | model-code-agent | handoff | Committed verified baseline at 66c580c, pushed feature/rsna-2-5d-baseline, and opened draft PR #1 | Pipeline, notebook schema, syntax, sanitizer idempotence, privacy scan, and diff checks passed; independent audit found test-order and weak-label-gating blockers | Keep PR draft; fix blockers with regression tests before any competition submission`

`2026-08-08 | model-code-agent | active | User approved the staged Version 5 direction; claimed the Version 5 design spec | Three read-only reviewers are checking safety gates, experiment order, and cache-runner boundaries | Write, self-review, commit, and present the specification before implementation planning`

`2026-08-08 | v5-task1-contracts | stopped | Local functional-test task stopped when the user clarified browser-only coding/testing | Partial local pytest additions removed before implementation | Revise the plan so Kaggle browser is authoritative and local storage contains only sanitized snapshots and documentation`

`2026-08-08 | model-code-agent | active | Replaced the local-execution plan with a browser-only implementation plan | Local workspace is storage/version-control only; functional tests and fixes move to Kaggle | Create isolated private V5 notebook with accelerator None and begin contract RED/GREEN cells`

`2026-08-08 | model-code-agent | completed | Created private exact-cache dataset and implemented all Version 5 contracts in the private Kaggle browser | Ordering, fold-local supervision, cache SHA, fold digests, parser digest, cached MIL, trainer/checkpoint, OOF, bootstrap, and guarded runner suites passed in a clean Run All | Permit H0 only after CPU contracts pass`

`2026-08-08 | model-code-agent | completed | Ran bounded T4 experiments H0, H1, and H2 | H0=0.6280362353 exact; H1=0.6447939998 screen-only; H2=0.6176430711 rejected; H3/H4 stopped by written gates | Reset saved execution cell to idle, stop T4, and reset accelerator None`

`2026-08-08 | model-code-agent | handoff | Saved successful private Kaggle Version 2 (scriptVersionId 340939456) and exported source-only notebook mirrors | Committed run: 50 seconds CPU, no traceback, no output files, no submission; browser sessions stopped and accelerator None | Review privacy and source parity, then commit/push to draft PR #1`

`2026-08-08 | model-code-agent | active | Corrected the browser implementation to use one paired study cohort per bootstrap replicate and enforce the maximum target-drop gate; reran RED/GREEN contracts and bounded H0/H1 T4 evidence | H1 common screen failed at P=0.7485 with two target drops over 0.10; private V5 Version 3 (scriptVersionId 340945234) completed in 53 seconds with no traceback; all sessions stopped and accelerator None | Verify source parity/privacy, obtain independent review, then commit/push and update draft PR #1`
