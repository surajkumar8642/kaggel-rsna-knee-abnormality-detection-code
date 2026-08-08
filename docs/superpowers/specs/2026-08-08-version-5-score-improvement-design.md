# Version 5 Competition Score Improvement Design

## Status

Approved direction. The user asked for a careful, sequential improvement program
aimed at the best defensible competition score and explicitly asked the agent not
to stop between approved phases. This document turns that direction into testable
contracts before any Version 5 model behavior is changed.

## Objective

Improve the verified Version 3 report-supervised DINOv2 MIL baseline without
losing reproducibility, leaking held-out labels, misaligning hidden-test rows, or
wasting Kaggle GPU quota. A candidate becomes the new champion only through
identical-fold out-of-fold evidence. A public leaderboard result is a final
external check, not an experiment-selection loop.

The current verified reference is Kaggle Version 3 (`scriptVersionId=340880172`):

- 4,407/4,407 training studies encoded;
- 58/58 official-label studies assigned exactly one OOF prediction;
- 12/12 targets scorable;
- OOF macro ROC AUC `0.6280`;
- zero train and visible-test fallbacks;
- 1h32m29s runtime on T4 x2;
- exact visible-test `3 x 13` submission validation.

The downloaded Version 4 notebook is a source snapshot whose retained output is
from a smoke run. It is not independent evidence for the Version 3 full-run
metric. Version 3's saved log and output remain the authoritative execution
evidence.

## External reference and originality boundary

The public notebook *RSNA Knee: read the report, then the knee* was reviewed only
for measured design evidence. Its reported Version 9 public score is `0.847` with
a 4h50m59s T4 x2 runtime. Its most useful high-level observations are that report
label quality dominates, anatomical slice coverage can matter more than image
resolution, larger encoders do not improve monotonically, and rank averaging is
appropriate for macro AUC.

No source code, prose, lexicon, constants, or generated labels will be copied.
Any similar mechanism will be independently implemented from this specification,
covered by our own tests, and judged only on our own leakage-safe OOF protocol.

## Non-goals

- No claim or guarantee of first place or a particular leaderboard rank.
- No competition submission during experimentation.
- No target-specific blend weights selected on 58 gold labels.
- No calibration whose only effect is a monotonic probability transform; it
  cannot improve ROC AUC.
- No repeated public-leaderboard probing.
- No TPU work; this remains a PyTorch pipeline.
- No competition data, reports, identifiers, derived labels, embeddings,
  checkpoints, or submissions in Git.

## Mandatory defects to repair first

### Hidden-test row alignment

The current notebook extracts test features in `test.csv` order but constructs
the output identifier column from `sample_submission.csv`. The visible three rows
happen to pass, but the code does not prove that hidden-test order is identical.

Version 5 will derive the ID column, target order, and row count only from the
runtime sample submission. It will require unique, non-null IDs in both frames,
set equality, and exactly one test row per sample ID. The inference frame will be
created by a left reindex from sample IDs. Exact order equality is asserted both
immediately before inference and immediately before the atomic write.

### Fold-local weak-supervision plumbing

The current parser creates structurally gated `supervision_values` and
`supervision_weights`, but fold training and fallback prevalence consume raw
`weak_values` and `weak_weights`. This bypasses the disable gate for four
implausibly one-sided targets.

Version 5 will build a fresh supervision bundle per fold. Eligibility, agreement
gates, gold multipliers, sampling weights, and positive weights are computed only
from that fold's non-held-out training data. Non-gold weights for disabled targets
are exactly zero and their values are unknown. Non-held-out official labels then
override parser values with exact hard labels. Held-out rows are zero-weighted
defensively and excluded from the loader.

## Invariants and regression tests

Correctness gates are unconditional and override any score improvement.

1. Mutating every held-out report, parser value, parser weight, or official label
   must leave the fold's training indices, enabled mask, target tensor, weight
   tensor, sampler probabilities, and positive weights byte-identical.
2. The feature cache must be label-independent and unchanged by that mutation.
3. Disabled non-gold target weights must be zero; official training values must
   remain exact `0/1` with the configured trusted-gold weight.
4. Raw global weak arrays must never be passed directly to the trainer outside
   the explicit Version 3 reproduction experiment.
5. OOF storage is exactly `(58, 12)` with integer assignment counters. Every gold
   row must be assigned once per CV repetition and only after all predeclared
   seeds for its fold have been averaged.
6. The final metric must be recomputed from official labels and final OOF values;
   the stored and displayed macro values must agree within `1e-12`.
7. Test predictions must be `(len(sample_submission), 12)`, numeric, finite, and
   within `[0, 1]`; columns and ID order must exactly match the sample.
8. Tests and logs print only aggregate counts, hashes, metrics, and runtimes.

## Phase 1: immutable cache-only experiment laboratory

Create a separate private Kaggle notebook that attaches the exact saved Version 3
output and does not decode DICOMs or load Transformers. It resolves one explicit
`train_full_8.npz` under the pinned input, never a recursive "latest" match.

The cache contract requires:

- source script version `340880172`;
- a pinned full-file SHA-256;
- features `(4407, 24, 384)` float16;
- planes `(4407, 24)` int8;
- masks `(4407, 24)` boolean;
- finite active features, valid plane indices, zero inactive features, and at
  least one active instance per study;
- eight slices per plane and zero stored fallback studies;
- a current train-order digest plus an explicit legacy note that Version 3 did
  not embed its own row-order digest.

The next full encoder run must embed train order, schema, encoder resource,
processor, sampler, plane map, and cache hashes so later use is cryptographically
bound to the same data flow.

## Browser notebook component boundaries

The private Kaggle notebook is the only implementation and execution source. It
uses module-shaped cells with one responsibility each:

1. **Contracts:** immutable cache, target, fold, and experiment configurations.
2. **Supervision:** parser outputs and the pure `build_fold_supervision` policy.
3. **Cache:** exact resolution, provenance, validation, and read-only payload.
4. **Folds:** one pinned deterministic Version 3-compatible split and digest.
5. **Datasets:** cached bags only; no identifiers returned by items.
6. **Models:** common `forward(features, planes, mask) -> [batch, 12]` interface.
7. **Training:** one trainer with deterministic seed, early stop, and checkpoint
   selection rules.
8. **Metrics:** per-target and macro AUC, exact OOF accounting, seed averaging,
   and paired bootstrap.
9. **Registry:** immutable named experiments; ad hoc configuration is rejected.
10. **Artifacts:** private aggregate manifests and checkpoints under a fresh run
    nonce; no submission is created by the head laboratory.

After a browser-tested saved version passes review, its exact notebook is
downloaded and sanitized for Git storage. Local files are not used to execute,
test, or independently modify model behavior.

## Phase 2: one-factor cached experiment ladder

Every candidate starts from the current champion. Failed variants are not stacked.

| Order | Experiment | Single changed factor | Promotion rule |
|---|---|---|---|
| H0 | `v3_repro` | Exact Version 3 raw supervision and head | Must reproduce `0.6280 +/- 0.005` |
| H1 | `fold_gated` | Correct fold-local gated supervision | Common screen gate |
| H2 | `gold_weight_2` | Trusted-gold loss weight `2.0` | Common screen gate |
| H3 | `gold_weight_3` | Trusted-gold loss weight `3.0` | Run only if H2 is promising but inconclusive |
| H4 | `seed_ensemble_3` | Three fixed seeds, averaged without seed selection | Confirmation gate |
| H5 | `bag_mask_tta` | Full/even/odd masks with fixed `0.50/0.25/0.25` blend | Common screen gate |
| H6 | `ema_head` | Fixed predeclared EMA | Common screen gate |
| H7 | `residual_statistics` | Attention plus masked mean/max residual summaries | Architecture screen gate |
| H8 | `hierarchical_plane` | Per-plane target attention and masked plane gates | Architecture screen gate |
| H9 | `second_cv_repeat` | Second fixed fold repetition | Confirmation gate |

Gold weighting changes every known official positive and negative equally; it is
not target-specific. Multi-seed and repeated-CV predictions are averaged before a
single final metric is computed. No best seed is selected from held-out results.

## Phase 3: report-supervision improvement

Only begin after the safety-corrected cached champion is known. The new parser is
an original implementation with explicit target concepts, clause boundaries,
negation, uncertainty, prior-history, postoperative, anatomy, and compartment
scope. It will support the languages observed in aggregate corpus diagnostics
without printing or exporting report text.

Parser development is separated from architecture experiments. Fold-local
agreement and coverage gates decide whether a target's weak labels are enabled.
One-sided prevalence, insufficient positive/negative evidence, low agreement, or
unstable fold behavior disables that target rather than manufacturing negatives.
Official training labels always override parser output.

Compare the old and new parsers with aggregate coverage, prevalence, conflict,
and training-fold-only agreement. The held-out gold rows are used only for the
final OOF score, never to choose fold-local rules.

## Phase 4: imaging improvement ladder

Only candidates that pass the cached laboratory are eligible for fresh encoding.
Each imaging change runs contract, bounded smoke, then identical-fold OOF.

| Order | Experiment | Cost expectation | Required gain |
|---|---|---:|---:|
| I1 | Save DINO CLS plus mean-patch descriptor in the same forward | Similar encoder time, about 2x cache | Common screen gate |
| I2 | Increase from 8 to 12 slices per plane | About 50% more encoder work | OOF delta at least `0.010` |
| I3 | Alternate central slice-grid TTA | About 23% more inference encoding | Common screen gate and runtime margin |
| I4 | Second protocol-distinct series per plane | Potentially near 2x imaging | OOF delta at least `0.010` |
| I5 | Partially fine-tune a bounded number of DINO blocks | Highest risk and memory | Must beat all frozen candidates materially |

Resolution increases, large encoders, flip TTA, and broad multi-arm ensembles are
deferred because reference evidence indicates weaker gain per GPU-hour. A second
series is selected by independently defined protocol diversity, not copied slot
rules.

## Metrics and promotion gates

The champion and candidate must use identical official labels, target order,
folds, cache, metric code, and OOF transformation.

### Reproduction gate

- 58 OOF rows, each assigned exactly once;
- 12 scorable targets;
- macro AUC within `+/- 0.005` of `0.6280`.

Failure stops all score experiments until explained.

### Common single-factor screen

- all correctness contracts pass;
- at least 10 scorable targets;
- macro AUC improves by at least `0.005` over the current champion;
- at least 7/12 targets are non-worse within `0.01`;
- no target drops by more than `0.10` during early screening;
- 2,000 paired study-level bootstrap samples yield at least 1,500 valid samples
  and `P(delta macro AUC > 0) >= 0.75`.

### Confirmed champion

- rank-averaged OOF macro AUC at least `0.6450` for the first Version 5 promotion;
- mean single-seed macro AUC at least `0.6380`;
- seed macro-AUC standard deviation at most `0.015`;
- no more than two targets drop by more than `0.03`;
- at least 7/12 targets improve;
- all provenance and leakage tests pass.

If no candidate passes, Version 3 remains authoritative. A candidate is never
promoted merely because it is newer or more complex.

## Runtime and accelerator policy

- Contract, privacy, cache, and most head tests run on CPU.
- T4 is enabled only for a promoted smoke or measured full run.
- A full recipe must project below 7.5 hours, preferably below 6 hours before
  optional inference TTA, leaving Kaggle's nine-hour limit a safety margin.
- Runtime projection includes training extraction, hidden-test extraction, head
  training, inference, validation, and artifact writes.
- Stop the Kaggle session and reset the accelerator to **None** immediately after
  every GPU task, success or failure.
- TPU is not enabled.

## Final hidden-test and submission-readiness contract

The final recipe is frozen before hidden inference. The notebook requires exact
checkpoint count, fold/seed metadata, source hashes, and current nonce. It
reindexes test studies from sample IDs, applies the same preprocessing and
fold-specific fallbacks, and requires fallback rate no greater than 1%.

It writes a nonce-specific CSV and manifest atomically, then writes
`submission.csv`, reloads both, checks modification times and SHA-256, and repeats
all schema, order, uniqueness, numeric, finiteness, and range assertions. It logs
only aggregates.

Saving a validated Kaggle version, publishing code, merging the GitHub draft PR,
and making a competition submission are separate external actions. The final
competition Submit control is not clicked without action-time user approval.

## Failure handling

- Any contract, leakage, privacy, or row-order failure aborts immediately.
- Any reproduction failure blocks later comparisons.
- Any non-finite loss, gradient, cache value, prediction, or metric rejects the
  run and preserves the prior champion.
- Any runtime projection above 7.5 hours aborts before a full run.
- Interrupted runs are never treated as evidence and their checkpoints are not
  mixed with later nonces.
- A browser or Kaggle failure does not authorize a local-data workaround.

## Completion criteria

Version 5 is ready for a submission decision only when:

1. mandatory safety regressions pass in the private Kaggle browser notebook;
2. Version 3 reproduction passes on the pinned cache;
3. one candidate passes the confirmed-champion gate, or Version 3 is explicitly
   retained after all bounded candidates fail;
4. a smoke run and a fresh full saved Kaggle run complete every cell;
5. the final submission artifact passes the hidden-test contract;
6. the accelerator is off;
7. code-only artifacts and aggregate evidence are synchronized to GitHub;
8. an explicit final submission decision is requested from the user.
