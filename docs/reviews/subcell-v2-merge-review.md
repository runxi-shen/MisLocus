# SubCell allele-v2 merge review

**Base:** `main` = `87c0ec71f676a775e9fad743f3d018d0bf2036b9`, confirmed by the user.
**Code endpoint:** `e3060373ac8bf512bc53a7c910ff8101bc834a11`; subsequent changes in this
review reconcile documentation/evidence, not model mathematics or produced artifacts.
Review command: `git diff 87c0ec7...HEAD`; commit list:

```text
9569b57  Pin validated dependencies
f45f344  Evaluation-only zero-mask MAE determinism
b9befa6  Shared verified native crops
7af25c7  Canonical allele training/native lifecycle
2e054e5  Frozen versus selected extraction
262d13f  Resource-gated paired campaign
03b1961  Forward initialized PATH to supervised workers
0a2ceb9  Selection/terminal-resume/campaign safeguards
35416e2  Read-only crop payload verification
e306037  External export provenance and bounded pilots
```

**Method:** separate direct Standards and Spec passes, followed by artifact verification.
No independent parallel reviewer was available; these are not independent endorsements.
No repository `AGENTS.md`, `CONTRIBUTING` or `CODING_STANDARDS` was found. Standards
sources were the repository's documented CLI/data contracts and the review skill's
change-size/Fowler-smell guidance; formatting enforced by tooling was excluded.
The original approved planning contract, especially §§4–8/10, is archived unchanged
at SHA256 `1ae7ad94fb06b3db2ad0b46aed51bcebf5335561b17f3343beba429a157c67fe`;
[the reconciled protocol](../subcell_allele_v2.md) exposes the numerical contract and
execution limits without publishing the internal planning history.

## Standards

**S1 — documentation drift, corrected.** The held guide said “Only dependency pins
are committed; no production campaign has run.” It also showed T4-only/in-checkout
exports and linked historical diagnostic reports as current readiness. The reconciled
README/guide/workflow now distinguish the actual training commit, export commit,
finished artifacts, archive-only seeds and missing matched controls. The dataset
reference no longer recommends preprocessing published features twice, claims all
payloads are gitignored, promises downstream determinism from parameters alone, or
presents `--hf-repo repo@sha` as a supported revision argument.

**S2 — review-size advisory, outstanding.** Before documentation, the diff changes
5,620 lines across40 files, well beyond the skill's800-line guidance. This is a
coherent replacement pipeline, not a small patch. Existing focused commits provide
review stages; they do not eliminate the need for independent human review. The
smallest coherent stage to land first is dependency pins + the evaluation-only MAE
patch, followed by shared staging, training, extraction, and supervision/hardening.
A single PR is retained as requested; it can be split in that dependency order if
reviewers prefer. No speculative framework/type refactor is proposed solely to
satisfy a code-smell heuristic.

**Compatibility checked:** intentional legacy extractor/CLI/YAML/session-resume
breaks are documented, including removal of08a extraction, frozen mode moving from08d
to08b, and rejection of historical gene checkpoints/automatic resume. The MorphEm
`08b_extract_vit_embeddings.py` has no diff from the review base. Downstream-owned
NaN cleanup/data-root bootstrap changes and generated data are excluded from this PR.
No additional hard coding-standard violation was established by this direct pass.

## Spec

**SP1 — matched-control requirement remains incomplete.** §7 asks: “Use frozen
four-channel MAE and ViT counterparts with IDENTICAL input mapping, geometry,
normalization, extraction precision, cell eligibility and downstream preprocessing.”
The implementations, strict loading and bounded parity checks exist; this campaign
only produced adapted seed42 bulk exports. Complete or validate matched frozen
exports before adaptation-only comparative claims. Historical RBG features do not
satisfy this requirement. This is a comparison gate, not evidence that the completed
adapted fits/exports are invalid.

**SP2 — rank-RNG caveat, not a demonstrated numerical-contract violation.** §4 requires
“Separate and save sampling seeds/state from model/augmentation randomness.” Sampling
has its own deterministic generators and RNG is checkpointed, but `08c` seeds each
rank identically. All six final checkpoints have matching NumPy/CPU-Torch/CUDA states
across ranks within a run; those states differ across42/43/44 within each family.
Every audited global training batch still contains128 distinct cells and16 alleles.
Views are drawn separately, not copied. Reduced cross-rank stochastic independence
has unmeasured performance impact. Do not silently alter this experiment; a rank-seed
change requires a versioned scientific decision and fresh validation/retraining.

**SP3 — reporting/evidence limits.** The contract's fresh42/43/44 training requirement
was met; the user subsequently narrowed export/reporting to42 before T4 extraction.
That is not a failed seed experiment, but it supplies no downstream multi-seed
robustness, mean/std or ensemble evidence. §6 explicitly distinguishes FP32 tensors
from bitwise reproducibility. Historical determinism flags were not all recorded;
future logging cannot reconstruct them. Exact repeated pilots are bounded checks,
not proof of deterministic training on every backend. T3 macro allele AP must not
be presented as mislocalization performance or unseen-allele generalization.

The reviewed implementation and completed artifacts otherwise support the approved
allele vocabulary/splits, exact crop alignment,128→955→448 geometry, objectives,
optimizer/decay/schedule, fixed T3/native FP64 selection, fresh starts, exact export
coverage and no historical-artifact relabeling. No demonstrated execution defect
requiring discard or automatic retraining was found in this review.

## Artifact acceptance and pending work

[The evidence package](../evidence/README.md) provides relative-path checksum lists,
not private host paths or raw payloads. The final packaging pass rechecked native
selection/source/terminal guards on six checkpoints, RNG states for all three seeds,
and hashes of all12 exported Parquets. A fresh read-only training audit rechecked
all600 sampling-pass and60 validation receipts and LR traces; original full export
readback records cover schema, finite FP32 values and exact ordered cell coverage.
Original snapshots are preserved;
new documentation does not pretend those jobs ran at the PR's later head.

- **Complete:** six approved fits; two selected seed42 all-split exports; crop rehash;
  source/data/model/selection/output binding; supervised completion/readback;26 clean
  producer tests and bounded exact-repeat GPU pilots.
- **Before adaptation comparisons:** matched four-channel frozen bulk exports.
- **Before final downstream analyses:** independently review the downstream lane's
  split/context pools, normalization/zero-MAD, feature selection, control calibration,
  hit definitions and full stage lineage. No downstream result is certified here.
- **Before stronger robustness claims:** additional seeded exports/evaluations and,
  if desired, an approved independent-rank RNG experiment—not retrospective relabeling.
- **Before publication:** synchronize the manuscript in its own repository and decide
  how to distribute the new artifacts; this PR publishes an inventory, not downloads.

**Summary:** Standards:2 findings (documentation corrected; size advisory remains).
Spec:3 findings (matched-control gate plus RNG/reporting limitations); the outstanding
comparison prerequisite is the main Spec gap. No automatic merge or independent
scientific certification is implied by this report.
