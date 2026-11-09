# Minimal scoring workflow

The reproducible published-feature path is `20_feature_artifacts.py` (spec or
validate), then `21_evaluate_features.py` (xgb, pa or hpa). Input manifests and
42 specifications pin external payloads, ordered features, identities and
preprocessing; released cleaned features are never normalized again.

Install with `pixi install --locked -e default`. Use an allocated environment
and call the evaluator with `--features`, `--spec`, `--batch`, `--channels` and
a fresh `--output`; `--help` describes the task-specific arguments. Each scored
artifact includes controls, exclusions/support and input/source/runtime hashes.

The named T4 protocol uses experimental T1/T2/T3 fit and T4 test, explicit
complete/available NC/PC control training, supported PA T4 queries with full
galleries, and pooled T4 WT HPA medians. The five available-control exceptions
are declared in the settings. This is distinct from the general native LOPO
CLI; its model/default fold configuration remains available for compatibility.

CPU is the default. GPU execution requires one full allocated UUID and rejects
actual backend fallback. Native XGB exposes classifier workers and threads;
the explicit T4 evaluator fits one classifier at a time with one XGBoost thread
and seed 0. Controls are computed alongside experimental scores. There is no
cross-run calibration reuse API or custom worker-pool monkeypatching.

Third-party internal pools are not a whole-process resource allocation.
The caller's job executor must enforce CPU/RAM/task/time and GPU limits; no
thread count is advertised as a cap on all libraries. CPU/GPU equivalence and
full-input reproduction are separate from the focused synthetic tests.

The PA compatibility command and explicit evaluator use the same importable
implementation. Optional read-only callbacks collect diagnostics without
replacing module functions. The existing native HPA command remains a general
compatibility analysis; the explicit HPA protocol has its own declared T4/support
contract. Reporting and the short reproduction launcher are the next change.
