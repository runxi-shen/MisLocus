# Sequential released-feature analysis

`scripts/23_run_analysis.py` connects pinned inputs, the explicit evaluators,
clinical reports, tables and figures. It fits downstream XGBoost classifiers;
it does not fit an encoder or extract raw-image embeddings.

Install the locked environment with `pixi install --locked -e default`.
The default input matrix contains 42 feature payloads (about 101.5 GB), a
public clinical annotation, and the separately pinned bundled HPA annotation.
Use a data directory following the relative Hugging Face paths in
`configs/analysis/inputs.json`; older downloader alias layouts may differ.

## Review the plan

```bash
pixi run -e default python scripts/23_run_analysis.py plan \
  --data-root data/public-features --output results/corrected-t4
```

Planning reads code/configuration and checks specification hashes. It does
not download data, load numerical feature matrices, start jobs or create the
output directory. The released plan includes 54 XGB, 54 PA and seven HPA
scoring jobs, seven HPA pooling jobs, input admissions, reports and exports.
The five Batch 8 available-control exceptions and reporting ALL policy come
from the explicit settings; they are never chosen for better agreement.

## Run within an allocation

```bash
pixi run -e default python scripts/23_run_analysis.py run \
  --data-root data/public-features --output results/corrected-t4 \
  --cpus 8 --workers 2 --memory-gib 64 --seconds 1800
```

This uses CPU XGBoost and is an independent CPU analysis, not a claim of
bit-exact agreement with the qualified CUDA replay. Defaults are 2 CPUs,
2 scoring workers, 8 GiB and 1800 seconds per stage; larger released inputs
may need the existing maximum 8 CPU/64 GiB allocation. The runner monitors
aggregate worker-tree RSS, at most 512 tasks, and each stage's wall time. It
does not reduce scientific parameters to fit the budget. Failed attempts,
logs and resource measurements remain under `jobs/ID/attempt-NNN/`.

Add `--download` to opt into downloading missing files from the pinned dataset
revision. Existing files with incorrect hashes/sizes are rejected and preserved.
Each admission records whether a download actually occurred. File hashes identify
the declared payloads; existing local files are not labelled fresh downloads.

GPU scoring additionally requires the GPU environment and one allocated full
UUID, using `--backend gpu --gpu-uuid GPU-...`. The inherited allocation must
agree. Only XGB stages see that GPU; other stages disable CUDA. The runner
monitors the worker tree's GPU use with an 8192 MiB limit. It retains physical
RSS monitoring without imposing the address-space cap that breaks CUDA.
`--require-qualified-runtime` additionally checks the recorded Python/package
versions, CUDA XGBoost library hash and H100/driver condition. Without those
checks, completion is not labelled historical-runtime qualification.

The runner serializes jobs and holds its output lock. In a shared allocation,
also pass every required shared lock with repeated `--lock-file PATH`; for
this project's audit allocation these are both `campaign.lock` and
`qualification.lock`. A busy lock stops before starting a job. Do not nest
another supervisor that already holds those same locks. External scheduler
allocations still govern the controller and other unrelated processes.

## Resume and outputs

Repeat the exact run options with `--resume`. Code, manifest, settings,
runtime, allocation and data/output locations must match. Every completed
stage is verified in a bounded subprocess, including all output hashes and
admitted source-file hashes. Changed bytes stop the run; failed stages get
new attempts while completed scores are reused only after verification.
`--stop-after N` deliberately records partial status, never complete success.

HPA pooling selects T4 `disease_wt` cells from every declared batch, preserves
identities and values, and uses the sorted common feature intersection for
each selected channel. Direct per-gene medians are calculated by the evaluator.
No released feature normalization is repeated.

Each stage has a completion receipt; the root `status.json` distinguishes
new scoring work from verified reuse. Reports consume only completed evaluator
outputs. Generic figures and full-precision tables are generated from those
scores; manuscript assets and frozen historical values are not inputs.

The runner is validated with synthetic integration/resume/resource checks.
The newly assembled runner has not completed the 115-job released matrix,
a fresh installation, real GPU validation or biological qualification.
Supplementary SHAP, top-K and seed-sensitivity campaigns remain separate.
