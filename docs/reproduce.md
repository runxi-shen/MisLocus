# Reproduce the declared feature analysis

The supported route is: pinned downloaded inputs → explicit scoring → saved
score reports → tables and figures. Numerical functions live in `src/`;
the scripts are entrypoints. The launcher only calls them sequentially.

## Inputs and environment

Install the locked package/environment with `pixi install --locked -e default`.
Download the payloads listed in `configs/analysis/inputs.json` at its immutable
dataset revision. Keep their relative dataset paths below a chosen data root;
older downloader remapping into `data/interim` uses different paths. Inputs
total about 101.5 GB. Released cleaned features must not be normalized again.
The launcher requires the files locally and performs no automatic downloads.
Admission checks payload/specification hashes and separately checks the bundled
curated HPA annotation; it does not label existing files fresh downloads.

`configs/analysis/settings.json` is the named corrected T4 protocol: nine
settings, six batches, five available-control exceptions, declared seeds and
the `all_retained_batches` ALL definition. That definition requires every
retained batch across complete biological pairs to hit. Native reporting's
`any_complete_pair` default is retained for compatibility. These definitions
can give different results and are recorded explicitly, not chosen for agreement.

## Inspect and launch

```
pixi run -e default python scripts/23_run_analysis.py plan \
  --data-root data/public-features --output results/reproduction

pixi run -e default python scripts/23_run_analysis.py run \
  --data-root data/public-features --output results/reproduction \
  --executor-config /path/to/executor.json \
  --cpus 8 --workers 2 --memory-gib 64 --seconds 1800
```

Planning creates no outputs and runs no numerical work. The released plan
contains 54 XGB, 54 PA and seven HPA evaluations. `24_pool_hpa.py` first selects
T4 WT cells from every declared batch and the sorted common channel features;
the evaluator then calculates direct per-gene medians.

The executor configuration contains structured command prefixes, as shown in
`configs/analysis/executor.example.json`; replace the placeholder paths with
your existing job supervisor. Each prefix receives a unique `{job}` plus
`{cpus}`, `{memory_gib}`, `{seconds}` and `{gpu_uuid}` placeholders. Commands
are passed as argument lists, never shell text. The CPU example matches this
project's existing bounded supervisor, which acquires both shared allocation
locks. Do not wrap the whole launcher in another process holding the same locks.

The executor must enforce the allocation: at most 8 CPUs, 64 GiB actual RAM,
512 tasks and 1800 seconds per job. Internal third-party pools can reserve
substantial virtual memory even for small inputs; the 8-GiB address cap was
insufficient for unmodified DuckDB in validation. Configure an appropriate
existing allocation; do not change the numerical method to fit a smaller cap.

The default CPU analysis is not a bit-exact CUDA reproduction. GPU execution
requires `--backend gpu --gpu-uuid FULL_UUID`, the compatible GPU environment,
and a `gpu` executor prefix enforcing one allocated GPU, 8192 MiB GPU memory,
the CPU/RAM/task/time ceilings, and physical-RSS monitoring without the CUDA-
breaking address cap. The example deliberately does not invent a GPU supervisor.
The caller must supply the existing qualified executor. Runtime/build/hardware
conditions remain part of any exact historical reproduction claim.

## Outputs and failure behavior

Use a fresh output directory. There is no resume engine or cross-run
calibration reuse. Scores and controls are computed together. The launcher
stops on the first failed command and preserves its log and partial outputs.
`steps.jsonl` records commands and exit status; `completion.json` is written
only after all requested commands succeed. Supervisors retain resource receipts.

Reports consume only completed evaluator directories, verify saved scores,
controls and output hashes, and retain cohort denominators and annotation
exclusions. `22_report_clinical.py` can also consume explicitly declared raw
score tables, for which producer calibration remains the caller's responsibility.
Coarse and strict label eligibility are independent, consistent with #38.
Missing hypothesis tests keep adjusted p-values unavailable.

`25_export_tables.py` preserves full-precision CSVs and produces generic LaTeX
tables. `28_plot_results.py` saves the exact plotted data for distributions,
hit rates, F1 and HPA retrieval. These are not manuscript-specific violin assets.

The replacement is validated with focused numerical and small full-command
fixtures. No full released 115-job run, fresh install or actual GPU fit of this
new assembly has been performed. No scientific qualification is inferred from
fixture completion. Encoder training and supplementary campaigns remain separate.
