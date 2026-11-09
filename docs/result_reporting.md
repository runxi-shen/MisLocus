# Reports, tables and figures from saved scores

`22_report_clinical.py` takes explicit score paths, representation/channel
settings and a public allele annotation file. It checks coherent calibrated
hit flags, reports shared support and complete biological-pair cohorts, and
retains missing or conflicting annotations as explicit exclusions.

Supply one `--input MODEL BATCH SCORES` for every completed score table,
one `--setting MODEL CHANNEL` per comparison and `--annotations FILE`.
Use `--task xgb` or `--task pa`; select predictors with `--predictors`.
The published primary comparison declares nine settings. Generic smaller
families are reported as such, never labelled as the full primary analysis.

```
python scripts/22_report_clinical.py --help
python scripts/25_export_tables.py --report results/report --output results/tables
python scripts/28_plot_results.py --clinical-report results/report --output results/figures
python scripts/28_plot_results.py --evaluation results/evaluation --output results/score-figures
```

The existing ALL definition remains: both batches of at least one complete
biological pair must hit. An alternative across every retained batch is a
separate methodological change. Different cohorts retain separate denominators.
Binary Fisher enrichment and continuous Mann-Whitney comparisons have separate
BH families by predictor/cohort or predictor/label tier across declared settings.
An incomplete family has unavailable adjusted p-values, not a smaller family.

Coarse and strict annotation consensus and eligibility are independent. A
strict label can contribute even when the coarse annotation is unavailable,
consistent with the corrected native helper in #38. This differs from the
older portable draft's coarse-label prerequisite for strict comparisons;
it can change strict subgroup distributions and continuous statistics.
The new assembled reporting path has not had a full-input comparison.

Exports preserve full-precision CSVs and produce generic LaTeX displays.
Plots save the exact plotted data and verify upstream output hashes.
They cover observed score distributions, hit rates, F1 and HPA retrieval;
they do not recreate manuscript-specific violin artwork or supply a biological
validation. Raw score-file input alone does not establish producer calibration
adequacy; the workflow runner separately verifies completed evaluator receipts.
