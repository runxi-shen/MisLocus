# Explicit ALL aggregation policy

The default remains `any_complete_pair`: an allele passes ALL if both batches
of at least one complete biological pair hit. `all_retained_batches` requires
hits in every retained batch across all complete biological pairs. Unpaired
batches are excluded from both definitions, and ANY is unchanged.

For example, A/B pass and C/D fail: the default ALL passes; the alternative
fails. This is a reporting-definition choice and can change hit rates, F1,
enrichment and figures. It is not an execution optimization.

Both `10c_report_hits.py` and `22_report_clinical.py` accept `--all-rule` and
record it in their receipts. The corrected released-analysis settings explicitly
choose `all_retained_batches`; the generic CLI default remains unchanged.
Review the chosen protocol before comparing results across analysis versions.
