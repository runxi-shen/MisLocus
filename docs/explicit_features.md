# Explicit feature evaluation

Use `scripts/20_feature_artifacts.py spec` to declare ordered feature columns,
channels and preprocessing; use `validate` to check existing Parquet inputs.
The validator preserves input row order and feature values and verifies physical
cell identities. Published cleaned features must not be normalized again.

`configs/analysis/inputs.json` pins 42 public payloads and their specifications
at dataset revision `74f63113a76b4a832285da308f3df2932266e456`. These are file
identities, not bundled data or proof that the files were freshly downloaded.
`settings.json` records the corrected T4 protocol, including control exceptions,
qualified runtime and the explicitly chosen downstream ALL definition. Later
reporting/runner changes consume those settings; individual scorers do not
silently select a reporting rule from that file.

```bash
pixi install --locked -e default
pixi run -e default python scripts/21_evaluate_features.py pa \
  --features /path/to/features.parquet \
  --spec configs/analysis/specs/cellprofiler/2024_02_06_Batch_8.json \
  --batch 2024_02_06_Batch_8 --channels GFP --workers 2 \
  --output results/pa/cellprofiler-b8-gfp
```

The evaluator writes scores, controls, exclusions, support diagnostics and
input/source/runtime/output hashes. Each output directory must be new.
XGB explicitly fits experimental T1/T2/T3 and tests T4; its default NC/PC
control policy requires all three training plates. `--control-plate-policy
available` permits incomplete non-T4 control training only when deliberately
requested. The pinned protocol names the five released Batch 8 exceptions.
The explicit protocol uses 150 trees, learning rate 0.05, histogram fitting,
seed 0, one XGBoost thread and the existing WT-positive label orientation.
The native classifier CLI retains its existing fold/default configuration.

GPU scoring requires `--backend gpu --gpu-uuid FULL_ALLOCATED_UUID` and a
compatible GPU environment. It never silently substitutes CPU fitting.
CPU results are independent diagnostics, not bit-exact CUDA reproduction.

PA uses site medians, the full gallery, supported T4 queries, seed 42,
10,000 experimental and 1,000 control null draws, isolated caches, normalized
control p95 and experimental BH q < 0.05. HPA accepts an explicitly prepared
multi-batch artifact of T4 WT cells, takes direct gene medians, and retains
signed tested scores separately from the display filter/floor.

These commands evaluate one declared artifact. They do not train an encoder,
perform crop extraction or run the full released matrix. The HPA pooling and
whole-analysis launcher are separate additions. Synthetic checks establish
behavior on fixtures, not full-input numerical reproduction of this checkout.
