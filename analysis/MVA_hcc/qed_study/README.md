# H(cc) QED-background study

Why the QEDcc background survives the H(cc) MVA, how much of it can be removed,
and what would have to change to remove more. Self-contained: nothing here
modifies `analysis/MVA_hcc/data/` or `analysis/MVA_hcc/results/`.

Run everything from the repository root after `source setup_env.sh`.

## Datasets

`analysis/MVA_hcc/prepare_dataset.py` gained two default-preserving arguments for
this study; with no arguments it reproduces the locked dataset byte for byte.

```bash
# 55 features, exclusive components only (~35 s)
python3 analysis/MVA_hcc/prepare_dataset.py --feature-set full \
  --components Hcc QCDcc_superchic QEDcc_superchic \
  --data-dir analysis/MVA_hcc/qed_study/data/exclusive_wide

# the 20 locked features, charm components only, derived from the nominal dataset
python3 analysis/MVA_hcc/prepare_cc_only_dataset.py \
  --output-dir analysis/MVA_hcc/qed_study/data/cc_locked

# 55 features including MadGraph QCDcc (~10 min)
python3 analysis/MVA_hcc/prepare_dataset.py --feature-set full \
  --components Hcc QCDcc_superchic QEDcc_superchic QCDcc_madgraph \
  --data-dir analysis/MVA_hcc/qed_study/data/cc_wide
```

`--components` selects a strict subset: seeds are keyed to each component's
position in `COMPONENT_SPECS`, so a restricted build selects exactly the events a
full build would.

## Stages

| script | question | needs |
|---|---|---|
| `stage1_qed_variables.py` | which of the 55 observables separate H(cc) from the exclusive continuum, and what is the ceiling with QEDcc as the only background | `exclusive_wide` |
| `stage2_proton_pt.py` | how much separation the outgoing-proton pT carries, and what t-resolution would be needed to use it | SuperChic LHE records |
| `stage3_architecture.py` | class structure, score definition and categorisation | `cc_locked` |
| `stage4_features.py` | charm-specific feature ranking and feature-count scan against MadGraph QCDcc | `cc_wide` |
| `stage4b_locked_vs_charm.py` | the locked H(bb) feature set against the charm-ranked one, matched settings | `cc_wide`, stage 4 |
| `stage5_mc_support.py` | how much of the result is MonteCarlo statistics, and how much generation would fix it | `cc_wide` |
| `stage6_recoil_proxy.py` | whether the dijet recoil substitutes for a proton pT measurement | `exclusive_wide`, stage 2 |
| `plot_study.py` | figures for every stage | stage outputs |

Each writes a `*_report.yaml` plus CSV/NPZ under `output/<stage>/`.

## Shared code

- `common.py` — weighted separation (folded AUC with mid-rank ties), separation
  conditioned on another variable, mass-binned significance, a held-out
  threshold significance, group-safe folds, the validated palette.
- `harness.py` — the fast evaluation path. Fits two group-safe folds, then caches
  the calibrated class probabilities for every (MadGraph central event, proton
  pool grid cell, class). Any score definition, kappa choice or categorisation is
  then a numpy pass rather than a fresh XGBoost sweep. The training class
  assignment is a parameter (`class_map`), so alternative class structures are
  compared without rebuilding the dataset.

Three things make it fast relative to `analysis/MVA_hcc/train_model.py`: the
feature matrix is read once sequentially into RAM instead of being fancy-indexed
off a memmap on every pass; the threshold scan buckets scores once and takes a
reverse cumulative sum instead of rescanning per cut; and `--stop-cap` optionally
caps the pooled early-stopping rows, which are otherwise ~90% of a set that is
re-evaluated every boosting round. `--stop-cap 0` keeps every row, matching
`train_model.py` exactly.

## Baseline

`output/baseline_cc_only/` holds an unmodified `analysis/MVA_hcc/train_model.py`
run on `cc_locked`, as the independent reference the harness is checked against.

## Caveats

Classifier-only and stat-only, with perfectly known nominal backgrounds. The bb
components are excluded from every stage (1.5% of the background at the nominal
operating point). `combinatorial_acceptance_factor` stays at its nominal 0.005.
Stage 2 is truth-level: the PPS configuration models xi acceptance and xi
resolution only, with no |t| acceptance and no |t| resolution, and the min-bias
proton-pair pool stores no proton pT, so MadGraph cannot enter that stage.
