# Standalone 20-feature proton MVA report

This directory contains a self-contained copy of the locked 20-feature MVA
workflow. The original scripts and caches in `analysis/MVA` are not modified.

The workflow has three steps:

```bash
source setup_env.sh
python3 analysis/MVA_new/prepare_dataset.py
python3 analysis/MVA_new/train_feature_count_20.py
python3 analysis/MVA_new/plot_feature_count_20.py
python3 analysis/MVA_new/scan_survival_factors.py
```

All paths default to locations beside the scripts, so the commands can also be
run from another working directory. `prepare_dataset.py` creates a physical,
reduced copy under `data/`; it is only needed when rebuilding that copy from
the original repository cache. `train_feature_count_20.py` reads only the
local dataset and writes numerical results and archived cross-fit models under
`results/`. `plot_feature_count_20.py` reads only `results/report.yaml` and
`results/report_data.npz`. `scan_survival_factors.py` uses those same compact
results to optimize the locked-score threshold while scaling the H(bb) and
SuperChic QCD normalizations together over a configurable survival-factor
range. It records both the optimized single-cut significance and the combined
significance from the six locked score categories; it does not retrain or
rescore the classifier.

To use a Python environment instead of the repository setup, install the
versions recorded from the environment used for the reference report:

```bash
python3 -m pip install -r analysis/MVA_new/requirements.txt
```

Each script accepts `--help`. In particular, `--data-dir` and `--result-dir`
can redirect the local data and results. The saved models are the two
out-of-fold evaluation models; they are archived for reproducibility and are
not a production inference ensemble.
