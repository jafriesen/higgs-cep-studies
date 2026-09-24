# Five-class H(cc) proton MVA

This directory is an independent H(cc) study patterned after `analysis/MVA_new`.
It uses the same locked 20 features but rebuilds the data from the configured
H(cc), QCDcc, QEDcc, QCDbb, and H(bb) samples.

Run the workflow from the repository root:

```bash
source setup_env.sh
python3 analysis/MVA_hcc/prepare_dataset.py
python3 analysis/MVA_hcc/train_model.py
python3 analysis/MVA_hcc/plot_results.py
python3 analysis/MVA_hcc/scan_normalizations.py
```

Each script accepts `--help`. In particular, `--data-dir` and `--result-dir`
can redirect generated artifacts, and `prepare_dataset.py --max-files N` plus
`train_model.py --n-estimators N --support-floor N` support small smoke runs.

`prepare_dataset.py` also accepts `--feature-set {locked,full}` and
`--components NAME ...`. The defaults reproduce the locked dataset byte for
byte; `full` keeps all 55 columns of `FULL_FEATURES`, and a restricted component
list is renumbered contiguously while still selecting exactly the events a full
build would. Both exist for `qed_study/` and are not used by the nominal
workflow.

## Training labels and physical components

The classifier has five outputs:

- `Hcc`: SuperChic H(cc), the signal.
- `exclusive_QCD`: SuperChic QCDcc and QCDbb.
- `exclusive_QED`: SuperChic QEDcc.
- `nonexclusive_QCD`: MadGraph QCDcc and QCDbb combined with random min-bias
  proton pairs.
- `Hbb_resonant`: SuperChic H(bb).

The dataset retains seven component IDs, so the two QCD flavors remain separate
in physical yields, mass templates, plots, and normalization scans even when
they share a classifier label. The mapping is recorded in `data/metadata.yaml`
and copied into `results/report.yaml`; downstream scripts read it rather than
hardcoding numeric IDs.

All true-charm components use `eff_c**2`. All true-bottom components, including
H(bb), use `mistag_b_to_c**2`, because this is a single c-tagged channel. These
values come from the repository `parameters.yaml` at dataset-build time.

## Outputs

`prepare_dataset.py` writes memory-mappable NumPy arrays and a local copy of the
weighted proton-pair pool under `data/`. `train_model.py` writes the two
calibrated cross-fit models, out-of-fold scores, compact component templates,
the threshold scan, and numerical reports under `results/`. The saved fold
models reproduce the out-of-fold study; they are not a production inference
ensemble. Both output directories are ignored by Git.

For the direct Delphes-Pythia8 charm samples, the Delphes `Particle` collection
does not retain the outgoing status-1 protons. Their truth xi values are read
from the matching SuperChic LHE event record. The older bottom samples use the
matching HepMC record, as in the existing H(bb) workflow.

`plot_results.py` creates calibrated-probability, component-score, threshold,
and mass-spectrum plots. `scan_normalizations.py` holds the nominal score fixed
while scanning the common QCD-mediated SuperChic survival factor, charm-tag
efficiency, and b-to-c mistag probability. The default tag ranges are
sensitivity tests of 0.5--1.5 times nominal, not assigned uncertainties. The
survival output includes both the re-optimized single-cut significance and the
combined significance from the six locked score categories.

## Coverage limitations

MadGraph QCDbb uses v02 and v03. QCDbb v01 remains excluded to match the locked
H(bb) report, leaving its wider low-pT/forward phase space unmodelled. MadGraph
QCDcc currently has only v01, with the configured parton-level coverage
`30 < pT < 100 GeV`, `|eta| < 1.5`, and `70 < m(cc) < 140 GeV`. The workflow
does not extrapolate beyond these generated regions.

## Charm-only comparison

The four-class comparison containing only Hcc, SuperChic QCDcc, SuperChic
QEDcc, and MadGraph QCDcc is derived from the nominal prepared dataset so that
the retained events and weights are identical:

```bash
python3 analysis/MVA_hcc/prepare_cc_only_dataset.py
python3 analysis/MVA_hcc/train_model.py \
  --data-dir analysis/MVA_hcc/comparisons/cc_only/data \
  --result-dir analysis/MVA_hcc/comparisons/cc_only/results
python3 analysis/MVA_hcc/plot_results.py \
  --result-dir analysis/MVA_hcc/comparisons/cc_only/results
python3 analysis/MVA_hcc/scan_normalizations.py \
  --result-dir analysis/MVA_hcc/comparisons/cc_only/results
```

Comparison artifacts are kept under `comparisons/cc_only/` and ignored by Git;
the nominal `data/` and `results/` directories are not modified.

## QED-background study

`qed_study/` is a self-contained study of why the QEDcc background survives this
MVA and what would reduce it. See `qed_study/README.md` for how to run it and
`qed_study/FINDINGS.md` for the results. It writes nothing into `data/`,
`results/` or `comparisons/`.
