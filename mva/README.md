# Unified proton-tagged dijet MVA

The implementation is split into four entry points per channel and shared
physics under `mva/common/`:

```text
mva/Hcc/prepare_dataset.py -> train_model.py -> plot_results.py
                                             -> scan_normalizations.py
mva/Hbb/prepare_dataset.py -> train_model.py -> plot_results.py
                                             -> scan_normalizations.py
```

The nominal dataset has one row per selected central event.  SuperChic rows
carry their reconstructed proton pair.  MadGraph rows carry a central
cross-section weight and an event-dependent analytic pair intensity; rapidity
cells are generated only for capped fitting and streamed evaluation.  No full
event-by-cell probability array or per-fold inference model is saved.

## H(cc) profiles

- `parity_five_class`: v01 QCDcc, legacy sampled proton pool, five old labels,
  the legacy Hbb-ranked features, leading jets, and corrections off.  This is a
  regression control only.
- `nominal_full`: seven components and five labels, with exclusive QCD and QED
  separated and resonant Hbb retained.
- `nominal_no_hbb_resonant`: the nominal bottom continuum and mistag
  backgrounds but no resonant Hbb component; exclusive QCD and QED remain
  separate classes.
- `charm_only_three_class`: the merged-exclusive charm study baseline.
- `charm_only_four_class`: the same charm components with exclusive QCD and
  QED as separate classes.

The default Hcc `locked` set is the charm-ranked top 20 from the dedicated
feature study.  `legacy_locked` preserves the previous Hcc list for the parity
profile, and the Hbb channel keeps its independent locked feature set.

QCDcc v02--v07 are summed with independent campaign weights.  They are never
sent through the overlap stitcher.  The QCDbb v01--v03 path retains the generic
phase-space stitcher.

## Typical commands

Run commands from the repository root after setting up the analysis
environment:

```bash
source setup_env.sh
python3 mva/Hcc/prepare_dataset.py --profile charm_only_four_class
python3 mva/Hcc/train_model.py \
  --data-dir mva/Hcc/data/charm_only_four_class \
  --result-dir mva/Hcc/results/charm_only_four_class
python3 mva/Hcc/plot_results.py \
  --result-dir mva/Hcc/results/charm_only_four_class
python3 mva/Hcc/scan_normalizations.py \
  --result-dir mva/Hcc/results/charm_only_four_class
```

Use `--jets truth` when preparing the truth-pair variation.  The resulting
metadata forces `truth_matched` in both training and evaluation; the trainer
rejects an attempt to turn that requirement off.  Use `--corrections off` only
for a controlled variation.  Unsupported corrected pairs are excluded without
changing the generated denominator.  All nominal profiles define candidate
jets with `|eta| < 3.0` before applying the corrected-pT cut or choosing the
leading pair.  The parity profile alone retains the unrestricted legacy jet
pool.

For quick monitoring checks, pass `--max-files 1` to the builder and a small
`--n-estimators` value to the trainer.  Builders report component, campaign,
file, elapsed time, rate/ETA, correction validity, and matching fractions.
Trainers report sequential feature reads, cell preparation, fold sizes,
boosting progress, streamed scoring, mass-template progress, and elapsed time.

## Outputs and normalization

`metadata.yaml` defines every class/component ID, array shape, generated
denominator, campaign, correction key, matching fraction, and correction
invalid fraction.  `report.yaml` adds component yields, the dijet-mass peak,
central-event effective statistics, score operating points, exact vertex
likelihood parameters, and timing-resolution scans.  `report_data.npz` is a
compact set of histograms and threshold accumulators, not an event-level OOF
artifact.

For real SuperChic pairs, `proton_px_left.npy`, `proton_py_left.npy`,
`proton_px_right.npy`, and `proton_py_right.npy` store the unsmeared truth
event-record components in GeV.  Analytic MadGraph rows contain NaN because no
single proton pair is materialized.  These archival arrays are not MVA
features.

The analytic min-bias input is
`output/minbias/minbias_inelastic_100m_v1/protons.parquet`.  Pair intensities
are normalized to its `n_inelastic_generated = 100000000` denominator and
scaled by `mu**2`.  The timing likelihood is integrated at use time, so PPS and
central timing resolutions can be scanned without rebuilding or retraining.

## Jet-correction contract

The charm correction maps cover the nominal candidate acceptance.  A jet
outside `|eta| < 3.0` is not an MVA candidate and therefore cannot veto a valid
fiducial pair.  A fiducial candidate outside the map's raw-pT support still
makes the pair invalid: the dataset reports that loss and never substitutes raw
kinematics.
