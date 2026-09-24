# Unified H(bb)/H(cc) MVA implementation plan

This plan records the agreed implementation before code is moved.  The legacy
directories under `analysis/MVA/` and `analysis/MVA_hcc/` remain read-only
references throughout.

## Decisions already made

- H(cc) is the first production channel.  H(bb) is added after the common and
  H(cc) paths are stable.
- Nominal MadGraph QCDcc uses `QCDcc__v02` through `QCDcc__v07`.  These
  campaigns are disjoint and are summed with their individual
  `cross_section / n_generated` weights.  They are not stitched.
- Overlap-aware stitching remains available in `mva/common/` for samples that
  need it, notably the H(bb) path.
- `QCDcc__v01` and the legacy proton-pair pool are permitted only in an
  explicitly selected parity profile.  Parity is a bounded regression control,
  not a reason to delay the nominal result.
- A MadGraph central event is stored once.  Proton-cell rows are constructed
  lazily, with analytic weights, only for capped training and streamed
  evaluation.  `group_id` always denotes the central event.
- The truth-matched variation requires `truth_matched` in both training and
  reported evaluation.  Unmatched events remain in the prepared dataset using
  the leading pair as a fallback, and never reduce `n_generated`.
- When corrections are enabled, an event whose selected jet pair is outside a
  correction map is excluded from the corrected selected numerator.  Raw
  kinematics are never substituted.  The generated denominator remains
  unchanged and the invalid fraction is reported per component.
- Nominal MVA jet candidates satisfy `|eta| < 3.0` before the leading pair is
  chosen.  The bounded parity profile retains the unrestricted legacy pool.
- The timing likelihood uses the exact likelihood ratio of the matched and
  unrelated two-dimensional Gaussian models.  Its bin probabilities and
  resolution scan are evaluated at use time and are not dataset features.
- MadGraph charm uses one component-level correction map.  Its derivation and
  validation files are selected round-robin across v02--v07 rather than taking
  the first files from a combined glob.

## H(cc) configurations

All configurations preserve physical components independently of classifier
labels.  The component-to-class mapping is data in the channel configuration
and is copied to dataset/result metadata.

1. `parity_five_class`
   - Existing seven physical components and five legacy classes.
   - MadGraph QCDcc v01, legacy proton pool, legacy locked features, leading
     jets, corrections off.
   - Used only for bounded regression checks unless a full parity run is
     explicitly requested.
2. `nominal_full`
   - All seven physical components.
   - Five classes: Hcc, exclusive QCD, exclusive QED, non-exclusive QCD, and
     resonant Hbb.
   - MadGraph QCDcc v02--v07.
3. `nominal_no_hbb_resonant`
   - The nominal component set without `Hbb_superchic`.
   - Four classes: Hcc, exclusive QCD, exclusive QED, and non-exclusive QCD.
   - Bottom continuum and MadGraph bottom mistag components remain present.
4. `charm_only_three_class`
   - Hcc, SuperChic QCDcc, SuperChic QEDcc, and MadGraph QCDcc only.
   - Three classes: Hcc, exclusive continuum, and non-exclusive QCD.
   - Retained as the merged-exclusive comparison.
5. `charm_only_four_class`
   - The same four charm components.
   - Four classes: Hcc, exclusive QCD, exclusive QED, and non-exclusive QCD.

The dataset builder's `--components` and training profile selection permit
additional comparisons without adding one-off scripts.

## On-disk dataset contract

The nominal format is central-event based.  Every aligned array has one entry
per selected central event except the feature matrix's second dimension.

- `x.npy`: the selected central features; the proton delta-y column is not
  duplicated for combinatorial components.
- `class.npy`, `component.npy`, `group_id.npy`: configured identities.  IDs are
  contiguous and their mappings live in `metadata.yaml`.
- `dijet_rapidity.npy`, `dijet_mass.npy`: inputs needed for analytic proton-cell
  expansion and diagnostics.
- `physical_weight.npy`: complete event weight for real-proton components;
  central hard-process weight before pair intensity for combinatorial ones.
- `training_mixture_weight.npy`: the corresponding luminosity-free mixture
  weight.
- `proton_yx.npy`, `proton_mx.npy`: measured/smeared values for real-proton
  components and NaN for combinatorial components.
- `proton_px_left.npy`, `proton_py_left.npy`, `proton_px_right.npy`, and
  `proton_py_right.npy`: unsmeared truth event-record components in GeV for
  real-proton rows and NaN for analytic combinatorial rows.  They are stored
  for later studies and are not MVA inputs.
- `truth_matched.npy`, `truth_is_leading.npy`, `match_dr1.npy`, `match_dr2.npy`:
  truth-pair diagnostics, stored for every component.
- `correction_valid.npy`: whether both selected jets were supported by the
  requested correction map.
- Optional `parton.npy`: the existing five parton fields.

For a combinatorial event, a logical `(event, proton-cell)` training row is
formed by copying the central features, setting
`yx_minus_dijet_rapidity` to the cell centre, and multiplying the central
weight by the min-bias pair intensity in that cell.  All cells keep the same
`group_id`.  Evaluation processes these cells in chunks and immediately
accumulates compact score/category/mass sufficient statistics; it does not save
an OOF array or a full event-by-cell probability tensor.

The parity backend may construct the legacy sampled rows because exact legacy
row counts refer to that representation.  It is isolated from the nominal
backend.

## Physics and normalization invariants

- Delphes branches are converted to float64 before four-vector arithmetic.
- Corrections are applied before the jet-pT cut and scale both jet pT and mass.
- Nominal candidate jets satisfy `|eta| < 3.0`; the corrected-pT cut and pT
  ranking operate only within that fiducial pool.
- Hard-parton -> GenJet -> JetPUPPI matching uses the validated 0.4/0.2 delta-R
  chain, unique smallest-distance-first matching, and `|eta| < 3.0`.
- `n_generated` is always the sum of `tree.num_entries` over input files before
  selection, correction validity, truth matching, or subsampling.
- Any file/event subsampling is rescaled by `n_total / n_drawn` and seeded by
  the global component ID.
- Tracks entering activity features satisfy pT >= 2 GeV.
- The Hcc locked feature set is the charm-ranked top 20; the Hbb locked set is
  unchanged.
- Folds, training caps, early stopping, and effective counts operate on central
  groups rather than expanded rows.
- MadGraph v02--v07 event weights are independent campaign weights.  A generic
  stitching strategy is invoked only when the component explicitly declares
  overlapping phase space.
- The new min-bias normalization is candidate intensity at Poisson pileup,
  `mu**2 * pair_intensity`, not a probability inferred from stored parquet
  rows and not the legacy constant 0.005.

## Exact timing likelihood

For matched covariance `A = diag(a_z^2, a_t^2)` and unrelated covariance
`B = diag(b_z^2, b_t^2)`, the discriminating quadratic is

```
q = dz^2 * (1/a_z^2 - 1/b_z^2)
  + dct^2 * (1/a_t^2 - 1/b_t^2)
log LR = log((b_z*b_t)/(a_z*a_t)) - q/2.
```

This is the exact Gaussian likelihood ratio even though the z and ct variance
ratios are unequal.  `minbias/vertex.py` will expose normalized bin
probabilities under both hypotheses, with tests for normalization, limiting
cases, monotonicity, and the existing rectangular-cut values.  The H(cc)
trainer/evaluator will spread each component's weight over these bins and add
the bin log-LR to the central/proton score.  Resolution scans recompute only
these small probability tables.

## Work sequence and verification gates

### 1. Charm calibration first

- Extend `jet-energy/datasets.yaml` for Hcc, SuperChic QCDcc, SuperChic QEDcc,
  and component-level MadGraph QCDcc.
- Extend the manifest loader minimally to accept `input_globs` and select files
  round-robin, with unit tests.
- Add progress output to the cache and correction study: dataset/file index,
  events, elapsed time, rate, and ETA.
- Build caches, derive maps, and stop if any trigger eta region lacks support.

Gate: all charm map keys exist; closure and unsupported-region summaries are
recorded before MVA corrections are enabled.

### 2. Shared physics and dataset builder

- Port, rather than import, the required legacy symbols and color-flow helpers.
- Add corrected/matched jet-pair selection while preserving all original cuts,
  feature definitions, constants, and names.
- Add component loading, disjoint weighting, optional stitching, real-proton
  handling, and central-event memmap writing.
- Emit progress at each component, campaign, file block, write block, and
  summary stage.

Gate: no `analysis.MVA` import below `mva/`; compilation and focused feature,
weight, matching, and deterministic-subset tests pass.

### 3. H(cc) training and reports

- Port the fast harness method: sequential feature reads, capped group-safe
  early stopping, fixed production parameters, calibrated two-fold cross-fit,
  chunked cell scoring, and bucketed accumulation.
- Implement all four named profiles.
- Keep only compact report arrays needed by plotting and normalization scans.
- Port the existing plots and locked-score scans for survival, charm-tag
  efficiency, and b-to-c mistag.
- Emit per-stage timing, memory-size estimates, row/group counts, training
  progress, scoring progress, rate, and ETA.

Gate: bounded parity control passes within seed noise or its discrepancy is
reported; every nominal profile completes a small-input smoke run.

### 4. Staged physics changes

Starting from the refactored leading-jet/corrections-off baseline, run and
report separately:

1. jet corrections only;
2. truth-matched jets plus the required truth-matched selection;
3. analytic min-bias pairs plus exact vertex likelihood.

Each report includes component yields, dijet-mass peak, significance,
`truth_is_leading`, correction-invalid fraction, and effective statistics.
Large unsupported correction regions or large truth-match losses are stop
conditions and are reported rather than tuned away.

### 5. H(bb)

- Add the same four entry points and an H(bb) channel configuration.
- Use the existing FSR/noFSR map keys explicitly per component.
- Retain v01--v03 MadGraph QCDbb inputs; v04--v09 remain deferred.

Gate: the documented H(bb) parity metrics are reproduced or the mismatch is
reported without changing physics constants.

### 6. Completion audit

- Compile every new Python file after `source setup_env.sh`.
- Run unit tests for calibration manifests, matching, weighting, vertex bins,
  storage, group safety, profiles, and compact accumulation.
- Run small local end-to-end builds for H(cc), then the staged production runs
  as resources allow.
- Check every success criterion in `REFACTOR_PROMPT.md` against an explicit
  file, command output, or generated report.
