# Analysis status

Last updated 2026-09-24.  Read this first; `FSR_MTD_STATUS.md` (2026-09-16) keeps
the jet-calibration decisions and closure tables and is otherwise superseded.
Paper-facing numbers are traced in the paper repo (`higgs-cep-paper/JHEP_example/`:
`claims.yaml`, `verify_claims.py`, `paper-notes.md`).

## Current nominal results

**H(bb):** `mva/Hbb/results/fsr_mtd_binary_7p1ps_allrows_gg_v04_base_g256_s12345`
- dataset `mva/Hbb/data/fsr_mtd_five_class_gg_v04` (profile `fsr_mtd_five_class_gg`):
  FSR on, MTD card, per-sample recovered-GenJet jet maps, QCDgg v04
- binary architecture, all rows, logloss, seed 12345, 10 ps PPS arms, 7.1 ps PV time
- single-cut Z = 0.892 (quote the single cut, not the six-category ladder);
  S = 15.7, B = 387 at 3 ab^-1
- bitwise identical to `..._gg_v04_g256_s12345` in every shared array; the `_base`
  rerun also stores `base_mass` and `madgraph_base_above_squared`
- paper table inputs: `mva/Hbb/results/fsr_mtd_binary_7p1ps_gg_v04_cutflow/`
  (`pre_mva_counts.yaml`, `table_{stitched,v04}/hbb_table.*`)

**H(cc):** not final.  FSR+MTD binary trainings exist
(`mva/Hcc/results/fsr_mtd_binary_*`); the `sc20_*` runs are at 20x survival
normalisation, not nominal, and `wpnew`/`wpold` differ in the c-tag working point.

## In progress

- H(bb) with mistagged charm: profile `fsr_mtd_eight_class_gg_cc` adds
  `QCDcc_fsr`, `QEDcc_fsr`, `QCDcc_madgraph_fsr`.  Dataset
  `mva/Hbb/data/fsr_mtd_eight_class_gg_cc` built 2026-09-24 (the five existing
  processes are identical to v04; charm preselection 7.55 / 30.9 / 1.80e8).
  Training `mva/Hbb/results/fsr_mtd_binary_7p1ps_allrows_gg_cc_g256_s12345`
  submitted with the nominal settings.  Estimate before the build: +9 background
  events (+2.4%, Z -1.2%) at c->b mistag 0.1.

## Established this round (details in the paper repo's `paper-notes.md`)

- **Survival retraining is a no-op for the binary architecture.**
  `balanced_weights` renormalises each class, and MadGraph dominates the
  background training mixture by ~1e8, so retraining at 2-20x survival changes
  Z by < 1%.  Use one nominal training and `mva/Hbb/scan_normalizations.py`
  (score held fixed) for sensitivity curves.
- **Noise floor ~5% in Z** (spread of those retrainings; matches the MC
  statistics of ~90 effective MadGraph events at the operating point).
- **The H(bb) operating point is support-floor limited** (Z still rising at
  50 effective events), so the quoted Z is a statistics-limited lower bound.
- **At the operating point the timing likelihood is a vertex cut**: no event's
  BDT-only score is high enough to pass from outside the vertex-compatible core
  (q < 4.15, 87.5% of genuine pairs, 0.20% of accidental pairs).  Finer binning or
  a 2 sigma pre-cut changes Z by < 1%.  See `mva/Hbb/timing_selection.py`.
- **The training's `vertex.resolution_scan` ignores the support floor** -- do not
  quote it.
- **Photon-exchange backgrounds are not survival-scaled** (near-unit soft
  survival): `QEDbb_fsr` now matches `QEDcc` in the H(cc) config.  Nominal weights
  never read this flag; it affects survival scans and `make_scaled_dataset.py`.
- **QCDgg v04 has a generator cut 110 < M_X < 140 GeV**; it agrees with the
  unrestricted v03 to 2.4% once the 117-133 GeV window is applied.

## Open items

- b/c tagging working points are optimistic (b->c 0.055 at eff_c 0.65; c->b 0.1
  and light->b 0.01 at eff_b 0.85).  Left as configured for now; mistagged
  backgrounds grow as the square of these.
- Training-setup robustness: compare trainings that differ only in which
  negligible backgrounds are included and in the relative class strengths,
  always evaluated on the full background set, against the ~5% noise floor.
- More inclusive MadGraph statistics would lift the H(bb) support-floor limit.
- MadGraph bb slices v04-v09 leave parton m(jj) 50-70 GeV uncovered for pT > 25.
- MTD track times in Delphes are unsmeared, so track-activity gains are an
  upper bound; the vertex timing itself is parameterised (7.1 ps).
- The trigger study still uses 30 ps central timing; whether 7.1 ps applies at L1
  is undecided.

## Tools added this round

- `mva/Hbb/make_scaled_dataset.py` -- clone a dataset at another survival factor
- `mva/Hbb/timing_selection.py` -- vertex cut / likelihood comparisons, support floor exact
- `mva/Hbb/build_paper_table.py` -- H(bb) efficiency table from the cutflow and timing study
- `mva/common/training.py` now also writes `base_mass` and
  `madgraph_base_above_squared` to `report_data.npz`
