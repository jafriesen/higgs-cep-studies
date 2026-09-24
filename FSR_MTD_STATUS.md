# FSR + MTD reprocessing: status and next steps

> **Superseded 2026-09-24 by `STATUS.md`** for the current state and next steps.
> Kept for the jet-calibration decisions and closure tables below, which still hold.

Handoff written 2026-09-16. Read this before touching the jet calibration or the
MVA datasets.

## What this campaign is

Regenerate every MVA input with **final-state radiation on** (the old samples had
FSR switched off, which is unphysical for the exclusivity variables) and with the
**MTD timing card** (which cuts leaked pileup tracks from 14.3 to 5.0 per event),
then rebuild the jet object and the MVA datasets on top.

## Current state

**Production is complete**: all 23,300 jobs, zero failures, 283 GB. All 18 samples
are finished and readable (spot-check of 300 random files: none empty or broken):

| Sample | Note |
|---|---|
| `Hbb`, `Hcc` `_DPy8_FSR_200PU_MTD__v01` | signal |
| `QCDbb`, `QEDbb`, `QCDcc`, `QEDcc` `_DPy8_FSR_200PU_MTD__v01` | exclusive backgrounds |
| `QCDbb_DPy8_FSR_200PU_MTD__v04`..`v09` | inclusive bb background, disjoint slices |
| `QCDcc_DPy8_FSR_200PU_MTD__v02`..`v07` | inclusive cc background, disjoint slices |

## What changed in the code

| File | Change |
|---|---|
| `sim/scripts/DelphesPythia8Filter.cpp` | activity veto (charged pT outside the two jets < 25 GeV), thresholds from env vars, `DELPHES_RANDOM_SEED` |
| `sim/scripts/run_processes_delphes_pythia8.py` | `--filter-activity-max`, per-job Delphes seed, shared build dir |
| `sim/Cards/CMS_PhaseII_200PU_v04_trim_MTD35ps.tcl` | production card (trimmed + MTD) |
| `jet-energy/derive_parton_corrections.py` | derives correction maps, `--target genjet` (production) or `parton` (study) |
| `mva/common/features.py` | `recovered_pair()` + recovery-aware pipeline |
| `mva/common/dataset.py` | part-offset proton matching, MadGraph `sim_subcampaign` override |
| `common/jet_calibration.py` | accepts schema 4 (legacy) and 5 (recovered) |
| `mva/Hbb/config.yaml` | `*_fsr` components + `fsr_mtd_four_class` profile |
| `mva/Hcc/config.yaml`, `mva/Hcc/prepare_dataset.py` | `*_fsr` components + `fsr_mtd_no_hbb_resonant` profile; profiles read from config |

Production maps: `jet-energy/output/fsr_mtd_recovered/corrections.yaml`, under
`maps[FSR][<key>]`:

- one map per SuperChic sample: `Hbb`, `QCDbb_superchic`, `QEDbb`, `Hcc`,
  `QCDcc_superchic`, `QEDcc` (derived on the first 100 files of each);
- **one map per MadGraph slice**: `QCDbb_madgraph_v04`..`v09`,
  `QCDcc_madgraph_v02`..`v07` (first 40 files of each). A component selects them
  with `source_sample: QCDbb_madgraph_{campaign}`.

Kept for reference only, failed closure: `bottom`, `charm` (shared per flavour) and
`QCDbb_madgraph`, `QCDcc_madgraph` (shared across a component's slices).

## Closure

`jet-energy/check_recovered_closure.py` applies each map to that sample's held-out
files (the half not used in the derivation) and reports the median corrected
response overall and in bins of truth (GenJet) pT and |eta|.

**Shared per-flavour maps failed:** exclusive samples closed near 1 but the
inclusive MadGraph backgrounds were overcorrected by 7-11% in every pT bin. Busy
inclusive events keep more residual pileup/UE in the reco jet after PUPPI than the
GenJet contains, a detector effect correlated with event activity.

**Per-sample maps (adopted):**

| Sample | Held-out response | | Sample | Held-out response |
|---|---|---|---|---|
| Hbb signal | 1.003 | | Hcc signal | 1.001 |
| QCDbb exclusive | 1.038* | | QCDcc exclusive | 1.017* |
| QEDbb exclusive | 1.015 | | QEDcc exclusive | 1.009 |
| QCDbb MG v08 | 1.013 | | QCDcc MG v02 | 0.993 |
| QCDbb MG v09 | 0.991 | | QCDcc MG v03 | 1.006 |

In truth-pT bins from 12 to 90 GeV every sample closes to about 1%.

\* **Read the overall medians carefully.** The soft exclusive samples have 15-22%
of their jets below 12 GeV truth pT, where the 5 GeV reco threshold only keeps
upward fluctuations, so their response there is 1.16-1.31. That threshold bias,
not the map, pulls the overall median up. Likewise, never quote closure after
cutting on *reco* pT (e.g. "jets passing calibrated pT >= 15"): that selects
upward fluctuations and gives 1.06-1.14 by construction. Always bin in truth pT.

**MadGraph: per-slice, not per-component.** A map shared across a component's slices
looks fine on overall medians (0.96-1.01) but misses by 5-15% inside pT bins, because
slices respond differently at the same truth pT. Per-slice maps, held-out:

| bb slice | shared | per-slice | | cc slice | shared | per-slice |
|---|---|---|---|---|---|---|
| v04 | 1.000 | 1.002 | | v02 | 1.001 | 1.001 |
| v05 | 1.010 | 1.003 | | v03 | 1.005 | 1.000 |
| v06 | 1.003 | 0.999 | | v04 | 1.003 | 1.000 |
| v07 | 1.008 | 0.997 | | v05 | 1.004 | 1.000 |
| v08 | 0.993 | 1.002 | | v06 | 0.996 | 0.999 |
| v09 | 0.957 | 1.004 | | v07 | 0.980 | 1.005 |

Per-slice maps close to 1-2% in almost every truth-pT bin. The few bins still off
are each slice's kinematic tail, far from its generated range and sparsely populated
(bb v07 at 12-22 GeV: 0.83; bb v09 at 33-50: 0.95; cc v05 at 12-33: 1.11/0.96).

Confirmation in the dataset: with its own map the MadGraph background m(jj) drops
from 85.9 to 80.0 GeV, the same 7% the shared map had been overcorrecting — it was
pushing the background mass toward the signal peak.

## Decisions already settled — do not re-litigate

These were each measured; the measurements are in the session and summarised in
the `jet-calibration-decisions` and `hbb-reco-card-test` memories.

1. **Calibrate to the recovered GenJet pair, never to the parton.** FSR out-of-cone
   costs the exclusive signal 12–16% and the inclusive background only 4–5%. That
   gap is colour flow (a singlet dipole radiates into the gap between the jets) and
   it is real signal/background separation. A parton target erases it and is 34%
   process dependent. A GenJet target removes only the 5–7% detector piece.
2. **One map per sample.** A shared map fails on the inclusive backgrounds (see
   "Closure"). This is not the per-sample problem the parton target had: at the
   GenJet level the difference is detector response tracking event activity, so
   per-sample maps are a stand-in for an activity-dependent offset correction, not
   an erasure of colour-flow physics.
3. **The gen side of the calibration must be recovered too**, otherwise the
   correction simply undoes the recovery.
4. **The signal m(jj) peak sits near 104 GeV, not 125, and that is physics.**
   Energy genuinely leaves the R=0.4 cones. Do not "fix" it with calibration.
   A single universal factor applied to all samples is fine for presentation.
5. **Rejected:** parton target, b-jet energy regression, shared per-flavour maps,
   AK8 (same scale as AK4+recovery but worse resolution).

## Next steps, in order

1. **Full-statistics H(bb) build and training** on the `fsr_mtd_four_class`
   profile. Everything needed is complete, and the component already includes all
   six slices (v04-v09) with per-slice maps. Compare significance against the noFSR baseline
   (`mva/Hbb/results/study_bbrank20`, ladder Z 0.854).

       python3 mva/Hbb/prepare_dataset.py --profile fsr_mtd_four_class
       python3 mva/Hbb/train_model.py --data-dir mva/Hbb/data/fsr_mtd_four_class \
           --result-dir mva/Hbb/results/fsr_mtd_four_class

2. **Full-statistics H(cc) build and training** on `fsr_mtd_no_hbb_resonant`
   (the FSR+MTD version of `nominal_no_hbb_resonant`, tracks from 0.5 GeV). Wiring is
   done and smoke-tested (2 files per sample: all subcampaigns FSR, zero invalid
   corrections, signal M_X 125.0, the b-mistag slices reproduce the H(bb) smoke
   masses exactly). The b-mistag MadGraph background is now disjoint v04-v09 instead
   of stitched v02-v03. Compare against `mva/Hcc/results/nominal_no_hbb_resonant`.

   The full build (all features) was submitted 2026-09-16 with
   `python3 mva/Hcc/prepare_dataset_condor.py submit --feature-set full` (425 shards,
   copy of the H(bb) sharded builder). Shards/logs in
   `mva/Hcc/condor/fsr_mtd_no_hbb_resonant_full`; check with `... status --manifest
   <that dir>/manifest.yaml`. Output: `mva/Hcc/data/fsr_mtd_no_hbb_resonant_full`.
   Do not edit `mva/common/*.py`, `mva/Hcc/config.yaml` or the correction file
   until the merge finishes: the jobs verify their checksums and will refuse to run.

3. **Jet pT threshold scan.** A naive S/√B on acceptance says 15 GeV is the worst
   point and 30 GeV is ~16% better, but that ignores the MVA. Rebuild at 25 and 30
   GeV and compare trained significance.
4. **Re-rank features.** The current lists were ranked on samples without MTD,
   which changes exactly the track-activity variables that dominate them. Fix the
   `rank_features.py` bug first: it scores with all background class weights set
   to 1 (`kappas = np.ones(...)`, line ~175) while production weights each class
   by its yield. That is the likely reason the H(cc) `ranked20` list is 9% worse
   than the older `locked` list.

## Open items

- **H(bb) datasets:** use `mva/Hbb/data/fsr_mtd_four_class_full_per_sample`
  (per-slice maps). `fsr_mtd_four_class_full` was built with the shared
  `QCDbb_madgraph` map, which failed closure.

- Whether to keep `QEDbb` as a separate MVA class (currently yes, in
  `fsr_mtd_four_class`).
- **b->c mistag working point.** The config assumes eff_c 0.65 with b->c 0.055, which
  makes bb only ~0.3% of the MadGraph background. Realistic c-taggers at that
  efficiency have b->c ~0.2-0.3, which would make it ~4-9%.
- **Parton mass gap.** The new bb slicing (v04–v09) inherits the charm partition,
  which leaves parton m(jj) 50–70 GeV uncovered for pT > 25. The old broad v01
  covered it. Check whether anything migrates into the mass window from there.
- **MTD is optimistic.** Delphes compares *true* track and vertex times with no
  smearing, so the track-variable gain is an upper bound.
- **Trial/scratch outputs** that can be deleted: `QCDcc_DPy8_FSR_200PU_MTDtest__v03`,
  `mva/Hbb/data/fsr_mtd_smoke`, and the AK8/PhaseI/0PU-v04 reco-test tags once the
  card study is no longer needed.
