# Agent brief: build `mva/` — one MVA setup for H(bb) and H(cc)

Read the repo `CLAUDE.md` first. Read the files named below before writing code.

## The problem

`analysis/MVA_hcc/` is a clean four-script pipeline
(`prepare_dataset.py` → `train_model.py` → `plot_results.py` →
`scan_normalizations.py`) but it imports its physics out of the H(bb) monolith
`analysis/MVA/run_dijet_mva_multiclass_protons.py` (82 KB), which in turn pulls
from `run_dijet_mva_exclusive_central.py`, `run_dijet_mva.py` and
`color_flow_geometry.py`. Changing a charm feature means editing an H(bb) script.

`analysis/MVA/` is also ~10 overlapping scripts, most of them dead prototypes.

## What to do

Build `mva/common/`, `mva/Hbb/`, `mva/Hcc/` (already scaffolded, empty). Keep the
four-script shape. Channel directories hold config plus entry points; anything
identical in both channels, or differing only by a constant, goes in `common/`.

**Do not touch `analysis/MVA/` or `analysis/MVA_hcc/`.** They stay as the
reference for parity — read freely, write nothing, delete nothing.

### 1. Move the shared physics into `common/`

Exactly nine symbols cross the boundary today. Port them; do not import them.

From `run_dijet_mva_multiclass_protons.py`: `central_features` (line 550),
`FEATURE_NAMES` (315), `physical_event_weight` (253), `conditional_pair_sample`
(401), `real_proton_pass` (870), `load_bootstrap_pool` (1036), `branch_name` (489).
From `run_dijet_mva.py`: `parse_hepmc_proton_xi`, `load_pps_config`.

Take the preselection constants, the 55 feature names, and the dataset array
schema from those files unchanged. `color_flow_geometry.py` is 67 lines and
clean — lift as-is.

Two structural ideas to preserve:

- **A row is an (event, proton-cell) pair, not an event.** Combinatorial
  components are expanded over a grid in `yx_minus_dijet_rapidity` within
  `|yx − y_jj| < 0.2`. `group_id` is the central event. Folds, subsampling and
  significance must respect groups.
- **`class` (classifier label) and `component` (physical sample) stay separate**,
  mapped in `metadata.yaml`. Downstream code never hardcodes numeric IDs.

For the training machinery follow `analysis/MVA_hcc/qed_study/harness.py`, not
`train_model.py`. It is the same method, already fixed for speed: one sequential
column read instead of six fancy-indexed passes over a 2 GB memmap, a capped
early-stopping set, cached cell probabilities, bucketed threshold accumulation.

### 2. Jet energy corrections

Use `common/jet_calibration.py` as-is — `load_correction_map(path, fsr_state,
source_sample)` then `correct_jet_kinematics(...)`.

Apply **immediately after reading JetPUPPI pT and before the `pt >= MIN_JET_PT`
cut.** Correcting after the cut selects on raw pT and reports corrected pT.
Corrected mass goes into the dijet four-vector.

`valid = False` (outside `raw_pt_support` [8, 250] or an unsupported `|eta|` bin)
must not silently fall back to raw. Flag the event and report the fraction per
component.

**The charm maps do not exist.** `jet-energy/output/stage2_3/corrections.yaml`
has FSR: Hbb/QCDbb/HardQCD, noFSR: Hbb/QCDbb. Extend `jet-energy/datasets.yaml`
to cover Hcc, QCDcc, QEDcc and MadGraph QCDcc with `match_selection:
hard_flavor`, `truth_pid_abs: 4`, then run `jet_cache.py` and
`study_jet_corrections.py`. **Start this first — it is the long pole.**

The map key is per component, not global: H(bb) uses `Hbb_FSR_200PU__v01` while
every charm sample on disk is noFSR. Put `fsr_state` and `source_sample` in the
component spec.

### 3. Truth-matched jets

At 200 pileup there are 2.77 pileup jets per event above 15 GeV against 2 from
the hard process, so the leading pair is often not the hard pair.

Reuse the chain already validated in `jet-energy/jet_cache.py`: status-23 parton
with `|PID| = 4` or `5` and `IsPU = 0`, exactly one quark and one antiquark
(`hard_process_partons`, line 263) → GenJet within ΔR < 0.4 → JetPUPPI within
ΔR < 0.2, `|eta| < 3.0`, greedy unique smallest-first (`match_jets`, line 194).

Applies to **every component with a hard pair** — signal and background alike.
Matching signal only would bias every shape comparison.

**Do not drop unmatched events at build time.** Build features from the matched
pair when both legs match and the leading pair otherwise, and store
`truth_matched`, `truth_is_leading`, `match_dr1`, `match_dr2`. The cut then lives
at training time. Report `truth_is_leading` per component — it measures the
pileup-fake dijet rate and is a headline number.

`n_generated = int(tree.num_entries)` is the weight denominator. Truth matching
is a numerator efficiency. **Do not filter the denominator or rescale by the
match rate.**

### 4. Min-bias protons and the vertex likelihood

**Use `output/minbias/minbias_inelastic_100m_v1/protons.parquet`.** 10^8 inelastic
interactions, Pythia 8.317 tune 14 at 14 TeV, elastic off so the generator's own
counter is the denominator (`n_inelastic_generated = 100000000` in the adjacent
`metadata.json` — the file is filtered to proton-bearing interactions, so never
count parquet rows). 8,725,884 stored protons over the loose truth window
`0.001 < xi < 0.25`. `validation.json` records the checks: analytic vs
brute-force pair closure to 0.16%, FFT to 5e-12, and no arm with two protons from
one interaction.

Do not use `output/minbias/minbias/`, `minbias__v01/`, or
`minbias_inelastic_pilot_100k/` — legacy and pilot. Do not use
`analysis/MVA_hcc/data/proton_pairs.parquet`, the old sampled pair pool, except
for the parity check in the success criteria.

The `minbias` package replaces two things at once.

**Pair rates.** Instead of materialising a pair pool and sampling four pairs per
central event, take intensities from the API:

```python
from minbias import Acceptance, PairDensity, ProtonFlux, Resolution
flux   = ProtonFlux.load(".../protons.parquet")
pairs  = PairDensity(flux, Acceptance(windows), Resolution(0.0003, seed))
pairs.integrate((117.0, 133.0))        # per ordered pair of distinct interactions
pairs.expected_pairs_poisson(200.0, (117.0, 133.0))
pairs.marginal_mass(range(117, 134))
```

Intensities are per generated inelastic interaction, so fixed multiplicity scales
as `N*(N-1)` and Poisson as `mu**2`. This removes both the sampling noise and the
illusory pool size — the old 81,363-pair pool held only 4,218 independent protons.

**The vertex cut.** `combinatorial_acceptance_factor = 0.005` becomes a per-event
quantity. From PPS timing, `z = c(t_R - t_L)/2` and `t = (t_L + t_R)/2`; compare
against the reconstructed primary vertex to form `dz` and `dt`, then
`chi2 = (dz/sigma_z)**2 + (dct/sigma_ct)**2`. Both hypotheses are zero-mean
Gaussians differing only in width, so the likelihood ratio is monotone in chi2 and
closed-form. Widths at 10 ps per arm and 30 ps central timing:

| | real pair | coincidental |
|---|---|---|
| sigma(dz) | 0.212 cm | 8.06 cm |
| sigma(dct) | 0.924 cm | 8.11 cm |

**Do not sample chi2 per event, and do not cut on it.** dz and dt are not
measured — the vertices are drawn from the beam spot — so integrate instead.
Spread each event's weight across chi2 bins by the analytic fraction from
`minbias/vertex.py`. Those fractions are identical for every event in a component
(coincidental pairs are unrelated by construction, signal pairs are matched), so
the fraction is a common factor and effective statistics are preserved exactly:
n_eff is unchanged in every bin, where sampling would cost a factor ~650 in the
tightest band. This is the same device as the existing `band_probability` grid.

Then add `log LR(chi2)` to the plug-in score, or use chi2 as a category axis. The
vertex quantities are independent of the central kinematics, so the likelihood
factorises and the log-likelihoods add — **no new classifier feature, no
retraining.** Relative to an honest 2-sigma cut this is worth +23.5% on the
combinatorial background.

**Resolutions stay a use-time configuration, never baked into the dataset.**
They enter only through `sigma_PPS = c*sigma_arm/sqrt(2)`,
`sigma_matched = sqrt(sigma_PPS**2 + sigma_PV**2)` and
`sigma_unrelated = sqrt(sigma_matched**2 + 2*sigma_beam**2)`, all already keyword
arguments of `vertex_overlap_probability`. Changing one must be a few `erf` calls
with the score held fixed — no regeneration, no retraining. Provide that scan; it
spans a factor 2.4 in Z between 30 ps / 30 ps and 5 ps / 10 ps, and PPS timing is
the more leveraged number because it enters both coordinates.

### 5. Drop

Everything in `analysis/MVA/` except the symbols named in §1. Also: the
hyperparameter grid search (use `PRODUCTION_PARAMS` from `train_model.py:28`),
the 830 MB `oof_results.npz`, per-fold model files kept for inference, and
`prepare_cc_only_dataset.py` (the `--components` option covers it).

Keep plotting to what `plot_results.py` already does, and keep
`scan_normalizations.py` in spirit — score fixed, scanning survival, `eff_c`,
`mistag_b_to_c`.

### 6. Keep configurable

The dataset build, because every open question is "what if we trained on a
different configuration". Carry forward from `MVA_hcc/prepare_dataset.py`:
`--feature-set {locked,full}`, `--components`, `--extra-campaigns`,
`--store-parton`, `--max-files`. Add `--jets {truth,leading}` and
`--corrections {on,off}`.

Class structure is channel config, not hardcoded: H(cc) has five classes but
merging to three gains ~9%, and H(bb) has no QED class at all.

## Traps — each of these was a real bug here

1. **Seed on the global component index**, never on position in a filtered list,
   or a subset build selects different events than the full build.
2. **Cast Delphes float32 branches to float64 before invariant-mass math.**
3. **Rescale when subsampling** by `n_total / n_drawn`. Missing this caused a
   137× error once.
4. **Nothing may shrink `n_generated`.**
5. **Group-safe folds** — rows from one event never straddle a split.
6. **`IsRecoPU` leaks soft pileup tracks**; track features need pT ≥ 2 GeV.
7. **Never train the non-exclusive class on a narrow generation box.** MadGraph
   QCDcc previously covered only `|eta| < 1.5`, so the classifier never learned
   to reject forward QCD; its survival at `|eta| 2.5–3.0` was 13× the central
   value, and retrained on the wide region it drops to zero. Use the new
   `QCDcc__v02`–`v07`, which span `|eta| < 3.0` and pT 15–120.

## Success criteria, in order

1. `grep -rn "analysis.MVA" mva/` returns nothing.
2. `python3 -m py_compile` clean.
3. **Parity.** With the legacy proton pool, corrections off and truth matching
   off, H(cc) reproduces `analysis/MVA_hcc/data/metadata.yaml` row counts and
   per-component yields, and `results/report.yaml` significance to within seed
   noise. This is the only reason the legacy pool is still readable — it is the
   control that separates a refactor bug from an intended physics change.
4. H(bb) reproduces
   `analysis/MVA/output/mva_bb_multiclass_protons/significance_v3_all_campaigns/summary_final.yaml`:
   single-cut `Z = 0.9322`, categories `Z = 0.9709`, `QCDbb.auc = 0.7152`,
   `QCDbb_madgraph.auc = 0.9961`, 48 features.
5. **Then enable the three changes one at a time**, reporting yields, `m(jj)`
   peak and `Z` for each, per channel. They move the answer in different
   directions and a combined number hides that.
   - jet energy corrections
   - truth-matched jets
   - the min-bias package and the vertex likelihood
6. Report `truth_is_leading` per component, the uncorrected-jet fraction per
   component, and the vertex resolution scan.

The H(bb) `m(bb)` peak sits near 99 GeV against a true 125 (unclustered FSR ~16%,
neutrinos ~5%). How far the corrections move it is the headline result.

## Stop and report rather than deciding

- Charm corrections coming back with large unsupported regions.
- Truth matching removing a large fraction of any component — in the
  combinatorial background that may mean it is removing genuine pileup-fake
  background rather than cleaning the sample.
- H(bb) parity not reached. Do not adjust constants to force it.
- MadGraph `QCDbb__v04`–`v09` are declared in `processes-madgraph.yaml` but
  marked DEFERRED and not generated. Use `v01`–`v03` and say so.
