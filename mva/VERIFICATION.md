# Refactor verification record

This record distinguishes structural/smoke verification from production
physics results.  Small-input fits below intentionally used only 1--8 boosting
rounds and must not be quoted as sensitivities.

## Calibration

The charm cache was built from 30 ROOT files and the map study completed in
about 322 seconds.  `noFSR` map keys are present for `Hcc`,
`QCDcc_superchic`, `QEDcc_superchic`, and the component-level
`QCDcc_madgraph` map.  The MadGraph derivation and validation inputs contain
one file from each of v02--v07 in round-robin order.

Validation used 15,000 events per SuperChic sample and 60,000 MadGraph events.
Complete truth-matched pairs with two correctable jets were:

| component | correctable / complete pair | fraction |
|---|---:|---:|
| Hcc | 12,907 / 13,178 | 97.94% |
| QCDcc SuperChic | 5,688 / 6,379 | 89.17% |
| QEDcc SuperChic | 8,090 / 8,768 | 92.27% |
| QCDcc MadGraph | 29,577 / 30,430 | 97.20% |

All eta bins have usable nodes.  The lower SuperChic fractions remain relevant
to the truth-pair variation.  Before the nominal candidate acceptance was
defined, a direct one-file leading-pair check found invalid correction pairs of
346/4,358 (7.94%) for Hcc and 397/1,753 (22.65%) for QCDcc: forward pileup jets
were entering the global-leading validity veto.

Nominal MVA candidates now satisfy strict `|eta| < 3.0` before the corrected-pT
cut and pT ranking.  Repeating the same checks within that pool found 0/4,044
invalid pairs for Hcc and 0/1,390 for QCDcc.  Selected central rows changed
from 3,900 to 3,937 and from 1,341 to 1,382, respectively, because an
out-of-acceptance jet no longer vetoes a valid fiducial dijet.  The v01 parity
profile deliberately retains the legacy acceptance and still produces its
reference 1,602 Hcc rows after the proton selection.

A one-file-per-campaign `nominal_full` build then exercised all seven
components, including QCDcc v02--v07.  It stored 15,195 events, and all seven
components reported zero invalid correction pairs within their fiducial
candidate pools.  A direct full-feature Hcc check confirmed the strict bound
(largest observed `|eta| = 2.99897`).

## Feature and legacy-pool parity

With leading jets, corrections off, and the legacy 0.5 GeV track setting, one
Hcc ROOT file produced 4,195 central rows in both implementations.  Event
indices, dijet mass, dijet rapidity, and all 54 central feature columns agreed;
the maximum finite feature difference was exactly zero.

The single-file seven-component parity build was also compared to the untouched
reference builder.  All five real-proton row counts matched exactly.  The new
format intentionally stored each MadGraph central event once instead of four
times, but physical yields agreed:

| component | reference rows | central rows | reference yield | yield residual |
|---|---:|---:|---:|---:|
| Hcc | 1,602 | 1,602 | 4.5135403 | `+4.9e-14` |
| QCDcc SuperChic | 83 | 83 | 444.63030 | `-5.1e-13` |
| QCDbb SuperChic | 88 | 88 | 4.9943809 | `-3.6e-15` |
| QEDcc SuperChic | 157 | 157 | 1,953.5072 | `-5.5e-12` |
| QCDcc MadGraph | 9,064 | 2,307 | 10,037,279 | `-5.2e-6` |
| QCDbb MadGraph | 14,084 | 3,914 | 115,534.20 | `+9.4e-9` |
| Hbb resonant | 623 | 623 | 0.6329942 | `-3.3e-16` |

The MadGraph row-count difference is the intended central-event storage
change; its group count and normalization, rather than materialized proton
copies, are the invariant.

## Charm-ranked features, split classes, and stored proton vectors

The nominal Hcc profiles now use separate `exclusive_QCD` and `exclusive_QED`
classifier classes.  The merged `charm_only_three_class` study control remains
available, with `charm_only_four_class` providing its split counterpart.  The
Hcc locked feature list is the exact charm-ranked top 20 from the earlier
permutation study.  The parity profile selects the preserved `legacy_locked`
list automatically, and Hbb remains unchanged.

LHE and HepMC parser fixtures verify the outgoing-proton arm convention and
exact unsmeared px/py values.  A 3,139-row integration build exercised LHE Hcc
and QEDcc, HepMC QCDbb, and one analytic QCDcc MadGraph campaign.  All four
real-proton transverse arrays were finite for every SuperChic row and NaN for
every MadGraph row.  Their one-file mean pair-averaged proton pT values were
0.361 GeV for Hcc and 0.145 GeV for QEDcc, consistent with the unsmeared study.

The same dataset loaded as 19 stored central columns plus the dynamic proton
delta-y column: exactly the configured 20 charm features.  None of the four
stored transverse arrays appeared in the feature list.  A four-class tiny-fit
smoke, all five plots, and all normalization scans completed; this verifies the
schema and class plumbing only, not physics performance.

## End-to-end smoke checks

- Hcc charm-only corrected leading build: 9,964 stored central events.  Build,
  three-class calibrated training, analytic cell evaluation, exact vertex
  likelihood, plots, and normalization scans completed.
- Before the fiducial-candidate update, an Hcc nominal-full corrected leading
  build stored 14,542 events across all seven components and four classes.
  Every component's dataset yield and report preselection yield agreed to
  floating-point rounding.  The post-update dataset-only check is recorded in
  the calibration section above; its model stages were not rerun.
- Hcc truth-jet build: 10,479 events stored.  The trainer automatically
  required matching; 6,415 of 8,688 stored QCDcc MadGraph events entered the
  physical evaluation after matching and nonzero pair support.
- Hbb nominal build: 5,196 events with the locked 48 features.  Its three-class
  training, evaluation, plots, and `eff_b` normalization scan completed, with
  exact preselection-yield conservation.

The nominal v02--v07 QCDcc yield is much larger than v01 because the campaigns
cover the previously missing forward, low-pT, and broad-mass regions and the
stored weight is before the unrelated-vertex likelihood.  In the one-file
nominal-full smoke build it was `1.64e10` events before timing discrimination,
versus `1.00e7` for the v01 parity control after its legacy 0.005 factor.  This
large, intentional phase-space change makes production correction coverage and
background support checks mandatory.

## Automated checks

After `source setup_env.sh`:

```text
python3 -m unittest mva.tests.test_common minbias.tests.test_flux minbias.tests.test_vertex
23 tests passed

(cd jet-energy && python3 -m unittest test_jet_cache.py)
11 tests passed

python3 -m py_compile <all Python files under mva, minbias, and jet-energy>
passed
```

No Python file under `mva/` imports the legacy MVA packages.  A literal grep of
the whole directory still finds their names in the supplied refactor prompt and
planning documents; the code-only check is clean.

## Not claimed yet

- No full-statistics corrected Hcc significance is claimed; the fiducial
  one-file correction checks pass, but the production dataset and fit were not
  rerun after defining the candidate acceptance.
- The documented full Hbb parity metrics were not rerun; Hcc was the requested
  implementation focus and the Hbb path has only a component-level smoke check.
- Tiny-fit smoke significances are deliberately omitted here.
