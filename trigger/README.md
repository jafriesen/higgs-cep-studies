# HardQCD and PPS trigger studies

This package contains the bunch-crossing trigger studies. The minbias package
continues to own the inelastic proton artifact and the reusable analytic vertex
probabilities.

Run all commands from the repository root after setting up the environment:

```bash
source setup_env.sh
```

## Inclusive HardQCD production

Prepare a 150,000-event HTCondor campaign with 50,000 correction-derivation
events and a disjoint 100,000-event validation/rate library:

```bash
python3 -m trigger.submit_hardqcd_condor \
  --campaign-dir output/trigger/hardqcd_pthat10_150k_v1
```

This writes the canonical `manifest.json`, 30 queue entries of 5,000 events,
and Condor artifacts without submitting. After inspection, submit the prepared
campaign explicitly with:

```bash
condor_submit output/trigger/hardqcd_pthat10_150k_v1/condor/submit.sub
```

Alternatively, add `--submit` to the Python preparation command to create and
submit a fresh campaign in one operation.

Jobs 0--9 are tagged `derivation`; jobs 10--29 are tagged `rate_validation`.
Each job records generator `pTHat`, runs the PU200 Delphes card, validates event
alignment, and keeps ROOT, Parquet, metadata, and logs beneath its shard
directory. Temporary HepMC is removed only after Delphes validation succeeds.
ROOT and Parquet shards are not merged.

```text
hardqcd_pthat10_150k_v1/
  manifest.json
  condor/{queue.txt,run_job.sh,submit.sub,...}
  shards/job_NNNNN/{delphes.root,events.parquet,metadata.json,delphes.log,...}
```

For a small interactive singleton that retains HepMC, use:

```bash
python3 -m trigger.produce_hardqcd \
  --campaign-dir output/trigger/hardqcd_pthat10_pilot \
  --events 10000
```

## Derive HardQCD jet corrections

The dataset manifest at `jet-energy/datasets.yaml` configures b-matched Hbb and
QCDbb samples plus inclusive hard-interaction GenJet matches for HardQCD. Cache
and derive all configured maps with:

```bash
python3 jet-energy/jet_cache.py
python3 jet-energy/study_jet_corrections.py
python3 jet-energy/plot_jet_corrections.py
```

The first ten HardQCD shards derive the map. The other twenty provide held-out
single-jet closure and the trigger-rate library. HardQCD does not produce dijet
profiles or residuals. Corrections reject raw jets outside `8 <= pT < 250 GeV`
or outside supported eta bins and report their coverage.

## JetPUPPI response check

```bash
python3 -m trigger.jet_response \
  --campaign-dir output/trigger/hardqcd_pthat10_pilot
```

Corrected `JetPUPPI` jets are uniquely matched to hard-interaction `GenJet`
jets within `deltaR < 0.2`. The JSON and plots report matching and 15 GeV
turn-on efficiencies, median response, central-68% relative resolution, and
dijet-rapidity closure in `pT` and `|eta|` bins. This is a descriptive Delphes
check, not a hardware-L1 performance requirement.

## Signal and background jet shapes

Compare the leading and subleading corrected JetPUPPI transverse momenta in
the H(bb) and HardQCD PU200 samples with:

```bash
python3 -m trigger.jet_kinematics
```

The default uses five H(bb) files and at most 10,000 events from each sample.
An event is included when it contains at least two corrected JetPUPPI jets
inside `|eta| < 2.4`; no additional jet-pT selection is imposed. Each sample is
corrected with its own map, since the raw PUPPI response differs between them by
about 15% in the barrel and an uncorrected shape comparison shows that artifact
rather than physics. The default signal directory is `noFSR` while the only
HardQCD map is `FSR`; use `--signal-fsr-state` if you change the signal sample. The plots are
normalized to unit event count, and `summary.json` records event counts and
kinematic quantiles. Use `--signal-max-files`, `--max-events`, and
`--output-dir` to change the pilot size or output location.

## Trigger rates

The standalone minbias rate study now lives here:

```bash
python3 -m trigger.minbias_rate \
  --artifact output/minbias/minbias_inelastic_100m_v1/protons.parquet
```

Run the dijet-PPS study with:

```bash
python3 -m trigger.dijet_rate \
  --hardqcd-campaign output/trigger/hardqcd_pthat10_150k_v1 \
  --jet-corrections jet-energy/output/stage2_3/corrections.yaml \
  --pps-artifact output/minbias/minbias_inelastic_100m_v1/protons.parquet \
  --pps-time-resolution-ps 10 \
  --output output/trigger/hardqcd_pthat10_150k_v1/dijet_rate.json
```

For each pseudo-BX, the HardQCD multiplicity is Poisson with mean

```text
mu * sigma(HardQCD) / sigma(inelastic).
```

The sampled record with the largest stored generator `pTHat` supplies the
central event. The required `FSR/HardQCD` correction is applied to JetPUPPI
`pT` and mass before ordering. Both leading and subleading corrected jets must
satisfy strict `pT > 20 GeV` and `|eta| < 2.4`; their summed four-vector defines
`y_jj`. A crossing without a sampled HardQCD event fails. The output records
the correction hash, identity, support losses, and coverage.

The selected Delphes event supplies the exact pileup occupancy as
`Vertex.GetEntriesFast() - 1`. The PPS sampler conditions on this same count
using binomial thinning of the inelastic proton artifact. The central and PPS
vertex coordinates are nevertheless sampled from the established equal-width
5.7 cm `(z, ct)` luminous model; individual Delphes and PPS pileup interactions
are not matched.

PPS acceptance is applied to truth xi. Each proton is smeared once and reused
in every pair, and the 117--133 GeV mass and PPS rapidity are reconstructed from
the smeared xi values. The full JSON contains all eight PPS/mass/vertex modes at
no rapidity cut and at `|y_jj-y_pp| < 1, 0.75, 0.5, 0.25, 0.1`. The printed
table emphasizes the mass-window trigger rates.

Every rate reports the pseudo-BX binomial error, a cluster bootstrap over the
finite HardQCD library, propagated HardQCD cross-section error, and their
quadrature sum. The bootstrap quantifies reuse of the central-event pilot; it
does not model additional detector-systematic or L1-emulation uncertainty.

## H(bb) signal efficiency

Add the existing SuperChic H(bb) FSR+PU200 sample to the same run with:

```bash
python3 -m trigger.dijet_rate \
  --hardqcd-campaign output/trigger/hardqcd_pthat10_150k_v1 \
  --jet-corrections jet-energy/output/stage2_3/corrections.yaml \
  --pps-artifact output/minbias/minbias_inelastic_100m_v1/protons.parquet \
  --pps-time-resolution-ps 10 \
  --with-hbb-signal \
  --output output/trigger/hardqcd_pthat10_150k_v1/dijet_rate_with_hbb.json
```

Signal and background are calibrated the same way: the `FSR/Hbb` map is read
from the same corrections file that supplies `FSR/HardQCD`, and each channel is
corrected to its own generator truth before the shared `pT > 20 GeV` and
`|eta| < 2.4` selection. The two maps differ because they undo a real, process-
dependent PUPPI response difference: in the barrel the H(bb) factor is about
1.02 while the HardQCD factor is about 0.89. Comparing an uncorrected signal
against a corrected background would apply a different effective true threshold
to each channel. The signal report records its own correction hash, identity,
support losses, and coverage.

The signal map is evaluated only on files it was not derived from: the loader
reads `derivation_files` out of the map and skips them, so `--hbb-max-files 5`
starts after the three derivation files. This mirrors the disjoint
`derivation`/`rate_validation` shard split the HardQCD campaign already uses.
Holding them out moves the 20 GeV efficiency from 63.0% to 62.7% +/- 0.5%, so
the map is not overfitting its derivation sample.

The signal loader aligns each Delphes file with its HepMC file and checks the
event numbers. JetPUPPI supplies the central dijet, while the two outgoing
H(bb) protons are read from HepMC. Truth PPS acceptance is applied before the
usual xi smearing. The accepted signal protons are placed at the primary
vertex and overlaid with exactly `Vertex entries - 1` independently sampled
minbias interactions, so signal and background use the same PPS reconstruction
and working-point definitions.

The reported signal efficiency is unconditional: its denominator is every
generated H(bb) event, including events that fail the central dijet or truth
double-tag requirements. The JSON also reports those stage efficiencies and
separates a genuine signal-proton pair from mixed or pileup-only pairs. A
`rescue` is an event that fails with the genuine pair but passes through an
accidental mixed or pileup pair. One PPS overlay is sampled per signal event;
the statistical error is therefore the finite-signal-sample binomial error.

Use `--hbb-max-files 1` for a quick integration check. The default input paths
point to the repository's `Hbb_FSR_200PU__v01` Delphes files and aligned
`Hbb_FSR__v01` HepMC files; they can be overridden with `--hbb-root-dir` and
`--hbb-hepmc-dir`.
