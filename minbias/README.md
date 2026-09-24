# Min-bias forward protons

This package generates a reusable truth-proton artifact and turns its two
single-arm proton intensities into analytic cross-interaction pair densities.
It does not enumerate proton pairs. "All pairs" means that every left/right
combination is included through the product of the two single-arm intensities.

It also provides analytic vertex-overlap probabilities and a correlated
bunch-crossing sampler for trigger-rate studies. Random pair selection and MVA
integration remain outside this package.

## Environment and configuration

Run commands from the repository root after setting up the analysis environment:

```bash
source setup_env.sh
```

Defaults live in `minbias/config.yaml`: 14 TeV collisions, inelastic SoftQCD,
a truth storage window of `0.001 < xi < 0.25`, the current PPS truth-xi
acceptance, `sigma_xi = 0.0003`, and 4096 uniform `ln(xi)` bins. Acceptance is
applied to truth xi before the reproducible detector smearing. No proton-pT or
t acceptance is applied.

## Local generation and validation

Generate same-seed filtered and unfiltered controls:

```bash
python3 -m minbias.generate_protons \
  --events 100000 --seed 12345 --output /tmp/minbias-filtered/protons.parquet

python3 -m minbias.generate_protons \
  --events 100000 --seed 12345 --no-filter \
  --output /tmp/minbias-unfiltered/protons.parquet

python3 -m minbias.validate \
  --artifact /tmp/minbias-filtered/protons.parquet \
  --unfiltered-control /tmp/minbias-unfiltered/protons.parquet \
  --expected-events 100000
```

`protons.parquet` contains `event`, `arm`, `xi`, `px`, `py`, and `process`.
`event` is dense only over generated interactions with stored proton rows. The
true inelastic denominator, including empty interactions, is
`n_inelastic_generated` in the adjacent `metadata.json`.

## API

```python
from minbias import Acceptance, PairDensity, ProtonFlux, Resolution

flux = ProtonFlux.load("output/minbias/minbias_inelastic_100m_v1/protons.parquet")
acceptance = Acceptance([
    (0.00325, 0.0116),
    (0.014, 0.0263),
    (0.0375, 0.0688),
    (0.08, 0.1967),
])
pairs = PairDensity(flux, acceptance, Resolution(0.0003, 12345))

per_ordered_interaction_pair = pairs.integrate((117.0, 133.0))
expected_at_fixed_200 = pairs.expected_pairs_fixed_n(200, (117.0, 133.0))
expected_at_poisson_200 = pairs.expected_pairs_poisson(200.0, (117.0, 133.0))
mass_template = pairs.marginal_mass(range(117, 134))
```

The single-arm `ln(xi)` histograms are intensities per generated inelastic
interaction, not unit-normalized templates. Consequently, `density()` and
`integrate()` are pair intensities per ordered pair of distinct interactions.
Fixed multiplicity scales by `N * (N - 1)` and Poisson multiplicity by `mu**2`.
The API intentionally has no `p_at_least_one`: shared protons make candidate
multiplicity non-Poisson, so that event-level calculation belongs downstream.

## Vertex overlap

`vertex_overlap_probability` calculates rectangular PPS-to-primary-vertex
compatibility probabilities. Resolutions are named with explicit units. PPS
timing is a single-arm resolution; its resolution on both `z` and `ct` is
`c * sigma_t / sqrt(2)`.

```python
from minbias import vertex_overlap_probability

probability = vertex_overlap_probability(
    truth_status="unrelated",
    timing_mode="central",  # z_only, luminous, or central
    beam_sigma_z_cm=5.7,
    single_arm_time_resolution_ps=10.0,
    pv_z_resolution_cm=0.001,
    pv_time_resolution_ps=30.0,
    nsigma=2.0,
)
```

For `truth_status="matched"`, the PPS protons originate from the central
primary vertex. For `"unrelated"`, the PPS-implied and central vertices are
independent draws from a Gaussian luminous region with equal widths in `z` and
`ct`. `z_only` applies only the reconstructed-z match. `luminous` also requires
the PPS-implied `ct` to lie within `nsigma * beam_sigma_z_cm` of zero. `central`
instead compares it with central-detector timing using the combined resolution.

For one coordinate with measurement variance `d` and unrelated luminous
variance `2 * beam_sigma_z_cm**2`, the rectangular-cut probability is

```
erf(nsigma * sqrt(d) / sqrt(2 * (d + 2 * beam_sigma_z_cm**2)))
```

The two-coordinate central-timing probability is the product of the independent
`z` and `ct` factors.

For likelihood-based analyses, use the exact two-Gaussian timing likelihood:

```python
from minbias import vertex_likelihood_table

table = vertex_likelihood_table(
    bins=8,
    single_arm_time_resolution_ps=10.0,
    pv_time_resolution_ps=30.0,
)
```

The default bins have equal probability for matched protons.  The table gives
the analytic matched and unrelated fractions in every bin plus the exact
discrete `log(P_matched/P_unrelated)` score.  The discriminant retains separate
z and ct coefficients; it does not assume that their variance ratios are equal.
Changing a timing or beam resolution only rebuilds this small table.

## Trigger-rate study

Sample Poisson pileup and print both BX probabilities and rates:

```bash
python3 -m trigger.minbias_rate \
  --artifact output/minbias/minbias_inelastic_100m_v1/protons.parquet \
  --bunch-crossings 1000000 \
  --output /tmp/minbias-trigger-rate.json
```

Defaults are `mu = 200`, a 117--133 GeV reconstructed mass window, 10 ps
single-arm PPS timing, 0.1 cm central-PV z resolution, 30 ps central timing, a
two-sigma rectangular cut, and a 31.6 MHz BX frequency. All are CLI options.
The CLI reports artifact loading, interaction preparation, and sampling progress
with elapsed time and an updated ETA approximately ten times per run.

The sampler first applies PPS acceptance to truth xi. It samples only
interactions containing an accepted proton using Poisson thinning against the
full inelastic denominator, smears every sampled proton independently, and
applies the mass window to reconstructed xi. Every left/right combination is
included, including same-interaction pairs.

Each sampled interaction receives one truth `(z, ct)`, each proton one timing
measurement, and each BX one independent central primary vertex. Measurements
are reused by all pairs containing that object, preserving shared-proton and
shared-PV correlations. The report contains mean candidates per BX, the BX
fraction with at least one candidate, binomial uncertainties, rates, and
same-interaction diagnostics for the inclusive and mass-window selections.

## Condor production

Prepare a dry-run campaign by omitting `--submit`; add it only after inspecting
the manifest and queue:

```bash
python3 -m minbias.submit_condor \
  --events 100000000 --jobs 500 --seed-base 100000 \
  --campaign minbias_inelastic_100m_v1 \
  --campaign-dir output/minbias/minbias_inelastic_100m_v1
```

Each job writes one Parquet shard and adjacent metadata. After every manifest
job succeeds:

```bash
python3 -m minbias.merge_protons \
  --manifest output/minbias/minbias_inelastic_100m_v1/manifest.json

python3 -m minbias.validate \
  --artifact output/minbias/minbias_inelastic_100m_v1/protons.parquet \
  --expected-events 100000000 \
  --report output/minbias/minbias_inelastic_100m_v1/validation.json
```
