# Min-bias forward-proton package — refactor plan

A self-contained generation and analysis setup for min-bias forward protons.
Downstream consumers (the H(cc)/H(bb) MVA, and later the trigger rate study)
depend on it; it depends on nothing else in the repository.

## Why

The current chain generates min-bias events, enumerates accepted proton *pairs*
into `proton_pairs.parquet`, samples four pairs per MadGraph central event, and
multiplies by two flat constants (`bx_pair_acceptance = 0.23609` and
`combinatorial_acceptance_factor = 0.005`). Four problems:

1. **The 81,363-pair pool contains only 4,218 independent numbers.** Every pair
   is built from 2,062 left and 2,156 right protons, so the pool's apparent size
   is illusory and every template inherits the proton statistics.
2. **Sampling four pairs per event adds Monte Carlo noise on top of that**, which
   the MVA harness then has to recover with a 32-cell numerical integration.
3. **The vertex-matching cut is a scalar**, so it contributes exactly zero
   discrimination despite being worth a factor of ~130 in rejection, and it is
   applied to the background only — the signal is implicitly given efficiency 1.
4. **The per-interaction rate is normalised to a denominator containing 22.6%
   elastic events**, which can never produce an accepted proton (verified: 0 of
   4,218). Pileup counts inelastic collisions, so the combinatorial background is
   low by a factor 1.26 per arm and **1.59 on the pair rate**.

The replacement stores single protons and builds cross-interaction candidate
densities analytically. It includes every left/right combination without
materialising pair rows or selecting a pair.

## Established decisions

- **Pileup means inelastic.** mu = 200 counts inelastic collisions, the standard
  convention, and the Delphes pileup file is taken to match it. Measure the proton
  intensities from the new inelastic sample rather than fixing legacy values.
- **Inelastic only.** `SoftQCD:elastic = off`. With elastic off every generated
  event is inelastic, so Pythia's event counter *is* the denominator that
  multiplies mu = 200. This removes the normalisation ambiguity structurally
  rather than by convention; the 23% CPU saving is incidental.
- **Store raw protons, never sample from them.** Evaluate pair densities by
  convolution. Sampling adds noise; convolution uses all the information.
- **Store truth quantities; apply acceptance and resolutions at use time.**
  Acceptance is applied to truth xi. Resolution scans become "add noise to each
  proton and re-histogram", which is exact because the two protons' measurement
  errors are independent.
- **Store on a loose xi window**, not the PPS acceptance, so station layouts stay
  a use-time configuration.
- **Vertex matching and event-level probabilities are deferred.** This package
  exposes candidate intensities; trigger and MVA consumers will be migrated later.
- **No out-of-time pileup.**
- **Target: 10^8 inelastic interactions.** This is a fixed production request,
  not a sample size tuned from a statistical-precision target.
- **No process biasing.** The generation is cheap enough that the ~5x saving is
  not worth the cross-section bookkeeping.
- **Acceptance is a function of xi alone**, applied to truth xi; the xi resolution
  is flat at 0.0003 across all four stations; there is no t acceptance.

## Provisional consequences for the existing results

The following projection predates the candidate-intensity API and is not a
validation target for this implementation. Recompute it during downstream MVA
migration before quoting it.

Two independent normalisation corrections fall out, and they compound. Measured
against the charm-only baseline (`train_model.py` on `cc_locked`, Z = 0.0405):

| correction | MadGraph scale | signal / exclusive scale | Z | ratio |
|---|---|---|---|---|
| as published | 1.00 | 1.00 | 0.0405 | 1.000 |
| vertex acceptance only | 1.52 | 0.911 | 0.0330 | 0.816 |
| pair rate only | 1.59 | 1.00 | 0.0352 | 0.869 |
| **both** | **2.42** | **0.911** | **0.0280** | **0.692** |

The combinatorial background rises by a factor 2.4 in total: 1.59 from removing
elastic events from the denominator, 1.52 from an honest vertex acceptance
(0.0076 rather than 0.005 at 10 ps PPS and 30 ps central timing). The signal and
the exclusive backgrounds fall by 0.911, the vertex-cut efficiency that is
currently applied to neither.

Taken together the H(cc) significance falls by about 31% before any of the
improvements identified in `analysis/MVA_hcc/qed_study/FINDINGS.md` are applied.
Those improvements — merging the exclusive classes, two-axis categorisation, the
charm-ranked feature set, and keeping the vertex position in the likelihood —
recover a substantial part of it, but the corrected number is the one to quote.

## Layout

```
minbias/
  README.md
  config.yaml            defaults: acceptance windows, resolutions, beam
  generate_protons.py    CLI: Pythia -> protons.parquet + metadata.json
  submit_condor.py       CLI: production manifest and Condor submission
  merge_protons.py       CLI: verify and merge production shards
  flux.py                library: protons -> occupancy, f(xi), pair densities
  validate.py            CLI: closure tests
```

Dependencies: `numpy`, `scipy`, `pyarrow`, `pyyaml`, and `pythia8` for generation only.
**No imports from `analysis/`, `common/`, or `bkg-generation/`.** `config.yaml`
mirrors the repository values but is owned by this package; callers may pass
their own.

## The artifact

`protons.parquet`, one row per stored proton:

| column | type | meaning |
|---|---|---|
| `event` | int32 | dense index over *written* events |
| `arm` | int8 | -1 / +1 from the sign of pz |
| `xi` | float32 | truth 1 - E/E_beam |
| `px`, `py` | float32 | truth transverse momentum |
| `process` | int16 | Pythia process code |

`metadata.json` alongside (essentials duplicated in the parquet key-value
metadata):

- `schema_version`
- `n_inelastic_generated` — **the denominator**, from `pythia.info.nAccepted()`
- `n_events_written`, `n_protons_written`
- `store_window` — the loose xi range used for the filter
- `sqrt_s_gev`, `beam_energy_gev`
- `pythia`: version, tune, process switches, seed
- `cross_sections`: `sigmaGen` and the per-process breakdown
- `git` (commit and dirty state), `timestamp`, `content_sha256`

`event` is kept so the within-interaction structure survives. Per-arm
multiplicity, multi-proton events, and the observed both-arm occupancy remain
directly measurable even though no same-interaction pair table is written.

## Library

### `flux.py`

```
Acceptance(windows)                     xi windows; mask(xi)
Resolution(xi_sigma=0.0003, seed=12345) applied at use time

ProtonFlux.load(path)                   protons + denominators
  .arm_statistics(acceptance)           -> multiplicity, occupancy, joint 2x2
  .log_xi_intensity(acceptance, res, bins)
                                        -> edges, f_left, f_right

PairDensity(flux, acceptance, resolution, bins)
  .density(mx, yx)                      -> pair intensity in (m_X, y_X)
  .integrate(mass_range, yx_range)      -> intensity in a rectangle
  .marginal_mass(mass_bins, yx_range)   -> m_X template
  .conditional_yx(mx, yx_range)         -> normalised density over y_X
  .expected_pairs_fixed_n(N, mass_range, yx_range)
                                        -> N(N-1) times pair intensity
  .expected_pairs_poisson(mu, mass_range, yx_range)
                                        -> mu^2 times pair intensity
```

**The package knows nothing about jets, mass windows or rapidity bands.** It
exposes p(m_X, y_X) and integrals over it; the caller supplies the ranges.
Nothing in `config.yaml` fixes 117-133 GeV or a rapidity half-width.

The single-arm densities integrate to the mean proton multiplicity per inelastic
interaction. Their product is the candidate intensity per ordered pair of
distinct interactions. The package does not turn that intensity into an
event-level probability.

The pair density is computed in `u = ln xi`, where the transform to physical
coordinates is linear with constant Jacobian:

```
u_L + u_R = 2 ln(m_X / sqrt(s))        u_R - u_L = 2 y_X
```

so **the m_X distribution is the convolution of the two ln-xi intensities**
(one FFT, O(n log n), no pair enumeration at any statistics), and the y_X
distribution at fixed m_X is their profile along the anti-diagonal. The disjoint
acceptance windows are handled automatically — all window pairings (420x420,
420x220, ...) appear in the convolution.

Acceptance is applied to truth xi, followed by one reproducible resolution smear
per proton. The physical density includes the `2 / m_X` Jacobian. There is never
a two-dimensional convolution.

## Generation request

| | |
|---|---|
| target | 10^8 inelastic interactions |
| processes | `SoftQCD:inelastic` — nonDiffractive, SD, DD, CD; **elastic off** |
| filter | write the event if any final-state proton has 0.001 < xi < 0.25 |
| expected output | ~8.7M protons, ~130 MB, 8.7% of events written |

Also generate **10^5 events with `--no-filter`** as a control, to validate that
the filter selects exactly the subset it should.

Do not bias the process mix. It would buy ~5x CPU that is not needed, and the
estimate says single diffraction accounts for only about two thirds of accepted
protons (sigma_SD/sigma_inel ~ 0.18 times a ~20% acceptance fraction gives 0.036
against the observed 0.054 per inelastic interaction), so the rest would have to
be measured and added back. Record `process` per proton anyway — it is one
column, it validates that breakdown, and it is what would be needed to bias
safely at 10^9.

## Verification

Checks 1-2 run on a same-seed filtered/unfiltered 100k pilot. The old sample is
used only for an optional, non-gating shape diagnostic.

1. **Generation.** Denominators, process totals, elastic-off settings, schemas,
   hashes, and dense written-event IDs are consistent.
2. **Filter control.** Applying the loose xi filter offline to the same-seed
   unfiltered artifact exactly reproduces the filtered rows and event grouping.
3. **Flux.** Arm multiplicities, occupancies, and zero-resolution intensity
   integrals reproduce direct counting; nonzero smearing is deterministic.
4. **Pair closure.** Generic rectangle integrals and the FFT mass marginal agree
   with direct calculations within discretisation tolerance, principally in the
   117-133 GeV window.

## Deferred downstream work

Vertex matching, trigger rates, event-level at-least-one probabilities, pair
selection, and MVA migration are intentionally deferred. The legacy analyzer
and `proton_pairs.parquet` remain unchanged as diagnostic references.

## Phases

| phase | content | gate |
|---|---|---|
| 0 | scaffolding, config, schema; same-seed 10^5 filtered/unfiltered pilot | checks 1-2 |
| 1 | `flux.py` and pair-density closure | checks 3-4 |
| 2 | **launch the 10^8 generation** | shard validation |
| 3 | merge, re-run every check, publish the artifact | all current checks |
| 4 | vertex, trigger-rate, and MVA work | deferred |

The full generation starts only after the pilot, API checks, and production
manifest dry run pass.

## Assumptions recorded

Conventions rather than measurements, listed so they stay auditable.

- **mu = 200 counts inelastic collisions**, and the Delphes pileup file is taken
  to match. If that ever needs verifying, the test is to count how many of the
  ~200 pileup vertices in one Delphes file have at least one charged particle in
  the tracker: inelastic-only gives essentially 200, `SoftQCD:all` gives ~78%.
- **The xi resolution is flat at 0.0003** across all four stations.
- **PPS acceptance depends on xi alone.** No t acceptance, which is adequate
  while proton pT remains exploratory.
- **The 10^8-interaction target is fixed independently of a precision claim.**
  Its statistical uncertainty should not be confused with the accuracy of
  Pythia's diffractive model.
- `process` is recorded per proton but not used. It exists so the process
  composition of accepted protons can be checked later without regenerating.
