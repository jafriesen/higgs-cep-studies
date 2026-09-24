# H(cc) versus the QED continuum — findings

Produced by `analysis/MVA_hcc/qed_study/`. Metric throughout is the mass-binned
significance `Z = sqrt(sum_m s^2/(s+b))` over 1 GeV bins in mx from 117 to 133,
at 3000 fb^-1 with 200 pileup. Classifier-only and stat-only, with perfectly
known nominal backgrounds.

Reference point, the nominal five-class workflow (`analysis/MVA_hcc/results/`):
**Z = 0.0426** at its single-cut operating point, **Z = 0.0508** from the
six-category ladder. At that operating point the background is QEDcc 488.4,
MadGraph QCDcc 661.1, exclusive QCDcc 103.7, everything else 19.6, against a
signal of 1.34.

## Headline

1. **QEDcc is very nearly irreducible with central-detector observables.** All 55
   available variables together do no better than `delta_eta_jj` alone. With
   QEDcc as the only background the ceiling is Z = 0.131 against a preselection
   value of 0.113 — a 16% gain, and that is the whole of it. Notably
   `cos_theta_star`, the obvious missing candidate, is `delta_eta_jj` in
   different coordinates.
2. **The existing model already learns the QED separation, and the score throws
   it away.** `log p_Hcc - log p_QED` from the trained model reaches 13.5% QEDcc
   efficiency at 30% signal efficiency, matching a dedicated binary classifier;
   the plug-in score actually cut on reaches 26.5%. The loss is in the
   one-dimensional projection, not in the training.
3. **Recovering it is free and worth 17%.** Merging the two exclusive classes
   (which the model cannot tell apart) and categorising in two score axes with
   the pooled background factorised takes the charm-only Z from **0.0488 to
   0.0571**, against an unmodified `train_model.py` baseline on the same data.
   No new samples, no new variables, no retuning. Swapping nine of the twenty
   features for charm-ranked ones adds a further 6%.
4. **The one observable that would change the problem is the outgoing proton
   pT.** Photon exchange leaves a 0.135 GeV proton against 0.361 GeV for the
   signal. At truth level it cuts QEDcc to 0.7% at 30% signal efficiency where
   the best central variable leaves 13.6%, and it keeps most of that at a 50-100
   MeV resolution. Nothing in the current PPS configuration measures |t|, and
   there is **no central-detector substitute**: the equivalent observable is the
   dijet recoil, which would need the transverse momentum of a 125 GeV system to
   0.24% (section 6).
5. **The MadGraph request is for wider phase space, not just more events.**
   Nearly half the top category's weight sits within 5 GeV of the 30 GeV parton
   pT generation edge of QCDcc__v01.

## 1. Which observables separate H(cc) from the exclusive continuum

`stage1_qed_variables.py`, on 55 features and the SuperChic components only.

The physics is the production angle. gamma-gamma -> ccbar and exclusive
gg -> ccbar are t-channel and forward-peaked (median `delta_eta_jj` 1.80 and
1.84); H -> ccbar is a scalar decay and central (median 0.91). Every
activity, gap and track variable is statistically identical across the three
exclusive processes, because all three are exclusive with the same 200 pileup.

Weighted separation (folded AUC) against QEDcc, and the same measured inside
bins of `delta_eta_jj`:

| variable | separation | conditioned on the angle |
|---|---|---|
| `delta_eta_jj` | 0.7062 | — (the conditioning variable) |
| `cos_theta_star` | 0.7062 | 0.5266 |
| `jet1_pt_over_mjj` | 0.6967 | 0.5183 |
| `jet2_pt_over_mjj` | 0.6883 | 0.5176 |
| `sum_gap_size` | 0.6418 | 0.5147 |
| `minimum_gap_size` | 0.6281 | 0.5123 |
| `dijet_pt` | 0.5871 | 0.5082 |
| all 49 others | < 0.58 | < 0.512 |

Nothing survives the conditioning. The largest residual, `cos_theta_star` at
0.527, exists only because `cos_theta_star` is `|tanh((y1-y2)/2)|` in rapidity
while `delta_eta_jj` is `|eta1-eta2|` in pseudorapidity; their Spearman
correlation is 0.999994. **`cos_theta_star` is not a missing variable** — it is
the same variable in different coordinates, and a tree model cannot use the
difference.

Achievable Z with QEDcc as the only background (held-out threshold in brackets):

| selection | Z | gain |
|---|---|---|
| preselection | 0.1126 | — |
| `delta_eta_jj` alone | 0.1305 (0.1307) | x1.16 |
| the 20 locked features | 0.1300 (0.1294) | x1.15 |
| all 55 features | 0.1298 (0.1299) | x1.15 |

Background efficiency at 30% signal efficiency is 0.1361, 0.1359 and 0.1343 for
those three. Adding exclusive QCDcc changes the numbers but not the conclusion:
0.1017 preselection, 0.1179 with the angle alone, 0.1181 with all 55.

**So the QEDcc-specific MVA work is worth about 16%, and one variable delivers
all of it.** Expanding the feature set for this purpose is not worth doing.

## 2. The proton side

`stage2_proton_pt.py`, truth-level, from the SuperChic LHE records. pT and the
proton azimuth are smeared together, since both would come from the same
measurement.

Photon exchange is coherent and leaves the proton nearly untouched; pomeron
exchange does not. Mean outgoing proton pT: **H(cc) 0.361 GeV, QEDcc 0.135 GeV,
exclusive QCDcc 0.468 GeV**. The proton azimuthal correlation is the classic CEP
spin-parity observable, and the 0++ signal and the QCD continuum trend in
opposite directions in `delta_phi_pp`, so it carries information the pT
magnitude does not.

Weighted separation, and how it survives a finite pT resolution:

| resolution | mean proton pT vs QEDcc | `delta_phi_pp` vs QEDcc | mean pT vs exclusive QCDcc |
|---|---|---|---|
| truth | 0.8868 | 0.6363 | 0.6815 |
| 20 MeV | 0.8835 | 0.6312 | 0.6811 |
| 50 MeV | 0.8691 | 0.6215 | 0.6772 |
| 100 MeV | 0.8300 | 0.6040 | 0.6644 |
| 200 MeV | 0.7340 | 0.5744 | 0.6325 |
| 500 MeV | 0.5805 | 0.5200 | 0.5551 |
| *`delta_eta_jj`, for comparison* | *0.7054* | | *0.7129* |

**A proton pT resolution of about 250 MeV is where this stops beating the best
central-detector variable**, and at 50-100 MeV most of the truth-level power
survives. The azimuthal correlation degrades faster, as it must: the azimuth of
a 0.1 GeV vector is destroyed by a smearing comparable to its length.

QEDcc efficiency at 30% signal efficiency — the stable way to read this, since
it needs no threshold optimisation:

| selection | QEDcc only | QEDcc + exclusive QCDcc |
|---|---|---|
| `delta_eta_jj` alone | 0.1361 | 0.1361 |
| proton pT, truth | 0.0067 | 0.0761 |
| angle + proton pT, truth | 0.0048 | 0.0370 |
| angle + proton pT, 50 MeV | 0.0061 | 0.0434 |
| angle + proton pT, 100 MeV | 0.0117 | 0.0534 |
| + `delta_phi_pp`, truth | 0.0017 | 0.0203 |

Against QEDcc alone that is a factor of **20-30** beyond what the central
detector can do. Against the whole exclusive continuum it is a factor of 3-4,
because exclusive QCDcc is pomeron-mediated like the signal and only the
detailed t-slope separates them.

In mass-binned significance with QEDcc as the only background, held-out
threshold, against a preselection 0.1126: the angle alone gives 0.131, proton pT
at 50 MeV resolution gives 0.42, and angle + pT 0.54. Adding `delta_phi_pp`
gives roughly 0.8, but by then the surviving QEDcc yield is a handful of events
and both the point and held-out estimates swing by tens of percent between runs
— treat the `delta_phi_pp` rows as indicative of a large gain, not as a
measurement. Including exclusive QCDcc, the realistic figure is 0.118 -> 0.173
at 50 MeV, a factor 1.5.

This is exploratory and there is no detector model behind it. The PPS
configuration in `analysis/scripts/new/config.yaml` models xi acceptance and xi
resolution only — there is no |t| acceptance and no |t| resolution anywhere in
the chain — and the min-bias proton-pair pool stores no proton pT, so MadGraph
cannot enter this stage at all. What the numbers establish is that the
observable is worth the cost of a study that models the measurement.

## 3. Score construction and categorisation

`stage3_architecture.py`, on the four charm components, at the production
hyperparameters and the nominal 800 estimators with the full early-stopping set.

**The harness is checked against the real thing.** An unmodified
`analysis/MVA_hcc/train_model.py` run on the same charm-only dataset
(`output/baseline_cc_only/`) gives Z = 0.0405 single-cut and 0.0488 for the
six-category ladder; the harness reproduces 0.0404 and 0.0488 with the same
class structure, and the same band-probability residual 1.229e-05.

| configuration | single cut | 6-bin ladder | 12-bin ladder | best partition |
|---|---|---|---|---|
| four-class (current structure) | 0.0404 | 0.0488 | 0.0499 | 0.0560 |
| three-class (merged continuum) | **0.0440** | **0.0530** | **0.0543** | **0.0571** |

### Merging the two exclusive classes is worth about 9%

The model cannot distinguish exclusive QCDcc from QEDcc — their separation
profiles are identical to three decimal places, and its mean predicted
probabilities on QEDcc rows are 0.291 for `exclusive_QCD` against 0.288 for
`exclusive_QED`. Splitting them spends a softmax output on a distinction that
does not exist. Merging gains 9% at every categorisation, and the physical
yields stay per-component exactly as they do today.

### The single plug-in score is where the QED separation is lost

QEDcc efficiency at 30% signal efficiency, out-of-fold, three-class:

| ranking variable | QEDcc efficiency |
|---|---|
| plug-in score T, what is actually cut on | 0.241 |
| the continuum axis, `log p0 - log p_continuum` | **0.137** |
| the non-exclusive axis, `log p0 - log p_nonexclusive` | 0.346 |

0.137 is the stage 1 ceiling: a dedicated binary Hcc-vs-QEDcc classifier reaches
0.136. **The model has already learned everything there is to learn about QEDcc,
and the one-dimensional projection discards half of it.** T is the correct
likelihood ratio for a counting experiment, but this is a mass-shape fit over
components with different mx spectra, and a single projection of a 3-simplex is
lossy regardless.

### A two-axis partition recovers it

| partition | Z | min pooled n_eff |
|---|---|---|
| single cut | 0.0440 | 296 |
| ladder, 6 bins | 0.0530 | 106 |
| ladder, 12 bins | 0.0543 | 43 |
| grid 6x4, direct count | 0.0538 | 14 |
| grid 6x4, factorised | 0.0564 | 80 |
| grid 8x4, direct count | 0.0545 | 7 |
| **grid 8x4, factorised** | **0.0571** | **51** |

**0.0571 against the 0.0488 the current workflow gets from its ladder is +17%,
and +41% against its single cut.** The continuum axis costs no MadGraph
statistics at all: QEDcc and exclusive QCDcc carry real protons and their own mx,
so that axis can be binned finely for free.

### The pooled background has to be factorised, and it is safe to do so

Spreading the combinatorial background over a two-dimensional partition strands
it — the direct 8x4 count reaches min n_eff = 7, which is not a measurement.
Estimating it as `N(axis-0 bin) x f(axis-1)`, the same treatment the mass
template already gets, lifts min n_eff to 51 and *raises* Z, because the direct
count's empty cells were biasing `s^2/(s+b)` upward where there was no MonteCarlo
and downward where there was.

The independence this assumes is measured, not asserted: the largest deviation
between the factorised estimate and the directly counted two-dimensional yields
is **0.1% of the total pooled yield** in every configuration tried.

## 4. Feature set for the MadGraph axis

`stage4_features.py` and `stage4b_locked_vs_charm.py`, on all 55 features with
MadGraph QCDcc, three-class, ladder metric, judged against the seed spread.

### Twenty is the right count

| features | Z (2 seeds) | seed spread |
|---|---|---|
| 10 | 0.0471 | 0.0001 |
| 15 | 0.0486 | 0.0007 |
| **20** | **0.0520** | 0.0004 |
| 30 | 0.0514 | 0.0001 |
| 55 | 0.0516 | 0.0001 |

The seed spread is small enough to resolve this: 10 and 15 are genuinely worse,
and beyond 20 there is nothing. The current feature count is correct.

### But nine of the twenty are the wrong nine

The locked set came from an H(bb) permutation ranking. Redone for charm, nine
of the twenty locked features fall outside the charm top twenty:

| in the locked set, not in the charm top 20 | in the charm top 20, not locked |
|---|---|
| `interjet_bridge_asymmetry` | `sum_track_pt_outside_jets` |
| `interjet_bridge_asymmetry_projected` | `n_tracks_outside_jets` |
| `minimum_gap_size` | `sum_gap_size` |
| `n_tracks_interjet` | `max_track_pt_outside_jets` |
| `n_tracks_projected_bridge` | `sum_track_pt_interjet` |
| `n_tracks_outer_positive` | `eta_rms_outside` |
| `sum_track_pt_outer_negative` | `jet2_eta` |
| `n_outer_track_jets` | `jet2_pt_over_mjj` |
| `n_vertices` | `dijet_mass_fsr` |

The pattern is consistent: charm prefers the summed-momentum and total-gap forms
of the activity variables over the counting and per-region forms, and wants the
second jet's kinematics that the bb set drops. The top of the charm ranking is
`delta_phi_jj` (+0.033), `yx_minus_dijet_rapidity` (+0.023), `sum_gap_size`
(+0.019), `sum_track_pt_outside_jets` (+0.017), `dijet_mass` (+0.011).

Head to head at identical settings, three seeds each (`stage4b`):

| feature set | n | Z | seed spread |
|---|---|---|---|
| locked 20, from H(bb) | 20 | 0.0492 | 0.0004 |
| **charm-ranked 20** | 20 | **0.0522** | 0.0005 |
| union of both | 29 | 0.0520 | 0.0005 |

**The charm ranking is worth +6.1%, at six times the seed spread — a real
difference, not a fluctuation.** The union of the two sets is no better than the
charm twenty alone, so the nine bb-specific variables genuinely add nothing once
the nine charm ones are present. This is a swap, not an addition.

For perspective against the count scan above: the locked twenty (0.0492)
performs about like the charm-ranked *fifteen* (0.0486). Five variables' worth of
the current set is being spent on the wrong channel.

Note the two `interjet_bridge_asymmetry` variables are among the nine dropped.
They are NaN for 74.5% of signal events and 59.7% of QEDcc events, at
component-dependent rates, so their learned default direction has been acting as
an undeclared proxy for the production angle.

## 5. MonteCarlo support and what to generate

`stage5_mc_support.py`, three-class, six-category ladder, 300 Poisson bootstrap
replicas at the hard-event group level.

### The ladder is not MonteCarlo limited; the tail of a single cut is

| category | Z | effective MadGraph central events |
|---|---|---|
| 0 | 0.0003 | 2 040 907 |
| 1 | 0.0053 | 11 371 |
| 2 | 0.0108 | 2 964 |
| 3 | 0.0179 | 1 019 |
| 4 | 0.0257 | 400 |
| 5 | 0.0389 | 106 |

Bootstrap sigma on the total is **0.0009 on 0.0514, that is 1.8%**, and the
median sits 0.0001 above the point estimate — no meaningful upward bias. The
categorised measurement is sound as it stands. What is fragile is the deep tail
of a *single-cut* scan and any fine two-dimensional partition, which is why the
factorised estimate in section 3 matters.

The top category rests on 173 contributing events for an effective 106, so the
weights are reasonably even (concentration 0.61, largest single event 2.3% of
the category).

### More events is the smaller half of the request

Reaching an effective 200 in the top category needs **1.9x** the current
generation, so about 1.9e7 events against the present 1.0e7.

But the surviving background is pressed against the generated boundary.
QCDcc__v01 is generated with parton pT 30-100 GeV and |eta| < 1.5, and in the
top category:

- **46.7% of the weight sits within 5 GeV of the 30 GeV lower pT edge**, and the
  5th percentile of the reconstructed softer jet pT is 24.0 GeV — below the
  generation cut, reached only by resolution migration from above it. The true
  background with parton pT below 30 GeV is not generated at all.
- the 95th percentile of the larger |eta| is 1.56, against a 1.5 generation cut;
  only 7.4% of the weight is within 0.2 of that edge, so the eta boundary is the
  less urgent of the two.

**The request should therefore be for a wider region, not merely more of the
same**: parton pT down to about 20-25 GeV, matching what QCDbb__v02 already
covers for bottom (pT 25-100, |eta| < 2.5), and roughly twice the statistics in
the existing region.

## 6. Can the proton-pT discrimination be had without measuring proton pT?

`stage6_recoil_proxy.py`. The answer is no, and the reason is quantitative
rather than a matter of taste.

**The proxy exists.** Exclusivity forces the central system to balance the
proton pair, so `p_T(central) = -(p_T1 + p_T2)` — a central-detector quantity
carrying the same information. It does:

| observable | separation vs QEDcc | vs exclusive QCDcc |
|---|---|---|
| individual proton pT (section 2) | 0.8868 | 0.6815 |
| **central-system pT, truth** | **0.8512** | 0.5330 |
| `delta_eta_jj`, for reference | 0.7054 | 0.7129 |

Mean central-system pT: H(cc) 0.592 GeV, exclusive QCDcc 0.646 GeV, **QEDcc
0.219 GeV**.

**The detector destroys it.** The reconstructed `dijet_pt` has a median of
**8.52 GeV** where the true recoil is **0.53 GeV**, and its correlation with the
true recoil is **+0.0022**. It carries none of it. `puppi_met` is the same story
(median 13.48 GeV, correlation +0.0018).

`dijet_pt` does show a separation of 0.587 against QEDcc, which is why it looks
promising in section 1. It is not recoil information: conditioned on
`delta_eta_jj` it falls to 0.508. It is measuring out-of-cone and FSR losses,
which correlate with the production angle.

**How good the measurement would have to be**, smearing each transverse
component of the central-system momentum:

| resolution per component | separation vs QEDcc |
|---|---|
| truth | 0.8512 |
| 0.1 GeV | 0.8227 |
| 0.2 GeV | 0.7618 |
| 0.5 GeV | 0.6200 |
| 1 GeV | 0.5413 |
| 2 GeV | 0.5112 |
| 5 GeV | 0.5024 |

Beating the `delta_eta_jj` ceiling of 0.705 needs **sigma < 0.3 GeV per
component**, that is the transverse momentum of a 125 GeV system to about 0.24%.
Jet energy resolution at these pT is several GeV per jet, an order of magnitude
and a half short, and no tracker-based recoil closes that gap because the
neutral fraction alone fluctuates by GeV.

**The proton longitudinal kinematics carry nothing either.** |yx| separates at
0.512 and mx at 0.529, both against QEDcc — the photon and pomeron fluxes are
near-flat in the ratio direction over the accepted range. Note the model already
has yx implicitly: `dijet_rapidity + yx_minus_dijet_rapidity` reconstructs it
exactly, and both are features.

So the proton pT result of section 2 stands or falls on an actual t-measurement.
There is no central-detector substitute.

## Recommendations for `analysis/MVA_hcc/`

In rough order of value per unit of work.

**1. Merge `exclusive_QCD` and `exclusive_QED` into one training class.** Worth
about 9% at every categorisation, costs nothing, and the physical yields stay
per-component exactly as they are today. The model demonstrably cannot separate
the two, so the split spends a softmax output on a distinction that does not
exist. `Hbb_resonant` should go the same way: it is 0.01% of the background at
the operating point and consumes a fifth of the class-balanced training weight.

**2. Stop cutting on a single plug-in score.** Categorise in two axes — one
against the non-exclusive background, one against the exclusive continuum —
and estimate the pooled background in the product partition as
`N(non-exclusive bin) x f(continuum bin)`. Together with (1) this takes
Z from 0.0488 to 0.0571 on the charm components, **+17%**. The continuum axis
is free of MadGraph statistics, so it can be binned finely.

**3. Swap nine of the twenty features for the charm-ranked ones.** Worth +6.1%
head to head at six times the seed spread. Twenty is the right count — the union
of both sets (29 variables) is no better than the charm twenty alone. See
section 4 for the swap list.

**4. Ask for wider QCDcc generation, not just more of it.** Parton pT down to
about 20-25 GeV, matching QCDbb__v02, plus roughly 2x the statistics in the
existing region. Nearly half the top category's weight is within 5 GeV of the
30 GeV generation edge.

**5. Do not spend more effort on central-detector variables for QEDcc.** The
ceiling is a 16% gain and one variable already delivers it.

**6. Smaller items.**

- `train_model.py:74` hard-asserts exactly 20 features. Read the count from
  metadata instead, or the workflow cannot run on any other feature set.
- The mx window 117-133 is only **+/- 2.05 sigma** (the in-window Hcc mx
  distribution implies sigma ~ 3.9 GeV, wider than the naive xi-resolution
  estimate because sigma(mx)/mx grows with |yx|). Roughly 4% of the signal is
  outside it. Widening it is cheap and the mass-binned metric will use the extra
  bins correctly.
- `interjet_bridge_asymmetry` and `interjet_bridge_asymmetry_projected` are NaN
  for 74.5%/55.5% of signal events and 59.7%/50.7% of QEDcc events. Their
  learned default direction is an undeclared proxy for the production angle.
  Both fall out of the charm ranking anyway.
- Runtime. Three changes in `harness.py` cut a full evaluation from ~14 minutes
  to ~4: read the feature matrix once sequentially into RAM rather than
  fancy-indexing a memmap on every pass; bucket the threshold scan once and take
  a reverse cumulative sum instead of rescanning per cut (`accumulate_above`);
  and optionally cap the pooled early-stopping rows, which are ~90% of a 460k
  set that is re-evaluated every boosting round.
- When restricting components in `prepare_dataset.py`, seeds must be keyed to
  each component's position in `COMPONENT_SPECS`. Keyed to the position in the
  restricted list, a subset build silently selects a different event sample
  (QEDcc came out with 16 075 events instead of 16 186). Fixed in this branch.

## Caveats

- Classifier-only and stat-only throughout, with perfectly known nominal
  backgrounds. `combinatorial_acceptance_factor` stays at its nominal 0.005; it
  is a parameter of a detector that does not exist, not a fitted uncertainty.
- The bb components are excluded from every stage. They are 1.5% of the
  background at the nominal operating point.
- MadGraph QCDcc has only campaign v01 (parton pT 30-100 GeV, |eta| < 1.5,
  m 70-140 GeV). The wider region covered for bb by QCDbb__v02 is unmodelled
  for charm.
- Stage 2 is truth-level and has no detector model.
