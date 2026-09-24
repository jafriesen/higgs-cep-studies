# Where the MadGraph combinatorial background actually lives

Produced by `analysis/MVA_hcc/lowpt_pilot/run_pilot.py`. The question: the QCD
samples are generated inside boxes in parton pT, |eta| and m(jj), while the
analysis selection asks only for two jets above 15 GeV, roughly back to back,
plus a proton pair. How much background falls outside the boxes?

QCDbb spans pT 15–100, |eta| < 3.0, m 50–180 across its three campaigns, so the
answer can be measured rather than extrapolated. QCDcc has one campaign,
pT 30–100, |eta| < 1.5, m 70–140 — identical to `QCDbb__v03`.

Everything below is at **parton level**, scored through the proton-pool grid, on
a dataset stitching all three bb campaigns (23.97M rows, 6.13M groups).

## Headline

**The cc generation box contains 18% of the surviving background.** Extrapolating
to cc, the combinatorial background is **2.9x to 5.6x** larger than currently
estimated, depending on whether the classifier is retrained on the wider region.

| | narrow-trained | wide-trained |
|---|---|---|
| QCDbb surviving yield | 4.768 | **2.329** |
| share inside the cc box | 17.9% | 33.9% |
| implied cc background | 1490 (x5.6) | **783 (x2.95)** |

"narrow-trained" restricts the non-exclusive training class to the current cc box
and then scores everything — the model the analysis actually uses, applied to
events it has never seen. "wide-trained" trains on the full region.

## Most of the excess is a training artifact, not physics

Share of surviving background, and survival relative to the most central bin:

| parton max \|eta\| | presel | survivors (narrow) | rel | survivors (wide) | rel |
|---|---|---|---|---|---|
| 0.0–1.0 | 0.344 | 0.197 | 1.0 | 0.404 | 1.0 |
| 1.0–1.5 | 0.326 | 0.074 | 0.4 | 0.131 | 0.3 |
| 1.5–2.0 | 0.215 | 0.312 | 2.5 | 0.317 | 1.3 |
| 2.0–2.5 | 0.089 | 0.221 | 4.3 | 0.148 | 1.4 |
| 2.5–3.0 | 0.025 | **0.195** | **13.4** | **0.000** | **0.0** |

| parton softer pT | presel | survivors (narrow) | survivors (wide) |
|---|---|---|---|
| 15–20 | 0.101 | 0.065 | 0.000 |
| 20–25 | 0.210 | 0.130 | 0.002 |
| 25–30 | 0.250 | 0.114 | 0.088 |
| 30–40 | 0.280 | 0.256 | 0.194 |
| 50–65 | 0.046 | 0.194 | 0.333 |
| 65–80 | 0.008 | 0.082 | 0.170 |

Under the narrow model the survival **rises monotonically** into the forward
region — 13x the central value at |eta| 2.5–3.0, which is the generation edge —
and pT < 30 supplies 31% of survivors. That looked like an irreducible forward
background still climbing past the boundary.

It is not. Shown those events in training, the classifier rejects them: survival
at |eta| 2.5–3.0 drops to **exactly zero**, the eta dependence flattens to within
a factor 1.4, and pT < 30 falls from 31% to 9% of survivors. Total surviving bb
background halves, 4.768 to 2.329.

**The classifier could always reject these events — `jet1_eta`, `dijet_rapidity`
and `delta_eta_jj` are all inputs. It had no reason to, because its non-exclusive
training class is dominated by QCDcc (1.01e7 against 1.13e5), and QCDcc is
central by construction.**

The mechanism for the forward excess is physical: with the tracker ending at
|eta| = 2.5, a forward dijet has most of its surrounding activity outside the
tracking acceptance, so the gap and track-counting variables see an artificial
rapidity gap. Forward QCD fakes exclusivity — and a model that has never seen it
cannot know.

## The mass window

| parton m(jj) | presel | survivors (narrow) |
|---|---|---|
| 40–70 | 0.409 | **0.000** |
| 70–100 | 0.375 | 0.090 |
| 100–140 | 0.181 | 0.713 |
| 140–180 | 0.036 | 0.197 |

The **lower** bound is irrelevant: m(jj) 40–70 is 41% of the preselected weight
and contributes exactly nothing. The **upper** bound matters — 140–180 gives 20%
of survivors at the highest survival, and the cc box stops at 140. Behaviour
above 180 is unmeasured; no campaign covers it.

## Flavour transfer, and its limit

The extrapolation to cc assumes the outside/inside survivor ratio carries over
from bb. Delphes does not distinguish c from b jets — flavour enters only as a
per-event tag weight — so the check is direct, in the box both flavours cover:

| | bb x tag x sigma | cc | ratio |
|---|---|---|---|
| preselection | 9.474e6 | 1.014e7 | **1.07** |
| survivors (narrow) | 122.9 | 266.2 | 2.17 |
| survivors (wide) | 114.0 | 265.3 | 2.33 |

**Preselection transfers to 7%.** Survivors do not: c-flavoured events survive
the MVA ~2.2x more than b-flavoured ones in the same box. b jets carry more
tracks and more jet mass, and the model is trained with cc dominating, so it is
tuned to the cc distribution.

The extrapolation uses only the *ratio* of outside-box to inside-box survivors,
which is unaffected if the flavour difference is a constant factor. That it sits
at 2.17 and 2.33 under two very different models is mildly reassuring, but it is
not proof of kinematic independence. **This is the main caveat on the 2.9x–5.6x
numbers, and only a wide-eta cc sample can remove it.**

## Consequences for the generation

1. **|eta| < 3.0, not 1.5.** This is the dominant axis. Under the narrow model
   73% of survivors are at |eta| > 1.5. |eta| < 3.0 is sufficient *provided the
   model is trained on it* — the wide-trained survival is exactly zero at 2.5–3.0.
2. **Parton pT down to ~20.** Needed for the estimate under the current model
   (31% of survivors) and, more importantly, needed **in training** so the
   classifier learns to reject it. Once trained, pT < 30 falls to 9%.
3. **Drop the lower mass cut** (it contributes nothing) and **raise the upper one
   above 180**, which is currently unmeasured.
4. **Retraining is not optional.** Generating wider without retraining leaves the
   x5.6 in place; retraining on the wider sample is what converts it to x2.95.

## Caveats

- The 2.5–3.0 |eta| bin rests on 4,846 central events. The narrow-model value
  there is thin; the wide-model zero is robust in direction but not in precision.
- Nothing above parton m(jj) = 180 or |eta| = 3.0 is generated by any campaign.
- The bb sample carries the mistag suppression, so its own contribution to H(cc)
  is negligible; it is used here purely as a kinematic probe.
- Stat-only, classifier-only, and independent of the two normalisation
  corrections identified separately (the elastic denominator, 1.59, and the
  vertex acceptance, 1.52).
