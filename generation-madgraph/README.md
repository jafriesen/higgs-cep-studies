# MadGraph generation

This workflow generates non-CEP heavy-flavor dijets as parton-level LHE
events. Processes and campaigns are defined in `processes-madgraph.yaml`.
Outputs use:

```text
output-madgraph/<process>/<campaign>/gen-MadGraph/
├── cards/
├── condor/
├── evrecs/
├── init/
├── logs/
└── metadata.yaml
```

The runtime is MadGraph 3.6.4 from the LCG 110 view. The initialized cards use
`NNPDF31_lo_as_0118` (LHAPDF ID 315000) and 7 TeV proton beams. The generated
campaigns have different parton-level phase spaces:

| Campaign | Parton pT (GeV) | max abs(eta) | Dijet mass (GeV) |
| --- | ---: | ---: | ---: |
| `QCDbb__v01` | 15--80 | 3.0 | 50--160 |
| `QCDbb__v02` | 25--100 | 2.5 | 70--180 |
| `QCDbb__v03` | 30--100 | 1.5 | 70--140 |
| `QCDcc__v01` | 4--unbounded | 2.4 | 90--160 |

These values are read back from `init/run_card.dat` when writing metadata,
rather than being duplicated in the submission scripts.

The configured matrix elements are:

```text
p p > b b~
p p > c c~ QCD<=2 QED<=2
```

The coupling-order bounds on the charm process retain the QCD and
electroweak tree diagrams and their interference; they do not restrict that
sample to pure QCD.
MadGraph's default proton definition is retained:
`p = g u c d s u~ c~ d~ s~`. Charm-initiated channels can therefore
contribute to `ccbar`, while bottom-initiated channels are absent.

## Run locally

Build the reusable gridpack and generate events from the repository root:

```bash
generation-madgraph/scripts/run_madgraph.sh \
  --process QCDbb \
  --campaign QCDbb__v01 \
  --nev 1000 \
  --seed 1001 \
  --init
```

The initialization key covers the process, cuts, beam energy, PDF, and
MadGraph version. Later runs reuse the gridpack. If any keyed setting changes,
run again with `--init` to rebuild it. With `--job N`, the default seed is
`1001 + N - 1`. Existing run outputs are protected unless `--overwrite` is
supplied. `--dry-run` prints the resolved paths and initialization card
without generating files.

Each completed job also saves the `MGRunCard` embedded in its LHE header under
`cards/`. The XML/CDATA wrapper is removed, leaving a directly readable
MadGraph run card.

## Submit to Condor

```bash
generation-madgraph/scripts/submit_madgraph_condor.sh \
  --process QCDbb \
  --campaign QCDbb__v01 \
  --jobs 100 \
  --nev-per-job 2000 \
  --init
```

The defaults are 100 jobs and 2000 events per job. The initialized gridpack
is transferred to each worker, while the LCG 110 view is loaded from CVMFS.
Use `--dry-run` to write the Condor files without submitting.
