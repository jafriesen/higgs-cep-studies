# FPMC generation

FPMC processes and campaigns are defined in `processes-fpmc.yaml`. Generation
outputs use:

```text
output-fpmc/<process>/<campaign>/gen-FPMC/
├── cards/
├── condor/
├── evrecs/
├── logs/
└── metadata.yaml
```

Campaign names are required but may be absent from the configuration.
Hadronization is selected at runtime with `--hadr Y|N` and defaults to `Y`.
Campaign names do not set this value automatically: a campaign containing
`_HadrN` still requires `--hadr N`.

Configured campaigns record downstream relationships. For example:

```yaml
Hbb:
  default_campaign:
    main: Hbb_HadrN__v01
    parton-pythia: Hbb__v01
    hadr-pythia: Hbb_FSR__v01
    sim-delphes: Hbb_FSR__v01
  campaigns:
    Hbb_HadrN__v01:
      parton-pythia: [Hbb__v01]
      hadr-pythia: [Hbb_noFSR__v01, Hbb_FSR__v01]
      sim-delphes: [Hbb_noFSR__v01, Hbb_FSR__v01]
```

## Run locally

From the repository root:

```bash
generation-fpmc/scripts/run_fpmc.sh \
  --process QCDbb \
  --campaign QCDbb_HadrN__v01 \
  --hadr N \
  --nev 1000 \
  --seed 33799
```

The script loads the legacy FPMC runtime in a clean subprocess. `--job N`
appends the job number to output names and, unless `--seed` is supplied, uses
seed `33799 + N - 1`. `--dry-run` prints the resolved card and paths without
running FPMC. A campaign cannot mix HADR modes after its metadata records one;
use `--overwrite` to replace the campaign generation outputs with another
mode.

## Submit to Condor

```bash
generation-fpmc/scripts/submit_fpmc_condor.sh \
  --process QCDbb \
  --campaign QCDbb_HadrN__v01 \
  --hadr N \
  --jobs 100 \
  --nev-per-job 2000
```

Defaults are 100 jobs, 2000 events per job, 2048 MB, one CPU, and at most 50
idle jobs. Workers run from Condor scratch using the shared repository and
FPMC installation, so `/isilon` and CVMFS must be available on worker nodes.
`--dry-run` builds the worker and Condor files without submitting. Existing
generation outputs are protected unless `--overwrite` is supplied.

## Resolve downstream stage paths

Configured stage relationships can be resolved without running downstream
scripts:

```bash
python3 common/path_helper.py generation-stage-root \
  --generator fpmc \
  --process Hbb \
  --campaign Hbb_HadrN__v01 \
  --stage hadr-pythia
```

This uses the configured default `Hbb_FSR__v01`. Pass
`--subcampaign Hbb_noFSR__v01` to select another registered relationship.
Ad hoc main generation campaigns are allowed, but downstream stage paths
require a registered campaign and subcampaign relationship.
