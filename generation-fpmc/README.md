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

Campaign names are required but may be absent from the configuration. An
undeclared campaign uses `HADR 'Y'`. Declare `hadronize: false` in a campaign
to generate a parton-level LHE file with `HADR 'N'`.

## Run locally

From the repository root:

```bash
generation-fpmc/scripts/run_fpmc.sh \
  --process QCDbb \
  --campaign QCDbb_parton \
  --nev 1000 \
  --seed 33799
```

The script loads the legacy FPMC runtime in a clean subprocess. `--job N`
appends the job number to output names and, unless `--seed` is supplied, uses
seed `33799 + N - 1`. `--dry-run` prints the resolved card and paths without
running FPMC.

## Submit to Condor

```bash
generation-fpmc/scripts/submit_fpmc_condor.sh \
  --process QCDbb \
  --campaign QCDbb_parton \
  --jobs 100 \
  --nev-per-job 2000
```

Defaults are 100 jobs, 2000 events per job, 2048 MB, one CPU, and at most 50
idle jobs. Workers run from Condor scratch using the shared repository and
FPMC installation, so `/isilon` and CVMFS must be available on worker nodes.
`--dry-run` builds the worker and Condor files without submitting. Existing
generation outputs are protected unless `--overwrite` is supplied.
