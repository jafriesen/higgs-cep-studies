# SuperChic generation

SuperChic processes and campaigns are defined in `processes-superchic.yaml`.
All processes use `cards/template.DAT`; the run scripts replace `[intag]`,
`[proc]`, `[outtg]`, `[iseed]`, and `[nev]` from the process configuration and
runtime arguments.

Generation outputs use:

```text
output-superchic/<process>/<campaign>/gen-SuperChic/
├── cards/
├── condor/
├── evrecs/
├── init/
├── logs/
├── output/
└── metadata.yaml
```

## Initialize and run locally

```bash
generation-superchic/scripts/run_superchic.sh \
  --process Hbb \
  --campaign Hbb__v01 \
  --nev 1000 \
  --seed 1001 \
  --init
```

Campaign names are required but may be absent from the configuration.
`--card FILE` selects another complete SuperChic card as the template; the
same five tagged fields are still replaced. Initialization is reused when its
energy, survival model, input tag, PDF name, and PDF member match.

With `--job N`, the default seed is `1001 + N - 1`.

## Submit to Condor

```bash
generation-superchic/scripts/submit_superchic_condor.sh \
  --process Hbb \
  --campaign Hbb__v01 \
  --jobs 100 \
  --nev-per-job 2000 \
  --init
```

Defaults are 100 jobs, 2000 events per job, 2048 MB, one CPU, and at most 50
idle jobs. Workers run from Condor scratch using the shared repository,
SuperChic installation, and initialized inputs, so `/isilon` and CVMFS must
be available on worker nodes. `--card FILE` is also supported for batch jobs.
`--dry-run` builds the worker and Condor files without submitting.
`--overwrite` clears non-initialization generation outputs and preserves
initialized inputs.
