# signal-generation

This directory contains SuperChic-side MC generation tools.

## What is here

- `templates/`
  - `h_bb_template.DAT` — H→bb signal card template for SuperChic.
  - `qcd_bb_template.DAT` — QCD bb background template.
- `scripts/run_superchic_signal.sh` — wrapper to generate a single SuperChic job.
- `output/` and `logs/` for local outputs.

## Setup

From repo root, source the common environment first:

```bash
source ../setup_env.sh
```

## Run a single H→bb job

```bash
cd signal-generation/scripts
./run_superchic_signal.sh --nev 10000 --seed 1001 --out-tag hbb_001 1 \
  --template ../templates/h_bb_template.DAT
```

Arguments:

- `--template` card template in this directory (default: `h_bb_template.DAT`)
- `--nev` number of generated events
- `--seed` RNG seed
- `--out-tag` output tag used in `evrec_*.root`
- `<job_index>` required positional argument

## Run a QCD bb card

```bash
./run_superchic_signal.sh --nev 10000 --seed 2025 --out-tag qcd_001 1 \
  --template ../templates/qcd_bb_template.DAT
```

The script copies the template to a temporary work directory, substitutes `@SEED@`, `@NEVT@`, and `@OUT@`, then runs SuperChic and stores produced files under `signal-generation/output/<out-tag>/`.
