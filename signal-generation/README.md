# signal-generation

Signal-generation instructions using an **external** SuperChic checkout.

## Contents

- `scripts/run_superchic_signal.sh` - runs one SuperChic job (`h_bb` or `qcd_bb`) using cards from `${SUPERCHIC_DIR}`.
- `output/` - produced files, grouped by process/tag.
- `logs/` - per-job logs.

No SuperChic source/cards are stored in this repo.

## Setup

```bash
cd /home/mstamenk/superchic/CMSSW_15_0_0/src/higgs-cep-studies
source setup_env.sh
```

## Run `h_bb` signal

```bash
cd signal-generation/scripts
./run_superchic_signal.sh --process h_bb --nev 10000 --seed 1001 --out-tag hbb_001 1
```

Default card for `h_bb`:
- `${SUPERCHIC_DIR}/bin/h_bb/h_bb.DAT`

The wrapper patches these entries in a temporary card copy:
- `[outtg]`
- `[iseed]`
- `[nev]`

Output location:
- `signal-generation/output/h_bb/<out-tag>/`

## Optional: custom card

```bash
./run_superchic_signal.sh --process h_bb --card /path/to/my_card.DAT --out-tag test 1
```
