# signal-generation

Signal-generation instructions using an **external** SuperChic checkout.

## Contents

- `scripts/run_superchic_signal.sh` - runs one SuperChic job (`h_bb` or `qcd_bb`) using local templates by default.
- `templates/` - default card templates (`h_bb_template.DAT`, `qcd_bb_template.DAT`).
- `output/` - produced files, grouped by process/tag.
- `logs/` - per-job logs.

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
- `signal-generation/templates/h_bb_template.DAT`
- fallback: `${SUPERCHIC_DIR}/bin/h_bb/h_bb.DAT`, then `${SUPERCHIC_DIR}/Cards/h_bb.DAT`

The wrapper patches these entries in a temporary card copy:
- `[outtg]`
- `[iseed]`
- `[nev]`

The wrapper also runs `init` automatically when needed (for the card's
`[rts]`, `[isurv]`, `[intag]`, `[PDFname]`, `[PDFmember]` combination) and
caches the generated `inputs/` files under:
- `signal-generation/workspace/init_cache/`

Output location:
- `signal-generation/output/h_bb/<out-tag>/`

## Optional: custom card

```bash
./run_superchic_signal.sh --process h_bb --card /path/to/my_card.DAT --out-tag test 1
```
