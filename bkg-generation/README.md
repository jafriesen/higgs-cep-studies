# bkg-generation

This directory contains the Pythia8 workflow for combinatorial background studies.

## Scripts

- `run_pythia8_minbias_study.sh` — main runner for generating min-bias protons and writing pair ROOT.
- `generate_minbias_protons.py` — copied from the project source; generates final-state protons into `.npz`.
- `build_minbias_pairs.py` — copied converter from `.npz` to `ProtonPairs` ROOT tree.
- `analyze_minbias_protons.py` — proton-level quality plots from `.npz`.
- `quick_check_minbias.py` — quick distribution sanity checks.

## Run

```bash
cd bkg-generation/scripts
source ../../setup_env.sh
./run_pythia8_minbias_study.sh --n-bx 1000 --mu 200 --label minbias
```

Default outputs:

- `bkg-generation/output/minbias/minbias.npz`
- `bkg-generation/output/minbias/minbias_pairs.root`
