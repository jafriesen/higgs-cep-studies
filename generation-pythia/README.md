# bkg-generation

Background generation is split into two pieces:

- SuperChic `qcd_bb` generation.
- Pythia8 min-bias proton study for combinatorial rates.

## Scripts

- `scripts/run_superchic_qcd.sh` - wrapper for one SuperChic `qcd_bb` job.
- `scripts/run_pythia8_minbias_study.sh` - end-to-end min-bias generation (`.npz`) and pair-building (`.root`).
- `scripts/generate_minbias_protons.py` - reused Pythia8 generator script.
- `scripts/build_minbias_pairs.py` - `.npz -> ProtonPairs` converter.
- `scripts/analyze_minbias_protons.py` - station-level acceptance/rate summaries.
- `scripts/quick_check_minbias.py` - quick quantile sanity plots.

## Setup

```bash
cd /home/mstamenk/superchic/CMSSW_15_0_0/src/higgs-cep-studies
source setup_env.sh
```

## Run SuperChic QCD background

```bash
cd bkg-generation/scripts
./run_superchic_qcd.sh --nev 10000 --seed 2001 --out-tag qcd_001 1
```

Output:
- `bkg-generation/output/qcd_bb/<out-tag>/`

## Run Pythia8 min-bias combinatorial study

```bash
./run_pythia8_minbias_study.sh --n-bx 1000 --mu 200 --label minbias
```

For BX-by-BX pileup fluctuations with mean 200 interactions:

```bash
./run_pythia8_minbias_study.sh --n-bx 1000 --mu 200 --mu-mode poisson --label minbias_poisson
```

Outputs:
- `bkg-generation/output/minbias/minbias.npz`
- `bkg-generation/output/minbias/minbias_pairs.root`
