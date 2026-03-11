# Higgs CEP Studies (CMSSW-friendly student layout)

This folder is prepared to run quickly in the CMSSW release already present in this tree:
`CMSSW_15_0_0` (detected from the checkout path).

It is split into three working areas:

- `signal-generation` — run SuperChic signal (and QCD background) jobs.
- `bkg-generation` — run Pythia8 min-bias proton production for combinatorial background studies.
- `analysis` — scripts to compare signal vs background and estimate trigger-rate working points.

## 1) Global setup

From the machine where CMSSW and SuperChic are available:

```bash
cd CMSSW_15_0_0/src
source /cvmfs/cms.cern.ch/cmsset_default.sh
cmsenv

cd SuperChic
source env_setup.sh

cd ../higgs-cep-studies
source setup_env.sh
```

`setup_env.sh` also exports:
- `HIGGS_SIGNAL_DIR`
- `HIGGS_BKG_DIR`
- `HIGGS_ANALYSIS_DIR`

## 2) Run flow (suggested)

1. Generate signal using `signal-generation/scripts/run_superchic_signal.sh`
2. Generate Pythia8 min-bias samples using `bkg-generation/scripts/run_pythia8_minbias_study.sh`
3. Run analysis scripts under `analysis/scripts/`.

## 3) Example one-line workflow

```bash
# 1) signal
cd signal-generation/scripts
./run_superchic_signal.sh --nev 10000 --seed 1001 --out-tag hbb_001 1

# 2) background
cd ../../bkg-generation/scripts
./run_pythia8_minbias_study.sh --n-bx 1000 --mu 200 --label minbias

# 3) rate scan
cd ../../analysis
python3 scripts/optimize_trigger_rate.py \
  --sig-root ../signal-generation/output/hbb_001/evrec_hbb_001*.root \
  --bkg-root ../bkg-generation/output/minbias/minbias_pairs.root \
  --top-n 10
```

Paths in these examples match the defaults of the helper scripts in this repository.
