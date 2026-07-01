# Higgs CEP Studies 

This repository is intentionally lightweight: it does **not** vendor `SuperChic`.

It is organized in three clean work areas:

- `signal-generation` - SuperChic signal production (`h_bb`).
- `bkg-generation` - background production (`qcd_bb` via SuperChic + Pythia8 min-bias study).
- `analysis` - reusable analysis and rate-estimate scripts.

The CMSSW release used here is:
- `CMSSW_15_0_0`

## 1) One-time environment setup

```bash
source /cvmfs/cms.cern.ch/cmsset_default.sh
cmsrel CMSSW_15_0_0
cd CMSSW_15_0_0/src
cmsenv
```

If `CMSSW_15_0_0` already exists on your machine, reuse it.

## 2) Download and build SuperChic (online)

```bash
cd CMSSW_15_0_0/src
git clone https://github.com/LucianHL/SuperChic.git
cd SuperChic

# Ensure CMSSW runtime is active in the current shell
cmsenv

# In CMSSW, pass LHAPDF explicitly so CMake can find include/lib/data paths
LHAPDF_PREFIX="$(lhapdf-config --prefix)"
cmake -S . -B build \
  -DCMAKE_INSTALL_PREFIX="$PWD/install" \
  -DLHAPDF_DIR="$LHAPDF_PREFIX"
cmake --build build -j 8
cmake --install build
if [[ -f env_setup.sh ]]; then
  source env_setup.sh
fi
```

If CMake reports:

`Could NOT find LHAPDF (missing: LHAPDF_INCLUDE_DIR LHAPDF_PATH LHAPDF_DATA_PATH)`

check that `lhapdf-config` is visible (`which lhapdf-config`) and rerun with `-DLHAPDF_DIR="$(lhapdf-config --prefix)"` as above.

## 3) Setup this repo

```bash
cd CMSSW_15_0_0/src/higgs-cep-studies
source setup_env.sh
```

`setup_env.sh` remains the interactive setup for the modern SuperChic,
Pythia, Delphes, and analysis environment. Individual runners use the narrower
tool setup scripts under `env/`.

FPMC uses an incompatible legacy GCC runtime and must be loaded separately in
a fresh shell:

```bash
source env/setup_fpmc.sh
```

The FPMC runner isolates this environment automatically. See
`generation-fpmc/README.md` for configuration and campaign examples.

## 4) Typical workflow

1. Generate events with SuperChic or FPMC.
2. Process events with Pythia and Delphes as required.
3. Run scans and plots in `analysis`.

## 5) Minimal command examples

```bash
# A) Signal (SuperChic h_bb)
cd signal-generation/scripts
./run_superchic_signal.sh --process h_bb --nev 10000 --seed 1001 --out-tag hbb_001 1

# B) Background (SuperChic qcd_bb)
cd ../../bkg-generation/scripts
./run_superchic_qcd.sh --nev 10000 --seed 2001 --out-tag qcd_001 1

# C) Pythia8 min-bias study (combinatorial)
./run_pythia8_minbias_study.sh --n-bx 1000 --mu 200 --label minbias

# D) Rate estimate (reused analysis script)
cd ../../analysis
python3 scripts/build_signal_pairs.py \
  --input ../signal-generation/output/h_bb/hbb_001/outputs \
  --root-out output/hbb_001_pairs.root

python3 scripts/optimize_trigger_rate.py \
  --sig-root output/hbb_001_pairs.root \
  --bkg-root ../bkg-generation/output/minbias/minbias_pairs.root \
  --csv-out examples/rate_scan.csv

# E) Optional shape comparison (h_bb vs qcd_bb)
python3 scripts/compare_sig_bkg_pps.py \
  --sig-root ../signal-generation/output/h_bb/hbb_001/evrecs/evrec_hbb_001.root \
  --bkg-root ../bkg-generation/output/qcd_bb/qcd_001/evrecs/evrec_qcd_001.root \
  --out-prefix examples/comp_hbb_vs_qcd
```

See local READMEs in each subdirectory for details.
