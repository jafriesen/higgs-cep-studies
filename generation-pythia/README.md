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
- `scripts/run_superchic_pythia.sh` - hadronize SuperChic LHE/evrec files with Pythia8 and write HepMC3.
- `scripts/run_processes_pythia.py` - run Pythia for SuperChic campaigns listed in `processes.yaml`.

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

## Run SuperChic events through Pythia8

```bash
./scripts/run_superchic_pythia.sh \
  --input ../generation-superchic/output/h_cc/superchic_h_cc_nev100000_j100_20260618_102604/evrecs \
  --output /tmp/h_cc_pythia.hepmc \
  --max-events 100
```

The script reads SuperChic LHE-style `evrec*.dat` files, disables Pythia beam remnants for CEP events with intact outgoing protons, keeps final-state showering and hadronization on, and writes HepMC3 ASCII output for Delphes.

To run every process default campaign from `processes.yaml` and write into each campaign directory:

```bash
./scripts/run_processes_pythia.py --max-events 100
```

Outputs are written to `<superchic campaign>/GEN-pythia/<process>_<campaign>.hepmc`.
If a process entry in `processes.yaml` defines `max_files`, that file limit is used unless `--max-files` is passed.
