# analysis

Analysis scripts reused from your existing workflow, with defaults adjusted for this clean layout.

## Scripts

- `scripts/build_signal_pairs.py` - builds a `ProtonPairs` tree from SuperChic signal `.dat/.lhe` outputs.
- `scripts/optimize_trigger_rate.py` - rate working-point scan (signal efficiency vs combinatorial rate).
- `scripts/compare_sig_bkg_pps.py` - compares `h_bb` vs `qcd_bb` `Events` trees.

## Setup

```bash
cd /home/mstamenk/superchic/CMSSW_15_0_0/src/higgs-cep-studies
source setup_env.sh
```

## A) Build signal pairs + run rate scan

```bash
cd analysis

python3 scripts/build_signal_pairs.py \
  --input ../signal-generation/output/h_bb/hbb_001/outputs \
  --root-out output/hbb_001_pairs.root

python3 scripts/optimize_trigger_rate.py \
  --sig-root output/hbb_001_pairs.root \
  --bkg-root ../bkg-generation/output/minbias/minbias_pairs.root \
  --csv-out examples/rate_scan.csv
```

## B) Compare signal vs QCD (shape-level)

```bash
python3 scripts/compare_sig_bkg_pps.py \
  --sig-root ../signal-generation/output/h_bb/hbb_001/evrecs/evrec_hbb_001.root \
  --bkg-root ../bkg-generation/output/qcd_bb/qcd_001/evrecs/evrec_qcd_001.root \
  --out-prefix examples/comp_hbb_vs_qcd
```
