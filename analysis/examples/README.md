# analysis/examples

Starter command sequence:

```bash
cd /home/mstamenk/superchic/CMSSW_15_0_0/src/higgs-cep-studies
source setup_env.sh
cd analysis

python3 scripts/build_signal_pairs.py \
  --input ../signal-generation/output/h_bb/hbb_001/outputs \
  --root-out output/hbb_001_pairs.root

python3 scripts/optimize_trigger_rate.py \
  --sig-root output/hbb_001_pairs.root \
  --bkg-root ../bkg-generation/output/minbias/minbias_pairs.root \
  --csv-out examples/rate_scan.csv

python3 scripts/compare_sig_bkg_pps.py \
  --sig-root ../signal-generation/output/h_bb/hbb_001/evrecs/evrec_hbb_001.root \
  --bkg-root ../bkg-generation/output/qcd_bb/qcd_001/evrecs/evrec_qcd_001.root \
  --out-prefix examples/comp_hbb_vs_qcd
```
