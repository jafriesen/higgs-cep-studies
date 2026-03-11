# analysis/examples

Small starting commands for students:

```bash
# generate a quick rate table and save to csv
cd analysis
python3 scripts/optimize_trigger_rate.py \
  --sig-root ../signal-generation/output/hbb_001/evrec_hbb_001*.root \
  --bkg-root ../bkg-generation/output/minbias/minbias_pairs.root \
  --csv-out examples/rate_scan.csv

# compare one distribution
python3 scripts/compare_sig_bkg_pps.py \
  --sig-root ../signal-generation/output/hbb_001/evrec_hbb_001*.root \
  --bkg-root ../bkg-generation/output/minbias/minbias_pairs.root \
  --out-prefix examples/comp_hbb_vs_minbias
```
