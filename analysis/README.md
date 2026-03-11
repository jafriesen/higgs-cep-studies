# analysis

This folder contains the analysis scripts used for the CEP study comparisons and trigger-rate estimates.

## Reused scripts

- `optimize_trigger_rate.py` — scans timing / kinematics windows and prints target working points.
- `compare_sig_bkg_pps.py` — compares kinematic shapes between signal and background trees.

## Example usage

From repo root (after running `source setup_env.sh`):

```bash
cd analysis

# Compare signal and background
python3 scripts/compare_sig_bkg_pps.py \
  --sig-root ../signal-generation/output/hbb_001/evrec_hbb_001*.root \
  --bkg-root ../bkg-generation/output/minbias/minbias_pairs.root \
  --sig-xsec-pb 5.708103e-03 --bkg-xsec-pb 8.997422e-01 \
  --lumi-ab 3.0

# Optimize rate scan
python3 scripts/optimize_trigger_rate.py \
  --sig-root ../signal-generation/output/hbb_001/evrec_hbb_001*.root \
  --bkg-root ../bkg-generation/output/minbias/minbias_pairs.root \
  --top-n 10
```

`optimize_trigger_rate.py` is designed to run against:
- `ProtonPairs` tree in the signal and background ROOT files
- station-tag columns `tag200_L`, `tag200_R`, `tag400_L`, `tag400_R`

## Next steps

1. Tune `--sig-root` and `--bkg-root` for the latest generated files.
2. Save CSV outputs with `--csv-out` for reproducibility.
3. Store plots under your own versioned directory in `analysis/plots/`.
