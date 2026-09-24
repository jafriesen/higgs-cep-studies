# Paper plots

This directory contains the dedicated `mplhep` figures used by the Higgs CEP
paper.  They use the `mplhep.style.plothist` style.  The scripts reuse the
reduced MVA arrays and mass templates already in the repository; they do not
train classifiers or generate events.

Run from the repository root after loading the analysis environment:

```bash
source setup_env.sh
python3 paper-plots/scan_trigger_timing.py
python3 paper-plots/delta_y_histograms.py
python3 paper-plots/make_plots.py
```

`scan_trigger_timing.py` evaluates only timing points without a cached report.
The established 3 and 10 ps reports in `output/` are reused.  New reports are
written to `paper-plots/data/trigger-timing/`.  The new grid points use
100,000 pseudo-bunch crossings; their statistical uncertainties are shown in
the trigger figure.

`delta_y_histograms.py` writes the proton--dijet rapidity-difference
histograms before the $|\Delta y|$ cut to `paper-plots/data/delta_y_histograms.npz`;
it rereads the SuperChic Delphes files, which the reduced dataset cannot replace.

`make_plots.py` writes PDF and PNG versions to `paper-plots/output/`, together
with `plot_summary.json` containing the plotted trigger rates and sensitivity
values.
