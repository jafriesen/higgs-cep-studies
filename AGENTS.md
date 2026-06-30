# AGENTS.md

Preflight for this repo:
1. Read this AGENTS.md before edits or verification.
2. Run analysis commands from the repo root.
3. Before Python analysis checks, run:
   source setup_env.sh

## Working Style

This is a lightweight scientific analysis codebase. Optimize for readable, minimal, fast-enough code over reusable architecture.

Prefer:
- simple, direct implementations
- small diffs
- clear names
- plain functions
- self-contained scripts when reasonable

Avoid:
- premature generalization
- unnecessary classes, factories, plugin systems, or broad configuration
- splitting one readable file into many tiny files
- new dependencies unless clearly justified
- broad refactors unless requested

## Layout

- `signal-generation/`: SuperChic signal production scripts/templates.
- `bkg-generation/`: background and min-bias production scripts.
- `analysis/`: analysis, plotting, and rate-estimate scripts.

Keep changes in the relevant area. Do not move code across areas unless requested.

## Commands

- Run project commands from the repository root unless a README or script says otherwise.
- Most commands in this repo, including Python scripts, require a step to set up the environment.
- Before running analysis, generation, or validation commands, run:

```bash
source setup_env.sh
```

- If a command fails because Python, ROOT, LHAPDF, SuperChic, or another expected dependency is missing, re-run it only after confirming both setup commands were run in the current shell.

## Verification

Prefer the smallest check that exercises the change.

For Python scripts:
- run `python3 -m py_compile path/to/script.py`
- when possible, run the script on a small local input or existing example file

For shell scripts:
- run `bash -n path/to/script.sh`

For environment changes:
- run `source setup_env.sh`
- verify expected variables with `env | rg 'HIGGS_|SUPERCHIC|LHAPDF'`

## Analysis Discipline

Preserve physics meaning over code neatness. Do not change cuts, constants, units, event weights, branch names, histogram definitions, random seeds, or normalization logic unless the task explicitly asks for it.

When changing analysis logic, state what physics quantity changed and how it was verified.

## Safety

Do not edit:
- `.env`
- secrets
- credentials

Do not edit or delete generated analysis artifacts unless explicitly asked:
- `analysis/output/`
- `analysis/examples/*.csv` unless the task is about examples
- `analysis/scripts/*.png`
- `analysis/scripts/**/*.png`
- `analysis/scripts/__pycache__/`
- `signal-generation/output/`
- `signal-generation/logs/`
- `bkg-generation/output/`
- `bkg-generation/logs/`
- `bkg-generation/plots_minbias/`
- `*.root`
- generated SuperChic `evrec*.dat` / `output*.dat`
