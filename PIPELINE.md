# Garmin Analysis Pipeline

Quick reference for running the analysis workflow.

## Quick Start

```bash
# Run full pipeline (fit model → predictions → plots → reports)
make pipeline

# Or run just the predictions and reports (faster, no model fitting)
make predictions plots reports
```

## Available Commands

### Main Workflows

| Command | Purpose |
|---------|---------|
| `make pipeline` | Full pipeline: fit base model, generate predictions, plots, reports |
| `make pipeline-base` | Same as `pipeline` |
| `make pipeline-trend` | Fit trend model variant instead of base |

### Individual Steps

```bash
# Model Fitting
make fit-model              # Fit base constrained AR model (Stan)
make fit-trend              # Fit trend model variant

# Generate Outputs
make predictions            # Generate component predictions from model
make plots                  # Regenerate visualizations with proper scaling
make reports                # Generate all HTML reports
make report-base            # Generate base model report only
make report-trend           # Generate trend model report only

# Utilities
make metadata               # Extract posterior metadata from base model
make metadata-trend         # Generate trend model metadata from comparison results
make deploy                 # Deploy reports (same as reports)
make status                 # Show project status overview
```

### Cleanup

```bash
make clean-cache            # Clear CmdStanPy cache (~/.cmdstanpy)
make clean-plots            # Remove regenerated PNG files
make clean-output           # Archive old output/ experiment directories
make clean-all              # Full cleanup (all of above)
```

## Workflow Examples

### Scenario 1: Update plots & reports (no model fitting)
```bash
make plots reports
```
**Use when:** You've edited the plotting code or report templates but don't need to refit the model.
**Time:** ~2 minutes

### Scenario 2: Full analysis with base model
```bash
make pipeline-base
```
**Use when:** Starting fresh or after updating the base Stan model.
**Time:** 30-60 minutes (depends on sampling)

### Scenario 3: Compare base vs trend model
```bash
make fit-base reports
make fit-trend report-trend
```
**Use when:** Investigating trend confounding in fitness effects.
**Time:** 60-120 minutes total

### Scenario 4: Quick iteration on reports only
```bash
make report-base report-trend
```
**Use when:** Reports already exist but you're tweaking captions/interpretations.
**Time:** <1 minute

## Project Structure

```
garmin-analysis-v2/
├── Makefile                    ← You are here (orchestrator)
├── PIPELINE.md                 ← This file
├── CLAUDE.md                   ← Project guidelines
│
├── stan/                       ← Stan models
│   ├── weight_state_space_training_decay_aerobic_symmetric_student_ar_spline_constrained.stan
│   └── weight_state_space_training_decay_aerobic_symmetric_student_ar_spline_constrained_trend.stan
│
├── src/analysis/               ← Python analysis scripts
│   ├── run_constrained_ar_model.py          (fit base model)
│   ├── run_trend_comparison.py              (fit trend model)
│   ├── generate_component_predictions.py    (posteriors → CSVs)
│   ├── regenerate_plots.py                  (CSVs → PNGs)
│   ├── generate_model_report.py             (CSVs → HTML report)
│   ├── generate_trend_report.py             (CSVs → HTML report)
│   └── extract_posterior_metadata.py        (model → JSON metadata)
│
├── data/                       ← Garmin export data (gitignored)
│   └── DI_CONNECT/
│
├── output/                     ← Stan model outputs
│   ├── constrained_ar_spline_current/      (current model run)
│   ├── trend_model_current/                (trend model run)
│   └── archive/                            (old experiments)
│
└── docs/                       ← Published reports
    ├── constrained_ar_model_report/        (main report)
    ├── trend_model_report/                 (trend model report)
    ├── component_predictions/              (visualizations & CSVs)
    └── index.html                          (landing page)
```

## Current Models

### Base Model: Constrained AR(1) Spline
- **File:** `stan/weight_state_space_training_decay_aerobic_symmetric_student_ar_spline_constrained.stan`
- **Key Feature:** AR(1) coefficient ρ constrained to [-0.5, 0.5] to prevent overfitting
- **Components:** Intercept + Strength Fitness + Aerobic Fitness + Daily Spline + AR(1) Error
- **Likelihood:** Student-t (robust to outliers)
- **Report:** `docs/constrained_ar_model_report/index.html`

### Trend Model: With Explicit Linear Trend
- **File:** `stan/weight_state_space_training_decay_aerobic_symmetric_student_ar_spline_constrained_trend.stan`
- **Key Feature:** Adds δ parameter for secular weight change
- **Use Case:** Test whether γ_s is confounded with trend
- **Report:** `docs/trend_model_report/index.html`

## Requirements

- Python 3.8+ with `uv` package manager
- CmdStanPy (installed via `pyproject.toml`)
- Garmin data export in `data/DI_CONNECT/` (required for full pipeline)

## Troubleshooting

### "FileNotFoundError: data/DI_CONNECT/..."
You're missing the Garmin export data. Either:
1. Extract the Garmin export to `data/DI_CONNECT/`
2. Run just `make predictions plots reports` (if predictions already exist)

### "Stan model compilation failed"
Usually a typo in the `.stan` file. Check the error message and verify syntax.

### "No matching functions named 'target'"
Stan version mismatch. Ensure CmdStanPy is up to date: `pip install --upgrade cmdstanpy`

### "permission denied" when running make
```bash
chmod +x Makefile   # (shouldn't be needed, but just in case)
```

## Tips

- **Use `make status`** to see what's already computed
- **Chain commands:** `make clean-plots && make plots && make reports`
- **Dry run:** `make -n pipeline` to see what would run
- **Verbose:** `make V=1 pipeline` for more output

## Next Steps

See [CLAUDE.md](CLAUDE.md) for project guidelines and development workflow.
