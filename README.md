# Garmin Health Data Analysis

Bayesian analysis of personal health data from Garmin, using Stan for statistical modeling and D3.js for interactive visualization.

## Overview

This project provides tools for:

1. **Bayesian modeling** of health metrics using Stan (weight, sleep, activity, etc.)
2. **Interactive D3.js visualizations** for exploring health trends and relationships
3. **Web interface** for viewing model results and predictions

## Current Status

- [x] Project structure and tooling setup
- [x] Weight data extraction from Garmin export
- [x] Proof-of-concept Gaussian Process model for weight trends
- [ ] Interactive web visualization
- [ ] Additional health metrics (sleep, activity, heart rate)
- [ ] Cross-metric relationship modeling

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for Python package management
- CmdStan (installed automatically on first run)

### Setup

```bash
# Install dependencies
uv sync

# Install CmdStan (if not already installed)
uv run install_cmdstan

# Run the weight analysis
uv run python -m src.models.fit_weight
```

### Data

This project analyzes personal Garmin data from the Garmin Connect export. Place your export in the `data/` directory:

```
data/
└── DI_CONNECT/
    ├── DI-Connect-Wellness/    # Biometrics, sleep, heart rate
    ├── DI-Connect-Fitness/     # Activities, workouts
    ├── DI-Connect-Aggregator/  # Daily summaries
    └── DI-Connect-Metrics/     # VO2 max, training metrics
```

The `data/` directory is gitignored to protect personal health information.

## Project Structure

```
garmin-analysis-v2/
├── data/                    # Garmin export data (gitignored)
├── stan/                    # Stan model files
│   └── weight_gp.stan      # Gaussian Process weight model
├── src/
│   ├── data/               # Data loading utilities
│   │   └── weight.py       # Weight data extraction
│   ├── models/             # Model fitting code
│   │   └── fit_weight.py   # Weight model fitting
│   └── analysis/           # Analysis scripts
├── web/                     # D3.js visualization (future)
├── notebooks/              # Jupyter notebooks for exploration
└── output/                 # Model outputs and plots
```

## Models

### Weight Trend Model (`stan/weight_gp.stan`)

A Gaussian Process model for weight over time that:
- Captures smooth temporal trends
- Quantifies uncertainty in predictions
- Provides posterior predictive checks

## Development

This project uses:
- **uv** for Python package management
- **CmdStanPy** for Stan model compilation and sampling
- **ArviZ** for Bayesian visualization and diagnostics
- **TDD** for development methodology

## License

MIT License - see [LICENSE](LICENSE)
