# Model and Report Consolidation Summary

**Date**: 2026-02-28
**Goal**: Focus analysis on the "four state-space fitness" model with baseline fitness equilibrium at 0

## Problem Identified

The repository had become bloated with:
- 25+ Stan models (many outdated/experimental)
- 8+ HTML reports with overlapping content
- Confusing organization with reports referencing different models
- Lack of clear focus on the most advanced model

## Solution Implemented

### 1. Simplified Documentation Structure

**Main documentation index** (`docs/index.html`) now features only 5 essential reports:

1. **Four-Fitness State-Space Analysis** (`docs/four_fitness_comprehensive/`)
   - Primary analysis report
   - Complete parameter estimates and variance decomposition
   - Mathematical foundations and predictions

2. **Fitness Time Series Visualizations** (`docs/fitness_time_series_report/`)
   - Interactive time series plots
   - Four fitness states and weight contributions
   - Aerobic vs strength, short-term vs long-term

3. **Technical Model Documentation** (`docs/four_fitness_educational/`)
   - Complete mathematical details
   - Stan implementation walkthrough
   - Parameter sensitivity analysis

4. **Project Guidelines** (`docs/CLAUDE.html`)
   - Project-specific instructions for AI assistants
   - Tooling requirements and development methodologies

5. **Workout Data Report** (`docs/workout_data_report.html`)
   - Garmin data quality analysis
   - Aggregation methods and modeling recommendations

### 2. Archived Redundant Content

**Moved to `docs/archive/`:**
- `fitness_model_report/` - Duplicate model comparison
- `improved_fitness_report/` - Redundant visualization
- `weight_gp_model_explanation.html` - Outdated focus (GP models)

**Moved to `scripts_archive/`:**
- `create_fitness_model_report.py`
- `create_fitness_model_report_matplotlib.py`
- `create_improved_fitness_report.py`
- `create_simple_four_fitness_report.py`

### 3. Updated Report Focus

**Primary Analysis Report** (`docs/four_fitness_comprehensive/index.html`):
- Updated title: "Four-Fitness State-Space Analysis"
- Emphasizes baseline fitness equilibrium at 0
- Clear analysis focus (not just educational)

**Technical Documentation** (`docs/four_fitness_educational/index.html`):
- Updated title: "Four-Fitness Model Technical Documentation"
- Clear purpose as technical reference

### 4. New Analysis Runner

Created `run_four_fitness_analysis.py`:
- Simple, focused interface for running the four-fitness model
- Automatically runs model, generates reports, updates visualizations
- Clear output showing progress and results
- Emphasizes key insights about the model

## Key Model: Four-Fitness State-Space

**File**: `stan/weight_state_space_four_fitness.stan`

**Key Features**:
1. **Four fitness components**:
   - Aerobic short-term (dehydration, hours-days)
   - Strength short-term (inflammation, hours-days)
   - Aerobic long-term (fat loss, weeks-months)
   - Strength long-term (muscle gain, weeks-months)

2. **Baseline fitness equilibrium at 0**:
   - Fitness states initialized at 0
   - With no workouts, fitness decays to 0
   - Physiologically meaningful baseline

3. **State-space formulation**:
   - Impulse dynamics: `I[t] = ψ·I[t-1] + X[t]`
   - Fitness dynamics: `F[t] = α·F[t-1] + β·I[t-1]`
   - Weight observation: `W[t] = Σ γ_i·F_i[t] + GP(t) + f_daily(t) + ε`

## How to Use the New Structure

### Quick Start:
```bash
# Run complete four-fitness analysis
uv run python run_four_fitness_analysis.py

# Or run individual components
uv run python run_four_fitness_with_states.py  # Fit model
uv run python create_fitness_time_series_report.py  # Generate visualizations
```

### View Results:
1. Open `docs/index.html` in browser
2. Click "Four-Fitness State-Space Analysis" for primary report
3. Click "Fitness Time Series Visualizations" for interactive plots
4. Click "Technical Model Documentation" for mathematical details

## Biological Interpretation

The four-fitness model captures key physiological processes:

| Component | Time Scale | Weight Effect | Biological Process |
|-----------|------------|---------------|-------------------|
| Aerobic short-term | Hours-days | Negative | Dehydration, glycogen depletion |
| Strength short-term | Hours-days | Positive | Inflammation, water retention |
| Aerobic long-term | Weeks-months | Negative | Fat loss, metabolic adaptation |
| Strength long-term | Weeks-months | Positive | Muscle gain, hypertrophy |

## Next Steps

1. **Consider updating `weight_fitness_report/`** to focus on four-fitness model
2. **Archive more outdated Stan models** to `stan/archive/`
3. **Ensure consistency** across all documentation
4. **Add model validation** and comparison metrics

## Files Modified

- `docs/index.html` - Simplified main index
- `docs/four_fitness_comprehensive/index.html` - Updated primary report
- `docs/four_fitness_educational/index.html` - Updated technical docs
- `run_four_fitness_analysis.py` - New analysis runner
- `.claude-current-status` - Updated progress tracking