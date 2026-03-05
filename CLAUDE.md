# CLAUDE.md - Project AI Guidelines

This project is a Bayesian analysis of Garmin health data using Stan.

## State Tracking

**Use two complementary systems for tracking work:**

1. **TodoWrite tool** (primary) - Use for task management visible to the user
   - Break work into discrete, actionable items
   - Mark tasks in_progress/completed as you work
   - Good for: task lists, progress tracking, user visibility

2. **`.claude-current-status` file** (supplementary) - Higher-resolution notes
   - Timestamps, context, decisions, file references
   - Details that don't fit in todo items
   - Session continuity across conversations
   - Good for: debugging context, decision rationale, file locations

**Workflow:** Start tasks with TodoWrite, always add detailed notes to `.claude-current-status`. When you add notes, re-assess and clean up old notes.

**Status Management Commands:**
- `/continue` - Efficiently resume work by extracting recent session context
- `/condense [N]` - Archive old content, keep last N lines (default: 200)
- `scripts/status-helper.sh` - Auto-cleanup helper for updates

**Auto-Cleanup Pattern:** When updating `.claude-current-status`, check if file exceeds 300 lines and consider running `/condense`.

## Project Overview

**Goal**: Build a personal health analytics system with:
1. Bayesian models in Stan for analyzing Garmin data (weight, sleep, activity, etc.)
2. Interactive D3.js visualizations for exploring health relationships
3. A web interface for exploring model results and data trends

## Tooling Requirements

**CRITICAL: Always use these tools for their respective domains:**

- **Python**: Always use `uv` for Python package management and execution
  - `uv add <package>` - Add dependencies
  - `uv run python script.py` - Run Python scripts
  - `uv sync` - Sync dependencies
  - Never use raw `python` or `pip` commands
  - **Note**: When running analysis scripts, you may see a warning about VIRTUAL_ENV mismatch if an external conda environment is active. This warning can be ignored as uv uses the project's `.venv`.

- **JavaScript/Node**: Use `npm` for package management
  - `npm install` - Install dependencies
  - `npm run <script>` - Run scripts

- **Stan**: Use CmdStanPy for Stan model compilation and sampling
  - Models go in `stan/` directory with `.stan` extension

## Data Location

Garmin export data is in `data/DI_CONNECT/`:
- `DI-Connect-Wellness/` - Biometrics, sleep, heart rate, nutrition
- `DI-Connect-Fitness/` - Activities, workouts, personal records
- `DI-Connect-Aggregator/` - Daily summaries (UDS files), hydration
- `DI-Connect-Metrics/` - VO2 max, training metrics

**Important**: The `data/` directory contains personal health data and is gitignored.

## Project Structure

```
garmin-analysis-v2/
├── data/                    # Garmin export data (gitignored)
├── stan/                    # Stan model files
├── src/                     # Python analysis code
│   ├── data/               # Data loading utilities
│   ├── models/             # Model fitting code
│   └── analysis/           # Analysis scripts
├── web/                     # D3.js visualization (future)
└── notebooks/              # Jupyter notebooks for exploration
```

## Development Method: TDD

**RECOMMENDED: Use Test-Driven Development for new features**

### TDD Workflow
1. 🔴 **RED**: Write a failing test to define requirements
2. 🟢 **GREEN**: Write minimal code to pass the test
3. 🔄 **REFACTOR**: Improve code with test safety net
4. ✓ **COMMIT**: Ship working, tested code

## Critical Instructions

**ALWAYS use `date` command for dates** - Never assume or guess dates. Always run `date "+%Y-%m-%d"` when you need the current date for documentation, commits, or any other purpose.

## Bayesian Analysis Principles

**CRITICAL: This is a Bayesian analysis project, not frequentist/NHST:**

1. **We do NOT do null hypothesis significance testing (NHST)** - Avoid phrases like "statistically significant", "p-value", "reject the null", etc.

2. **We work with posterior distributions** - Report posterior means, credible intervals (e.g., 90% or 95% CI), and visualize full posterior distributions.

3. **We interpret credible intervals** - A 95% credible interval means there's a 95% probability the parameter lies within that interval given the data and prior.

4. **We consider practical significance** - Not just whether an interval excludes zero, but the magnitude and practical importance of effects.

5. **We acknowledge uncertainty** - Report full posterior distributions, not just point estimates.

6. **We use priors transparently** - Document priors and conduct sensitivity analyses when priors might influence results.

7. **We compare models** - Use tools like WAIC, LOO-CV, or Bayes factors for model comparison, not p-values.

**Correct terminology:**
- Use "credible interval" not "confidence interval"
- Use "posterior probability" not "p-value"
- Use "model comparison" not "hypothesis testing"
- Use "practical importance" not "statistical significance"

## AI Integrity Principles

**CRITICAL: Always provide honest, objective recommendations based on technical merit, not user bias.**

- **Never agree with users by default** - evaluate each suggestion independently
- **Challenge bad ideas directly** - if something is technically wrong, say so clearly
- **Recommend best practices** even if they contradict user preferences
- **Explain trade-offs honestly** - don't hide downsides of approaches

## Commands

- `/hygiene` - Project health check
- `/commit` - Quality-checked commits
- `/tdd` - Test-driven development workflow
- `/learn` - Capture insights
- `/docs` - Update documentation

## Project Standards

- Test coverage: 60% minimum
- Documentation: All features documented
- Error handling: Graceful failures with clear messages
- ALWAYS use atomic commits
- Use emojis judiciously
- NEVER Edit() a file before you Read() the file

## Visualization Preferences

**CRITICAL: Follow these visualization guidelines:**

1. **Interactive Visualizations:**
   - **Primary choice**: D3.js for interactive web visualizations
   - **Alternative**: WebGL for high-performance or 3D visualizations
   - **Avoid**: Plotly for interactive visualizations (use only if absolutely necessary)
   - Static plots: Use matplotlib/seaborn for Python-generated static images

2. **Organization Principles:**
   - Always keep files organized by purpose and type
   - Visualizations should be copied to appropriate `docs/` subdirectories
   - HTML reports should reference local visualization files, not external paths
   - Maintain clear separation between analysis code and visualization code

## Analytics Tracking

**CRITICAL:** All HTML pages MUST include Simple Analytics tracking.

The tracking script is configured in `docs/convert_to_html.py` HTML template:
```html
<script async src="https://scripts.simpleanalyticscdn.com/latest.js"></script>
```

**When generating or modifying HTML:**
- Always ensure this script is present in the `<head>` section of the template
- Run `npm run generate-docs` to rebuild all HTML files
- Verify tracking is present: `grep -r "simpleanalyticscdn" docs/`
