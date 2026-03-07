.PHONY: help clean-cache clean-output clean-plots clean help-all \
        fit-model predictions plots reports deploy \
        fit-base fit-trend \
        report-base report-trend \
        pipeline pipeline-base pipeline-trend

# Color output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m # No Color

help:
	@echo "$(BLUE)Garmin Analysis Pipeline$(NC)"
	@echo ""
	@echo "$(GREEN)Main Workflows:$(NC)"
	@echo "  make pipeline              Run full pipeline (fit model → predictions → plots → reports)"
	@echo "  make pipeline-base         Pipeline for base constrained AR model"
	@echo "  make pipeline-trend        Pipeline for trend model variant"
	@echo ""
	@echo "$(GREEN)Individual Steps:$(NC)"
	@echo "  make fit-model             Fit constrained AR model (Stan)"
	@echo "  make fit-base              Alias for fit-model"
	@echo "  make fit-trend             Fit trend model variant"
	@echo ""
	@echo "  make predictions           Generate component predictions"
	@echo "  make plots                 Regenerate all visualizations"
	@echo "  make reports               Generate all HTML reports"
	@echo ""
	@echo "  make report-base           Generate base model report"
	@echo "  make report-trend          Generate trend model report"
	@echo ""
	@echo "$(GREEN)Deployment:$(NC)"
	@echo "  make deploy                Deploy reports to docs/"
	@echo ""
	@echo "$(GREEN)Cleanup:$(NC)"
	@echo "  make clean-cache           Clear CmdStanPy cache"
	@echo "  make clean-output          Archive old output/ directories"
	@echo "  make clean-plots           Remove regenerated plots"
	@echo "  make clean-all             Full cleanup (cache + output + plots)"
	@echo ""

help-all: help
	@echo "$(GREEN)Available Models:$(NC)"
	@echo "  BASE:  weight_state_space_training_decay_aerobic_symmetric_student_ar_spline_constrained.stan"
	@echo "  TREND: weight_state_space_training_decay_aerobic_symmetric_student_ar_spline_constrained_trend.stan"
	@echo ""
	@echo "$(YELLOW)Note:$(NC) Requires data/ directory with Garmin export files"

# ============================================================================
# MAIN WORKFLOWS
# ============================================================================

pipeline: pipeline-base
	@echo "$(GREEN)✓ Full pipeline complete$(NC)"

pipeline-base: fit-base predictions plots report-base
	@echo "$(GREEN)✓ Base model pipeline complete$(NC)"

pipeline-trend: fit-trend predictions plots report-trend
	@echo "$(GREEN)✓ Trend model pipeline complete$(NC)"

# ============================================================================
# MODEL FITTING
# ============================================================================

fit-model: fit-base

fit-base:
	@echo "$(BLUE)Fitting base constrained AR model...$(NC)"
	uv run python src/analysis/run_constrained_ar_model.py
	@echo "$(GREEN)✓ Base model fit complete$(NC)"

fit-trend:
	@echo "$(BLUE)Fitting trend model...$(NC)"
	uv run python src/analysis/run_trend_comparison.py
	@echo "$(GREEN)✓ Trend model fit complete$(NC)"

# ============================================================================
# PREDICTIONS & VISUALIZATIONS
# ============================================================================

predictions:
	@echo "$(BLUE)Generating component predictions...$(NC)"
	uv run python src/analysis/generate_component_predictions.py
	@echo "$(GREEN)✓ Predictions generated$(NC)"

plots:
	@echo "$(BLUE)Regenerating plots with proper scaling...$(NC)"
	uv run python src/analysis/regenerate_plots.py
	@echo "$(GREEN)✓ Plots regenerated$(NC)"

# ============================================================================
# REPORT GENERATION
# ============================================================================

reports: report-base report-trend
	@echo "$(GREEN)✓ All reports generated$(NC)"

report-base:
	@echo "$(BLUE)Generating base model report...$(NC)"
	uv run python src/analysis/generate_model_report.py
	@echo "$(GREEN)✓ Base model report generated: docs/constrained_ar_model_report/index.html$(NC)"

report-trend:
	@echo "$(BLUE)Generating trend model report...$(NC)"
	uv run python src/analysis/generate_trend_report.py
	@echo "$(GREEN)✓ Trend model report generated: docs/trend_model_report/index.html$(NC)"

# ============================================================================
# METADATA & UTILITIES
# ============================================================================

metadata:
	@echo "$(BLUE)Extracting posterior metadata...$(NC)"
	uv run python src/analysis/extract_posterior_metadata.py
	@echo "$(GREEN)✓ Metadata extracted$(NC)"

# ============================================================================
# DEPLOYMENT
# ============================================================================

deploy: reports
	@echo "$(BLUE)Reports deployed to docs/$(NC)"
	@echo "View at: file://$(shell pwd)/docs/constrained_ar_model_report/index.html"

# ============================================================================
# CLEANUP
# ============================================================================

clean-cache:
	@echo "$(YELLOW)Cleaning CmdStanPy cache...$(NC)"
	rm -rf ~/.cmdstanpy/
	@echo "$(GREEN)✓ Cache cleaned$(NC)"

clean-plots:
	@echo "$(YELLOW)Removing regenerated plots...$(NC)"
	rm -f docs/component_predictions/*.png
	@echo "$(GREEN)✓ Plots cleaned$(NC)"

clean-output:
	@echo "$(YELLOW)Archiving old output directories...$(NC)"
	@if [ ! -d output/archive ]; then mkdir -p output/archive; fi
	@for dir in output/*/; do \
		if [ "$$dir" != "output/archive/" ] && [ "$$dir" != "output/current/" ]; then \
			mv "$$dir" output/archive/ 2>/dev/null || true; \
		fi; \
	done
	@echo "$(GREEN)✓ Output archived$(NC)"

clean-all: clean-cache clean-plots clean-output
	@echo "$(GREEN)✓ Full cleanup complete$(NC)"

# ============================================================================
# STATUS
# ============================================================================

status:
	@echo "$(BLUE)Project Status:$(NC)"
	@echo ""
	@echo "Stan Models:"
	@echo "  Current: $$(ls stan/weight_state_space*constrained*.stan 2>/dev/null | wc -l) production models"
	@echo "  Archive: $$(ls stan/weight_gp*.stan 2>/dev/null | wc -l) experimental models"
	@echo ""
	@echo "Analysis Scripts: $$(find src/analysis -name '*.py' -type f | wc -l) files"
	@echo ""
	@echo "Output Directories: $$(ls -d output/*/ 2>/dev/null | wc -l) directories"
	@if [ -d "output/archive" ]; then \
		echo "  Archived: $$(ls -d output/archive/*/ 2>/dev/null | wc -l)"; \
	fi
	@echo ""
	@echo "Reports: $$(find docs -maxdepth 1 -name '*_report' -type d | wc -l) reports"
	@echo ""

.DEFAULT_GOAL := help
