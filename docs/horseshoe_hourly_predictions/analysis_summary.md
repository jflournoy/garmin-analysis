# Horseshoe Model with Hourly Predictions Analysis

Generated: 2026-03-04 16:39:28

## Model Configuration
- Days (D): 924
- Weight observations: 147
- Fourier harmonics (K): 2
- Prediction hours (H): 5
- Hours predicted at: [np.float64(0.0), np.float64(6.0), np.float64(12.0), np.float64(18.0), np.float64(24.0)]

## AR(1) Component Analysis
- ρ (autocorrelation): 0.932 [0.901, 0.964]
- σ_ε (innovation scale): 0.152 ± 0.021
- τ (global shrinkage): 0.255 ± 0.209

## Generated Files
1. `daily_patterns.png` - Daily weight patterns for sample days
2. `noon_time_series.png` - Time series of noon predictions vs actual measurements
3. `heatmap_daily_patterns.png` - Heatmap of daily patterns over time
4. `ar_component_analysis.png` - Distribution of AR(1) parameters
5. `hourly_predictions.csv` - All hourly predictions
6. `analysis_summary.md` - This summary file
