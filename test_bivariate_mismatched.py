#!/usr/bin/env python3
"""Quick test of bivariate mismatched model fitting (weight vs VO2 max)."""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.data.weight import load_weight_data
from src.data.vo2max import load_vo2max_data
from src.models.fit_weight import fit_bivariate_model_mismatched
from src.data.align import prepare_bivariate_stan_data_mismatched

def prepare_weight_daily(df_weight: pd.DataFrame) -> pd.DataFrame:
    """Aggregate weight measurements to daily mean."""
    df = df_weight.copy()
    df["date"] = df["timestamp"].dt.date
    df["date"] = pd.to_datetime(df["date"])
    daily = df.groupby("date")["weight_lbs"].agg(["mean", "std", "count"]).reset_index()
    daily = daily.rename(columns={"mean": "weight", "date": "timestamp"})
    return daily

print("=== Bivariate Mismatched Model Test (Weight vs VO2 Max) ===")

# Load data
print("\n1. Loading data...")
df_weight_raw = load_weight_data()
df_vo2max = load_vo2max_data()
print(f"   Weight raw measurements: {len(df_weight_raw)}")
print(f"   VO2 max measurements: {len(df_vo2max)}")

# Prepare weight daily aggregated
df_weight_daily = prepare_weight_daily(df_weight_raw)
print(f"   Weight daily aggregated: {len(df_weight_daily)}")

# Prepare data frames for mismatched model
# Weight dataframe needs columns 'timestamp' and 'weight_lbs' (value column)
df_weight_for_model = df_weight_daily.rename(columns={"weight": "weight_lbs"})
# VO2 max dataframe has columns 'date' and 'vo2_max'
df_vo2_for_model = df_vo2max.rename(columns={"date": "timestamp", "vo2_max": "value"})

# Test data preparation
print("\n2. Testing data preparation...")
try:
    stan_data = prepare_bivariate_stan_data_mismatched(
        df_weight=df_weight_for_model,
        df_other=df_vo2_for_model,
        weight_time_col="timestamp",
        weight_value_col="weight_lbs",
        other_time_col="timestamp",
        other_value_col="value",
        use_sparse=True,
        n_inducing_points=30,
        inducing_point_method="uniform",
        include_prediction_grid=False,
    )
    print(f"   Success! Keys: {list(stan_data.keys())}")
    print(f"   N_weight: {stan_data['N_weight']}, N_other: {stan_data['N_other']}")
    print(f"   M: {stan_data['M']}, use_sparse: {stan_data['use_sparse']}")
except Exception as e:
    print(f"   Data preparation error: {e}")
    sys.exit(1)

# Test model fitting (minimal iterations)
print("\n3. Testing model fitting (minimal iterations)...")
try:
    fit, idata, stan_data = fit_bivariate_model_mismatched(
        df_weight=df_weight_for_model,
        df_other=df_vo2_for_model,
        weight_time_col="timestamp",
        weight_value_col="weight_lbs",
        other_time_col="timestamp",
        other_value_col="value",
        chains=1,
        iter_warmup=5,
        iter_sampling=5,
        use_sparse=True,
        n_inducing_points=30,
        cache=False,
        force_refit=True,
        include_prediction_grid=False,
    )
    print("   Success!")
    print(f"   Parameters: {list(fit.stan_variables().keys())}")
    # Check correlation parameter
    if 'correlation' in fit.stan_variables():
        corr = fit.stan_variables()['correlation']
        print(f"   Latent correlation mean: {corr.mean():.3f}")
        print(f"   Latent correlation std: {corr.std():.3f}")
    # Check convergence diagnostics
    # Divergent transitions info not needed for quick test
except Exception as e:
    print(f"   Model fitting error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n=== Test completed successfully ===")