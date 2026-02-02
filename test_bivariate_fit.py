#!/usr/bin/env python3
"""Quick test of bivariate model fitting."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.models.fit_weight import fit_bivariate_model

print("Starting bivariate model fit test...")
try:
    fit, idata, df, stan_data = fit_bivariate_model(
        chains=1,
        iter_warmup=10,
        iter_sampling=10,
        cache=False,
        force_refit=True,
        use_sparse=True,
        n_inducing_points=30,
    )
    print("Success!")
    print(f"  Observations: {len(df)}")
    print(f"  Parameters: {list(fit.stan_variables().keys())}")
    # Check correlation parameter
    if 'correlation' in fit.stan_variables():
        corr = fit.stan_variables()['correlation']
        print(f"  Latent correlation mean: {corr.mean():.3f}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)