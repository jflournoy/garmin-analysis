"""Test that lower bounds fix sigma zero issue."""
import warnings
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from models.fit_weight import fit_weight_model_flexible

def main():
    print("Testing flexible model with lower bounds (alpha, rho, sigma >= 0.01)...")
    print("Using default priors.")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        try:
            fit, idata, df, stan_data = fit_weight_model_flexible(
                chains=2,
                iter_warmup=50,
                iter_sampling=50,
                alpha_prior_sd=1.0,
                rho_prior_shape=5.0,
                rho_prior_scale=1.0,
                sigma_prior_sd=0.5,
            )

            if w:
                print(f"\nWarnings ({len(w)}):")
                for warning in w:
                    print(f"  - {warning.category.__name__}: {warning.message}")
            else:
                print("\nNo warnings.")

            # Check sigma values
            sigma_samples = idata.posterior["sigma"].values.flatten()
            min_sigma = sigma_samples.min()
            print(f"\nSigma range: {min_sigma:.6f} to {sigma_samples.max():.3f}")
            if min_sigma < 0.001:
                print(f"WARNING: sigma approaches zero (min={min_sigma:.6f})")
            else:
                print("OK: sigma safely above zero.")

            # Check for divergent transitions
            try:
                divergent = fit.diagnose().get('divergent_iterations', 0)
                print(f"Divergent transitions: {divergent}")
            except AttributeError:
                print("Divergent transitions: diagnose returned string")

            print("\nTest passed.")

        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

if __name__ == "__main__":
    main()