"""Test increased jitter for numerical stability."""
import warnings
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models.fit_weight import fit_weight_model

def main():
    print("Testing weight GP model with increased diagonal jitter (1e-4)...")
    print("Running with 2 chains, 100 warmup, 100 sampling for quick test.")

    # Capture warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        try:
            fit, idata, df, stan_data = fit_weight_model(
                chains=2,
                iter_warmup=100,
                iter_sampling=100,
            )

            # Check for warnings
            if w:
                print(f"\nWarnings captured ({len(w)}):")
                for warning in w:
                    print(f"  - {warning.category.__name__}: {warning.message}")
            else:
                print("\nNo warnings captured.")

            # Check for divergent transitions (diagnose returns string)
            try:
                divergent = fit.diagnose().get('divergent_iterations', 0)
                print(f"\nDivergent transitions: {divergent}")
            except AttributeError:
                print("\nDivergent transitions: diagnose method returned string")

            # Check parameter means
            posterior = idata.posterior[["alpha", "rho", "sigma"]]
            print("\nParameter summary (mean):")
            for param in ["alpha", "rho", "sigma"]:
                mean_val = posterior[param].mean().item()
                print(f"  {param}: {mean_val:.3f}")

            print("\nTest completed successfully.")

        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

if __name__ == "__main__":
    main()