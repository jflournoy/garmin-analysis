#!/usr/bin/env python3
"""Extract posterior metadata from saved MCMC chain CSVs.

Loads saved chain CSVs via cmdstanpy, extracts posterior summaries and MCMC diagnostics,
saves as JSON for reproducible HTML report generation.
"""

import sys
import json
from pathlib import Path
import numpy as np
import cmdstanpy

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def extract_posterior_from_csvs(csv_dir):
    """Load chains from CSV files and extract posterior summaries.

    Args:
        csv_dir: Directory containing stan-* CSV files from MCMC chains

    Returns:
        Dictionary with posterior summaries and diagnostics
    """
    print(f"Loading chains from {csv_dir}...")

    # Load chains using cmdstanpy
    try:
        fit = cmdstanpy.from_csv(str(csv_dir))
    except Exception as e:
        print(f"Error loading chains: {e}")
        return None

    # Extract summaries
    summaries = fit.summary()
    print(f"Loaded {fit.num_draws_sampling} draws across {fit.num_chains} chains")

    # Parameters to extract (scalar parameters only)
    scalar_params = [
        'rho', 'sigma_epsilon', 'gamma_s', 'gamma_a', 'weight_intercept', 'nu',
        'sigma_w', 'sigma_fourier', 'beta_s', 'beta_a',
        'alpha_d_s', 'alpha_m_s', 'alpha_d_a', 'alpha_m_a'
    ]

    posterior_metadata = {}

    # Extract each parameter
    for param in scalar_params:
        if param in summaries.index:
            row = summaries.loc[param]
            posterior_metadata[param] = {
                'mean': float(row['Mean']),
                'sd': float(row['StdDev']),
                'lower_ci': float(row['2.5%']),
                'upper_ci': float(row['97.5%']),
                'rhat': float(row['R_hat']) if not np.isnan(row['R_hat']) else None,
                'ess_bulk': float(row['Bulk_ESS']) if not np.isnan(row['Bulk_ESS']) else None,
                'ess_tail': float(row['Tail_ESS']) if not np.isnan(row['Tail_ESS']) else None,
            }
            print(f"  {param:20s}: {posterior_metadata[param]['mean']:8.4f} [{posterior_metadata[param]['lower_ci']:8.4f}, {posterior_metadata[param]['upper_ci']:8.4f}]")
        else:
            print(f"  {param:20s}: NOT FOUND")

    # MCMC diagnostics
    diagnostics = {
        'num_chains': fit.num_chains,
        'num_draws_sampling': fit.num_draws_sampling,
        'num_draws_warmup': fit.num_draws_warmup,
    }

    # Count divergent transitions (if available)
    if hasattr(fit, 'diagnose'):
        diag_output = fit.diagnose()
        diagnostics['diagnose_output'] = str(diag_output)

    # Check for divergences in metadata
    if fit.metadata() is not None:
        try:
            metadata = fit.metadata()
            if hasattr(metadata, 'divergences'):
                diagnostics['total_divergences'] = fit.metadata().divergences
        except:
            pass

    print(f"\nMCMC Diagnostics:")
    print(f"  Chains: {diagnostics['num_chains']}")
    print(f"  Sampling draws: {diagnostics['num_draws_sampling']}")
    print(f"  Warmup draws: {diagnostics['num_draws_warmup']}")

    return {
        'posterior': posterior_metadata,
        'diagnostics': diagnostics,
    }


def save_metadata_json(metadata, output_file):
    """Save metadata to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved metadata to {output_file}")


def main():
    """Main entry point."""
    # Try to find most recent temp directory with saved chains
    import tempfile
    import glob

    # Look for saved chains in common locations
    search_paths = [
        Path('/tmp'),
        Path.home() / 'tmp',
    ]

    csv_dir = None
    for search_path in search_paths:
        if search_path.exists():
            # Look for directories with stan CSV files
            pattern = str(search_path / 'tmp*')
            dirs = glob.glob(pattern)
            if dirs:
                # Use most recent
                dirs.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
                for d in dirs:
                    csv_files = list(Path(d).glob('*-1.csv'))
                    if csv_files:
                        csv_dir = Path(d)
                        break
        if csv_dir:
            break

    if not csv_dir:
        print("Error: Could not find saved chain CSV directory.")
        print("Please specify CSV directory as argument: python extract_posterior_metadata.py <csv_dir>")
        if len(sys.argv) > 1:
            csv_dir = Path(sys.argv[1])
            if not csv_dir.exists():
                print(f"Error: Directory {csv_dir} does not exist")
                sys.exit(1)
        else:
            sys.exit(1)

    print(f"Using CSV directory: {csv_dir}")

    # Extract metadata
    metadata = extract_posterior_from_csvs(csv_dir)
    if metadata is None:
        sys.exit(1)

    # Save to JSON in docs directory
    output_dir = Path(__file__).parent.parent.parent / 'docs' / 'component_predictions'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'posterior_metadata.json'

    save_metadata_json(metadata, output_file)
    print(f"\n✓ Successfully extracted posterior metadata")
    print(f"  Output: {output_file}")


if __name__ == '__main__':
    main()
