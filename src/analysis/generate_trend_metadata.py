#!/usr/bin/env python3
"""Generate posterior_metadata_trend.json from comparison results.

This script creates the metadata file needed by generate_trend_report.py
when the trend model fitting has been completed but metadata extraction
hasn't been run separately.
"""

import json
from pathlib import Path


def main():
    """Generate metadata_trend.json from comparison results."""

    comparison_file = Path(__file__).parent.parent.parent / 'docs' / 'trend_comparison' / 'comparison_results.json'

    if not comparison_file.exists():
        print(f"Error: {comparison_file} not found")
        print("Please run: make fit-trend")
        return False

    # Load comparison results
    with open(comparison_file, 'r') as f:
        comparison = json.load(f)

    # Extract trend model posteriors
    posteriors_trend = comparison.get('posteriors_trend')
    if not posteriors_trend:
        print("Error: No trend model posteriors found in comparison results")
        return False

    # Build metadata structure (same format as posterior_metadata.json)
    metadata = {
        'posterior': posteriors_trend,
        'diagnostics': {
            'num_chains': comparison.get('data_summary', {}).get('num_chains', 4),
            'num_draws_sampling': comparison.get('data_summary', {}).get('num_draws_sampling', 500),
            'num_draws_warmup': comparison.get('data_summary', {}).get('num_draws_warmup', 500),
        }
    }

    # Save to standard location
    output_dir = Path(__file__).parent.parent.parent / 'docs' / 'component_predictions'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'posterior_metadata_trend.json'

    with open(output_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Generated posterior_metadata_trend.json")
    print(f"  Source: {comparison_file}")
    print(f"  Output: {output_file}")

    return True


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
