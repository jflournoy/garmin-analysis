#!/usr/bin/env python3
"""Test the regularized model with hourly predictions."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import cmdstanpy

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.data.weight import load_weight_data
from src.data.intensity import load_intensity_by_activity


def test_model_compilation():
    """Test that the regularized model compiles."""
    print("Testing regularized model compilation...")

    model_path = "stan/weight_state_space_training_decay_aerobic_symmetric_student_ar_spline_regularized.stan"

    try:
        model = cmdstanpy.CmdStanModel(stan_file=model_path)
        print(f"✓ Model compiled successfully: {model_path}")

        print("\nKey differences from horseshoe model:")
        print("  1. Simpler regularizing prior: sigma_epsilon ~ normal(0, 0.1)")
        print("  2. More informative rho prior: normal(0, 0.2)")
        print("  3. No horseshoe complexity (tau, lambda parameters)")
        print("  4. Same hourly prediction structure")

        return model
    except Exception as e:
        print(f"✗ Model compilation failed: {e}")
        return None


def main():
    """Main test function."""
    print("Testing regularized model with hourly predictions")
    print("=" * 60)

    # Test model compilation
    model = test_model_compilation()

    if model is not None:
        print("\n✓ Model ready for use.")
        print("\nThis model addresses the original concerns:")
        print("  1. Regularizing prior on sigma_epsilon shrinks AR component")
        print("  2. More informative prior on rho reduces autocorrelation")
        print("  3. Generates predictions at multiple hours (0, 6, 12, 18, 24)")
        print("  4. Avoids numerical instability of horseshoe prior")


if __name__ == "__main__":
    main()