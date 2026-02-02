#!/usr/bin/env python3
"""Test spline model compilation."""
import sys
from pathlib import Path

# Add project root to path if needed
sys.path.insert(0, str(Path(__file__).parent))

from cmdstanpy import CmdStanModel

def main():
    print("Testing spline Stan model compilation...")
    try:
        CmdStanModel(stan_file="stan/weight_gp_spline.stan")
        print("✓ Spline model compiled successfully")
        return 0
    except Exception as e:
        print(f"✗ Compilation failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())