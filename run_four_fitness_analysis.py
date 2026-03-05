#!/usr/bin/env python3
"""Simple runner for four-fitness state-space model analysis.

This script provides a clean, focused interface for running the four-fitness
state-space model with baseline fitness equilibrium at 0. It generates
essential visualizations and updates key reports.

Key features:
- Uses `weight_state_space_four_fitness.stan` model
- Focuses on baseline fitness equilibrium at 0
- Generates time series visualizations
- Updates comprehensive analysis report
"""

import sys
from pathlib import Path
import subprocess
import shutil

def main():
    print("\n" + "=" * 70)
    print("FOUR-FITNESS STATE-SPACE MODEL ANALYSIS")
    print("=" * 70)
    print("Model: weight_state_space_four_fitness.stan")
    print("Focus: Baseline fitness equilibrium at 0")
    print("=" * 70)

    # Check if required scripts exist
    required_scripts = [
        "run_four_fitness_with_states.py",
        "analyze_four_fitness_model.py",
        "create_fitness_time_series_report.py",
        "create_comprehensive_four_fitness_report.py"
    ]

    for script in required_scripts:
        if not Path(script).exists():
            print(f"❌ Missing required script: {script}")
            print("Please ensure all analysis scripts are available.")
            return 1

    print("\n1. Running four-fitness model with state saving...")
    print("   (This may take 10-20 minutes)")

    # Run the model with state saving
    try:
        result = subprocess.run(
            [sys.executable, "run_four_fitness_with_states.py"],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"❌ Model fitting failed:")
            print(result.stderr[:500])  # Show first 500 chars of error
            return 1

        print("   ✓ Model fitted successfully")

        # Extract key information from output
        for line in result.stdout.split('\n'):
            if "divergent transitions" in line:
                print(f"   ⚠️  {line.strip()}")
            elif "Sampling completed" in line:
                print(f"   {line.strip()}")

    except Exception as e:
        print(f"❌ Error running model: {e}")
        return 1

    print("\n2. Generating fitness time series report...")
    try:
        result = subprocess.run(
            [sys.executable, "create_fitness_time_series_report.py"],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"❌ Time series report generation failed:")
            print(result.stderr[:500])
            return 1

        print("   ✓ Time series report generated")
        print("   → Available at: docs/fitness_time_series_report/index.html")

    except Exception as e:
        print(f"❌ Error generating time series report: {e}")
        return 1

    print("\n3. Updating comprehensive analysis report...")
    try:
        result = subprocess.run(
            [sys.executable, "create_comprehensive_four_fitness_report.py"],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"❌ Comprehensive report update failed:")
            print(result.stderr[:500])
            return 1

        print("   ✓ Comprehensive report updated")
        print("   → Available at: docs/four_fitness_comprehensive/index.html")

    except Exception as e:
        print(f"❌ Error updating comprehensive report: {e}")
        return 1

    print("\n4. Copying visualizations to report directories...")
    try:
        # Copy visualizations from output to docs directories
        output_dir = Path("output/four_fitness_full")
        docs_viz_dir = Path("docs/fitness_time_series_report/visualizations")

        if output_dir.exists() and docs_viz_dir.exists():
            # Copy any new PNG files
            for png_file in output_dir.glob("*.png"):
                shutil.copy2(png_file, docs_viz_dir)
                print(f"   ✓ Copied {png_file.name}")

        print("   ✓ Visualizations updated")

    except Exception as e:
        print(f"⚠️  Warning copying visualizations: {e}")
        # Non-critical error, continue

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\n📊 Reports available:")
    print("   Primary Analysis: docs/four_fitness_comprehensive/index.html")
    print("   Time Series: docs/fitness_time_series_report/index.html")
    print("   Technical Docs: docs/four_fitness_educational/index.html")
    print("\n🔑 Key insights from four-fitness model:")
    print("   • Separates aerobic vs strength training effects")
    print("   • Distinguishes short-term (hours-days) vs long-term (weeks-months)")
    print("   • Baseline fitness equilibrium at 0 (no workouts → fitness → 0)")
    print("   • Aerobic: fat loss (negative weight effect)")
    print("   • Strength: muscle gain (positive weight effect)")
    print("\n📈 Next steps:")
    print("   • Review parameter estimates in comprehensive report")
    print("   • Explore time series visualizations")
    print("   • Check model diagnostics in output/four_fitness_full/")

    return 0

if __name__ == "__main__":
    sys.exit(main())