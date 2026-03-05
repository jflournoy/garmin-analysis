#!/usr/bin/env python3
"""Create preview plots showing what component visualizations will look like."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def create_preview_plots():
    """Create preview plots of component visualizations."""
    print("Creating preview plots of component visualizations...")

    output_dir = Path("docs/component_previews")
    output_dir.mkdir(exist_ok=True)

    # Create sample date range
    dates = pd.date_range(start='2023-07-12', end='2024-01-20', freq='D')
    pred_hours = np.array([0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0])

    # 1. Preview of component time series at noon
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    component_names = ['Intercept', 'Strength', 'Aerobic', 'Spline', 'Total', 'Full Model']
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']

    for idx, (name, color) in enumerate(zip(component_names, colors)):
        ax = axes[idx]

        # Create sample time series
        time = np.arange(len(dates))
        if name == 'Intercept':
            values = np.zeros_like(time)
        elif name == 'Strength':
            values = 0.1 * np.sin(2 * np.pi * time / 180)  # ~6 month cycle
        elif name == 'Aerobic':
            values = -0.05 * np.sin(2 * np.pi * time / 90)  # ~3 month cycle
        elif name == 'Spline':
            values = 0.02 * np.sin(2 * np.pi * time / 7)  # Weekly variation
        elif name == 'Total':
            values = 0.1 * np.sin(2 * np.pi * time / 180) - 0.05 * np.sin(2 * np.pi * time / 90) + 0.02 * np.sin(2 * np.pi * time / 7)
        else:  # Full Model
            values = 0.1 * np.sin(2 * np.pi * time / 180) - 0.05 * np.sin(2 * np.pi * time / 90) + 0.02 * np.sin(2 * np.pi * time / 7) + 0.01 * np.random.randn(len(time))

        # Plot with credible interval (simulated)
        ax.fill_between(dates, values - 0.05, values + 0.05,
                       alpha=0.3, color=color, label='95% CI (simulated)')
        ax.plot(dates, values, '-', color=color, linewidth=1.5, label='Mean')
        ax.axhline(0, color='black', linestyle='-', alpha=0.3)

        ax.set_xlabel('Date')
        ax.set_ylabel('Standardized Weight')
        ax.set_title(f'{name} Component at 12:00\n(PREVIEW)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Format x-axis
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / 'preview_component_time_series.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Preview of heatmap
    fig, ax = plt.subplots(figsize=(14, 6))

    # Create sample heatmap data
    D = len(dates)
    H = len(pred_hours)
    heatmap_data = np.zeros((D, H))

    for d in range(D):
        for h_idx, hour in enumerate(pred_hours):
            # Base pattern + daily cycle + some noise
            base = 0.1 * np.sin(2 * np.pi * d / 180)
            daily = 0.05 * np.sin(2 * np.pi * hour / 24 + d/10)
            heatmap_data[d, h_idx] = base + daily

    im = ax.imshow(heatmap_data.T, aspect='auto', cmap='viridis',
                  extent=[dates[0], dates[-1], pred_hours[-1], pred_hours[0]])

    ax.set_xlabel('Date')
    ax.set_ylabel('Hour of Day')
    ax.set_title('Total Predictions Heatmap (PREVIEW)')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Standardized Weight')

    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / 'preview_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Preview of component contributions
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    sample_dates = [
        dates[0],
        dates[len(dates)//4],
        dates[len(dates)//2],
        dates[3*len(dates)//4],
        dates[-1],
        dates[len(dates)//3]  # Extra date
    ]

    for idx, sample_date in enumerate(sample_dates):
        ax = axes[idx]

        # Create sample component contributions through the day
        components = ['Intercept', 'Strength', 'Aerobic', 'Spline']
        colors = ['blue', 'green', 'red', 'purple']

        for comp_idx, (comp_name, color) in enumerate(zip(components, colors)):
            # Different patterns for each component
            if comp_name == 'Intercept':
                values = np.zeros_like(pred_hours)
            elif comp_name == 'Strength':
                values = 0.08 * np.sin(2 * np.pi * pred_hours / 24 + idx)
            elif comp_name == 'Aerobic':
                values = -0.04 * np.cos(2 * np.pi * pred_hours / 24 + idx)
            else:  # Spline
                values = 0.03 * np.sin(4 * np.pi * pred_hours / 24 + idx)

            ax.plot(pred_hours, values, 'o-', linewidth=2, markersize=4,
                   color=color, label=comp_name)

        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Standardized Weight')
        ax.set_title(f'Components: {sample_date.date()}\n(PREVIEW)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(pred_hours)

    plt.tight_layout()
    plt.savefig(output_dir / 'preview_component_contributions.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Create a preview summary
    summary_path = output_dir / 'preview_summary.md'
    with open(summary_path, 'w') as f:
        f.write("# Component Visualization Previews\n\n")
        f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Note\n\n")
        f.write("These are **preview plots** showing what the actual visualizations will look like.\n")
        f.write("The actual plots will be generated from the fitted model with real posterior predictions.\n\n")

        f.write("## Preview Plots Generated\n\n")
        f.write("### 1. Component Time Series at Noon\n")
        f.write("![Component Time Series](preview_component_time_series.png)\n")
        f.write("- Shows each component's contribution over time at 12:00\n")
        f.write("- Includes simulated 95% credible intervals\n")
        f.write("- 6 components: Intercept, Strength, Aerobic, Spline, Total, Full Model\n\n")

        f.write("### 2. Total Predictions Heatmap\n")
        f.write("![Heatmap](preview_heatmap.png)\n")
        f.write("- Shows total predictions across time (x-axis) and hours (y-axis)\n")
        f.write("- Color represents standardized weight value\n")
        f.write("- Reveals daily and longer-term patterns\n\n")

        f.write("### 3. Component Contributions at Sample Dates\n")
        f.write("![Component Contributions](preview_component_contributions.png)\n")
        f.write("- Shows how each component varies through the day\n")
        f.write("- 6 sample dates across the time range\n")
        f.write("- Reveals daily patterns in each component\n\n")

        f.write("## Actual Plots Coming Soon\n\n")
        f.write("The full model is currently running and will generate:\n")
        f.write("1. **Actual posterior predictions** with real credible intervals\n")
        f.write("2. **CSV files** with component predictions at each hour\n")
        f.write("3. **Detailed visualizations** based on the fitted model\n")
        f.write("4. **Summary statistics** and analysis\n")

    print(f"✓ Preview plots saved to {output_dir}/")
    print(f"✓ Preview summary: {summary_path}")

    return output_dir


def main():
    """Main function to create preview plots."""
    print("Creating Component Visualization Previews")
    print("=" * 60)

    output_dir = create_preview_plots()

    print("\n" + "="*60)
    print("Preview Plots Created")
    print("="*60)

    print(f"\nPreview plots show what the actual visualizations will look like.")
    print(f"Output directory: {output_dir}/")
    print(f"\nThe full model is still running and will generate actual predictions soon.")


if __name__ == "__main__":
    main()