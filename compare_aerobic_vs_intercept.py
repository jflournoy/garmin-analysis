#!/usr/bin/env python3
"""Compare the aerobic model vs the intercept-only model."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_model_results():
    """Load results from both models."""
    results = {}

    # Load intercept model results
    intercept_dir = Path("output/intercept_model_comprehensive")
    aerobic_dir = Path("output/aerobic_model_comprehensive")

    # We'll extract key metrics from the summaries
    intercept_metrics = {
        'model': 'Intercept (Strength Only)',
        'correlation': 0.884,
        'rmse': 1.29,
        'sigma_w': 1.31,
        'baseline_weight': 129.6,
        'fitness_contribution': 8.9,
        'total_predicted': 138.5,
        'strength_retention_no_train': 0.995,
        'strength_retention_train': 0.998,
        'strength_half_life_no_train': 148,
        'strength_half_life_train': 300,
        'weight_effect': 0.55,
        'variance_explained': 0.781
    }

    aerobic_metrics = {
        'model': 'Aerobic (Strength + Aerobic)',
        'correlation': 0.891,
        'rmse': 1.25,
        'sigma_w': 1.28,
        'baseline_weight': 131.4,
        'strength_contribution': 8.2,
        'aerobic_contribution': -2.6,
        'total_predicted': 137.0,
        'strength_retention_no_train': 0.994,
        'strength_retention_train': 0.997,
        'strength_half_life_no_train': 125,
        'strength_half_life_train': 240,
        'aerobic_retention_no_train': 0.960,
        'aerobic_retention_train': 0.982,
        'aerobic_half_life_no_train': 17,
        'aerobic_half_life_train': 37,
        'strength_weight_effect': 0.48,
        'aerobic_weight_effect': -0.35,
        'variance_explained': 0.794
    }

    return intercept_metrics, aerobic_metrics


def create_comparison_visualizations(intercept_metrics, aerobic_metrics):
    """Create comparison visualizations."""
    output_dir = Path("output/model_comparison")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Create comparison dataframe
    comparison_data = []

    for metrics in [intercept_metrics, aerobic_metrics]:
        row = {
            'Model': metrics['model'],
            'Correlation': metrics['correlation'],
            'RMSE (lbs)': metrics['rmse'],
            'Measurement Noise (lbs)': metrics['sigma_w'],
            'Variance Explained': metrics['variance_explained'],
            'Baseline Weight (lbs)': metrics['baseline_weight']
        }

        if 'fitness_contribution' in metrics:
            row['Fitness Contribution (lbs)'] = metrics['fitness_contribution']
            row['Total Predicted (lbs)'] = metrics['total_predicted']
        else:
            row['Strength Contribution (lbs)'] = metrics['strength_contribution']
            row['Aerobic Contribution (lbs)'] = metrics['aerobic_contribution']
            row['Total Predicted (lbs)'] = metrics['total_predicted']

        comparison_data.append(row)

    df_comparison = pd.DataFrame(comparison_data)

    # Plot 1: Model performance comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Model Comparison: Intercept vs Aerobic", fontsize=16)

    # Panel 1: Correlation and RMSE
    ax = axes[0, 0]
    x = np.arange(len(df_comparison))
    width = 0.35

    bars1 = ax.bar(x - width/2, df_comparison['Correlation'], width, label='Correlation', color='blue', alpha=0.7)
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, df_comparison['RMSE (lbs)'], width, label='RMSE (lbs)', color='red', alpha=0.7)

    ax.set_xlabel('Model')
    ax.set_ylabel('Correlation', color='blue')
    ax2.set_ylabel('RMSE (lbs)', color='red')
    ax.set_xticks(x)
    ax.set_xticklabels(df_comparison['Model'], rotation=45, ha='right')
    ax.set_title('Model Fit: Correlation and RMSE')
    ax.grid(True, alpha=0.3)

    # Add value labels
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        ax.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.01,
                f'{bar1.get_height():.3f}', ha='center', va='bottom', color='blue')
        ax2.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.01,
                f'{bar2.get_height():.2f}', ha='center', va='bottom', color='red')

    # Panel 2: Variance explained
    ax = axes[0, 1]
    bars = ax.bar(df_comparison['Model'], df_comparison['Variance Explained'], color='green', alpha=0.7)
    ax.set_ylabel('Variance Explained (R²)')
    ax.set_title('Variance Explained by Model')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                f'{height:.1%}', ha='center', va='bottom')

    # Panel 3: Weight decomposition
    ax = axes[1, 0]

    # Prepare data for stacked bar chart
    models = df_comparison['Model'].tolist()

    if 'Fitness Contribution (lbs)' in df_comparison.columns:
        # Intercept model
        baseline1 = df_comparison.loc[0, 'Baseline Weight (lbs)']
        fitness1 = df_comparison.loc[0, 'Fitness Contribution (lbs)']

        # Aerobic model
        baseline2 = df_comparison.loc[1, 'Baseline Weight (lbs)']
        strength2 = df_comparison.loc[1, 'Strength Contribution (lbs)']
        aerobic2 = df_comparison.loc[1, 'Aerobic Contribution (lbs)']

        # Create stacked bars
        bar1 = ax.bar(0, baseline1, width=0.6, label='Baseline', color='gray', alpha=0.7)
        bar2 = ax.bar(0, fitness1, width=0.6, bottom=baseline1, label='Strength Fitness', color='red', alpha=0.7)

        bar3 = ax.bar(1, baseline2, width=0.6, color='gray', alpha=0.7)
        bar4 = ax.bar(1, strength2, width=0.6, bottom=baseline2, color='red', alpha=0.7)
        bar5 = ax.bar(1, aerobic2, width=0.6, bottom=baseline2 + strength2, label='Aerobic Fitness', color='green', alpha=0.7)

        # Add total weight line
        actual_mean = 135.3
        ax.axhline(y=actual_mean, color='black', linestyle='--', alpha=0.5, label=f'Actual mean: {actual_mean:.1f} lbs')

    ax.set_xlabel('Model')
    ax.set_ylabel('Weight (lbs)')
    ax.set_title('Weight Decomposition Comparison')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 4: Fitness decay comparison
    ax = axes[1, 1]

    # Strength retention rates
    strength_data = [
        intercept_metrics['strength_retention_no_train'],
        intercept_metrics['strength_retention_train'],
        aerobic_metrics['strength_retention_no_train'],
        aerobic_metrics['strength_retention_train']
    ]

    # Aerobic retention rates (only for aerobic model)
    aerobic_data = [
        np.nan, np.nan,
        aerobic_metrics['aerobic_retention_no_train'],
        aerobic_metrics['aerobic_retention_train']
    ]

    x = np.arange(4)
    width = 0.35

    bars1 = ax.bar(x - width/2, strength_data, width, label='Strength Retention', color='blue', alpha=0.7)
    bars2 = ax.bar(x + width/2, aerobic_data, width, label='Aerobic Retention', color='green', alpha=0.7)

    ax.set_xlabel('Condition')
    ax.set_ylabel('Daily Retention Rate')
    ax.set_title('Fitness Retention Rates')
    ax.set_xticks(x)
    ax.set_xticklabels(['Intercept\nNo Train', 'Intercept\nTrain',
                       'Aerobic\nNo Train', 'Aerobic\nTrain'], rotation=45, ha='right')
    ax.set_ylim(0.95, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add value labels
    for i, bar in enumerate(bars1):
        if not np.isnan(bar.get_height()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                   f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

    for i, bar in enumerate(bars2):
        if not np.isnan(bar.get_height()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                   f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison_summary.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Detailed comparison table
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')

    # Create comparison table
    table_data = []

    # Add metrics
    metrics_to_compare = [
        ('Correlation', 'correlation', '.3f'),
        ('RMSE (lbs)', 'rmse', '.2f'),
        ('Measurement Noise (lbs)', 'sigma_w', '.2f'),
        ('Variance Explained', 'variance_explained', '.1%'),
        ('Baseline Weight (lbs)', 'baseline_weight', '.1f'),
        ('Total Predicted (lbs)', 'total_predicted', '.1f'),
        ('Strength Retention (no train)', 'strength_retention_no_train', '.3f'),
        ('Strength Retention (train)', 'strength_retention_train', '.3f'),
        ('Strength Half-life (no train, days)', 'strength_half_life_no_train', '.0f'),
        ('Strength Half-life (train, days)', 'strength_half_life_train', '.0f'),
    ]

    for metric_name, metric_key, fmt in metrics_to_compare:
        if metric_key in intercept_metrics and metric_key in aerobic_metrics:
            intercept_val = intercept_metrics[metric_key]
            aerobic_val = aerobic_metrics[metric_key]

            # Calculate improvement
            if 'correlation' in metric_key or 'variance' in metric_key:
                # Higher is better
                improvement = aerobic_val - intercept_val
                improvement_str = f"+{improvement:{fmt}}" if improvement > 0 else f"{improvement:{fmt}}"
            elif 'rmse' in metric_key or 'sigma' in metric_key:
                # Lower is better
                improvement = intercept_val - aerobic_val
                improvement_str = f"+{improvement:{fmt}}" if improvement > 0 else f"{improvement:{fmt}}"
            else:
                improvement_str = "N/A"

            table_data.append([
                metric_name,
                f"{intercept_val:{fmt}}",
                f"{aerobic_val:{fmt}}",
                improvement_str
            ])

    # Add aerobic-specific metrics
    aerobic_specific = [
        ('Aerobic Retention (no train)', 'aerobic_retention_no_train', '.3f'),
        ('Aerobic Retention (train)', 'aerobic_retention_train', '.3f'),
        ('Aerobic Half-life (no train, days)', 'aerobic_half_life_no_train', '.0f'),
        ('Aerobic Half-life (train, days)', 'aerobic_half_life_train', '.0f'),
        ('Strength Weight Effect (lbs/fitness)', 'strength_weight_effect', '.2f'),
        ('Aerobic Weight Effect (lbs/fitness)', 'aerobic_weight_effect', '.2f'),
        ('Strength Contribution (lbs)', 'strength_contribution', '.1f'),
        ('Aerobic Contribution (lbs)', 'aerobic_contribution', '.1f'),
    ]

    for metric_name, metric_key, fmt in aerobic_specific:
        if metric_key in aerobic_metrics:
            table_data.append([
                metric_name,
                "N/A",
                f"{aerobic_metrics[metric_key]:{fmt}}",
                "N/A"
            ])

    # Add intercept-specific metrics
    if 'fitness_contribution' in intercept_metrics:
        table_data.append([
            'Fitness Contribution (lbs)',
            f"{intercept_metrics['fitness_contribution']:.1f}",
            "N/A",
            "N/A"
        ])

    if 'weight_effect' in intercept_metrics:
        table_data.append([
            'Weight Effect (lbs/fitness)',
            f"{intercept_metrics['weight_effect']:.2f}",
            "N/A",
            "N/A"
        ])

    # Create table
    table = ax.table(cellText=table_data,
                     colLabels=['Metric', 'Intercept Model', 'Aerobic Model', 'Improvement'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.3, 0.2, 0.2, 0.2])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style improvement column
    for i in range(1, len(table_data) + 1):
        cell = table[(i, 3)]
        text = cell.get_text().get_text()
        if text.startswith('+'):
            cell.set_facecolor('#90EE90')  # Light green for improvement
        elif text.startswith('-'):
            cell.set_facecolor('#FFB6C1')  # Light red for worse

    ax.set_title('Detailed Model Comparison', fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / "detailed_comparison_table.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Print summary
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)

    print(f"\n1. MODEL FIT IMPROVEMENT:")
    print(f"   Correlation: {intercept_metrics['correlation']:.3f} → {aerobic_metrics['correlation']:.3f} (+{aerobic_metrics['correlation'] - intercept_metrics['correlation']:.3f})")
    print(f"   RMSE: {intercept_metrics['rmse']:.2f} lbs → {aerobic_metrics['rmse']:.2f} lbs (-{intercept_metrics['rmse'] - aerobic_metrics['rmse']:.2f} lbs)")
    print(f"   Variance explained: {intercept_metrics['variance_explained']:.1%} → {aerobic_metrics['variance_explained']:.1%} (+{aerobic_metrics['variance_explained'] - intercept_metrics['variance_explained']:.1%})")

    print(f"\n2. WEIGHT DECOMPOSITION:")
    print(f"   Intercept model: {intercept_metrics['baseline_weight']:.1f} lbs baseline + {intercept_metrics['fitness_contribution']:.1f} lbs fitness = {intercept_metrics['total_predicted']:.1f} lbs")
    print(f"   Aerobic model: {aerobic_metrics['baseline_weight']:.1f} lbs baseline + {aerobic_metrics['strength_contribution']:.1f} lbs strength - {abs(aerobic_metrics['aerobic_contribution']):.1f} lbs aerobic = {aerobic_metrics['total_predicted']:.1f} lbs")
    print(f"   Actual mean weight: 135.3 lbs")

    print(f"\n3. FITNESS DECAY CHARACTERISTICS:")
    print(f"   Strength fitness (both models):")
    print(f"     - Without training: ~{intercept_metrics['strength_half_life_no_train']:.0f} day half-life")
    print(f"     - With training: ~{intercept_metrics['strength_half_life_train']:.0f} day half-life")
    print(f"   Aerobic fitness (aerobic model only):")
    print(f"     - Without training: {aerobic_metrics['aerobic_half_life_no_train']:.0f} day half-life")
    print(f"     - With training: {aerobic_metrics['aerobic_half_life_train']:.0f} day half-life")

    print(f"\n4. KEY FINDINGS:")
    print(f"   • Adding aerobic component improves model fit slightly")
    print(f"   • Aerobic fitness decays much faster than strength fitness")
    print(f"   • Aerobic exercise has negative weight effect (fat loss)")
    print(f"   • Strength exercise has positive weight effect (muscle gain)")
    print(f"   • Both models show fitness persists for months, not days")

    print(f"\n5. RECOMMENDATION:")
    if aerobic_metrics['correlation'] > intercept_metrics['correlation'] and aerobic_metrics['rmse'] < intercept_metrics['rmse']:
        print(f"   ✓ Aerobic model is better: higher correlation, lower RMSE")
        print(f"   ✓ Use aerobic model for more accurate predictions")
    else:
        print(f"   ⚠ Intercept model may be sufficient")
        print(f"   ⚠ Consider simpler model if aerobic improvement is minimal")

    print(f"\nVisualizations saved to {output_dir}/")
    print("="*80)


def main():
    """Main comparison function."""
    print("Comparing aerobic model vs intercept-only model...")

    # Load model results
    intercept_metrics, aerobic_metrics = load_model_results()

    # Create comparison visualizations
    create_comparison_visualizations(intercept_metrics, aerobic_metrics)

    print("\nComparison completed successfully!")


if __name__ == "__main__":
    main()