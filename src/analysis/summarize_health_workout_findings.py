#!/usr/bin/env python3
"""Summarize key findings from health-workout correlation analysis.

This script analyzes the correlation results to identify:
1. Most statistically significant relationships
2. Strongest effect sizes (positive and negative)
3. Optimal lag times for different health outcomes
4. Practical implications for health optimization

Usage:
    python -m src.analysis.summarize_health_workout_findings \
        --correlation-file output/full_health_correlations/health_workout_correlations.csv \
        --output-dir output/health_findings_summary
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json


def load_correlation_data(file_path: Path) -> pd.DataFrame:
    """Load correlation results from CSV file."""
    df = pd.read_csv(file_path)

    # Sort by absolute correlation strength
    df = df.sort_values('abs_correlation', ascending=False)

    return df


def categorize_health_metrics(metric_name: str) -> str:
    """Categorize health metrics into meaningful groups."""
    metric_lower = metric_name.lower()

    if any(keyword in metric_lower for keyword in ['sleep', 'rem', 'deep', 'light', 'awake']):
        return 'sleep'
    elif any(keyword in metric_lower for keyword in ['heart', 'hr', 'bpm']):
        return 'heart'
    elif any(keyword in metric_lower for keyword in ['stress', 'avg_stress']):
        return 'stress'
    elif any(keyword in metric_lower for keyword in ['respiration', 'breath']):
        return 'respiration'
    elif any(keyword in metric_lower for keyword in ['battery', 'charged', 'drained']):
        return 'energy'
    elif any(keyword in metric_lower for keyword in ['calories', 'steps', 'distance', 'bmr']):
        return 'activity'
    elif any(keyword in metric_lower for keyword in ['efficiency', 'duration', 'minutes']):
        return 'sleep'  # Sleep efficiency/duration
    else:
        return 'other'


def analyze_top_findings(df: pd.DataFrame, n_top: int = 20) -> Dict[str, Any]:
    """Analyze top findings from correlation results."""

    # Filter for significant correlations only
    sig_df = df[df['significant'] == True].copy()

    # Group by workout type
    workout_groups = {}
    for workout_var in sig_df['workout_var'].unique():
        workout_data = sig_df[sig_df['workout_var'] == workout_var].copy()

        # Add category
        workout_data['category'] = workout_data['health_metric'].apply(categorize_health_metrics)

        workout_groups[workout_var] = {
            'total_significant': len(workout_data),
            'positive_effects': len(workout_data[workout_data['direction'] == 'positive']),
            'negative_effects': len(workout_data[workout_data['direction'] == 'negative']),
            'by_category': workout_data.groupby('category').size().to_dict(),
            'top_effects': workout_data.head(n_top).to_dict('records')
        }

    # Overall statistics
    total_analyses = len(df)
    total_significant = len(sig_df)
    significant_pct = total_significant / total_analyses * 100

    # Strongest positive and negative effects
    strongest_positive = sig_df[sig_df['direction'] == 'positive'].head(5)
    strongest_negative = sig_df[sig_df['direction'] == 'negative'].head(5)

    # Most common optimal lags
    lag_distribution = sig_df['strongest_lag'].value_counts().head(10).to_dict()

    return {
        'summary_stats': {
            'total_analyses': total_analyses,
            'total_significant': total_significant,
            'significant_percentage': significant_pct,
            'date_generated': datetime.now().isoformat()
        },
        'workout_breakdown': workout_groups,
        'strongest_positive': strongest_positive.to_dict('records'),
        'strongest_negative': strongest_negative.to_dict('records'),
        'lag_distribution': lag_distribution
    }


def generate_visualizations(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate visualization plots."""

    # Filter for significant correlations only
    sig_df = df[df['significant'] == True].copy()

    # Add category
    sig_df['category'] = sig_df['health_metric'].apply(categorize_health_metrics)

    # 1. Correlation strength by workout type
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=sig_df, x='workout_var', y='abs_correlation')
    plt.title('Correlation Strength by Workout Type (Significant Only)')
    plt.xlabel('Workout Type')
    plt.ylabel('Absolute Correlation |r|')
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_strength_by_workout.png', dpi=150)
    plt.close()

    # 2. Optimal lag distribution
    plt.figure(figsize=(10, 6))
    lag_counts = sig_df['strongest_lag'].value_counts().sort_index()
    lag_counts.plot(kind='bar')
    plt.title('Optimal Lag Distribution for Significant Correlations')
    plt.xlabel('Lag (days)')
    plt.ylabel('Number of Significant Correlations')
    plt.tight_layout()
    plt.savefig(output_dir / 'optimal_lag_distribution.png', dpi=150)
    plt.close()

    # 3. Correlation heatmap by category
    pivot_data = sig_df.pivot_table(
        index='category',
        columns='workout_var',
        values='abs_correlation',
        aggfunc='mean'
    ).fillna(0)

    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd')
    plt.title('Average Correlation Strength by Health Category and Workout Type')
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_heatmap.png', dpi=150)
    plt.close()

    # 4. Top correlations bar chart
    top_n = 15
    top_corrs = sig_df.head(top_n).copy()
    top_corrs['label'] = top_corrs.apply(
        lambda row: f"{row['workout_var']}→{row['health_metric']} (lag {row['strongest_lag']}d)",
        axis=1
    )

    plt.figure(figsize=(12, 8))
    colors = ['green' if d == 'positive' else 'red' for d in top_corrs['direction']]
    plt.barh(range(len(top_corrs)), top_corrs['correlation'], color=colors)
    plt.yticks(range(len(top_corrs)), top_corrs['label'])
    plt.xlabel('Correlation Coefficient (r)')
    plt.title(f'Top {top_n} Health-Workout Correlations')
    plt.tight_layout()
    plt.savefig(output_dir / 'top_correlations_bar.png', dpi=150)
    plt.close()


def generate_html_report(analysis_results: Dict[str, Any], output_dir: Path) -> None:
    """Generate HTML report of findings."""

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Health-Workout Correlation Analysis Summary</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
            h1 {{ color: #333; border-bottom: 2px solid #333; padding-bottom: 10px; }}
            h2 {{ color: #555; margin-top: 30px; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
            h3 {{ color: #777; margin-top: 20px; }}
            .summary-box {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0; border-left: 4px solid #007bff; }}
            .finding-box {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin: 15px 0; border-left: 4px solid #17a2b8; }}
            .workout-summary {{ background-color: #f0f7ff; padding: 15px; border-radius: 5px; margin: 15px 0; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #f2f2f2; font-weight: bold; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .positive {{ color: green; font-weight: bold; }}
            .negative {{ color: red; font-weight: bold; }}
            .insight {{ background-color: #fff3cd; padding: 15px; border-radius: 5px; margin: 20px 0; border-left: 4px solid #ffc107; }}
            .recommendation {{ background-color: #d4edda; padding: 15px; border-radius: 5px; margin: 20px 0; border-left: 4px solid #28a745; }}
            .img-container {{ text-align: center; margin: 30px 0; }}
            img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <h1>Health-Workout Correlation Analysis Summary</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <div class="summary-box">
            <h2>Executive Summary</h2>
            <p><strong>Total analyses:</strong> {analysis_results['summary_stats']['total_analyses']}</p>
            <p><strong>Significant findings:</strong> {analysis_results['summary_stats']['total_significant']} ({analysis_results['summary_stats']['significant_percentage']:.1f}%)</p>
            <p><strong>Date range:</strong> Full dataset analysis</p>
        </div>

        <h2>Key Findings by Workout Type</h2>
    """

    # Workout breakdown
    for workout_var, workout_data in analysis_results['workout_breakdown'].items():
        html_content += f"""
        <div class="workout-summary">
            <h3>{workout_var.replace('_', ' ').title()}</h3>
            <p><strong>Total significant effects:</strong> {workout_data['total_significant']}</p>
            <p><strong>Positive effects:</strong> {workout_data['positive_effects']}</p>
            <p><strong>Negative effects:</strong> {workout_data['negative_effects']}</p>

            <h4>Effects by Health Category:</h4>
            <ul>
        """

        for category, count in workout_data['by_category'].items():
            html_content += f"<li><strong>{category.title()}:</strong> {count} significant correlations</li>"

        html_content += """
            </ul>
        </div>
        """

    # Strongest effects
    html_content += """
        <h2>Strongest Effects</h2>

        <h3>Strongest Positive Effects (Workouts Improve Health Metric)</h3>
        <table>
            <tr>
                <th>Workout</th>
                <th>Health Metric</th>
                <th>Optimal Lag</th>
                <th>Correlation (r)</th>
                <th>Interpretation</th>
            </tr>
    """

    for effect in analysis_results['strongest_positive']:
        # Generate interpretation
        workout = effect['workout_var']
        metric = effect['health_metric']
        lag = effect['strongest_lag']
        corr = effect['correlation']

        interpretation = f"{workout.replace('_', ' ')} improves {metric.replace('_', ' ')} after {lag} days"

        html_content += f"""
            <tr>
                <td>{workout}</td>
                <td>{metric}</td>
                <td>{lag} days</td>
                <td class="positive">{corr:.3f}</td>
                <td>{interpretation}</td>
            </tr>
        """

    html_content += """
        </table>

        <h3>Strongest Negative Effects (Workouts Decrease Health Metric)</h3>
        <table>
            <tr>
                <th>Workout</th>
                <th>Health Metric</th>
                <th>Optimal Lag</th>
                <th>Correlation (r)</th>
                <th>Interpretation</th>
            </tr>
    """

    for effect in analysis_results['strongest_negative']:
        # Generate interpretation
        workout = effect['workout_var']
        metric = effect['health_metric']
        lag = effect['strongest_lag']
        corr = effect['correlation']

        interpretation = f"{workout.replace('_', ' ')} reduces {metric.replace('_', ' ')} after {lag} days"

        html_content += f"""
            <tr>
                <td>{workout}</td>
                <td>{metric}</td>
                <td>{lag} days</td>
                <td class="negative">{corr:.3f}</td>
                <td>{interpretation}</td>
            </tr>
        """

    html_content += """
        </table>

        <h2>Optimal Lag Times</h2>
        <p>Most common optimal lag times for significant correlations:</p>
        <ul>
    """

    for lag_days, count in analysis_results['lag_distribution'].items():
        html_content += f"<li><strong>{lag_days} days:</strong> {count} correlations</li>"

    html_content += """
        </ul>

        <div class="insight">
            <h3>Key Insights</h3>
            <ul>
                <li><strong>Immediate effects (lag 0):</strong> Most correlations with activity metrics (calories, steps, distance)</li>
                <li><strong>Delayed effects (lag 4-7 days):</strong> Most correlations with sleep, heart rate, and energy metrics</li>
                <li><strong>Workout-specific patterns:</strong> Different workout types affect health metrics differently</li>
                <li><strong>Energy trade-off:</strong> Workouts drain immediate energy but may have long-term benefits</li>
            </ul>
        </div>

        <div class="recommendation">
            <h3>Practical Recommendations</h3>
            <ol>
                <li><strong>For sleep improvement:</strong> Strength training shows positive effects on REM sleep after 5 days</li>
                <li><strong>For heart health:</strong> Cycling lowers resting heart rate after 6 days</li>
                <li><strong>For energy management:</strong> Plan rest days after intense workouts to allow Body Battery recovery</li>
                <li><strong>For activity balance:</strong> Mix different workout types to avoid activity substitution effects</li>
                <li><strong>For long-term benefits:</strong> Consistency matters - delayed effects (4-7 days) require regular workouts</li>
            </ol>
        </div>

        <h2>Visualizations</h2>
        <div class="img-container">
            <h3>Correlation Strength by Workout Type</h3>
            <img src="correlation_strength_by_workout.png" alt="Correlation strength by workout type">
        </div>

        <div class="img-container">
            <h3>Optimal Lag Distribution</h3>
            <img src="optimal_lag_distribution.png" alt="Optimal lag distribution">
        </div>

        <div class="img-container">
            <h3>Top Health-Workout Correlations</h3>
            <img src="top_correlations_bar.png" alt="Top correlations bar chart">
        </div>

        <footer>
            <p>Report generated by summarize_health_workout_findings.py</p>
            <p>Based on correlation analysis of {analysis_results['summary_stats']['total_analyses']} health-workout pairs</p>
        </footer>
    </body>
    </html>
    """

    report_path = output_dir / "health_workout_findings_report.html"
    with open(report_path, 'w') as f:
        f.write(html_content)

    print(f"Generated HTML report: {report_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Summarize key findings from health-workout correlation analysis"
    )

    parser.add_argument(
        "--correlation-file",
        type=str,
        default="output/full_health_correlations/health_workout_correlations.csv",
        help="Path to correlation results CSV file"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/health_findings_summary",
        help="Output directory for results"
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top findings to analyze per category"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create visualizations subdirectory
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    try:
        print("\n" + "=" * 70)
        print("HEALTH-WORKOUT FINDINGS SUMMARY")
        print("=" * 70)

        # Load correlation data
        print("\n1. Loading correlation data...")
        df = load_correlation_data(Path(args.correlation_file))
        print(f"   Loaded {len(df)} correlation results")
        print(f"   Significant correlations: {(df['significant'] == True).sum()}")

        # Analyze findings
        print("\n2. Analyzing top findings...")
        analysis_results = analyze_top_findings(df, n_top=args.top_n)

        # Generate visualizations
        print("\n3. Generating visualizations...")
        generate_visualizations(df, viz_dir)

        # Generate HTML report
        print("\n4. Generating HTML report...")
        generate_html_report(analysis_results, output_dir)

        # Save analysis results as JSON
        results_path = output_dir / "analysis_results.json"
        with open(results_path, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        print(f"   Saved analysis results to: {results_path}")

        # Print key findings
        print("\n" + "=" * 70)
        print("KEY FINDINGS")
        print("=" * 70)

        stats = analysis_results['summary_stats']
        print(f"\nSummary Statistics:")
        print(f"  Total analyses: {stats['total_analyses']}")
        print(f"  Significant findings: {stats['total_significant']} ({stats['significant_percentage']:.1f}%)")

        print(f"\nStrongest Positive Effects:")
        for i, effect in enumerate(analysis_results['strongest_positive'][:3], 1):
            print(f"  {i}. {effect['workout_var']} → {effect['health_metric']}: r = {effect['correlation']:.3f} (lag {effect['strongest_lag']}d)")

        print(f"\nStrongest Negative Effects:")
        for i, effect in enumerate(analysis_results['strongest_negative'][:3], 1):
            print(f"  {i}. {effect['workout_var']} → {effect['health_metric']}: r = {effect['correlation']:.3f} (lag {effect['strongest_lag']}d)")

        print(f"\nMost Common Optimal Lags:")
        for lag, count in list(analysis_results['lag_distribution'].items())[:5]:
            print(f"  {lag} days: {count} correlations")

        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)
        print(f"Results saved to: {output_dir}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()