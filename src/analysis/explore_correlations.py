"""Explore correlations between weight and other Garmin metrics."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import merge_weight_with_daily_metrics


def plot_correlation_matrix(df: pd.DataFrame, output_dir: Path = Path("output/correlations")):
    """Plot correlation matrix between weight and other variables."""
    # Select numeric columns of interest
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Filter to relevant metrics (exclude derived columns)
    exclude = ["weight_day_of_week", "weight_day_of_year", "weight_variable"]
    metric_cols = [c for c in numeric_cols if c not in exclude and not c.startswith("_")]
    # Keep top 20 most relevant (prioritize weight columns and key metrics)
    weight_cols = [c for c in metric_cols if "weight" in c]
    other_cols = [c for c in metric_cols if "weight" not in c]
    # Select up to 15 other columns with most non-missing values
    other_cols = sorted(other_cols, key=lambda c: df[c].notnull().sum(), reverse=True)[:15]
    selected_cols = weight_cols + other_cols

    corr_df = df[selected_cols].corr()

    # Create figure
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title("Correlation Matrix: Weight vs Daily Metrics", fontsize=16)
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "correlation_matrix.png", dpi=150)
    plt.close()
    print(f"Saved correlation matrix to {output_dir / 'correlation_matrix.png'}")

    # Also save correlation with weight_mean as a table
    weight_mean_corr = corr_df["weight_mean"].sort_values(key=abs, ascending=False)
    weight_mean_corr.to_csv(output_dir / "weight_mean_correlations.csv")
    print(f"Saved weight_mean correlations to {output_dir / 'weight_mean_correlations.csv'}")


def plot_scatter_pairs(df: pd.DataFrame, output_dir: Path = Path("output/correlations")):
    """Create scatter plots of weight vs key metrics."""
    key_metrics = [
        "resting_heart_rate",
        "total_steps",
        "active_kilocalories",
        "avg_stress_level",
        "moderate_intensity_minutes",
        "vigorous_intensity_minutes",
        "highly_active_seconds",
        "min_heart_rate",
        "max_heart_rate",
    ]
    # Filter to available columns
    key_metrics = [m for m in key_metrics if m in df.columns]

    # Create subplot grid
    n_cols = 3
    n_rows = (len(key_metrics) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]

    for idx, metric in enumerate(key_metrics):
        ax = axes[idx]
        # Drop NA pairs
        subset = df[["weight_mean", metric]].dropna()
        if len(subset) < 2:
            ax.text(0.5, 0.5, f"No data for {metric}", ha="center", va="center")
            ax.set_title(f"{metric} (n={len(subset)})")
            continue

        ax.scatter(subset[metric], subset["weight_mean"], alpha=0.6, s=30)
        ax.set_xlabel(metric)
        ax.set_ylabel("Weight (lbs)")
        # Add correlation text
        corr = subset.corr().iloc[0, 1]
        ax.set_title(f"{metric} (r={corr:.2f}, n={len(subset)})")

        # Add trend line
        if len(subset) > 1:
            z = np.polyfit(subset[metric], subset["weight_mean"], 1)
            p = np.poly1d(z)
            x_range = np.linspace(subset[metric].min(), subset[metric].max(), 100)
            ax.plot(x_range, p(x_range), color="red", alpha=0.8, linewidth=1.5)

    # Hide unused subplots
    for idx in range(len(key_metrics), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Weight vs Daily Metrics Scatter Plots", fontsize=16)
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "scatter_pairs.png", dpi=150)
    plt.close()
    print(f"Saved scatter pairs to {output_dir / 'scatter_pairs.png'}")


def plot_time_series(df: pd.DataFrame, output_dir: Path = Path("output/correlations")):
    """Plot time series of weight and key metrics."""
    # Select a few key metrics to plot alongside weight
    key_metrics = ["resting_heart_rate", "total_steps", "active_kilocalories", "avg_stress_level"]
    key_metrics = [m for m in key_metrics if m in df.columns]

    fig, axes = plt.subplots(len(key_metrics) + 1, 1, figsize=(14, 3 * (len(key_metrics) + 1)), sharex=True)

    # Plot weight
    ax = axes[0]
    ax.plot(df["date"], df["weight_mean"], label="Weight (mean)", color="blue", linewidth=1.5)
    ax.fill_between(df["date"], df["weight_min"], df["weight_max"], alpha=0.2, color="blue", label="Weight range")
    ax.set_ylabel("Weight (lbs)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Plot each metric
    for idx, metric in enumerate(key_metrics, start=1):
        ax = axes[idx]
        ax.plot(df["date"], df[metric], label=metric, color=f"C{idx}", linewidth=1.5)
        ax.set_ylabel(metric)
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Date")
    plt.suptitle("Time Series: Weight and Daily Metrics", fontsize=16)
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "time_series.png", dpi=150)
    plt.close()
    print(f"Saved time series to {output_dir / 'time_series.png'}")


def main():
    """Run all exploration plots."""
    output_dir = Path("output/correlations")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading merged data...")
    df = merge_weight_with_daily_metrics()
    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Basic statistics
    print("\nWeight statistics:")
    print(df[["weight_mean", "weight_std", "weight_count"]].describe())

    print("\nKey metrics statistics:")
    key = ["resting_heart_rate", "total_steps", "active_kilocalories", "avg_stress_level"]
    for metric in key:
        if metric in df.columns:
            print(f"{metric}: mean={df[metric].mean():.2f}, sd={df[metric].std():.2f}")

    # Generate plots
    print("\nGenerating correlation matrix...")
    plot_correlation_matrix(df, output_dir)

    print("\nGenerating scatter plots...")
    plot_scatter_pairs(df, output_dir)

    print("\nGenerating time series plots...")
    plot_time_series(df, output_dir)

    # Save merged dataset for reference
    df.to_csv(output_dir / "merged_weight_daily.csv", index=False)
    print(f"\nSaved merged dataset to {output_dir / 'merged_weight_daily.csv'}")

    print("\nExploration complete.")


if __name__ == "__main__":
    main()