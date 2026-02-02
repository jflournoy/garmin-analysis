"""Generate comprehensive report of workout data with visualizations."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.workout import load_workout_data, prepare_workout_aggregates


def analyze_exercise_categories(df_workouts):
    """Analyze exercise categories from detailed exercise data."""
    if df_workouts.empty or 'exercise_details' not in df_workouts.columns:
        return None

    # Flatten exercise details
    all_exercises = []
    for idx, row in df_workouts.iterrows():
        if not row['exercise_details']:
            continue
        for ex in row['exercise_details']:
            ex_record = ex.copy()
            ex_record['workout_date'] = row['date']
            ex_record['workout_id'] = row['activity_id']
            all_exercises.append(ex_record)

    if not all_exercises:
        return None

    df_exercises = pd.DataFrame(all_exercises)

    # Analyze by category
    category_stats = df_exercises.groupby('category').agg({
        'reps': ['count', 'sum', 'mean', 'std'],
        'volume': ['sum', 'mean', 'std'],
        'duration': ['sum', 'mean', 'std'],
        'sets': ['sum', 'mean', 'std'],
    }).round(2)

    # Get top categories by total volume
    if 'volume' in df_exercises.columns:
        volume_by_category = df_exercises.groupby('category')['volume'].sum().sort_values(ascending=False)
    else:
        volume_by_category = None

    return {
        'df_exercises': df_exercises,
        'category_stats': category_stats,
        'volume_by_category': volume_by_category,
        'total_exercises': len(df_exercises),
        'unique_categories': df_exercises['category'].nunique() if 'category' in df_exercises.columns else 0
    }


def create_workout_report():
    """Create comprehensive workout data report with visualizations."""
    output_dir = Path("output/workout_report")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Workout Data Analysis Report ===")
    print(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 1. Load data
    print("1. Loading workout data...")
    df_workouts = load_workout_data(activity_type="strength_training")
    print(f"   Total workouts: {len(df_workouts)}")
    print(f"   Date range: {df_workouts['date'].min().date()} to {df_workouts['date'].max().date()}")
    print(f"   Total days: {(df_workouts['date'].max() - df_workouts['date'].min()).days + 1}")
    print(f"   Workout frequency: {len(df_workouts) / ((df_workouts['date'].max() - df_workouts['date'].min()).days + 1):.2f} workouts/day")
    print()

    # 2. Basic statistics
    print("2. Basic workout statistics:")
    metrics = ['duration', 'calories', 'total_reps', 'total_sets', 'total_volume']
    for metric in metrics:
        if metric in df_workouts.columns:
            values = df_workouts[metric]
            print(f"   {metric}:")
            print(f"     Min: {values.min():.1f}")
            print(f"     Max: {values.max():.1f}")
            print(f"     Mean: {values.mean():.1f}")
            print(f"     Std: {values.std():.1f}")
            print(f"     Median: {values.median():.1f}")
            print(f"     Total: {values.sum():.1f}")
    print()

    # 3. Check volume data quality
    print("3. Volume data quality check:")
    zero_volume = (df_workouts['total_volume'] == 0).sum()
    print(f"   Workouts with zero volume: {zero_volume}/{len(df_workouts)} ({zero_volume/len(df_workouts)*100:.1f}%)")

    # Check exercise details for volume information
    if 'exercise_details' in df_workouts.columns:
        workouts_with_details = df_workouts['exercise_details'].apply(lambda x: len(x) > 0 if x else False).sum()
        print(f"   Workouts with exercise details: {workouts_with_details}/{len(df_workouts)}")

        # Sample exercise details
        for idx in range(min(3, len(df_workouts))):
            details = df_workouts.iloc[idx]['exercise_details']
            if details:
                print(f"   Sample workout {idx+1}: {len(details)} exercises")
                for ex in details[:2]:  # Show first 2 exercises
                    print(f"     - {ex.get('category', 'Unknown')}: {ex.get('reps', 0)} reps, {ex.get('volume', 0)} volume")
    print()

    # 4. Temporal patterns
    print("4. Temporal patterns:")

    # Aggregate by different time periods
    df_workouts['year_month'] = df_workouts['date'].dt.strftime('%Y-%m')
    df_workouts['week_of_year'] = df_workouts['date'].dt.isocalendar().week
    df_workouts['day_of_week'] = df_workouts['date'].dt.day_name()

    # Monthly statistics
    monthly = df_workouts.groupby('year_month').agg({
        'activity_id': 'count',
        'total_volume': 'sum',
        'total_reps': 'sum',
        'total_sets': 'sum',
        'calories': 'sum'
    }).rename(columns={'activity_id': 'workout_count'})

    print("   Monthly summary:")
    print(monthly.round(1).to_string())
    print()

    # Day of week patterns
    dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_counts = df_workouts['day_of_week'].value_counts().reindex(dow_order, fill_value=0)
    print("   Workouts by day of week:")
    for day, count in dow_counts.items():
        print(f"     {day}: {count} workouts")
    print()

    # 5. Create visualizations

    # Setup figure
    fig = plt.figure(figsize=(15, 20))

    # 5.1 Workout frequency over time
    ax1 = plt.subplot(5, 2, 1)
    df_workouts_sorted = df_workouts.sort_values('date')
    ax1.plot(df_workouts_sorted['date'], np.arange(1, len(df_workouts_sorted) + 1), 'b-', linewidth=2)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cumulative Workout Count')
    ax1.set_title('Cumulative Workouts Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # 5.2 Workout calendar heatmap
    ax2 = plt.subplot(5, 2, 2)
    # Create binary workout calendar
    all_dates = pd.date_range(df_workouts['date'].min(), df_workouts['date'].max(), freq='D')
    workout_dates = set(df_workouts['date'].dt.date)
    workout_calendar = [1 if d.date() in workout_dates else 0 for d in all_dates]

    # Create matrix for heatmap (weeks x days)
    dates = pd.DataFrame({'date': all_dates, 'workout': workout_calendar})
    dates['year'] = dates['date'].dt.year
    dates['week'] = dates['date'].dt.isocalendar().week
    dates['day'] = dates['date'].dt.dayofweek

    # Pivot for heatmap
    heatmap_data = dates.pivot_table(index='week', columns='day', values='workout', aggfunc='sum')
    im = ax2.imshow(heatmap_data.fillna(0).values, cmap='YlOrRd', aspect='auto')
    ax2.set_xlabel('Day of Week (0=Mon, 6=Sun)')
    ax2.set_ylabel('Week of Year')
    ax2.set_title('Workout Calendar Heatmap')
    plt.colorbar(im, ax=ax2, label='Workouts per day')

    # 5.3 Workout volume over time
    ax3 = plt.subplot(5, 2, 3)
    ax3.scatter(df_workouts['date'], df_workouts['total_volume'], alpha=0.6, s=20)
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Total Volume')
    ax3.set_title('Workout Volume Over Time')
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

    # 5.4 Workout duration vs calories
    ax4 = plt.subplot(5, 2, 4)
    ax4.scatter(df_workouts['duration'] / 3600, df_workouts['calories'], alpha=0.6)
    ax4.set_xlabel('Duration (hours)')
    ax4.set_ylabel('Calories')
    ax4.set_title('Duration vs Calories')
    ax4.grid(True, alpha=0.3)

    # 5.5 Reps vs sets
    ax5 = plt.subplot(5, 2, 5)
    ax5.scatter(df_workouts['total_sets'], df_workouts['total_reps'], alpha=0.6)
    ax4.set_xlabel('Total Sets')
    ax5.set_ylabel('Total Reps')
    ax5.set_title('Sets vs Reps')
    ax5.grid(True, alpha=0.3)

    # 5.6 Histogram of workout metrics
    metrics_to_plot = ['duration', 'calories', 'total_reps', 'total_sets']
    for i, metric in enumerate(metrics_to_plot):
        ax = plt.subplot(5, 2, 6 + i)
        values = df_workouts[metric]
        ax.hist(values, bins=20, alpha=0.7, edgecolor='black')
        ax.set_xlabel(metric.replace('_', ' ').title())
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {metric.replace("_", " ").title()}')
        ax.grid(True, alpha=0.3)

        # Add vertical line at mean
        ax.axvline(values.mean(), color='red', linestyle='--', label=f'Mean: {values.mean():.1f}')
        ax.legend()

    # 5.7 Day of week patterns
    ax10 = plt.subplot(5, 2, 10)
    dow_counts.plot(kind='bar', ax=ax10, color='steelblue')
    ax10.set_xlabel('Day of Week')
    ax10.set_ylabel('Number of Workouts')
    ax10.set_title('Workout Frequency by Day of Week')
    ax10.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'workout_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"5. Visualizations saved to {output_dir / 'workout_analysis.png'}")

    # 6. Detailed exercise analysis
    print("\n6. Detailed exercise analysis:")
    exercise_analysis = analyze_exercise_categories(df_workouts)
    if exercise_analysis:
        print(f"   Total exercises recorded: {exercise_analysis['total_exercises']}")
        print(f"   Unique exercise categories: {exercise_analysis['unique_categories']}")

        # Create exercise category visualization
        if exercise_analysis['volume_by_category'] is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            # Volume by category
            top_categories = exercise_analysis['volume_by_category'].head(10)
            ax1.bar(range(len(top_categories)), top_categories.values)
            ax1.set_xticks(range(len(top_categories)))
            ax1.set_xticklabels(top_categories.index, rotation=45, ha='right')
            ax1.set_xlabel('Exercise Category')
            ax1.set_ylabel('Total Volume')
            ax1.set_title('Top 10 Exercise Categories by Volume')
            ax1.grid(True, alpha=0.3, axis='y')

            # Frequency by category
            df_exercises = exercise_analysis['df_exercises']
            category_counts = df_exercises['category'].value_counts().head(10)
            ax2.bar(range(len(category_counts)), category_counts.values)
            ax2.set_xticks(range(len(category_counts)))
            ax2.set_xticklabels(category_counts.index, rotation=45, ha='right')
            ax2.set_xlabel('Exercise Category')
            ax2.set_ylabel('Number of Exercises')
            ax2.set_title('Top 10 Exercise Categories by Frequency')
            ax2.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            plt.savefig(output_dir / 'exercise_categories.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"   Exercise category analysis saved to {output_dir / 'exercise_categories.png'}")

            # Print top categories
            print("\n   Top exercise categories by volume:")
            for i, (category, volume) in enumerate(exercise_analysis['volume_by_category'].head(10).items(), 1):
                print(f"     {i:2d}. {category}: {volume:,.0f}")

    # 7. Summary statistics for modeling
    print("\n7. Summary for cross-lagged GP modeling:")

    # Aggregate to daily for modeling
    for metric in ['count', 'volume', 'reps', 'sets', 'calories', 'duration']:
        df_agg = prepare_workout_aggregates(df_workouts, aggregation='daily', metric=metric)
        if not df_agg.empty:
            metric_col = f'workout_{metric}'
            total_days = len(df_agg)
            workout_days = (df_agg[metric_col] > 0).sum()
            zero_days = (df_agg[metric_col] == 0).sum()

            print(f"\n   Metric: {metric}")
            print(f"     Total days: {total_days}")
            print(f"     Days with workouts: {workout_days} ({workout_days/total_days*100:.1f}%)")
            print(f"     Days without workouts: {zero_days} ({zero_days/total_days*100:.1f}%)")
            if workout_days > 0:
                workout_values = df_agg[df_agg[metric_col] > 0][metric_col]
                print(f"     Mean (non-zero days): {workout_values.mean():.1f}")
                print(f"     Std (non-zero days): {workout_values.std():.1f}")
                print(f"     Total: {workout_values.sum():.1f}")

    # 8. Save data for reference
    print("\n8. Saving data files...")

    # Save raw data
    df_workouts.to_csv(output_dir / 'workouts_raw.csv', index=False)

    # Save daily aggregates
    for metric in ['count', 'volume', 'reps', 'sets']:
        df_agg = prepare_workout_aggregates(df_workouts, aggregation='daily', metric=metric)
        if not df_agg.empty:
            df_agg.to_csv(output_dir / f'workouts_daily_{metric}.csv', index=False)

    # Save summary statistics
    with open(output_dir / 'summary.txt', 'w') as f:
        f.write("Workout Data Analysis Summary\n")
        f.write("=============================\n\n")
        f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Total workouts: {len(df_workouts)}\n")
        f.write(f"Date range: {df_workouts['date'].min().date()} to {df_workouts['date'].max().date()}\n")
        f.write(f"Total days: {(df_workouts['date'].max() - df_workouts['date'].min()).days + 1}\n")
        f.write(f"Workout frequency: {len(df_workouts) / ((df_workouts['date'].max() - df_workouts['date'].min()).days + 1):.2f} workouts/day\n\n")

        f.write("Basic Statistics:\n")
        for metric in metrics:
            if metric in df_workouts.columns:
                values = df_workouts[metric]
                f.write(f"{metric}:\n")
                f.write(f"  Min: {values.min():.1f}\n")
                f.write(f"  Max: {values.max():.1f}\n")
                f.write(f"  Mean: {values.mean():.1f}\n")
                f.write(f"  Std: {values.std():.1f}\n")
                f.write(f"  Median: {values.median():.1f}\n")
                f.write(f"  Total: {values.sum():.1f}\n\n")

    print(f"   Data files saved to {output_dir}")
    print(f"\n{'='*60}")
    print("REPORT COMPLETE")
    print(f"{'='*60}")
    print("\nKey findings:")
    print(f"- {len(df_workouts)} strength training workouts over {((df_workouts['date'].max() - df_workouts['date'].min()).days + 1)} days")
    print(f"- Average frequency: {len(df_workouts) / ((df_workouts['date'].max() - df_workouts['date'].min()).days + 1):.2f} workouts/day")
    print(f"- Total volume lifted: {df_workouts['total_volume'].sum():,.0f} (but check data quality)")
    print(f"- Most common workout day: {dow_counts.idxmax() if not dow_counts.empty else 'N/A'}")
    print("\nNext steps for cross-lagged modeling:")
    print("1. Use 'calories' as primary magnitude metric (most reliably recorded)")
    print("   - Alternative: 'reps' or 'sets' but be aware of potential counting errors")
    print("2. Test lags: 0, 1, 2, 3, 7 days for muscle gain hypothesis")
    print("3. Account for sparse data (many zero days)")

    return output_dir


if __name__ == "__main__":
    create_workout_report()