#!/usr/bin/env python3
"""Extract strength training metrics from Garmin workout data.

This script extracts detailed strength training metrics including:
- Exercise volume (weight × reps × sets)
- Total daily volume
- Exercise categories and types
- Reps and sets counts
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import sys

def load_strength_training_details(data_dir: Path = Path("data")) -> pd.DataFrame:
    """Load strength training workouts with exercise set details.

    Args:
        data_dir: Path to data directory containing DI_CONNECT folder.

    Returns:
        DataFrame with strength training workouts and exercise details.
    """
    # Path to summarized activities
    activities_path = data_dir / "DI_CONNECT" / "DI-Connect-Fitness" / "jocoflo@pm.me_0_summarizedActivities.json"

    if not activities_path.exists():
        raise FileNotFoundError(f"Activities file not found: {activities_path}")

    print(f"Loading activities from: {activities_path}")

    # Load JSON data
    with open(activities_path, 'r') as f:
        data = json.load(f)

    # Extract strength training activities
    strength_activities = []

    if 'summarizedActivitiesExport' in data[0]:
        activities = data[0]['summarizedActivitiesExport']

        for activity in activities:
            if activity.get('activityType') == 'strength_training':
                # Basic activity info
                activity_info = {
                    'activity_id': activity.get('activityId'),
                    'name': activity.get('name', 'Strength'),
                    'start_time_gmt': activity.get('startTimeGmt'),
                    'start_time_local': activity.get('startTimeLocal'),
                    'duration': activity.get('duration', 0) / 1000,  # Convert to seconds
                    'calories': activity.get('calories', 0),
                    'avg_hr': activity.get('avgHr'),
                    'max_hr': activity.get('maxHr'),
                    'min_hr': activity.get('minHr'),
                    'steps': activity.get('steps', 0),
                    'total_sets': activity.get('totalSets', 0),
                    'total_reps': activity.get('totalReps', 0),
                    'active_sets': activity.get('activeSets', 0),
                }

                # Convert timestamp to date
                if activity_info['start_time_gmt']:
                    dt = datetime.fromtimestamp(activity_info['start_time_gmt'] / 1000)
                    activity_info['date'] = dt.date()
                    activity_info['datetime'] = dt
                else:
                    activity_info['date'] = None
                    activity_info['datetime'] = None

                # Extract exercise sets if available
                if 'summarizedExerciseSets' in activity:
                    exercise_sets = activity['summarizedExerciseSets']

                    for i, exercise in enumerate(exercise_sets):
                        exercise_info = activity_info.copy()
                        exercise_info.update({
                            'exercise_index': i,
                            'category': exercise.get('category', 'UNKNOWN'),
                            'sub_category': exercise.get('subCategory', ''),
                            'reps': exercise.get('reps', 0),
                            'volume': exercise.get('volume', 0),  # in grams
                            'duration': exercise.get('duration', 0) / 1000,  # in seconds
                            'sets': exercise.get('sets', 0),
                            'max_weight': exercise.get('maxWeight', 0),  # in grams
                        })

                        # Convert grams to pounds for readability
                        exercise_info['volume_lbs'] = exercise_info['volume'] * 0.00220462
                        exercise_info['max_weight_lbs'] = exercise_info['max_weight'] * 0.00220462

                        strength_activities.append(exercise_info)
                else:
                    # No exercise details, just add activity-level info
                    activity_info.update({
                        'exercise_index': 0,
                        'category': 'NO_DETAILS',
                        'sub_category': '',
                        'reps': 0,
                        'volume': 0,
                        'duration': 0,
                        'sets': 0,
                        'max_weight': 0,
                        'volume_lbs': 0,
                        'max_weight_lbs': 0,
                    })
                    strength_activities.append(activity_info)

    # Convert to DataFrame
    df = pd.DataFrame(strength_activities)

    if len(df) > 0:
        # Sort by date and time
        df = df.sort_values(['date', 'datetime', 'activity_id', 'exercise_index'])
        df = df.reset_index(drop=True)

        print(f"Loaded {len(df)} exercise records from {df['activity_id'].nunique()} strength training workouts")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")

        # Summary statistics
        print(f"\nSummary statistics:")
        print(f"  Total volume: {df['volume_lbs'].sum():.1f} lbs")
        print(f"  Total reps: {df['reps'].sum():,}")
        print(f"  Total sets: {df['sets'].sum():,}")
        print(f"  Unique exercise categories: {df['category'].nunique()}")
        print(f"  Most common categories:")
        for category, count in df['category'].value_counts().head(10).items():
            print(f"    {category}: {count} records")

    return df

def compute_daily_strength_metrics(df_strength: pd.DataFrame) -> pd.DataFrame:
    """Compute daily aggregated strength training metrics.

    Args:
        df_strength: DataFrame from load_strength_training_details.

    Returns:
        DataFrame with daily strength metrics.
    """
    if len(df_strength) == 0:
        return pd.DataFrame(columns=['date', 'strength_volume', 'strength_reps',
                                     'strength_sets', 'n_workouts', 'n_exercises'])

    # Group by date
    daily_metrics = df_strength.groupby('date').agg({
        'volume': 'sum',  # in grams
        'reps': 'sum',
        'sets': 'sum',
        'activity_id': 'nunique',  # number of workouts
        'category': 'count',  # number of exercise records
        'duration': 'sum',  # total workout duration in seconds
        'calories': 'sum',
    }).reset_index()

    # Rename columns
    daily_metrics = daily_metrics.rename(columns={
        'volume': 'strength_volume',  # in grams
        'reps': 'strength_reps',
        'sets': 'strength_sets',
        'activity_id': 'n_workouts',
        'category': 'n_exercises',
        'duration': 'strength_duration',
        'calories': 'strength_calories',
    })

    # Convert volume to pounds for readability
    daily_metrics['strength_volume_lbs'] = daily_metrics['strength_volume'] * 0.00220462

    # Ensure date is datetime
    daily_metrics['date'] = pd.to_datetime(daily_metrics['date'])

    print(f"\nDaily strength metrics computed for {len(daily_metrics)} days")
    print(f"  Days with strength training: {len(daily_metrics)}")
    print(f"  Average daily volume: {daily_metrics['strength_volume_lbs'].mean():.1f} lbs")
    print(f"  Average daily reps: {daily_metrics['strength_reps'].mean():.0f}")
    print(f"  Average daily sets: {daily_metrics['strength_sets'].mean():.0f}")

    return daily_metrics

def categorize_exercises_by_muscle_groups(df_strength: pd.DataFrame) -> pd.DataFrame:
    """Categorize exercises by primary muscle groups for more nuanced analysis.

    Args:
        df_strength: DataFrame with exercise details.

    Returns:
        DataFrame with muscle group categorization.
    """
    # Define muscle group mappings
    muscle_group_map = {
        # Upper body pushing
        'BENCH_PRESS': 'chest',
        'PUSH_UP': 'chest',
        'DIPS': 'chest_triceps',
        'SHOULDER_PRESS': 'shoulders',
        'OVERHEAD_PRESS': 'shoulders',

        # Upper body pulling
        'ROW': 'back',
        'PULL_UP': 'back',
        'LAT_PULLDOWN': 'back',
        'FACE_PULL': 'rear_delts',

        # Arms
        'BICEP_CURL': 'biceps',
        'TRICEPS_EXTENSION': 'triceps',
        'SKULLCRUSHER': 'triceps',

        # Lower body
        'SQUAT': 'legs',
        'DEADLIFT': 'posterior_chain',
        'LUNGE': 'legs',
        'LEG_PRESS': 'legs',
        'LEG_EXTENSION': 'quads',
        'LEG_CURL': 'hamstrings',
        'CALF_RAISE': 'calves',

        # Shoulders
        'LATERAL_RAISE': 'shoulders',
        'REAR_DELT_RAISE': 'rear_delts',
        'FRONT_RAISE': 'shoulders',

        # Chest
        'FLYE': 'chest',
        'CABLE_CROSSOVER': 'chest',

        # Core
        'CRUNCH': 'abs',
        'PLANK': 'core',
        'RUSSIAN_TWIST': 'obliques',
        'LEG_RAISE': 'abs',
    }

    # Apply mapping
    df = df_strength.copy()
    df['muscle_group'] = df['category'].map(muscle_group_map)
    df['muscle_group'] = df['muscle_group'].fillna('other')

    # Count by muscle group
    muscle_group_counts = df['muscle_group'].value_counts()

    print(f"\nExercise categorization by muscle group:")
    for group, count in muscle_group_counts.head(15).items():
        print(f"  {group}: {count} records")

    return df

def main():
    """Main function to extract and analyze strength training metrics."""
    data_dir = Path("data")

    try:
        # Load strength training details
        df_strength = load_strength_training_details(data_dir)

        if len(df_strength) == 0:
            print("No strength training data found!")
            return

        # Compute daily metrics
        df_daily = compute_daily_strength_metrics(df_strength)

        # Categorize exercises
        df_categorized = categorize_exercises_by_muscle_groups(df_strength)

        # Save results
        output_dir = Path("output/strength_analysis")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed data
        df_strength.to_csv(output_dir / "strength_exercise_details.csv", index=False)
        df_daily.to_csv(output_dir / "daily_strength_metrics.csv", index=False)
        df_categorized.to_csv(output_dir / "strength_exercises_categorized.csv", index=False)

        print(f"\nResults saved to: {output_dir}/")
        print(f"  strength_exercise_details.csv - Detailed exercise records")
        print(f"  daily_strength_metrics.csv - Daily aggregated metrics")
        print(f"  strength_exercises_categorized.csv - Exercises categorized by muscle group")

        # Create summary report
        summary = {
            'total_workouts': df_strength['activity_id'].nunique(),
            'total_exercise_records': len(df_strength),
            'date_range_start': str(df_strength['date'].min()),
            'date_range_end': str(df_strength['date'].max()),
            'total_volume_lbs': float(df_strength['volume_lbs'].sum()),
            'total_reps': int(df_strength['reps'].sum()),
            'total_sets': int(df_strength['sets'].sum()),
            'unique_exercise_categories': int(df_strength['category'].nunique()),
            'days_with_training': len(df_daily),
            'avg_daily_volume_lbs': float(df_daily['strength_volume_lbs'].mean()),
            'avg_daily_reps': float(df_daily['strength_reps'].mean()),
            'avg_daily_sets': float(df_daily['strength_sets'].mean()),
        }

        with open(output_dir / "strength_analysis_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nSummary saved to: {output_dir}/strength_analysis_summary.json")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()