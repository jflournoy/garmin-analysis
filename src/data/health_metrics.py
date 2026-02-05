"""Load and process health metrics from Garmin wellness data.

This module provides functions to load various health metrics from Garmin export:
- Sleep data (duration, stages, quality)
- Stress metrics (average stress, intensity, durations)
- Heart rate variability (HRV) if available
- Resting heart rate
- Body Battery (charged/drained values)
- Respiration data
- Other wellness metrics
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
from datetime import datetime


def load_sleep_data(data_dir: Path | str = "data") -> pd.DataFrame:
    """Load sleep data from Garmin wellness export.

    Args:
        data_dir: Path to data directory containing DI_CONNECT folder.

    Returns:
        DataFrame with sleep metrics:
        - date: Calendar date
        - sleep_start: Sleep start timestamp
        - sleep_end: Sleep end timestamp
        - total_sleep_minutes: Total sleep duration in minutes
        - deep_sleep_minutes: Deep sleep duration in minutes
        - light_sleep_minutes: Light sleep duration in minutes
        - rem_sleep_minutes: REM sleep duration in minutes
        - awake_minutes: Awake duration during sleep window
        - unmeasurable_minutes: Unmeasurable sleep duration
        - avg_respiration: Average respiration rate
        - lowest_respiration: Lowest respiration rate
        - highest_respiration: Highest respiration rate
    """
    data_dir = Path(data_dir)
    wellness_dir = data_dir / "DI_CONNECT/DI-Connect-Wellness"

    # Find all sleep data files
    sleep_files = list(wellness_dir.glob("*sleepData.json"))
    if not sleep_files:
        raise FileNotFoundError(f"No sleep data files found in {wellness_dir}")

    all_records = []

    for sleep_file in sleep_files:
        with open(sleep_file) as f:
            data = json.load(f)

        if not isinstance(data, list):
            continue

        for record in data:
            # Extract basic sleep metrics
            sleep_record = {
                'date': record.get('calendarDate'),
                'sleep_start': record.get('sleepStartTimestampGMT'),
                'sleep_end': record.get('sleepEndTimestampGMT'),
                'total_sleep_minutes': (
                    record.get('deepSleepSeconds', 0) +
                    record.get('lightSleepSeconds', 0) +
                    record.get('remSleepSeconds', 0) +
                    record.get('awakeSleepSeconds', 0)
                ) / 60,
                'deep_sleep_minutes': record.get('deepSleepSeconds', 0) / 60,
                'light_sleep_minutes': record.get('lightSleepSeconds', 0) / 60,
                'rem_sleep_minutes': record.get('remSleepSeconds', 0) / 60,
                'awake_minutes': record.get('awakeSleepSeconds', 0) / 60,
                'unmeasurable_minutes': record.get('unmeasurableSeconds', 0) / 60,
                'avg_respiration': record.get('averageRespiration'),
                'lowest_respiration': record.get('lowestRespiration'),
                'highest_respiration': record.get('highestRespiration'),
            }

            # Calculate sleep efficiency
            total_sleep = sleep_record['total_sleep_minutes']
            awake = sleep_record['awake_minutes']
            if total_sleep > 0:
                sleep_record['sleep_efficiency'] = (
                    (total_sleep - awake) / total_sleep * 100
                )
            else:
                sleep_record['sleep_efficiency'] = np.nan

            # Calculate sleep stage percentages
            if total_sleep > 0:
                sleep_record['deep_sleep_pct'] = (
                    sleep_record['deep_sleep_minutes'] / total_sleep * 100
                )
                sleep_record['light_sleep_pct'] = (
                    sleep_record['light_sleep_minutes'] / total_sleep * 100
                )
                sleep_record['rem_sleep_pct'] = (
                    sleep_record['rem_sleep_minutes'] / total_sleep * 100
                )
            else:
                sleep_record['deep_sleep_pct'] = np.nan
                sleep_record['light_sleep_pct'] = np.nan
                sleep_record['rem_sleep_pct'] = np.nan

            all_records.append(sleep_record)

    df = pd.DataFrame(all_records)

    # Convert date column to datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    # Sort by date
    if 'date' in df.columns and not df.empty:
        df = df.sort_values('date').reset_index(drop=True)

    return df


def load_daily_health_metrics(data_dir: Path | str = "data") -> pd.DataFrame:
    """Load daily health metrics from Garmin UDS (User Daily Summary) files.

    Args:
        data_dir: Path to data directory containing DI_CONNECT folder.

    Returns:
        DataFrame with daily health metrics:
        - date: Calendar date
        - resting_heart_rate: Resting heart rate (bpm)
        - avg_stress: Average stress level (0-100)
        - max_stress: Maximum stress level (0-100)
        - stress_duration_minutes: Total stress duration in minutes
        - rest_duration_minutes: Rest duration in minutes
        - activity_duration_minutes: Activity duration in minutes
        - body_battery_charged: Body Battery charged value
        - body_battery_drained: Body Battery drained value
        - body_battery_highest: Highest Body Battery value
        - body_battery_lowest: Lowest Body Battery value
        - avg_respiration: Average waking respiration rate
        - highest_respiration: Highest respiration rate
        - lowest_respiration: Lowest respiration rate
        - total_steps: Total steps
        - total_calories: Total calories burned
        - active_calories: Active calories burned
    """
    data_dir = Path(data_dir)
    aggregator_dir = data_dir / "DI_CONNECT/DI-Connect-Aggregator"

    # Find all UDS files
    uds_files = list(aggregator_dir.glob("UDSFile_*.json"))
    if not uds_files:
        raise FileNotFoundError(f"No UDS files found in {aggregator_dir}")

    all_records = []

    for uds_file in uds_files:
        with open(uds_file) as f:
            data = json.load(f)

        if not isinstance(data, list):
            continue

        for record in data:
            daily_record = {
                'date': record.get('calendarDate'),
                'resting_heart_rate': record.get('restingHeartRate'),
                'total_steps': record.get('totalSteps'),
                'total_calories': record.get('totalKilocalories'),
                'active_calories': record.get('activeKilocalories'),
                'bmr_calories': record.get('bmrKilocalories'),
                'wellness_calories': record.get('wellnessKilocalories'),
                'total_distance_meters': record.get('totalDistanceMeters'),
                'wellness_distance_meters': record.get('wellnessDistanceMeters'),
            }

            # Extract stress data
            stress_data = record.get('stress')
            if isinstance(stress_data, dict):
                aggregator_list = stress_data.get('aggregatorList', [])
                for agg in aggregator_list:
                    if agg.get('type') == 'TOTAL':
                        daily_record.update({
                            'avg_stress': agg.get('averageStressLevel'),
                            'max_stress': agg.get('maxStressLevel'),
                            'stress_duration_minutes': agg.get('stressDuration', 0) / 60,
                            'rest_duration_minutes': agg.get('restDuration', 0) / 60,
                            'activity_duration_minutes': agg.get('activityDuration', 0) / 60,
                            'uncategorized_duration_minutes': agg.get('uncategorizedDuration', 0) / 60,
                            'total_stress_count': agg.get('totalStressCount'),
                            'low_stress_minutes': agg.get('lowDuration', 0) / 60,
                            'medium_stress_minutes': agg.get('mediumDuration', 0) / 60,
                            'high_stress_minutes': agg.get('highDuration', 0) / 60,
                        })
                        break

            # Extract Body Battery data
            body_battery = record.get('bodyBattery')
            if isinstance(body_battery, dict):
                daily_record.update({
                    'body_battery_charged': body_battery.get('chargedValue'),
                    'body_battery_drained': body_battery.get('drainedValue'),
                })

                # Extract highest and lowest from stat list
                stat_list = body_battery.get('bodyBatteryStatList', [])
                for stat in stat_list:
                    stat_type = stat.get('bodyBatteryStatType')
                    if stat_type == 'HIGHEST':
                        daily_record['body_battery_highest'] = stat.get('statsValue')
                    elif stat_type == 'LOWEST':
                        daily_record['body_battery_lowest'] = stat.get('statsValue')

            # Extract respiration data
            respiration = record.get('respiration')
            if isinstance(respiration, dict):
                daily_record.update({
                    'avg_respiration': respiration.get('avgWakingRespirationValue'),
                    'highest_respiration': respiration.get('highestRespirationValue'),
                    'lowest_respiration': respiration.get('lowestRespirationValue'),
                    'latest_respiration': respiration.get('latestRespirationValue'),
                })

            all_records.append(daily_record)

    df = pd.DataFrame(all_records)

    # Convert date column to datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    # Sort by date
    if 'date' in df.columns and not df.empty:
        df = df.sort_values('date').reset_index(drop=True)

    return df


def load_combined_health_data(data_dir: Path | str = "data") -> pd.DataFrame:
    """Load and combine all available health metrics.

    This function combines sleep data and daily health metrics into a single
    DataFrame with one row per day.

    Args:
        data_dir: Path to data directory containing DI_CONNECT folder.

    Returns:
        Combined DataFrame with all health metrics.
    """
    # Load individual datasets
    sleep_df = load_sleep_data(data_dir)
    daily_df = load_daily_health_metrics(data_dir)

    if sleep_df.empty and daily_df.empty:
        return pd.DataFrame()

    # Merge datasets on date
    if not sleep_df.empty and not daily_df.empty:
        # Ensure both have date columns
        if 'date' in sleep_df.columns and 'date' in daily_df.columns:
            # Convert to date-only for merging (remove time component)
            sleep_df['date_only'] = sleep_df['date'].dt.date
            daily_df['date_only'] = daily_df['date'].dt.date

            # Rename conflicting columns in sleep data before merge
            # Both datasets have respiration columns - rename sleep ones
            sleep_rename = {}
            for col in sleep_df.columns:
                if col not in ['date', 'date_only'] and 'respiration' in col.lower():
                    sleep_rename[col] = f'sleep_{col}'

            if sleep_rename:
                sleep_df = sleep_df.rename(columns=sleep_rename)

            # Merge
            combined_df = pd.merge(
                daily_df,
                sleep_df.drop(columns=['date']),
                on='date_only',
                how='outer'
            )

            # Use date from daily_df if available, otherwise from sleep_df
            # First rename date columns to avoid conflict
            if 'date_x' in combined_df.columns and 'date_y' in combined_df.columns:
                combined_df['date'] = combined_df['date_x'].combine_first(combined_df['date_y'])
                # Drop helper columns
                combined_df = combined_df.drop(
                    columns=['date_only', 'date_x', 'date_y'],
                    errors='ignore'
                )
            elif 'date_x' in combined_df.columns:
                combined_df = combined_df.rename(columns={'date_x': 'date'})
                combined_df = combined_df.drop(columns=['date_only'], errors='ignore')
            elif 'date_y' in combined_df.columns:
                combined_df = combined_df.rename(columns={'date_y': 'date'})
                combined_df = combined_df.drop(columns=['date_only'], errors='ignore')
        else:
            # If one dataset is empty, use the other
            combined_df = daily_df if sleep_df.empty else sleep_df
    elif sleep_df.empty:
        combined_df = daily_df
    else:
        combined_df = sleep_df

    # Sort by date
    if 'date' in combined_df.columns and not combined_df.empty:
        combined_df = combined_df.sort_values('date').reset_index(drop=True)

    return combined_df


def prepare_health_metrics_for_analysis(
    health_df: pd.DataFrame,
    target_date: Optional[str] = None
) -> pd.DataFrame:
    """Prepare health metrics for cross-lagged analysis.

    This function:
    1. Filters to relevant date range if specified
    2. Handles missing values
    3. Creates derived metrics
    4. Ensures consistent date indexing

    Args:
        health_df: DataFrame with health metrics (from load_combined_health_data)
        target_date: Optional target date for filtering (format: 'YYYY-MM-DD')

    Returns:
        Prepared DataFrame ready for analysis.
    """
    if health_df.empty:
        return pd.DataFrame()

    df = health_df.copy()

    # Filter by date if specified
    if target_date and 'date' in df.columns:
        target_dt = pd.to_datetime(target_date)
        # Keep data up to target date
        df = df[df['date'] <= target_dt]

    # Create derived metrics
    # 1. Sleep quality score (composite metric)
    if all(col in df.columns for col in ['sleep_efficiency', 'deep_sleep_pct']):
        # Normalize metrics to 0-100 scale
        df['sleep_quality_score'] = (
            df['sleep_efficiency'].fillna(50) * 0.5 +
            df['deep_sleep_pct'].fillna(15) * 0.3 +
            (100 - df['awake_minutes'].fillna(30)).clip(0, 100) * 0.2
        )

    # 2. Stress recovery ratio
    if all(col in df.columns for col in ['rest_duration_minutes', 'stress_duration_minutes']):
        df['stress_recovery_ratio'] = (
            df['rest_duration_minutes'] / (df['stress_duration_minutes'] + 1)
        )

    # 3. Body Battery net change
    if all(col in df.columns for col in ['body_battery_charged', 'body_battery_drained']):
        df['body_battery_net'] = (
            df['body_battery_charged'] - df['body_battery_drained']
        )

    # 4. Activity intensity
    if all(col in df.columns for col in ['active_calories', 'total_steps']):
        df['activity_intensity'] = (
            df['active_calories'] / (df['total_steps'] + 1) * 1000
        )

    # Handle missing values
    # For time series analysis, we might want to interpolate or carry forward
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        # Simple forward fill for missing values (carry last observation forward)
        df[col] = df[col].ffill()

    # Ensure date is the index
    if 'date' in df.columns:
        df = df.set_index('date')

    return df


def get_available_health_metrics(data_dir: Path | str = "data") -> Dict[str, List[str]]:
    """Get list of available health metrics in the data.

    Args:
        data_dir: Path to data directory.

    Returns:
        Dictionary with metric categories and available metrics.
    """
    try:
        health_df = load_combined_health_data(data_dir)
    except Exception as e:
        return {"error": str(e)}

    if health_df.empty:
        return {"available_metrics": []}

    # Categorize metrics
    categories = {
        "sleep": [
            col for col in health_df.columns
            if 'sleep' in col.lower() or 'rem' in col.lower()
        ],
        "stress": [
            col for col in health_df.columns
            if 'stress' in col.lower()
        ],
        "heart": [
            col for col in health_df.columns
            if 'heart' in col.lower() or 'hr' in col.lower()
        ],
        "activity": [
            col for col in health_df.columns
            if 'step' in col.lower() or 'calori' in col.lower() or 'distance' in col.lower()
        ],
        "respiration": [
            col for col in health_df.columns
            if 'respir' in col.lower()
        ],
        "body_battery": [
            col for col in health_df.columns
            if 'battery' in col.lower()
        ],
        "other": [
            col for col in health_df.columns
            if not any(keyword in col.lower() for keyword in [
                'sleep', 'stress', 'heart', 'hr', 'step', 'calori',
                'distance', 'respir', 'battery', 'date'
            ])
        ]
    }

    # Remove empty categories
    categories = {k: v for k, v in categories.items() if v}

    # Add summary statistics
    date_range = {"start": None, "end": None}
    if 'date' in health_df.columns:
        try:
            date_range = {
                "start": health_df['date'].min().strftime('%Y-%m-%d'),
                "end": health_df['date'].max().strftime('%Y-%m-%d'),
            }
        except Exception:
            pass

    summary = {
        "total_metrics": len(health_df.columns),
        "total_days": len(health_df),
        "date_range": date_range,
        "categories": categories
    }

    return summary


if __name__ == "__main__":
    # Example usage
    print("Loading health metrics...")

    try:
        # Get available metrics
        available = get_available_health_metrics()
        print("\nAvailable Health Metrics:")
        print("=" * 50)

        if "error" in available:
            print(f"Error: {available['error']}")
        else:
            print(f"Total metrics: {available['total_metrics']}")
            print(f"Total days: {available['total_days']}")
            print(f"Date range: {available['date_range']['start']} to {available['date_range']['end']}")

            for category, metrics in available['categories'].items():
                print(f"\n{category.upper()} ({len(metrics)} metrics):")
                for metric in metrics[:10]:  # Show first 10
                    print(f"  - {metric}")
                if len(metrics) > 10:
                    print(f"  ... and {len(metrics) - 10} more")

        # Load combined data
        print("\n\nLoading combined health data...")
        health_df = load_combined_health_data()

        if not health_df.empty:
            print(f"\nLoaded {len(health_df)} days of health data")
            print(f"Columns: {len(health_df.columns)}")
            print("\nFirst few rows:")
            print(health_df.head())

            # Show missing values
            print("\nMissing values per column:")
            missing = health_df.isnull().sum()
            print(missing[missing > 0])
        else:
            print("No health data loaded")

    except Exception as e:
        print(f"Error: {e}")