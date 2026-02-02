"""Load sleep data from Garmin sleepData files."""
import json
from pathlib import Path

import pandas as pd


def load_sleep_data(data_dir: Path | str = "data") -> pd.DataFrame:
    """Load sleep metrics from Garmin sleepData files.

    Args:
        data_dir: Path to the data directory containing DI_CONNECT folder.

    Returns:
        DataFrame with columns: date, sleep_start_gmt, sleep_end_gmt,
        total_sleep_seconds, deep_sleep_seconds, light_sleep_seconds,
        rem_sleep_seconds, awake_seconds, unmeasurable_seconds,
        sleep_efficiency, deep_sleep_percent, light_sleep_percent,
        rem_sleep_percent, awake_percent, average_respiration,
        lowest_respiration, highest_respiration, etc.
    """
    data_dir = Path(data_dir)
    sleep_dir = data_dir / "DI_CONNECT/DI-Connect-Wellness"

    # Find all sleepData files
    sleep_files = list(sleep_dir.glob("*_sleepData.json"))
    if not sleep_files:
        raise FileNotFoundError(f"No sleepData files found in {sleep_dir}")

    records = []
    for filepath in sleep_files:
        with open(filepath) as f:
            data = json.load(f)

        for entry in data:
            # Extract date
            date_str = entry.get("calendarDate")
            if not date_str:
                continue

            # Sleep stage durations (seconds)
            deep = entry.get("deepSleepSeconds", 0)
            light = entry.get("lightSleepSeconds", 0)
            rem = entry.get("remSleepSeconds", 0)
            awake = entry.get("awakeSleepSeconds", 0)
            unmeasurable = entry.get("unmeasurableSeconds", 0)

            total_sleep = deep + light + rem + awake + unmeasurable
            # Sleep efficiency: (total - awake) / total
            sleep_efficiency = (total_sleep - awake) / total_sleep if total_sleep > 0 else None

            # Percentages
            deep_pct = deep / total_sleep if total_sleep > 0 else 0
            light_pct = light / total_sleep if total_sleep > 0 else 0
            rem_pct = rem / total_sleep if total_sleep > 0 else 0
            awake_pct = awake / total_sleep if total_sleep > 0 else 0

            record = {
                "date": pd.to_datetime(date_str),
                "sleep_start_gmt": pd.to_datetime(entry.get("sleepStartTimestampGMT")),
                "sleep_end_gmt": pd.to_datetime(entry.get("sleepEndTimestampGMT")),
                "total_sleep_seconds": total_sleep,
                "deep_sleep_seconds": deep,
                "light_sleep_seconds": light,
                "rem_sleep_seconds": rem,
                "awake_seconds": awake,
                "unmeasurable_seconds": unmeasurable,
                "sleep_efficiency": sleep_efficiency,
                "deep_sleep_percent": deep_pct,
                "light_sleep_percent": light_pct,
                "rem_sleep_percent": rem_pct,
                "awake_percent": awake_pct,
                "average_respiration": entry.get("averageRespiration"),
                "lowest_respiration": entry.get("lowestRespiration"),
                "highest_respiration": entry.get("highestRespiration"),
                "sleep_window_confirmation_type": entry.get("sleepWindowConfirmationType"),
                "retro": entry.get("retro", False),
            }
            records.append(record)

    df = pd.DataFrame(records)
    if df.empty:
        return df

    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)

    # Remove duplicate dates (keep first occurrence - assume one sleep session per day)
    df = df.drop_duplicates(subset="date", keep="first")

    return df


def merge_sleep_with_daily(
    df_daily: pd.DataFrame,
    df_sleep: pd.DataFrame,
) -> pd.DataFrame:
    """Merge sleep data with daily metrics DataFrame.

    Args:
        df_daily: Daily metrics DataFrame (from load_daily_metrics).
        df_sleep: Sleep DataFrame (from load_sleep_data).

    Returns:
        Merged DataFrame with sleep columns added.
    """
    merged = pd.merge(df_daily, df_sleep, on="date", how="left", suffixes=("", "_sleep"))
    return merged


if __name__ == "__main__":
    # Test loading
    df = load_sleep_data()
    print(f"Loaded {len(df)} sleep records")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print("\nColumns:", df.columns.tolist())

    # Show missing values
    print("\nMissing values per column:")
    print(df.isnull().sum())

    # Show summary statistics
    print("\nSleep duration summary (hours):")
    df["total_sleep_hours"] = df["total_sleep_seconds"] / 3600
    print(df["total_sleep_hours"].describe())

    # Merge with daily metrics to test
    try:
        from .activity import load_daily_metrics
        df_daily = load_daily_metrics()
        merged = merge_sleep_with_daily(df_daily, df)
        print(f"\nMerged dataset shape: {merged.shape}")
        print(f"Dates with both sleep and daily metrics: {merged['total_sleep_seconds'].notnull().sum()}")
    except ImportError:
        print("\nCould not import activity module for merge test")