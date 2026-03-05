#!/usr/bin/env python3
"""Test different scaling options for intensity."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.data.weight import load_weight_data
from src.data.intensity import load_intensity_by_activity

# Load data
data_dir = Path("data")
df_weight = load_weight_data(data_dir)
df_intensity = load_intensity_by_activity(
    data_dir=data_dir,
    activity_types=['walking', 'cycling', 'strength_training'],
    max_hr=185.0,
)

# Create daily dataframe
all_dates = set()
all_dates.update(df_weight['timestamp'].dt.date)
all_dates.update(df_intensity['date'].dt.date)

min_date = min(all_dates)
max_date = max(all_dates)
date_range = pd.date_range(start=min_date, end=max_date, freq='D')
df_daily = pd.DataFrame({'date': date_range})

# Merge intensity
for activity in ['walking', 'cycling', 'strength_training']:
    if activity in df_intensity.columns:
        df_act = df_intensity[['date', activity]].copy()
        df_daily = pd.merge(df_daily, df_act, on='date', how='left')
        df_daily[activity] = df_daily[activity].fillna(0.0)
    else:
        df_daily[activity] = 0.0

df_daily['aerobic_intensity'] = df_daily['walking'] + df_daily['cycling']
df_daily['strength_intensity'] = df_daily['strength_training']

print("Original intensity statistics:")
for col in ['aerobic_intensity', 'strength_intensity']:
    print(f"\n{col}:")
    print(f"  min={df_daily[col].min():.2f}")
    print(f"  max={df_daily[col].max():.2f}")
    print(f"  mean={df_daily[col].mean():.2f}")
    print(f"  std={df_daily[col].std():.2f}")
    print(f"  non-zero days={(df_daily[col] > 0).sum()}/{len(df_daily)}")

# Test different scaling options
scaling_options = [
    ("(value - min)/std", lambda x: (x - x.min()) / x.std() if x.std() > 0 else x - x.min()),
    ("(value - min)/std / 1000", lambda x: (x - x.min()) / x.std() / 1000.0 if x.std() > 0 else (x - x.min()) / 1000.0),
    ("(value - min)/10000", lambda x: (x - x.min()) / 10000.0),
    ("log(value - min + 1)", lambda x: np.log(x - x.min() + 1)),
    ("sqrt(value - min)", lambda x: np.sqrt(x - x.min())),
]

print("\n\nScaling comparisons:")
for col in ['aerobic_intensity', 'strength_intensity']:
    print(f"\n{col}:")
    for name, func in scaling_options:
        scaled = func(df_daily[col])
        print(f"  {name:30s}: min={scaled.min():.4f}, max={scaled.max():.4f}, mean={scaled.mean():.4f}, std={scaled.std():.4f}")

# What does a typical workout look like?
print("\n\nTypical workout examples:")
# Find days with workouts
aerobic_days = df_daily[df_daily['aerobic_intensity'] > 0]
strength_days = df_daily[df_daily['strength_intensity'] > 0]

if len(aerobic_days) > 0:
    sample = aerobic_days.sample(min(5, len(aerobic_days)))
    print("\nSample aerobic workouts (original units):")
    for idx, row in sample.iterrows():
        print(f"  Date: {row['date'].date()}, Intensity: {row['aerobic_intensity']:.0f}")
        # Convert to "minutes at 50 bpm above resting" equivalent
        # intensity = duration * HR_diff, so duration = intensity / HR_diff
        # Assuming HR_diff = 50 bpm (typical workout)
        equiv_min = row['aerobic_intensity'] / 50 / 60  # Convert to minutes
        print(f"    Equivalent: {equiv_min:.1f} minutes at 50 bpm above resting")

if len(strength_days) > 0:
    sample = strength_days.sample(min(5, len(strength_days)))
    print("\nSample strength workouts (original units):")
    for idx, row in sample.iterrows():
        print(f"  Date: {row['date'].date()}, Intensity: {row['strength_intensity']:.0f}")
        equiv_min = row['strength_intensity'] / 50 / 60
        print(f"    Equivalent: {equiv_min:.1f} minutes at 50 bpm above resting")

# Think about reasonable fitness scaling
print("\n\nThinking about reasonable fitness units:")
print("If we want fitness to range from 0 to ~10 (like your example):")
print("  And typical workout adds ~0.1-1.0 fitness units")
print("  Then intensity should be scaled so typical workout = 0.1-1.0")

# Check what scaling gives typical workout = 0.5
typical_aerobic = aerobic_days['aerobic_intensity'].median()
typical_strength = strength_days['strength_intensity'].median()

print(f"\nMedian aerobic workout: {typical_aerobic:.0f} units")
print(f"Median strength workout: {typical_strength:.0f} units")

scale_factor_aerobic = typical_aerobic / 0.5  # To make median = 0.5
scale_factor_strength = typical_strength / 0.5

print(f"\nTo make median = 0.5 fitness units:")
print(f"  Aerobic: divide by {scale_factor_aerobic:.0f}")
print(f"  Strength: divide by {scale_factor_strength:.0f}")

# Simpler: just divide by 100,000?
print(f"\nDivide by 100,000:")
print(f"  Aerobic median: {typical_aerobic/100000:.3f}")
print(f"  Strength median: {typical_strength/100000:.3f}")