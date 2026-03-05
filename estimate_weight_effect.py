#!/usr/bin/env python3
"""Estimate weight effect of training from model parameters."""

import numpy as np

def simulate_fitness_accumulation(alpha_d, alpha_m, beta, intensity=1.0, days=30):
    """Simulate fitness accumulation with and without training."""
    # With training: decay = alpha_d - alpha_m (training reduces decay)
    # Without training: decay = alpha_d

    alpha_with_training = alpha_d - alpha_m  # Actually alpha_d + alpha_m gives total decay, but alpha_m reduces decay
    # Wait, correction: total decay = alpha_d + alpha_m, but alpha_m is POSITIVE (training reduces decay)
    # So with training: decay = alpha_d + alpha_m (more decay reduction = higher persistence)
    # Actually no: fitness[t] = (alpha_d + alpha_m * trained) * fitness[t-1] + gain
    # If alpha_m > 0, then alpha_d + alpha_m > alpha_d, meaning MORE persistence (less decay)
    # Decay = 1 - persistence

    # Let me recalculate: persistence = alpha_d + alpha_m * trained
    # Decay = 1 - persistence = 1 - (alpha_d + alpha_m * trained)

    persistence_no_train = alpha_d  # 0.478
    persistence_with_train = alpha_d + alpha_m  # 0.478 + 0.251 = 0.729

    decay_no_train = 1 - persistence_no_train  # 0.522
    decay_with_train = 1 - persistence_with_train  # 0.271

    print(f"Persistence (fraction kept each day):")
    print(f"  No training: {persistence_no_train:.3f} (decay: {decay_no_train:.3f})")
    print(f"  With training: {persistence_with_train:.3f} (decay: {decay_with_train:.3f})")
    print(f"  Training reduces decay by: {decay_no_train - decay_with_train:.3f} ({((decay_no_train - decay_with_train)/decay_no_train*100):.1f}%)")

    # Simulate
    fitness_no_train = np.zeros(days)
    fitness_with_train = np.zeros(days)

    for t in range(1, days):
        # No training
        fitness_no_train[t] = persistence_no_train * fitness_no_train[t-1]

        # With training (every day)
        fitness_with_train[t] = persistence_with_train * fitness_with_train[t-1] + beta * intensity

    return fitness_no_train, fitness_with_train

def main():
    """Estimate weight effect."""
    # Parameters from simplified model
    alpha_d = 0.478  # decay without training
    alpha_m = 0.251  # training reduces decay
    beta = 0.233     # gain per unit intensity
    gamma = 0.182    # weight effect per unit fitness
    weight_std = 2.77  # lbs per standardized unit

    print("Training Decay Model Weight Effect Estimation")
    print("="*60)
    print(f"Parameters from model:")
    print(f"  alpha_d (decay without training): {alpha_d:.3f}")
    print(f"  alpha_m (training reduces decay): {alpha_m:.3f}")
    print(f"  beta (gain per intensity): {beta:.3f}")
    print(f"  gamma (weight effect): {gamma:.3f}")
    print(f"  Weight std: {weight_std:.2f} lbs per standardized unit")

    # Simulate 30 days
    days = 30
    intensity = 1.0  # medium intensity

    fitness_no_train, fitness_with_train = simulate_fitness_accumulation(
        alpha_d, alpha_m, beta, intensity, days
    )

    # Calculate weight effects
    weight_effect_no_train = fitness_no_train * gamma * weight_std
    weight_effect_with_train = fitness_with_train * gamma * weight_std

    print(f"\nAfter {days} days:")
    print(f"  Fitness without training: {fitness_no_train[-1]:.2f} units")
    print(f"  Fitness with training: {fitness_with_train[-1]:.2f} units")
    print(f"  Fitness difference: {fitness_with_train[-1] - fitness_no_train[-1]:.2f} units")

    print(f"\nWeight effect:")
    print(f"  Without training: {weight_effect_no_train[-1]:.2f} lbs")
    print(f"  With training: {weight_effect_with_train[-1]:.2f} lbs")
    print(f"  Weight difference: {weight_effect_with_train[-1] - weight_effect_no_train[-1]:.2f} lbs")

    # Calculate steady-state (infinite days)
    # For geometric series: sum = a / (1 - r) where a = daily gain, r = persistence
    # Steady-state fitness = beta * intensity / (1 - persistence)

    persistence_no_train = alpha_d
    persistence_with_train = alpha_d + alpha_m

    steady_state_no_train = beta * intensity / (1 - persistence_no_train)
    steady_state_with_train = beta * intensity / (1 - persistence_with_train)

    print(f"\nSteady-state (infinite training):")
    print(f"  Without training: {steady_state_no_train:.2f} fitness units")
    print(f"  With training: {steady_state_with_train:.2f} fitness units")
    print(f"  Difference: {steady_state_with_train - steady_state_no_train:.2f} units")

    steady_weight_no_train = steady_state_no_train * gamma * weight_std
    steady_weight_with_train = steady_state_with_train * gamma * weight_std

    print(f"\nSteady-state weight effect:")
    print(f"  Without training: {steady_weight_no_train:.2f} lbs")
    print(f"  With training: {steady_weight_with_train:.2f} lbs")
    print(f"  Difference: {steady_weight_with_train - steady_weight_no_train:.2f} lbs")

    # Realistic scenario: train 3x per week
    print(f"\n" + "="*60)
    print("Realistic scenario: Train 3x per week (Mon, Wed, Fri)")

    days = 90  # 3 months
    fitness = np.zeros(days)

    for t in range(1, days):
        # Check if training day (0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri, 5=Sat, 6=Sun)
        day_of_week = (t - 1) % 7
        trained_today = 1 if day_of_week in [0, 2, 4] else 0  # Mon, Wed, Fri

        if t > 1:  # Check yesterday's training
            day_of_week_yesterday = (t - 2) % 7
            trained_yesterday = 1 if day_of_week_yesterday in [0, 2, 4] else 0

            persistence = alpha_d + alpha_m * trained_yesterday
            gain = beta * intensity * trained_yesterday
            fitness[t] = persistence * fitness[t-1] + gain

    weight_effect = fitness * gamma * weight_std

    print(f"After {days} days ({days//7} weeks):")
    print(f"  Fitness: {fitness[-1]:.2f} units")
    print(f"  Weight effect: {weight_effect[-1]:.2f} lbs")

    # Compare to no training
    fitness_no_train_90 = np.zeros(days)
    for t in range(1, days):
        fitness_no_train_90[t] = alpha_d * fitness_no_train_90[t-1]

    weight_no_train_90 = fitness_no_train_90 * gamma * weight_std

    print(f"  Without any training:")
    print(f"    Fitness: {fitness_no_train_90[-1]:.2f} units")
    print(f"    Weight effect: {weight_no_train_90[-1]:.2f} lbs")
    print(f"  Difference: {weight_effect[-1] - weight_no_train_90[-1]:.2f} lbs")

if __name__ == "__main__":
    main()