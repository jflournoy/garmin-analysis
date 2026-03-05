#!/usr/bin/env python3
"""Test different diminishing returns functions for fitness accumulation."""

import numpy as np
import matplotlib.pyplot as plt

def linear_gain(current_fitness, impulse, beta=1.0):
    """Linear: gain = beta * impulse (no diminishing returns)"""
    return beta * impulse

def logistic_gain(current_fitness, impulse, beta=1.0, L=8.0):
    """Logistic: gain = beta * impulse * (1 - current_fitness/L)"""
    return beta * impulse * max(0, 1 - current_fitness / L)

def exponential_gain(current_fitness, impulse, beta=1.0, k=0.15):
    """Exponential: gain = beta * impulse * exp(-k * current_fitness)"""
    return beta * impulse * np.exp(-k * current_fitness)

def power_gain(current_fitness, impulse, beta=1.0, gamma=0.5):
    """Power law: gain = beta * impulse / (1 + gamma * current_fitness)"""
    return beta * impulse / (1 + gamma * current_fitness)

def simulate_fitness(impulse_sequence, gain_function, **kwargs):
    """Simulate fitness accumulation with given gain function."""
    fitness = [0.0]
    for impulse in impulse_sequence:
        gain = gain_function(fitness[-1], impulse, **kwargs)
        fitness.append(fitness[-1] + gain)
    return fitness

# Test sequence: add impulse=2 each time
impulse_sequence = [2.0] * 10

# Test different gain functions
functions = [
    ("Linear (no DR)", linear_gain, {}),
    ("Logistic (L=8)", logistic_gain, {"L": 8.0}),
    ("Exponential (k=0.15)", exponential_gain, {"k": 0.15}),
    ("Power (γ=0.5)", power_gain, {"gamma": 0.5}),
]

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for idx, (name, func, kwargs) in enumerate(functions):
    ax = axes[idx]

    # Simulate fitness
    fitness = simulate_fitness(impulse_sequence, func, **kwargs)

    # Plot fitness over time
    ax.plot(range(len(fitness)), fitness, 'o-', linewidth=2, markersize=6)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Fitness')
    ax.set_title(f'{name}')
    ax.grid(True, alpha=0.3)

    # Add gain annotations
    gains = []
    for i in range(len(fitness)-1):
        gain = func(fitness[i], 2.0, **kwargs)
        gains.append(gain)
        ax.text(i, fitness[i] + 0.1, f'+{gain:.2f}', fontsize=8, ha='center')

    # Print summary
    print(f"\n{name}:")
    print(f"  Final fitness: {fitness[-1]:.2f}")
    print(f"  Gains: {[f'{g:.2f}' for g in gains]}")

plt.tight_layout()
plt.savefig('test_diminishing_returns.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nYour example pattern (0→2→3.5→4.75):")
print("  Step 0: fitness=0, +2 → 2.0 (gain=2.0)")
print("  Step 1: fitness=2, +2 → 3.5 (gain=1.5, 75% of initial)")
print("  Step 2: fitness=3.5, +2 → 4.75 (gain=1.25, 62.5% of initial)")

# Try to match your pattern
print("\n\nFinding parameters to match your pattern...")

# Your pattern data
target_fitness = [0, 2, 3.5, 4.75]
target_gains = [2.0, 1.5, 1.25]

# Test exponential function
print("\nExponential function f(gain) = β * impulse * exp(-k * current_fitness):")
print("  With β=1, impulse=2:")
for k in [0.1, 0.125, 0.15, 0.175, 0.2]:
    gains = []
    fitness = 0
    for _ in range(3):
        gain = 2 * np.exp(-k * fitness)
        gains.append(gain)
        fitness += gain
    print(f"  k={k:.3f}: gains={[f'{g:.2f}' for g in gains]}, final fitness={fitness:.2f}")

# Test logistic function
print("\nLogistic function f(gain) = β * impulse * (1 - current_fitness/L):")
print("  With β=1, impulse=2:")
for L in [6.0, 8.0, 10.0, 12.0]:
    gains = []
    fitness = 0
    for _ in range(3):
        gain = 2 * max(0, 1 - fitness / L)
        gains.append(gain)
        fitness += gain
    print(f"  L={L:.1f}: gains={[f'{g:.2f}' for g in gains]}, final fitness={fitness:.2f}")

print("\nDone! Check test_diminishing_returns.png for visual comparison.")