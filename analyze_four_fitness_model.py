#!/usr/bin/env python3
"""Analyze four-fitness state-space model for weight with separate short/long term effects.

This script:
1. Prepares data for four-fitness model
2. Fits the model using CmdStanPy
3. Analyzes results with focus on short vs long term effects
4. Compares with simpler dual model
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import arviz as az
import cmdstanpy

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.data.weight import load_weight_data
from src.data.intensity import load_intensity_by_activity

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class FourFitnessModelAnalyzer:
    """Analyze four-fitness state-space model for weight prediction."""

    def __init__(
        self,
        output_dir: str = "output/four_fitness_analysis",
        data_dir: str = "data",
        chains: int = 4,
        iter_warmup: int = 500,
        iter_sampling: int = 500,
        adapt_delta: float = 0.95,
        max_treedepth: int = 12,
        n_inducing_points: int = 50,
        fourier_harmonics: int = 2,
    ):
        """Initialize analyzer with configuration.

        Args:
            output_dir: Path to directory for output files.
            data_dir: Path to data directory containing Garmin export data.
            chains: Number of MCMC chains.
            iter_warmup: Warmup iterations per chain.
            iter_sampling: Sampling iterations per chain.
            adapt_delta: Target acceptance probability for NUTS.
            max_treedepth: Maximum tree depth for NUTS.
            n_inducing_points: Number of inducing points for sparse GP.
            fourier_harmonics: Number of Fourier harmonics for daily cycle.
        """
        self.output_dir = Path(output_dir)
        self.data_dir = Path(data_dir)
        self.chains = chains
        self.iter_warmup = iter_warmup
        self.iter_sampling = iter_sampling
        self.adapt_delta = adapt_delta
        self.max_treedepth = max_treedepth
        self.n_inducing_points = n_inducing_points
        self.fourier_harmonics = fourier_harmonics

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Store loaded data and results
        self.df_weight = None
        self.df_daily = None
        self.stan_data = None
        self.fit = None
        self.idata = None
        self.results = {}

    def load_and_prepare_data(self) -> None:
        """Load and prepare data for four-fitness model."""
        print("\n" + "=" * 70)
        print("LOADING AND PREPARING DATA FOR FOUR-FITNESS MODEL")
        print("=" * 70)

        # Load weight data
        print("Loading weight data...")
        self.df_weight = load_weight_data(self.data_dir)
        print(f"  Loaded {len(self.df_weight)} weight measurements")
        print(f"  Date range: {self.df_weight['timestamp'].min()} to {self.df_weight['timestamp'].max()}")

        # Load aerobic intensity (walking + cycling)
        print("\nLoading aerobic intensity data...")
        df_intensity = load_intensity_by_activity(
            data_dir=self.data_dir,
            activity_types=['walking', 'cycling'],
            max_hr=185.0,
        )

        if len(df_intensity) == 0:
            raise ValueError("No aerobic intensity data found")

        # Sum walking and cycling
        df_intensity['aerobic_intensity'] = df_intensity.get('walking', 0) + df_intensity.get('cycling', 0)
        df_aerobic = df_intensity[['date', 'aerobic_intensity']].copy()

        print(f"  Loaded {len(df_aerobic)} days of aerobic intensity")
        print(f"  Non-zero days: {(df_aerobic['aerobic_intensity'] > 0).sum()}")

        # Load strength training intensity
        print("\nLoading strength training intensity data...")
        df_strength_intensity = load_intensity_by_activity(
            data_dir=self.data_dir,
            activity_types=['strength_training'],
            max_hr=185.0,
        )

        if len(df_strength_intensity) > 0 and 'strength_training' in df_strength_intensity.columns:
            df_strength_daily = df_strength_intensity[['date', 'strength_training']].copy()
            df_strength_daily = df_strength_daily.rename(columns={'strength_training': 'strength_intensity'})
            print(f"  Loaded {len(df_strength_daily)} days with strength training intensity")
        else:
            print("  No strength training intensity data found")
            df_strength_daily = pd.DataFrame(columns=['date', 'strength_intensity'])

        # Create full date range
        all_dates = set()
        all_dates.update(self.df_weight['timestamp'].dt.date)
        all_dates.update(df_aerobic['date'].dt.date)
        if len(df_strength_daily) > 0:
            all_dates.update(df_strength_daily['date'].dt.date)

        min_date = min(all_dates)
        max_date = max(all_dates)
        date_range = pd.date_range(start=min_date, end=max_date, freq='D')
        D = len(date_range)

        print(f"\nDate range for analysis: {min_date} to {max_date} ({D} days)")

        # Create daily dataframe
        self.df_daily = pd.DataFrame({'date': date_range})

        # Merge aerobic intensity
        self.df_daily = pd.merge(self.df_daily, df_aerobic, on='date', how='left')
        self.df_daily['aerobic_intensity'] = self.df_daily['aerobic_intensity'].fillna(0.0)

        # Merge strength intensity
        if len(df_strength_daily) > 0:
            self.df_daily = pd.merge(self.df_daily, df_strength_daily, on='date', how='left')
        self.df_daily['strength_intensity'] = self.df_daily['strength_intensity'].fillna(0.0)

        # Standardize inputs
        for col in ['aerobic_intensity', 'strength_intensity']:
            mean = self.df_daily[col].mean()
            std = self.df_daily[col].std()
            if std > 0:
                self.df_daily[f'{col}_std'] = (self.df_daily[col] - mean) / std
            else:
                self.df_daily[f'{col}_std'] = 0.0
            print(f"  {col}: mean={mean:.2f}, std={std:.2f}")

        # Standardize weight
        weight_mean = self.df_weight['weight_lbs'].mean()
        weight_std = self.df_weight['weight_lbs'].std()
        self.df_weight['weight_std'] = (self.df_weight['weight_lbs'] - weight_mean) / weight_std

        print(f"\nWeight standardization: mean={weight_mean:.2f} lbs, std={weight_std:.2f} lbs")

        # Map weight observations to days
        self.df_weight['date'] = self.df_weight['timestamp'].dt.date
        date_to_idx = {date: i+1 for i, date in enumerate(date_range.date)}

        self.df_weight['day_idx'] = self.df_weight['date'].map(date_to_idx)
        missing = self.df_weight['day_idx'].isna().sum()
        if missing > 0:
            print(f"  WARNING: {missing} weight observations outside date range, dropping")
            self.df_weight = self.df_weight[self.df_weight['day_idx'].notna()]

        # Scale time to [0, 1] for GP
        min_time = self.df_weight['timestamp'].min()
        max_time = self.df_weight['timestamp'].max()
        time_range = (max_time - min_time).total_seconds()

        if time_range > 0:
            self.df_weight['t_scaled'] = (self.df_weight['timestamp'] - min_time).dt.total_seconds() / time_range
        else:
            self.df_weight['t_scaled'] = 0.0

        # Hour of day for daily cycle
        self.df_weight['hour_of_day'] = self.df_weight['timestamp'].dt.hour + self.df_weight['timestamp'].dt.minute / 60.0

        # Create inducing points
        t_inducing = np.linspace(0, 1, self.n_inducing_points)

        # Prepare Stan data
        self.stan_data = {
            'D': D,
            'aerobic_intensity': self.df_daily['aerobic_intensity_std'].values.astype(float),
            'strength_intensity': self.df_daily['strength_intensity_std'].values.astype(float),

            'N_weight': len(self.df_weight),
            't_weight': self.df_weight['t_scaled'].values.astype(float),
            'y_weight': self.df_weight['weight_std'].values.astype(float),
            'day_idx': self.df_weight['day_idx'].values.astype(int),

            'hour_of_day': self.df_weight['hour_of_day'].values.astype(float),
            'K': self.fourier_harmonics,

            'use_sparse': 1,
            'M': self.n_inducing_points,
            't_inducing': t_inducing.astype(float),

            'N_pred': 0,
            't_pred': np.array([]).astype(float),
            'hour_of_day_pred': np.array([]).astype(float),
        }

        # Save standardization info
        self.standardization = {
            'weight_mean': float(weight_mean),
            'weight_std': float(weight_std),
            'aerobic_mean': float(self.df_daily['aerobic_intensity'].mean()),
            'aerobic_std': float(self.df_daily['aerobic_intensity'].std()),
            'strength_mean': float(self.df_daily['strength_intensity'].mean()),
            'strength_std': float(self.df_daily['strength_intensity'].std()),
            'min_time': min_time.isoformat(),
            'max_time': max_time.isoformat(),
        }

        print(f"\nStan data prepared:")
        print(f"  Days: {D}")
        print(f"  Weight observations: {len(self.df_weight)}")
        print(f"  Aerobic days > 0: {(self.df_daily['aerobic_intensity'] > 0).sum()}")
        print(f"  Strength days > 0: {(self.df_daily['strength_intensity'] > 0).sum()}")

    def fit_model(self) -> None:
        """Fit four-fitness state-space model."""
        print("\n" + "=" * 70)
        print("FITTING FOUR-FITNESS STATE-SPACE MODEL")
        print("=" * 70)

        print(f"Configuration:")
        print(f"  Chains: {self.chains}")
        print(f"  Warmup iterations: {self.iter_warmup}")
        print(f"  Sampling iterations: {self.iter_sampling}")
        print(f"  Adapt delta: {self.adapt_delta}")
        print(f"  Max treedepth: {self.max_treedepth}")
        print(f"  Inducing points: {self.n_inducing_points}")
        print(f"  Fourier harmonics: {self.fourier_harmonics}")

        # Compile model if needed
        model_path = Path("stan/weight_state_space_four_fitness.stan")
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        print(f"\nCompiling model: {model_path}")
        model = cmdstanpy.CmdStanModel(stan_file=str(model_path))

        # Fit model
        print("\nFitting model (this may take several minutes)...")
        self.fit = model.sample(
            data=self.stan_data,
            chains=self.chains,
            iter_warmup=self.iter_warmup,
            iter_sampling=self.iter_sampling,
            adapt_delta=self.adapt_delta,
            max_treedepth=self.max_treedepth,
            show_progress=True,
            seed=12345,
        )

        # Convert to ArviZ InferenceData
        self.idata = az.from_cmdstanpy(
            posterior=self.fit,
            posterior_predictive='y_weight_rep',
            log_likelihood='log_lik_weight',
        )

        print("\nModel fitting completed!")
        print(f"  Number of samples: {self.fit.num_draws_sampling}")
        print(f"  Number of chains: {self.fit.chains}")

        # Save results
        self._save_results()

    def _save_results(self) -> None:
        """Save model results to files."""
        print("\nSaving model results...")

        # Save parameter summary
        param_names = [
            # Impulse decay
            'psi_a_short', 'psi_s_short', 'psi_a_long', 'psi_s_long',
            # Fitness decay
            'alpha_a_short', 'alpha_s_short', 'alpha_a_long', 'alpha_s_long',
            # Fitness gain
            'beta_a_short', 'beta_s_short', 'beta_a_long', 'beta_s_long',
            # Weight effects
            'gamma_a_short', 'gamma_s_short', 'gamma_a_long', 'gamma_s_long',
            # Noise and GP
            'sigma_w', 'alpha_gp', 'rho_gp',
        ]

        summary = az.summary(self.idata, var_names=param_names)
        summary.to_csv(self.output_dir / "parameter_summary.csv")
        print(f"  Parameter summary saved to: {self.output_dir / 'parameter_summary.csv'}")

        # Save diagnostics
        diagnostics = {
            'divergences': getattr(self.fit, 'divergences', []),
            'max_treedepths': getattr(self.fit, 'max_treedepths', []),
            'step_size': getattr(self.fit, 'step_size', []),
            'accept_stat': getattr(self.fit, 'accept_stat', []),
        }

        # Convert to serializable format
        def to_serializable(obj):
            if obj is None:
                return None
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            if isinstance(obj, (list, tuple)):
                return [to_serializable(item) for item in obj]
            return obj

        diagnostics_serial = {k: to_serializable(v) for k, v in diagnostics.items()}

        with open(self.output_dir / "diagnostics.json", 'w') as f:
            json.dump(diagnostics_serial, f, indent=2)

        # Save samples for key parameters
        samples = {}
        for param in param_names:
            if param in self.idata.posterior:
                samples[param] = self.idata.posterior[param].values.flatten().tolist()

        with open(self.output_dir / "parameter_samples.json", 'w') as f:
            json.dump(samples, f, indent=2)

        # Save standardization info
        with open(self.output_dir / "standardization.json", 'w') as f:
            json.dump(self.standardization, f, indent=2)

        print(f"  Diagnostics saved to: {self.output_dir / 'diagnostics.json'}")
        print(f"  Parameter samples saved to: {self.output_dir / 'parameter_samples.json'}")

    def analyze_results(self) -> None:
        """Analyze model results with focus on short vs long term effects."""
        print("\n" + "=" * 70)
        print("ANALYZING FOUR-FITNESS MODEL RESULTS")
        print("=" * 70)

        if self.idata is None:
            print("No model results to analyze. Run fit_model() first.")
            return

        # 1. Weight effect parameters (gamma) comparison
        print("\n1. Weight effect parameters (gamma):")
        gamma_params = ['gamma_a_short', 'gamma_s_short', 'gamma_a_long', 'gamma_s_long']

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for i, param in enumerate(gamma_params):
            if i >= len(axes):
                break

            if param in self.idata.posterior:
                samples = self.idata.posterior[param].values.flatten()
                ax = axes[i]

                # Histogram
                ax.hist(samples, bins=30, alpha=0.7, density=True, edgecolor='black')

                # Add vertical line at 0
                ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)

                # Add mean and 95% CI
                mean = np.mean(samples)
                ci_low, ci_high = np.percentile(samples, [2.5, 97.5])

                ax.axvline(x=mean, color='blue', linestyle='-', alpha=0.8, label=f'Mean: {mean:.3f}')
                ax.axvline(x=ci_low, color='blue', linestyle=':', alpha=0.6)
                ax.axvline(x=ci_high, color='blue', linestyle=':', alpha=0.6)

                ax.fill_betweenx([0, ax.get_ylim()[1]], ci_low, ci_high, alpha=0.2, color='blue')

                # Set title based on parameter
                titles = {
                    'gamma_a_short': 'Aerobic Short-term\n(dehydration)',
                    'gamma_s_short': 'Strength Short-term\n(inflammation/water)',
                    'gamma_a_long': 'Aerobic Long-term\n(fat loss)',
                    'gamma_s_long': 'Strength Long-term\n(muscle gain)',
                }
                ax.set_title(titles.get(param, param))
                ax.set_xlabel('Effect on Weight')
                ax.set_ylabel('Density')
                ax.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(self.output_dir / "gamma_effects_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Gamma effects comparison saved to: {self.output_dir / 'gamma_effects_comparison.png'}")

        # Calculate probabilities for key hypotheses
        print("\n2. Key hypothesis probabilities:")
        if all(p in self.idata.posterior for p in gamma_params):
            gamma_a_short = self.idata.posterior['gamma_a_short'].values.flatten()
            gamma_s_short = self.idata.posterior['gamma_s_short'].values.flatten()
            gamma_a_long = self.idata.posterior['gamma_a_long'].values.flatten()
            gamma_s_long = self.idata.posterior['gamma_s_long'].values.flatten()

            hypotheses = {
                'Aerobic short-term reduces weight (dehydration)': np.mean(gamma_a_short < 0),
                'Strength short-term increases weight (inflammation)': np.mean(gamma_s_short > 0),
                'Aerobic long-term reduces weight (fat loss)': np.mean(gamma_a_long < 0),
                'Strength long-term increases weight (muscle gain)': np.mean(gamma_s_long > 0),
            }

            for hypothesis, prob in hypotheses.items():
                print(f"  {hypothesis}: {prob:.3f}")

        # 3. Time scale comparison
        print("\n3. Time scale comparison (half-lives in days):")
        if 'half_life_a_short' in self.idata.posterior:
            half_life_params = ['half_life_a_short', 'half_life_s_short',
                               'half_life_a_long', 'half_life_s_long']

            fig, ax = plt.subplots(figsize=(10, 6))

            colors = ['skyblue', 'lightcoral', 'deepskyblue', 'lightpink']
            labels = ['Aerobic Short', 'Strength Short', 'Aerobic Long', 'Strength Long']

            for i, param in enumerate(half_life_params):
                if param in self.idata.posterior:
                    samples = self.idata.posterior[param].values.flatten()
                    # Cap extreme values for visualization
                    samples_capped = np.clip(samples, 0, 100)

                    ax.hist(samples_capped, bins=30, alpha=0.5, label=labels[i],
                           color=colors[i], edgecolor='black', density=True)

                    mean = np.mean(samples)
                    median = np.median(samples)
                    print(f"  {labels[i]}: mean={mean:.1f}d, median={median:.1f}d")

            ax.set_xlabel('Half-life (days, capped at 100)')
            ax.set_ylabel('Density')
            ax.set_title('Fitness Half-life Comparison')
            ax.legend()

            plt.tight_layout()
            plt.savefig(self.output_dir / "half_life_comparison.png", dpi=150, bbox_inches='tight')
            plt.close()

            print(f"  Half-life comparison saved to: {self.output_dir / 'half_life_comparison.png'}")

        # 4. Impulse decay comparison (short vs long term)
        print("\n4. Impulse decay comparison (psi parameters):")
        psi_params = ['psi_a_short', 'psi_s_short', 'psi_a_long', 'psi_s_long']

        if all(p in self.idata.posterior for p in psi_params):
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Short-term impulse decay
            psi_a_short = self.idata.posterior['psi_a_short'].values.flatten()
            psi_s_short = self.idata.posterior['psi_s_short'].values.flatten()

            axes[0].hist(psi_a_short, bins=30, alpha=0.5, label='Aerobic', color='skyblue', edgecolor='black', density=True)
            axes[0].hist(psi_s_short, bins=30, alpha=0.5, label='Strength', color='lightcoral', edgecolor='black', density=True)
            axes[0].set_xlabel('Impulse Decay (psi)')
            axes[0].set_ylabel('Density')
            axes[0].set_title('Short-term Impulse Decay\n(smaller = faster decay)')
            axes[0].legend()

            # Long-term impulse decay
            psi_a_long = self.idata.posterior['psi_a_long'].values.flatten()
            psi_s_long = self.idata.posterior['psi_s_long'].values.flatten()

            axes[1].hist(psi_a_long, bins=30, alpha=0.5, label='Aerobic', color='skyblue', edgecolor='black', density=True)
            axes[1].hist(psi_s_long, bins=30, alpha=0.5, label='Strength', color='lightcoral', edgecolor='black', density=True)
            axes[1].set_xlabel('Impulse Decay (psi)')
            axes[1].set_ylabel('Density')
            axes[1].set_title('Long-term Impulse Decay\n(larger = slower decay)')
            axes[1].legend()

            plt.tight_layout()
            plt.savefig(self.output_dir / "impulse_decay_comparison.png", dpi=150, bbox_inches='tight')
            plt.close()

            print(f"  Impulse decay comparison saved to: {self.output_dir / 'impulse_decay_comparison.png'}")

            # Check if short-term decays faster than long-term
            prob_short_faster_a = np.mean(psi_a_short < psi_a_long)
            prob_short_faster_s = np.mean(psi_s_short < psi_s_long)
            print(f"  Probability aerobic short-term decays faster: {prob_short_faster_a:.3f}")
            print(f"  Probability strength short-term decays faster: {prob_short_faster_s:.3f}")

        # 5. Variance decomposition
        print("\n5. Variance decomposition:")
        var_params = ['prop_variance_a_short', 'prop_variance_s_short',
                     'prop_variance_a_long', 'prop_variance_s_long',
                     'prop_variance_gp', 'prop_variance_daily']

        if all(p in self.idata.posterior for p in var_params):
            # Create summary dataframe
            components = ['Aerobic Short', 'Strength Short', 'Aerobic Long', 'Strength Long', 'GP', 'Daily', 'Noise']
            means = []
            stds = []

            for param in var_params:
                samples = self.idata.posterior[param].values.flatten()
                means.append(np.mean(samples))
                stds.append(np.std(samples))

            # Noise is residual
            total_explained = sum(means)
            means.append(1 - total_explained)
            stds.append(0)

            var_df = pd.DataFrame({
                'Component': components,
                'Mean': means,
                'Std': stds
            })

            var_df.to_csv(self.output_dir / "variance_decomposition.csv", index=False)

            # Create bar plot
            fig, ax = plt.subplots(figsize=(12, 6))
            colors = ['skyblue', 'lightcoral', 'deepskyblue', 'lightpink', 'lightgreen', 'gold', 'lightgray']
            bars = ax.bar(var_df['Component'], var_df['Mean'], yerr=var_df['Std'],
                         capsize=5, alpha=0.7, color=colors, edgecolor='black')
            ax.set_ylabel('Proportion of Variance')
            ax.set_title('Variance Decomposition of Weight (Four-Fitness Model)')
            ax.set_ylim(0, 1)
            ax.tick_params(axis='x', rotation=45)

            # Add value labels on bars
            for bar, mean in zip(bars, var_df['Mean']):
                height = bar.get_height()
                if height > 0.05:  # Only label significant components
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{mean:.3f}', ha='center', va='bottom', fontsize=9)

            plt.tight_layout()
            plt.savefig(self.output_dir / "variance_decomposition.png", dpi=150, bbox_inches='tight')
            plt.close()

            print(f"  Variance decomposition saved to: {self.output_dir / 'variance_decomposition.png'}")
            for i, comp in enumerate(components):
                print(f"  {comp}: {means[i]:.3f} ± {stds[i]:.3f}")

        print("\nAnalysis complete!")

    def run(self) -> None:
        """Run full analysis pipeline."""
        self.load_and_prepare_data()
        self.fit_model()
        self.analyze_results()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Analyze four-fitness state-space model for weight prediction.")
    parser.add_argument("--output-dir", default="output/four_fitness_analysis",
                       help="Output directory for results")
    parser.add_argument("--data-dir", default="data",
                       help="Data directory containing Garmin export")
    parser.add_argument("--chains", type=int, default=4,
                       help="Number of MCMC chains")
    parser.add_argument("--iter-warmup", type=int, default=500,
                       help="Warmup iterations per chain")
    parser.add_argument("--iter-sampling", type=int, default=500,
                       help="Sampling iterations per chain")
    parser.add_argument("--adapt-delta", type=float, default=0.95,
                       help="Target acceptance probability for NUTS")
    parser.add_argument("--max-treedepth", type=int, default=12,
                       help="Maximum tree depth for NUTS")
    parser.add_argument("--n-inducing-points", type=int, default=50,
                       help="Number of inducing points for sparse GP")
    parser.add_argument("--fourier-harmonics", type=int, default=2,
                       help="Number of Fourier harmonics for daily cycle")

    args = parser.parse_args()

    # Create and run analyzer
    analyzer = FourFitnessModelAnalyzer(
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        chains=args.chains,
        iter_warmup=args.iter_warmup,
        iter_sampling=args.iter_sampling,
        adapt_delta=args.adapt_delta,
        max_treedepth=args.max_treedepth,
        n_inducing_points=args.n_inducing_points,
        fourier_harmonics=args.fourier_harmonics,
    )

    analyzer.run()

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()