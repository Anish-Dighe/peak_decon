#!/usr/bin/env python
"""
Peak Deconvolution using Optimization Only

This version uses only scipy optimization (no neural network required).
Good for immediate use without training a model first.

Usage:
    python deconvolve_opt_only.py input_spectrum.csv --n-components 3
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
from scipy.signal import find_peaks

from data_utils import load_spectrum_from_csv, save_deconvolution_results
from peak_generator import GEGPeak


class OptimizationDeconvolver:
    """
    Peak deconvolution using optimization only
    """

    def __init__(self, method='differential_evolution'):
        self.method = method

    def auto_detect_peaks(self, x, y, max_peaks=10):
        """
        Automatically detect number and approximate positions of peaks
        """
        # Find peaks in the spectrum
        peaks_idx, properties = find_peaks(y, prominence=0.1 * y.max(), width=5)

        n_peaks = min(len(peaks_idx), max_peaks)

        if n_peaks == 0:
            # If no peaks found, assume single peak at maximum
            n_peaks = 1
            peaks_idx = [np.argmax(y)]

        print(f"   Auto-detected {n_peaks} peak(s)")
        return n_peaks, peaks_idx

    def generate_initial_guess(self, x, y, n_components):
        """
        Generate initial parameter guess from spectrum
        """
        n_peaks, peaks_idx = self.auto_detect_peaks(x, y, n_components)
        n_components = n_peaks  # Use detected number

        initial_params = []
        for peak_idx in peaks_idx:
            # Estimate parameters from peak
            mu = x[peak_idx]
            amplitude = y[peak_idx]

            # Estimate width from half-maximum
            half_max = amplitude / 2
            left_idx = np.where(y[:peak_idx] < half_max)[0]
            right_idx = np.where(y[peak_idx:] < half_max)[0]

            if len(left_idx) > 0 and len(right_idx) > 0:
                fwhm = x[peak_idx + right_idx[0]] - x[left_idx[-1]]
                sigma = fwhm / 2.355  # Convert FWHM to sigma
            else:
                sigma = 0.05  # Default

            sigma = np.clip(sigma, 0.01, 0.15)

            initial_params.append({
                'alpha': 1.0,  # Start with Gaussian-like
                'tau': 0.1,
                'mu': mu,
                'sigma': sigma,
                'amplitude': amplitude
            })

        return n_components, initial_params

    def deconvolve(self, x, y, n_components=None, initial_params=None):
        """
        Perform deconvolution

        Parameters:
        -----------
        x : array
            Position values (normalized 0-1)
        y : array
            Intensity values (normalized)
        n_components : int, optional
            Number of components. If None, auto-detect
        initial_params : list of dict, optional
            Initial parameter guesses

        Returns:
        --------
        tuple
            (n_components, parameters, metrics)
        """
        # Auto-detect or use provided n_components
        if n_components is None or initial_params is None:
            if n_components is None:
                n_components = 3  # Default guess
            n_components, initial_params = self.generate_initial_guess(x, y, n_components)

        # Convert to flat array
        x0 = []
        for p in initial_params:
            x0.extend([p['alpha'], p['tau'], p['mu'], p['sigma'], p['amplitude']])
        x0 = np.array(x0)

        # Define bounds
        bounds = []
        for _ in range(n_components):
            bounds.extend([
                (0.5, 3.0),    # alpha
                (0.05, 0.3),   # tau
                (0.0, 1.0),    # mu
                (0.01, 0.15),  # sigma
                (0.0, 2.0)     # amplitude
            ])

        # Objective function
        def objective(params_flat):
            params_2d = params_flat.reshape(n_components, 5)
            y_pred = self._generate_spectrum(x, params_2d)
            mse = np.mean((y - y_pred) ** 2)

            # Add regularization to prefer fewer, distinct peaks
            penalty = 0
            for i in range(n_components):
                for j in range(i + 1, n_components):
                    mu_diff = abs(params_2d[i, 2] - params_2d[j, 2])
                    if mu_diff < 0.05:  # Penalize peaks too close together
                        penalty += 0.1 / (mu_diff + 0.01)

            return mse + 0.01 * penalty

        # Optimize
        print(f"   Optimizing {n_components} component(s)...")
        if self.method == 'differential_evolution':
            result = differential_evolution(
                objective, bounds,
                maxiter=300,
                popsize=15,
                seed=42,
                workers=1,
                updating='deferred',
                disp=False
            )
        else:
            result = minimize(
                objective, x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000}
            )

        # Extract parameters
        final_params = result.x.reshape(n_components, 5)

        # Sort by position
        sort_idx = np.argsort(final_params[:, 2])
        final_params = final_params[sort_idx]

        # Filter out negligible peaks
        significant_peaks = final_params[final_params[:, 4] > 0.05]
        if len(significant_peaks) < n_components:
            print(f"   Filtered to {len(significant_peaks)} significant peak(s)")
            final_params = significant_peaks
            n_components = len(significant_peaks)

        # Calculate metrics
        metrics = self._calculate_metrics(x, y, final_params)

        return n_components, final_params, metrics

    def _generate_spectrum(self, x, params):
        """Generate spectrum from parameters"""
        y = np.zeros_like(x)
        for param in params:
            peak = GEGPeak(
                alpha=param[0],
                tau=param[1],
                mu=param[2],
                sigma=param[3],
                amplitude=param[4]
            )
            _, peak_y = peak.generate_peak(x)
            y += peak_y
        return y

    def _calculate_metrics(self, x, y_true, params):
        """Calculate goodness of fit metrics"""
        y_pred = self._generate_spectrum(x, params)

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        nrmse = rmse / (y_true.max() - y_true.min()) if y_true.max() > y_true.min() else 0

        return {
            'R2': r2,
            'RMSE': rmse,
            'NRMSE': nrmse,
            'residual_sum': ss_res
        }


def main():
    parser = argparse.ArgumentParser(
        description='Peak Deconvolution using Optimization Only'
    )

    parser.add_argument('input', type=str, help='Input CSV file')
    parser.add_argument('--output', '-o', type=str, default='results/output.csv')
    parser.add_argument('--position', '-p', type=str, default='wavelength')
    parser.add_argument('--intensity', '-i', type=str, default='absorbance')
    parser.add_argument('--n-components', '-n', type=int, default=None,
                       help='Number of components (auto-detect if not specified)')
    parser.add_argument('--method', type=str, default='differential_evolution',
                       choices=['differential_evolution', 'L-BFGS-B'])

    args = parser.parse_args()

    print("=" * 70)
    print("PEAK DECONVOLUTION (OPTIMIZATION-ONLY)")
    print("=" * 70)

    # Load spectrum
    print(f"\n1. Loading spectrum from: {args.input}")
    position, intensity, metadata = load_spectrum_from_csv(
        args.input,
        position_col=args.position,
        intensity_col=args.intensity
    )
    print(f"   ✓ Loaded {len(position)} points")

    # Deconvolve
    print("\n2. Running deconvolution...")
    deconvolver = OptimizationDeconvolver(method=args.method)
    n_components, parameters, metrics = deconvolver.deconvolve(
        position, intensity, n_components=args.n_components
    )

    print(f"\n   ✓ Fitted {n_components} component(s)")
    print(f"   ✓ R² = {metrics['R2']:.4f}")
    print(f"   ✓ RMSE = {metrics['RMSE']:.6f}")

    # Generate fitted spectrum
    fitted = deconvolver._generate_spectrum(position, parameters)

    # Save results
    print("\n3. Saving results...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_deconvolution_results(
        output_path=output_path,
        n_components=n_components,
        params=parameters,
        metadata=metadata,
        fitted_spectrum=(position, fitted),
        original_spectrum=(position, intensity),
        goodness_of_fit=metrics
    )

    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
