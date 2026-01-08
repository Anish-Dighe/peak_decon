"""
Test deconvolution WITHOUT normalization
Simple plots with max 2 curves per plot
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

from peak_generator import (X_GRID, geg_peak, generate_complete_spectrum)
from deconvolve import (deconvolve_spectrum, calculate_r2_score,
                       calculate_parameter_mae)


# Results directory at project root
RESULTS_DIR = Path(__file__).parent.parent / 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)


def test_deconvolution(n_components, seed=None, max_iter=1000):
    """
    Test deconvolution without normalization

    Returns:
    --------
    results : dict
        Results dictionary
    """
    print(f"\n{'='*60}")
    print(f"Testing: {n_components} components")
    print(f"{'='*60}")

    # Generate synthetic test data (NO normalization)
    params_true, y_true = generate_complete_spectrum(n_components, seed=seed)

    if params_true is None:
        print(f"ERROR: Failed to generate {n_components}-component spectrum")
        return None

    # Do NOT normalize - keep absolute intensities
    print(f"True spectrum max intensity: {np.max(y_true):.4f}")

    print("\nGround Truth Parameters:")
    print("Comp |   α    |   τ    |   μ    |   σ")
    print("-" * 45)
    for i, p in enumerate(params_true):
        print(f"  {i+1}  | {p[0]:.4f} | {p[1]:.4f} | {p[2]:.4f} | {p[3]:.4f}")

    # Run deconvolution (no normalization)
    start_time = time.time()
    params_est, y_fit, opt_result = deconvolve_spectrum(
        y_true,
        n_components=n_components,
        max_iter=max_iter,
        verbose=False,
        seed=seed
    )
    elapsed_time = time.time() - start_time

    print(f"Fitted spectrum max intensity: {np.max(y_fit):.4f}")

    # Calculate metrics
    mae_overall, mae_per_param = calculate_parameter_mae(params_true, params_est)
    r2 = calculate_r2_score(y_true, y_fit)
    mse = opt_result.fun

    print(f"\nOptimization Results:")
    print(f"  MSE: {mse:.8f}")
    print(f"  R²: {r2:.6f}")
    print(f"  Parameter MAE: {mae_overall:.6f}")
    print(f"  Time: {elapsed_time:.2f}s")

    print("\nEstimated Parameters:")
    print("Comp |   α    |   τ    |   μ    |   σ")
    print("-" * 45)
    for i, p in enumerate(params_est):
        print(f"  {i+1}  | {p[0]:.4f} | {p[1]:.4f} | {p[2]:.4f} | {p[3]:.4f}")

    results = {
        'n_components': n_components,
        'params_true': params_true,
        'params_estimated': params_est,
        'y_true': y_true,
        'y_fitted': y_fit,
        'mse': mse,
        'r2': r2,
        'mae_overall': mae_overall,
        'mae_per_param': mae_per_param,
        'elapsed_time': elapsed_time
    }

    return results


def plot_simple_comparison(results_list):
    """
    Create simple comparison plots
    For each n_comp case, create 4 separate plots:
    1. True total only
    2. Fitted total only
    3. All true individual peaks
    4. All fitted individual peaks
    """
    for res in results_list:
        n_comp = res['n_components']
        params_true = res['params_true']
        params_est = res['params_estimated']
        y_true = res['y_true']
        y_fit = res['y_fitted']

        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown',
                  'pink', 'gray', 'olive', 'cyan']

        # Plot 1: True total only (black line)
        fig1 = go.Figure()
        fig1.add_trace(
            go.Scatter(
                x=X_GRID, y=y_true,
                mode='lines',
                name='True Total',
                line=dict(color='black', width=3)
            )
        )
        fig1.update_layout(
            title=f"{n_comp} Components - True Total Spectrum",
            xaxis_title="Position",
            yaxis_title="Intensity (absolute)",
            height=400
        )
        fig1.write_html(os.path.join(RESULTS_DIR, f'n{n_comp}_true_total.html'))

        # Plot 2: Fitted total only (red line)
        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=X_GRID, y=y_fit,
                mode='lines',
                name='Fitted Total',
                line=dict(color='red', width=3)
            )
        )
        fig2.update_layout(
            title=f"{n_comp} Components - Fitted Total Spectrum (R²={res['r2']:.6f})",
            xaxis_title="Position",
            yaxis_title="Intensity (absolute)",
            height=400
        )
        fig2.write_html(os.path.join(RESULTS_DIR, f'n{n_comp}_fitted_total.html'))

        # Plot 3: All true individual peaks
        fig3 = go.Figure()
        for i, param_row in enumerate(params_true):
            alpha, tau, mu, sigma = param_row
            y_peak = geg_peak(X_GRID, alpha, tau, mu, sigma)

            color = colors[i % len(colors)]
            fig3.add_trace(
                go.Scatter(
                    x=X_GRID, y=y_peak,
                    mode='lines',
                    name=f'Peak {i+1}',
                    line=dict(color=color, width=2)
                )
            )
        fig3.update_layout(
            title=f"{n_comp} Components - True Individual Peaks",
            xaxis_title="Position",
            yaxis_title="Intensity (absolute)",
            height=400,
            showlegend=True
        )
        fig3.write_html(os.path.join(RESULTS_DIR, f'n{n_comp}_true_individual.html'))

        # Plot 4: All fitted individual peaks
        fig4 = go.Figure()
        for i, param_row in enumerate(params_est):
            alpha, tau, mu, sigma = param_row
            y_peak = geg_peak(X_GRID, alpha, tau, mu, sigma)

            color = colors[i % len(colors)]
            fig4.add_trace(
                go.Scatter(
                    x=X_GRID, y=y_peak,
                    mode='lines',
                    name=f'Peak {i+1}',
                    line=dict(color=color, width=2)
                )
            )
        fig4.update_layout(
            title=f"{n_comp} Components - Fitted Individual Peaks",
            xaxis_title="Position",
            yaxis_title="Intensity (absolute)",
            height=400,
            showlegend=True
        )
        fig4.write_html(os.path.join(RESULTS_DIR, f'n{n_comp}_fitted_individual.html'))

        print(f"Saved 4 plots for n_comp={n_comp}")


def plot_side_by_side(results_list):
    """
    Side-by-side plots for each n_comp:
    Left: Individual peaks
    Right: Total spectrum
    Separate figures for True and Fitted
    """
    for res in results_list:
        n_comp = res['n_components']
        params_true = res['params_true']
        params_est = res['params_estimated']
        y_true = res['y_true']
        y_fit = res['y_fitted']

        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown',
                  'pink', 'gray', 'olive', 'cyan']

        # TRUE: Side-by-side (individual | total)
        fig_true = make_subplots(
            rows=1, cols=2,
            subplot_titles=['True Individual Peaks', 'True Total Spectrum'],
            horizontal_spacing=0.12
        )

        # Left: True individual peaks
        for i, param_row in enumerate(params_true):
            alpha, tau, mu, sigma = param_row
            y_peak = geg_peak(X_GRID, alpha, tau, mu, sigma)

            color = colors[i % len(colors)]
            fig_true.add_trace(
                go.Scatter(
                    x=X_GRID, y=y_peak,
                    mode='lines',
                    name=f'Peak {i+1}',
                    line=dict(color=color, width=2),
                    showlegend=True
                ),
                row=1, col=1
            )

        # Right: True total
        fig_true.add_trace(
            go.Scatter(
                x=X_GRID, y=y_true,
                mode='lines',
                name='Total',
                line=dict(color='black', width=3),
                showlegend=True
            ),
            row=1, col=2
        )

        fig_true.update_xaxes(title_text="Position", row=1, col=1)
        fig_true.update_xaxes(title_text="Position", row=1, col=2)
        fig_true.update_yaxes(title_text="Intensity", row=1, col=1)
        fig_true.update_yaxes(title_text="Intensity", row=1, col=2)

        fig_true.update_layout(
            title=f"{n_comp} Components - TRUE: Individual Peaks | Total Spectrum",
            height=400
        )
        fig_true.write_html(os.path.join(RESULTS_DIR, f'n{n_comp}_true_sidebyside.html'))

        # FITTED: Side-by-side (individual | total)
        fig_fitted = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Fitted Individual Peaks', 'Fitted Total Spectrum'],
            horizontal_spacing=0.12
        )

        # Left: Fitted individual peaks
        for i, param_row in enumerate(params_est):
            alpha, tau, mu, sigma = param_row
            y_peak = geg_peak(X_GRID, alpha, tau, mu, sigma)

            color = colors[i % len(colors)]
            fig_fitted.add_trace(
                go.Scatter(
                    x=X_GRID, y=y_peak,
                    mode='lines',
                    name=f'Peak {i+1}',
                    line=dict(color=color, width=2),
                    showlegend=True
                ),
                row=1, col=1
            )

        # Right: Fitted total
        fig_fitted.add_trace(
            go.Scatter(
                x=X_GRID, y=y_fit,
                mode='lines',
                name='Total',
                line=dict(color='red', width=3),
                showlegend=True
            ),
            row=1, col=2
        )

        fig_fitted.update_xaxes(title_text="Position", row=1, col=1)
        fig_fitted.update_xaxes(title_text="Position", row=1, col=2)
        fig_fitted.update_yaxes(title_text="Intensity", row=1, col=1)
        fig_fitted.update_yaxes(title_text="Intensity", row=1, col=2)

        fig_fitted.update_layout(
            title=f"{n_comp} Components - FITTED: Individual Peaks | Total Spectrum (R²={res['r2']:.6f})",
            height=400
        )
        fig_fitted.write_html(os.path.join(RESULTS_DIR, f'n{n_comp}_fitted_sidebyside.html'))

        print(f"Saved 2 side-by-side plots for n_comp={n_comp}")


if __name__ == '__main__':
    print("="*60)
    print("DECONVOLUTION TEST WITHOUT NORMALIZATION")
    print("Testing with 6, 7, 8, 9, and 10 components")
    print("="*60)

    component_counts = [6, 7, 8, 9, 10]
    seeds = [111, 222, 333, 444, 555]

    results = []

    for n_comp, seed in zip(component_counts, seeds):
        result = test_deconvolution(
            n_components=n_comp,
            seed=seed,
            max_iter=1000
        )
        if result is not None:
            results.append(result)
        time.sleep(0.5)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'n_comp':<8} {'MSE':<12} {'R²':<10} {'MAE':<10} {'Time(s)'}")
    print("-"*60)
    for res in results:
        print(f"{res['n_components']:<8} "
              f"{res['mse']:<12.8f} "
              f"{res['r2']:<10.6f} "
              f"{res['mae_overall']:<10.6f} "
              f"{res['elapsed_time']:.2f}")
    print("="*60)

    # Generate plots
    print("\nGenerating simple comparison plots...")
    plot_simple_comparison(results)
    plot_side_by_side(results)

    print("\n" + "="*60)
    print("TEST COMPLETE!")
    print("="*60)
    print("Generated files for each n_comp:")
    print("  - n{X}_true_total.html")
    print("  - n{X}_fitted_total.html")
    print("  - n{X}_true_individual.html")
    print("  - n{X}_fitted_individual.html")
    print("  - n{X}_true_sidebyside.html (individual | total)")
    print("  - n{X}_fitted_sidebyside.html (individual | total)")
    print("="*60)
