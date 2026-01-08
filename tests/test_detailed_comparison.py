"""
Detailed comparison of true vs fitted spectra for verification

Tests with 6, 7, 8, 9, and 10 components
Creates multiple visualizations:
- Total spectra comparison (True vs Fitted overlay)
- Individual peaks comparison (True vs Fitted overlay)
- Side-by-side decomposition (True | Fitted separate)
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

from peak_generator import (X_GRID, geg_peak, generate_complete_spectrum,
                            normalize_by_max)
from deconvolve import (deconvolve_spectrum, calculate_r2_score,
                       calculate_parameter_mae)


# Results directory at project root
RESULTS_DIR = Path(__file__).parent.parent / 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)


def test_and_plot_detailed(n_components, seed=None, max_iter=1000):
    """
    Test deconvolution and create detailed comparison plots

    Parameters:
    -----------
    n_components : int
        Number of peaks
    seed : int, optional
        Random seed
    max_iter : int
        Maximum optimization iterations

    Returns:
    --------
    results : dict
        Results dictionary
    """
    print(f"\n{'='*60}")
    print(f"Testing: {n_components} components")
    print(f"{'='*60}")

    # Generate synthetic test data
    params_true, y_true = generate_complete_spectrum(n_components, seed=seed)

    if params_true is None:
        print(f"ERROR: Failed to generate {n_components}-component spectrum")
        return None

    # Normalize
    y_true = normalize_by_max(y_true)

    print("\nGround Truth Parameters:")
    print("Comp |   α    |   τ    |   μ    |   σ")
    print("-" * 45)
    for i, p in enumerate(params_true):
        print(f"  {i+1}  | {p[0]:.4f} | {p[1]:.4f} | {p[2]:.4f} | {p[3]:.4f}")

    # Run deconvolution
    start_time = time.time()
    params_est, y_fit, opt_result = deconvolve_spectrum(
        y_true,
        n_components=n_components,
        max_iter=max_iter,
        verbose=False,
        seed=seed
    )
    elapsed_time = time.time() - start_time

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


def plot_total_comparison(results_list):
    """
    Plot 1: True Total vs Fitted Total (5 subplots in 2x3 grid)
    """
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[f"{res['n_components']} Components" for res in results_list],
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )

    positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)]

    for idx, (res, (row, col)) in enumerate(zip(results_list, positions)):
        y_true = res['y_true']
        y_fit = res['y_fitted']

        # Plot true total (black solid, thicker)
        fig.add_trace(
            go.Scatter(
                x=X_GRID, y=y_true,
                mode='lines',
                name='True Total',
                line=dict(color='black', width=3),
                showlegend=(idx == 0)
            ),
            row=row, col=col
        )

        # Plot fitted total (red dashed, thinner)
        fig.add_trace(
            go.Scatter(
                x=X_GRID, y=y_fit,
                mode='lines',
                name='Fitted Total',
                line=dict(color='red', width=2, dash='dash'),
                showlegend=(idx == 0)
            ),
            row=row, col=col
        )

        # Add metrics
        metrics_text = (
            f"<b>MSE:</b> {res['mse']:.8f}<br>"
            f"<b>R²:</b> {res['r2']:.6f}<br>"
            f"<b>Param MAE:</b> {res['mae_overall']:.4f}"
        )

        fig.add_annotation(
            text=metrics_text,
            xref=f"x{idx+1 if idx > 0 else ''}",
            yref=f"y{idx+1 if idx > 0 else ''}",
            x=0.02, y=0.98,
            xanchor='left', yanchor='top',
            showarrow=False,
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=10)
        )

    fig.update_xaxes(title_text="Position")
    fig.update_yaxes(title_text="Normalized Intensity")

    fig.update_layout(
        title="Total Spectra: True (black solid) vs Fitted (red dashed)",
        height=700,
        showlegend=True
    )

    output_path = os.path.join(RESULTS_DIR, 'detailed_total_comparison.html')
    fig.write_html(output_path)
    print(f"\nSaved: {output_path}")


def plot_individual_comparison(results_list):
    """
    Plot 2: True Individual Peaks vs Fitted Individual Peaks (5 subplots in 2x3 grid)
    """
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[f"{res['n_components']} Components (Individual Peaks)"
                       for res in results_list],
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )

    positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)]
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown',
              'pink', 'gray', 'olive', 'cyan']

    for idx, (res, (row, col)) in enumerate(zip(results_list, positions)):
        params_true = res['params_true']
        params_est = res['params_estimated']
        y_true = res['y_true']
        total_max = np.max(y_true)

        # Plot true individual peaks (solid lines)
        for i, param_row in enumerate(params_true):
            alpha, tau, mu, sigma = param_row
            y_peak = geg_peak(X_GRID, alpha, tau, mu, sigma)
            y_peak_norm = y_peak / total_max

            color = colors[i % len(colors)]
            fig.add_trace(
                go.Scatter(
                    x=X_GRID, y=y_peak_norm,
                    mode='lines',
                    name=f'True Peak {i+1}' if idx == 0 and i < 3 else None,
                    line=dict(color=color, width=2),
                    opacity=0.7,
                    showlegend=(idx == 0 and i < 3)
                ),
                row=row, col=col
            )

        # Plot estimated individual peaks (dashed lines)
        for i, param_row in enumerate(params_est):
            alpha, tau, mu, sigma = param_row
            y_peak = geg_peak(X_GRID, alpha, tau, mu, sigma)
            y_peak_norm = y_peak / total_max

            color = colors[i % len(colors)]
            fig.add_trace(
                go.Scatter(
                    x=X_GRID, y=y_peak_norm,
                    mode='lines',
                    name=f'Fitted Peak {i+1}' if idx == 0 and i < 3 else None,
                    line=dict(color=color, width=2, dash='dash'),
                    opacity=0.7,
                    showlegend=(idx == 0 and i < 3)
                ),
                row=row, col=col
            )

        # Plot true total (black solid, for reference)
        y_true_norm = normalize_by_max(y_true)
        fig.add_trace(
            go.Scatter(
                x=X_GRID, y=y_true_norm,
                mode='lines',
                name='True Total' if idx == 0 else None,
                line=dict(color='black', width=3),
                showlegend=(idx == 0)
            ),
            row=row, col=col
        )

        # Plot fitted total (gray dashed, for reference)
        y_fit_norm = normalize_by_max(res['y_fitted'])
        fig.add_trace(
            go.Scatter(
                x=X_GRID, y=y_fit_norm,
                mode='lines',
                name='Fitted Total' if idx == 0 else None,
                line=dict(color='gray', width=2, dash='dot'),
                opacity=0.5,
                showlegend=(idx == 0)
            ),
            row=row, col=col
        )

        # Add legend note
        legend_text = (
            "<b>Solid lines:</b> True peaks<br>"
            "<b>Dashed lines:</b> Fitted peaks"
        )

        fig.add_annotation(
            text=legend_text,
            xref=f"x{idx+1 if idx > 0 else ''}",
            yref=f"y{idx+1 if idx > 0 else ''}",
            x=0.98, y=0.98,
            xanchor='right', yanchor='top',
            showarrow=False,
            bgcolor="rgba(255,255,200,0.95)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=9)
        )

    fig.update_xaxes(title_text="Position")
    fig.update_yaxes(title_text="Normalized Intensity")

    fig.update_layout(
        title="Individual Peaks: True (solid) vs Fitted (dashed) - Same color = same peak",
        height=700,
        showlegend=True
    )

    output_path = os.path.join(RESULTS_DIR, 'detailed_individual_comparison.html')
    fig.write_html(output_path)
    print(f"Saved: {output_path}")


def plot_side_by_side_decomposition(results_list):
    """
    Plot 3: Side-by-side comparison of decompositions
    Left column: True individual peaks + True total
    Right column: Fitted individual peaks + Fitted total
    """
    n_tests = len(results_list)

    # Create subplot titles
    subplot_titles = []
    for res in results_list:
        subplot_titles.append(f"{res['n_components']} Components - TRUE")
        subplot_titles.append(f"{res['n_components']} Components - FITTED")

    fig = make_subplots(
        rows=n_tests, cols=2,
        subplot_titles=subplot_titles,
        vertical_spacing=0.08,
        horizontal_spacing=0.08
    )

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown',
              'pink', 'gray', 'olive', 'cyan']

    for test_idx, res in enumerate(results_list):
        row = test_idx + 1
        params_true = res['params_true']
        params_est = res['params_estimated']
        y_true = res['y_true']
        total_max = np.max(y_true)

        # LEFT COLUMN: True individual peaks + True total
        # Plot true individual peaks
        for i, param_row in enumerate(params_true):
            alpha, tau, mu, sigma = param_row
            y_peak = geg_peak(X_GRID, alpha, tau, mu, sigma)
            y_peak_norm = y_peak / total_max

            color = colors[i % len(colors)]
            fig.add_trace(
                go.Scatter(
                    x=X_GRID, y=y_peak_norm,
                    mode='lines',
                    name=f'True Peak {i+1}' if test_idx == 0 and i < 3 else None,
                    line=dict(color=color, width=2),
                    opacity=0.6,
                    showlegend=(test_idx == 0 and i < 3)
                ),
                row=row, col=1
            )

        # Plot true total (black solid, thick)
        y_true_norm = normalize_by_max(y_true)
        fig.add_trace(
            go.Scatter(
                x=X_GRID, y=y_true_norm,
                mode='lines',
                name='True Total' if test_idx == 0 else None,
                line=dict(color='black', width=3),
                showlegend=(test_idx == 0)
            ),
            row=row, col=1
        )

        # RIGHT COLUMN: Fitted individual peaks + Fitted total
        # Plot fitted individual peaks
        for i, param_row in enumerate(params_est):
            alpha, tau, mu, sigma = param_row
            y_peak = geg_peak(X_GRID, alpha, tau, mu, sigma)
            y_peak_norm = y_peak / total_max

            color = colors[i % len(colors)]
            fig.add_trace(
                go.Scatter(
                    x=X_GRID, y=y_peak_norm,
                    mode='lines',
                    name=f'Fitted Peak {i+1}' if test_idx == 0 and i < 3 else None,
                    line=dict(color=color, width=2),
                    opacity=0.6,
                    showlegend=(test_idx == 0 and i < 3)
                ),
                row=row, col=2
            )

        # Plot fitted total (red solid, thick)
        y_fit_norm = normalize_by_max(res['y_fitted'])
        fig.add_trace(
            go.Scatter(
                x=X_GRID, y=y_fit_norm,
                mode='lines',
                name='Fitted Total' if test_idx == 0 else None,
                line=dict(color='red', width=3),
                showlegend=(test_idx == 0)
            ),
            row=row, col=2
        )

        # Add metrics annotation on right plot
        metrics_text = (
            f"<b>R²:</b> {res['r2']:.6f}<br>"
            f"<b>MAE:</b> {res['mae_overall']:.4f}"
        )

        subplot_idx = test_idx * 2 + 2  # Right column subplot index
        fig.add_annotation(
            text=metrics_text,
            xref=f"x{subplot_idx if subplot_idx > 1 else ''}",
            yref=f"y{subplot_idx if subplot_idx > 1 else ''}",
            x=0.02, y=0.98,
            xanchor='left', yanchor='top',
            showarrow=False,
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=9)
        )

    fig.update_xaxes(title_text="Position")
    fig.update_yaxes(title_text="Normalized Intensity")

    fig.update_layout(
        title="Side-by-Side Decomposition: True (left) vs Fitted (right)<br>Individual peaks + Total spectrum",
        height=300 * n_tests,
        showlegend=True
    )

    output_path = os.path.join(RESULTS_DIR, 'side_by_side_decomposition.html')
    fig.write_html(output_path)
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    print("="*60)
    print("DETAILED DECONVOLUTION VERIFICATION")
    print("Testing with 6, 7, 8, 9, and 10 components")
    print("="*60)

    component_counts = [6, 7, 8, 9, 10]
    seeds = [111, 222, 333, 444, 555]

    results = []

    for n_comp, seed in zip(component_counts, seeds):
        result = test_and_plot_detailed(
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
    print("\nGenerating detailed comparison plots...")
    plot_total_comparison(results)
    plot_individual_comparison(results)
    plot_side_by_side_decomposition(results)

    print("\n" + "="*60)
    print("VERIFICATION COMPLETE!")
    print("="*60)
    print("Generated files:")
    print("  1. detailed_total_comparison.html - True vs Fitted total spectra overlay")
    print("  2. detailed_individual_comparison.html - Individual peaks overlay comparison")
    print("  3. side_by_side_decomposition.html - Side-by-side: True | Fitted decompositions")
    print("="*60)
