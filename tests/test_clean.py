"""
Clean deconvolution test WITHOUT normalization
Simple, clear 2x2 layout for each test
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
    """
    print(f"\n{'='*60}")
    print(f"Testing: {n_components} components")
    print(f"{'='*60}")

    # Generate synthetic test data (NO normalization)
    params_true, y_true = generate_complete_spectrum(n_components, seed=seed)

    if params_true is None:
        print(f"ERROR: Failed to generate {n_components}-component spectrum")
        return None

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


def plot_single_test(result):
    """
    Create 2x2 plot for a single test:
    Row 1, Col 1: True individual peaks
    Row 1, Col 2: True total spectrum
    Row 2, Col 1: Fitted individual peaks
    Row 2, Col 2: Fitted total spectrum

    Colors are consistent: Peak 1 (leftmost) = red in both true and fitted
    """
    n_comp = result['n_components']
    params_true = result['params_true']
    params_est = result['params_estimated']
    y_true = result['y_true']
    y_fit = result['y_fitted']

    # Sort both by μ to ensure consistent peak ordering
    # Peak 1 = leftmost (smallest μ), Peak 2 = next, etc.
    params_true = params_true[params_true[:, 2].argsort()]
    params_est = params_est[params_est[:, 2].argsort()]

    # Consistent colors: same color = same peak position
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown',
              'pink', 'gray', 'olive', 'cyan']

    # Create 2x2 subplot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'True Individual Peaks',
            'True Total Spectrum',
            'Fitted Individual Peaks',
            'Fitted Total Spectrum'
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )

    # Row 1, Col 1: True individual peaks
    for i, param_row in enumerate(params_true):
        alpha, tau, mu, sigma = param_row
        y_peak = geg_peak(X_GRID, alpha, tau, mu, sigma)

        color = colors[i % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=X_GRID, y=y_peak,
                mode='lines',
                name=f'Peak {i+1}',
                line=dict(color=color, width=2),
                showlegend=False
            ),
            row=1, col=1
        )

    # Row 1, Col 2: True total spectrum
    fig.add_trace(
        go.Scatter(
            x=X_GRID, y=y_true,
            mode='lines',
            name='True Total',
            line=dict(color='black', width=3),
            showlegend=False
        ),
        row=1, col=2
    )

    # Row 2, Col 1: Fitted individual peaks
    for i, param_row in enumerate(params_est):
        alpha, tau, mu, sigma = param_row
        y_peak = geg_peak(X_GRID, alpha, tau, mu, sigma)

        color = colors[i % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=X_GRID, y=y_peak,
                mode='lines',
                name=f'Peak {i+1}',
                line=dict(color=color, width=2),
                showlegend=False
            ),
            row=2, col=1
        )

    # Row 2, Col 2: Fitted total spectrum
    fig.add_trace(
        go.Scatter(
            x=X_GRID, y=y_fit,
            mode='lines',
            name='Fitted Total',
            line=dict(color='red', width=3),
            showlegend=False
        ),
        row=2, col=2
    )

    # Add metrics annotation to fitted total plot
    metrics_text = (
        f"<b>R²:</b> {result['r2']:.6f}<br>"
        f"<b>MAE:</b> {result['mae_overall']:.4f}<br>"
        f"<b>Time:</b> {result['elapsed_time']:.1f}s"
    )

    fig.add_annotation(
        text=metrics_text,
        xref="x4",
        yref="y4",
        x=0.02, y=0.98,
        xanchor='left', yanchor='top',
        showarrow=False,
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=10)
    )

    # Update axes
    fig.update_xaxes(title_text="Position")
    fig.update_yaxes(title_text="Intensity")

    fig.update_layout(
        title=f"{n_comp} Components - Deconvolution Results",
        height=700,
        showlegend=False
    )

    output_path = RESULTS_DIR / f'test_{n_comp}_components.html'
    fig.write_html(str(output_path))
    print(f"Saved: {output_path}")


def plot_parameter_comparison(results_list):
    """
    Create parameter comparison plots: true vs estimated
    with x=y reference line and equal axes
    """
    n_tests = len(results_list)

    # Parameter bounds for axis limits
    param_bounds = [
        (0.5, 3.0),    # α
        (0.05, 0.3),   # τ
        (0.0, 1.0),    # μ
        (0.01, 0.4)    # σ
    ]
    param_names = ['α (alpha)', 'τ (tau)', 'μ (mu)', 'σ (sigma)']

    fig = make_subplots(
        rows=n_tests, cols=4,
        subplot_titles=param_names * n_tests,
        vertical_spacing=0.08,
        horizontal_spacing=0.08
    )

    for test_idx, res in enumerate(results_list):
        row = test_idx + 1
        params_true = res['params_true']
        params_est = res['params_estimated']
        n_comp = res['n_components']

        for param_idx in range(4):
            col = param_idx + 1

            # Get parameter values
            true_vals = params_true[:, param_idx]
            est_vals = params_est[:, param_idx]

            # Get axis limits
            axis_min, axis_max = param_bounds[param_idx]

            # Add x=y reference line (behind markers)
            fig.add_trace(
                go.Scatter(
                    x=[axis_min, axis_max],
                    y=[axis_min, axis_max],
                    mode='lines',
                    line=dict(color='gray', dash='dash', width=2),
                    showlegend=False
                ),
                row=row, col=col
            )

            # Scatter: true vs estimated
            fig.add_trace(
                go.Scatter(
                    x=true_vals,
                    y=est_vals,
                    mode='markers',
                    marker=dict(size=12, color='blue',
                               line=dict(width=1, color='black')),
                    showlegend=False
                ),
                row=row, col=col
            )

            # Set equal axes with same range
            fig.update_xaxes(
                range=[axis_min, axis_max],
                constrain='domain',
                row=row, col=col
            )
            fig.update_yaxes(
                range=[axis_min, axis_max],
                scaleanchor=f"x{(test_idx)*4 + param_idx + 1}",
                scaleratio=1,
                row=row, col=col
            )

            # Add MAE annotation
            mae = res['mae_per_param'][param_idx]
            fig.add_annotation(
                text=f"MAE: {mae:.4f}",
                xref=f"x{(test_idx)*4 + param_idx + 1}",
                yref=f"y{(test_idx)*4 + param_idx + 1}",
                x=axis_min + 0.05 * (axis_max - axis_min),
                y=axis_max - 0.05 * (axis_max - axis_min),
                xanchor='left', yanchor='top',
                showarrow=False,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="black",
                borderwidth=1,
                font=dict(size=9)
            )

            # Add n_comp label on first column
            if param_idx == 0:
                fig.add_annotation(
                    text=f"<b>{n_comp} comp</b>",
                    xref=f"x{(test_idx)*4 + param_idx + 1}",
                    yref=f"y{(test_idx)*4 + param_idx + 1}",
                    x=axis_min + 0.05 * (axis_max - axis_min),
                    y=axis_min + 0.1 * (axis_max - axis_min),
                    xanchor='left', yanchor='bottom',
                    showarrow=False,
                    bgcolor="rgba(255,255,200,0.9)",
                    bordercolor="black",
                    borderwidth=1,
                    font=dict(size=10, color='black')
                )

    fig.update_xaxes(title_text="True Value")
    fig.update_yaxes(title_text="Estimated Value")

    fig.update_layout(
        title="Parameter Estimation: True vs Estimated (gray dashed line = x=y, perfect fit)",
        height=300 * n_tests,
        showlegend=False
    )

    output_path = RESULTS_DIR / 'parameter_comparison.html'
    fig.write_html(str(output_path))
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    print("="*60)
    print("CLEAN DECONVOLUTION TEST (NO NORMALIZATION)")
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
            # Plot individual test result
            plot_single_test(result)
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

    # Generate parameter comparison plot
    print("\nGenerating parameter comparison plot...")
    plot_parameter_comparison(results)

    print("\n" + "="*60)
    print("TEST COMPLETE!")
    print("="*60)
    print("Generated files:")
    print("  - test_6_components.html")
    print("  - test_7_components.html")
    print("  - test_8_components.html")
    print("  - test_9_components.html")
    print("  - test_10_components.html")
    print("  - parameter_comparison.html")
    print("="*60)
