"""
Test and validate deconvolution algorithm

Tests optimization with different numbers of components (1-5)
and generates visualization comparing fitted vs observed spectra.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os

from peak_generator import (X_GRID, geg_peak, generate_complete_spectrum,
                            normalize_by_max)
from deconvolve import (deconvolve_spectrum, calculate_r2_score,
                       calculate_parameter_mae)


RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)


def test_single_case(n_components, seed=None, max_iter=1000, verbose=True):
    """
    Test deconvolution for a single case

    Parameters:
    -----------
    n_components : int
        Number of peaks
    seed : int, optional
        Random seed for reproducibility
    max_iter : int
        Maximum optimization iterations
    verbose : bool
        Print detailed output

    Returns:
    --------
    results : dict
        Dictionary containing all results and metrics
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Test Case: {n_components} component(s)")
        print(f"{'='*60}")

    # Generate synthetic test data
    params_true, y_true = generate_complete_spectrum(n_components, seed=seed)

    if params_true is None:
        print(f"ERROR: Failed to generate {n_components}-component spectrum")
        return None

    # Normalize
    y_true = normalize_by_max(y_true)

    if verbose:
        print("\nGround Truth Parameters:")
        print("Comp |   α    |   τ    |   μ    |   σ")
        print("-" * 45)
        for i, p in enumerate(params_true):
            print(f"  {i+1}  | {p[0]:.4f} | {p[1]:.4f} | {p[2]:.4f} | {p[3]:.4f}")
        print()

    # Time the optimization
    start_time = time.time()

    # Run deconvolution
    params_est, y_fit, opt_result = deconvolve_spectrum(
        y_true,
        n_components=n_components,
        max_iter=max_iter,
        verbose=verbose,
        seed=seed
    )

    elapsed_time = time.time() - start_time

    # Calculate metrics
    mae_overall, mae_per_param = calculate_parameter_mae(params_true, params_est)
    r2 = calculate_r2_score(y_true, y_fit)
    mse = opt_result.fun

    # Package results
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
        'elapsed_time': elapsed_time,
        'n_iterations': opt_result.nfev,
        'success': opt_result.success
    }

    if verbose:
        print(f"\n{'='*60}")
        print("Results Summary:")
        print(f"{'='*60}")
        print(f"Success: {results['success']}")
        print(f"MSE: {mse:.8f}")
        print(f"R² Score: {r2:.6f}")
        print(f"Parameter MAE: {mae_overall:.6f}")
        print(f"  α MAE: {mae_per_param[0]:.6f}")
        print(f"  τ MAE: {mae_per_param[1]:.6f}")
        print(f"  μ MAE: {mae_per_param[2]:.6f}")
        print(f"  σ MAE: {mae_per_param[3]:.6f}")
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        print(f"Function evaluations: {results['n_iterations']}")
        print(f"{'='*60}")

    return results


def run_test_suite(component_counts=[1, 2, 3, 5], seeds=None, max_iter=1000):
    """
    Run test suite for multiple component counts

    Parameters:
    -----------
    component_counts : list of int
        List of component counts to test
    seeds : list of int, optional
        Random seeds (one per test)
    max_iter : int
        Maximum iterations per test

    Returns:
    --------
    all_results : list of dict
        List of result dictionaries
    """
    if seeds is None:
        seeds = [42 + i*10 for i in range(len(component_counts))]

    print("\n" + "="*60)
    print("DECONVOLUTION TEST SUITE")
    print("="*60)
    print(f"Testing with component counts: {component_counts}")
    print(f"Max iterations: {max_iter}")
    print("="*60)

    all_results = []

    for n_comp, seed in zip(component_counts, seeds):
        result = test_single_case(
            n_components=n_comp,
            seed=seed,
            max_iter=max_iter,
            verbose=True
        )

        if result is not None:
            all_results.append(result)

        # Pause between tests
        time.sleep(0.5)

    # Print summary
    print("\n" + "="*60)
    print("TEST SUITE SUMMARY")
    print("="*60)
    print(f"{'n_comp':<8} {'MSE':<12} {'R²':<10} {'MAE':<10} {'Time(s)':<10} {'Success'}")
    print("-"*60)

    for res in all_results:
        print(f"{res['n_components']:<8} "
              f"{res['mse']:<12.8f} "
              f"{res['r2']:<10.6f} "
              f"{res['mae_overall']:<10.6f} "
              f"{res['elapsed_time']:<10.2f} "
              f"{'✓' if res['success'] else '✗'}")

    print("="*60)

    return all_results


def plot_deconvolution_results(results_list):
    """
    Create visualization comparing true vs fitted spectra

    Parameters:
    -----------
    results_list : list of dict
        List of result dictionaries from test suite
    """
    n_tests = len(results_list)

    # Determine subplot layout
    if n_tests <= 2:
        rows, cols = 1, n_tests
    elif n_tests <= 4:
        rows, cols = 2, 2
    else:
        rows, cols = 2, 3

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f"{res['n_components']} Component(s) - R²={res['r2']:.4f}"
                       for res in results_list],
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )

    for idx, res in enumerate(results_list):
        row = idx // cols + 1
        col = idx % cols + 1

        y_true = res['y_true']
        y_fit = res['y_fitted']

        # Plot true total spectrum (black solid line)
        fig.add_trace(
            go.Scatter(
                x=X_GRID, y=y_true,
                mode='lines',
                name='True Spectrum',
                line=dict(color='black', width=2.5),
                showlegend=(idx == 0)
            ),
            row=row, col=col
        )

        # Plot fitted total spectrum (red dashed line)
        fig.add_trace(
            go.Scatter(
                x=X_GRID, y=y_fit,
                mode='lines',
                name='Fitted Spectrum',
                line=dict(color='red', width=2, dash='dash'),
                showlegend=(idx == 0)
            ),
            row=row, col=col
        )

        # Add metrics annotation
        metrics_text = (
            f"<b>n_comp: {res['n_components']}</b><br>"
            f"MSE: {res['mse']:.8f}<br>"
            f"R²: {res['r2']:.6f}<br>"
            f"Param MAE: {res['mae_overall']:.4f}<br>"
            f"Time: {res['elapsed_time']:.1f}s"
        )

        fig.add_annotation(
            text=metrics_text,
            xref=f"x{idx+1 if idx > 0 else ''}",
            yref=f"y{idx+1 if idx > 0 else ''}",
            x=0.02, y=0.98,
            xanchor='left', yanchor='top',
            showarrow=False,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=10)
        )

    fig.update_xaxes(title_text="Position")
    fig.update_yaxes(title_text="Normalized Intensity")

    fig.update_layout(
        title="Deconvolution Results: True vs Fitted Spectra",
        height=300 * rows,
        showlegend=True,
        legend=dict(x=1.02, y=1.0)
    )

    output_path = os.path.join(RESULTS_DIR, 'deconvolution_test_results.html')
    fig.write_html(output_path)
    print(f"\nVisualization saved to: {output_path}")


def plot_parameter_comparison(results_list):
    """
    Create detailed parameter comparison plots

    Parameters:
    -----------
    results_list : list of dict
        List of result dictionaries
    """
    n_tests = len(results_list)

    fig = make_subplots(
        rows=n_tests, cols=4,
        subplot_titles=['α (alpha)', 'τ (tau)', 'μ (mu)', 'σ (sigma)'] * n_tests,
        row_titles=[f'{res["n_components"]} comp' for res in results_list],
        vertical_spacing=0.08,
        horizontal_spacing=0.08
    )

    param_names = ['α', 'τ', 'μ', 'σ']

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

            # Scatter plot: true vs estimated
            fig.add_trace(
                go.Scatter(
                    x=true_vals,
                    y=est_vals,
                    mode='markers',
                    marker=dict(size=10, color='blue'),
                    showlegend=False
                ),
                row=row, col=col
            )

            # Add diagonal line (perfect fit)
            min_val = min(true_vals.min(), est_vals.min())
            max_val = max(true_vals.max(), est_vals.max())
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    line=dict(color='red', dash='dash', width=1),
                    showlegend=False
                ),
                row=row, col=col
            )

            # Add MAE annotation
            mae = res['mae_per_param'][param_idx]
            fig.add_annotation(
                text=f"MAE: {mae:.4f}",
                xref=f"x{test_idx*4 + param_idx + 1 if test_idx*4 + param_idx > 0 else ''}",
                yref=f"y{test_idx*4 + param_idx + 1 if test_idx*4 + param_idx > 0 else ''}",
                x=0.95, y=0.05,
                xanchor='right', yanchor='bottom',
                showarrow=False,
                bgcolor="rgba(255,255,255,0.8)",
                font=dict(size=8)
            )

    fig.update_xaxes(title_text="True Value")
    fig.update_yaxes(title_text="Estimated Value")

    fig.update_layout(
        title="Parameter Estimation Accuracy: True vs Estimated",
        height=300 * n_tests,
        showlegend=False
    )

    output_path = os.path.join(RESULTS_DIR, 'parameter_comparison.html')
    fig.write_html(output_path)
    print(f"Parameter comparison saved to: {output_path}")


if __name__ == '__main__':
    # Run test suite
    results = run_test_suite(
        component_counts=[1, 2, 3, 5],
        max_iter=1000
    )

    # Generate visualizations
    if results:
        print("\nGenerating visualizations...")
        plot_deconvolution_results(results)
        plot_parameter_comparison(results)
        print("\nAll tests completed!")
