"""
Demonstration of three normalization methods for GEG spectra

Shows how different normalization approaches affect the same spectrum,
especially for incomplete peaks (cut off at edges).
"""

import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import trapezoid
from peak_generator import (X_GRID, geg_peak, generate_random_params,
                            generate_spectrum, generate_complete_spectrum,
                            normalize_by_max, normalize_by_area,
                            normalize_by_complete_area)


RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)


def generate_incomplete_spectrum(n_components, seed=None, max_attempts=100):
    """Generate a spectrum with incomplete peaks (tail >= 0.7)"""
    if seed is not None:
        np.random.seed(seed)

    for attempt in range(max_attempts):
        params = generate_random_params(n_components)
        y_total = generate_spectrum(params)

        # Check if it's incomplete (tail >= 0.7)
        y_norm = y_total / np.max(y_total)
        tail_value = y_norm[-1]

        if tail_value >= 0.7:
            return params, y_total

    return None, None


def calculate_stats(y_spectrum, params):
    """Calculate statistics for a spectrum"""
    # Max value
    max_val = np.max(y_spectrum)

    # Observed area
    obs_area = trapezoid(y_spectrum, X_GRID)

    # Complete theoretical area
    complete_area = 0.0
    for param_row in params:
        alpha, tau, mu, sigma = param_row
        from peak_generator import calculate_peak_theoretical_area
        peak_area = calculate_peak_theoretical_area(alpha, tau, mu, sigma, 0.0, 1.0)
        complete_area += peak_area

    # Tail value
    y_norm = y_spectrum / max_val
    tail_value = y_norm[-1]

    return {
        'max': max_val,
        'obs_area': obs_area,
        'complete_area': complete_area,
        'tail': tail_value
    }


def plot_normalization_comparison():
    """Compare three normalization methods on same spectrum"""

    # Generate a 3-component spectrum (complete)
    params, y_raw = generate_complete_spectrum(3, seed=42)

    if params is None:
        print("Warning: Could not generate spectrum")
        return

    # Apply three normalizations
    y_norm1 = normalize_by_max(y_raw)
    y_norm2 = normalize_by_area(y_raw)
    y_norm3 = normalize_by_complete_area(y_raw, params)

    # Calculate stats
    stats_raw = calculate_stats(y_raw, params)
    stats1 = calculate_stats(y_norm1, params)
    stats2 = calculate_stats(y_norm2, params)
    stats3 = calculate_stats(y_norm3, params)

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Raw Spectrum (unnormalized)',
            'Method 1: Normalize by Max',
            'Method 2: Normalize by Observed Area',
            'Method 3: Normalize by Complete Area'
        ),
        vertical_spacing=0.12
    )

    # Raw spectrum
    fig.add_trace(
        go.Scatter(x=X_GRID, y=y_raw, mode='lines',
                  line=dict(color='black', width=2), name='Raw'),
        row=1, col=1
    )

    # Method 1
    fig.add_trace(
        go.Scatter(x=X_GRID, y=y_norm1, mode='lines',
                  line=dict(color='blue', width=2), name='By Max'),
        row=1, col=2
    )

    # Method 2
    fig.add_trace(
        go.Scatter(x=X_GRID, y=y_norm2, mode='lines',
                  line=dict(color='green', width=2), name='By Area'),
        row=2, col=1
    )

    # Method 3
    fig.add_trace(
        go.Scatter(x=X_GRID, y=y_norm3, mode='lines',
                  line=dict(color='red', width=2), name='By Complete'),
        row=2, col=2
    )

    # Add annotations with stats
    annotations_text = [
        f"Max: {stats_raw['max']:.3f}<br>Obs Area: {stats_raw['obs_area']:.3f}<br>Complete Area: {stats_raw['complete_area']:.3f}<br>Tail: {stats_raw['tail']:.3f}",
        f"Max: {stats1['max']:.3f}<br>Obs Area: {stats1['obs_area']:.3f}<br>Tail: {stats1['tail']:.3f}",
        f"Max: {stats2['max']:.3f}<br>Obs Area: {stats2['obs_area']:.3f}<br>Tail: {stats2['tail']:.3f}",
        f"Max: {stats3['max']:.3f}<br>Obs Area: {stats3['obs_area']:.3f}<br>Tail: {stats3['tail']:.3f}"
    ]

    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for idx, ((row, col), text) in enumerate(zip(positions, annotations_text)):
        fig.add_annotation(
            text=text,
            xref=f"x{idx+1 if idx > 0 else ''}", yref=f"y{idx+1 if idx > 0 else ''}",
            x=0.7, y=0.95,
            xanchor='left', yanchor='top',
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=10)
        )

    fig.update_xaxes(title_text="Position")
    fig.update_yaxes(title_text="Intensity")

    fig.update_layout(
        title="Normalization Methods Comparison (Complete Peaks)",
        height=800,
        showlegend=False
    )

    output_path = os.path.join(RESULTS_DIR, 'normalization_comparison.html')
    fig.write_html(output_path)
    print(f"Created: {output_path}")


def plot_complete_vs_incomplete():
    """Compare normalizations on complete vs incomplete peaks"""

    # Generate complete spectrum
    params_complete, y_complete = generate_complete_spectrum(3, seed=100)

    # Generate incomplete spectrum
    params_incomplete, y_incomplete = generate_incomplete_spectrum(3, seed=200)

    if params_complete is None or params_incomplete is None:
        print("Warning: Could not generate spectra")
        return

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            'Complete: By Max', 'Complete: By Obs Area', 'Complete: By Complete Area',
            'Incomplete: By Max', 'Incomplete: By Obs Area', 'Incomplete: By Complete Area'
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )

    # Complete peaks - three normalizations
    y_c1 = normalize_by_max(y_complete)
    y_c2 = normalize_by_area(y_complete)
    y_c3 = normalize_by_complete_area(y_complete, params_complete)

    stats_c1 = calculate_stats(y_c1, params_complete)
    stats_c2 = calculate_stats(y_c2, params_complete)
    stats_c3 = calculate_stats(y_c3, params_complete)

    fig.add_trace(go.Scatter(x=X_GRID, y=y_c1, mode='lines',
                            line=dict(color='blue', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=X_GRID, y=y_c2, mode='lines',
                            line=dict(color='green', width=2)), row=1, col=2)
    fig.add_trace(go.Scatter(x=X_GRID, y=y_c3, mode='lines',
                            line=dict(color='red', width=2)), row=1, col=3)

    # Incomplete peaks - three normalizations
    y_i1 = normalize_by_max(y_incomplete)
    y_i2 = normalize_by_area(y_incomplete)
    y_i3 = normalize_by_complete_area(y_incomplete, params_incomplete)

    stats_i1 = calculate_stats(y_i1, params_incomplete)
    stats_i2 = calculate_stats(y_i2, params_incomplete)
    stats_i3 = calculate_stats(y_i3, params_incomplete)

    fig.add_trace(go.Scatter(x=X_GRID, y=y_i1, mode='lines',
                            line=dict(color='blue', width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=X_GRID, y=y_i2, mode='lines',
                            line=dict(color='green', width=2)), row=2, col=2)
    fig.add_trace(go.Scatter(x=X_GRID, y=y_i3, mode='lines',
                            line=dict(color='red', width=2)), row=2, col=3)

    # Add statistics annotations
    stats_list = [stats_c1, stats_c2, stats_c3, stats_i1, stats_i2, stats_i3]

    for idx, stats in enumerate(stats_list):
        row = (idx // 3) + 1
        col = (idx % 3) + 1

        text = f"Max: {stats['max']:.3f}<br>Area: {stats['obs_area']:.3f}<br>Tail: {stats['tail']:.3f}"

        fig.add_annotation(
            text=text,
            xref=f"x{idx+1 if idx > 0 else ''}", yref=f"y{idx+1 if idx > 0 else ''}",
            x=0.65, y=0.95,
            xanchor='left', yanchor='top',
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=9)
        )

    fig.update_xaxes(title_text="Position")
    fig.update_yaxes(title_text="Intensity")

    fig.update_layout(
        title="Complete vs Incomplete Peaks: Normalization Comparison",
        height=700,
        showlegend=False
    )

    output_path = os.path.join(RESULTS_DIR, 'complete_vs_incomplete.html')
    fig.write_html(output_path)
    print(f"Created: {output_path}")


def plot_multicomponent_normalizations():
    """Show 2, 3, 5 component spectra with all three normalizations"""

    component_counts = [2, 3, 5]

    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=[f'{n}-comp: Max' if j==0 else f'{n}-comp: Obs Area' if j==1 else f'{n}-comp: Complete Area'
                       for n in component_counts for j in range(3)],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    colors = ['blue', 'green', 'red']

    for comp_idx, n_comp in enumerate(component_counts):
        row = comp_idx + 1
        seed = n_comp * 50

        params, y_raw = generate_complete_spectrum(n_comp, seed=seed)

        if params is None:
            continue

        # Three normalizations
        y_norm1 = normalize_by_max(y_raw)
        y_norm2 = normalize_by_area(y_raw)
        y_norm3 = normalize_by_complete_area(y_raw, params)

        # Plot each
        fig.add_trace(
            go.Scatter(x=X_GRID, y=y_norm1, mode='lines',
                      line=dict(color=colors[0], width=2)),
            row=row, col=1
        )
        fig.add_trace(
            go.Scatter(x=X_GRID, y=y_norm2, mode='lines',
                      line=dict(color=colors[1], width=2)),
            row=row, col=2
        )
        fig.add_trace(
            go.Scatter(x=X_GRID, y=y_norm3, mode='lines',
                      line=dict(color=colors[2], width=2)),
            row=row, col=3
        )

    fig.update_xaxes(title_text="Position")
    fig.update_yaxes(title_text="Intensity")

    fig.update_layout(
        title="Multi-Component Spectra: Three Normalization Methods",
        height=900,
        showlegend=False
    )

    output_path = os.path.join(RESULTS_DIR, 'multicomponent_normalizations.html')
    fig.write_html(output_path)
    print(f"Created: {output_path}")


def plot_method3_demonstration():
    """Detailed demonstration of why method 3 matters for incomplete peaks"""

    # Generate an incomplete spectrum where rightmost peak is cut off
    params_incomplete, y_incomplete = generate_incomplete_spectrum(3, seed=300)

    if params_incomplete is None:
        print("Warning: Could not generate incomplete spectrum")
        return

    # Show individual peaks to illustrate the cutoff
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Individual Peaks (rightmost cut off)',
            'Method 2: Normalized by Observed Area',
            'Method 3: Normalized by Complete Area',
            'Difference: Method 3 - Method 2'
        ),
        vertical_spacing=0.15
    )

    # Plot individual peaks
    colors_rgba = ['rgba(255,0,0,0.6)', 'rgba(0,0,255,0.6)', 'rgba(0,255,0,0.6)']

    for i, (param_row, color) in enumerate(zip(params_incomplete, colors_rgba)):
        alpha, tau, mu, sigma = param_row
        y_peak = geg_peak(X_GRID, alpha, tau, mu, sigma)

        fig.add_trace(
            go.Scatter(x=X_GRID, y=y_peak, mode='lines',
                      name=f'Peak {i+1} (μ={mu:.2f})',
                      line=dict(color=color, width=2),
                      fill='tozeroy'),
            row=1, col=1
        )

    # Total spectrum
    fig.add_trace(
        go.Scatter(x=X_GRID, y=y_incomplete, mode='lines',
                  name='Total', line=dict(color='black', width=3)),
        row=1, col=1
    )

    # Method 2
    y_norm2 = normalize_by_area(y_incomplete)
    fig.add_trace(
        go.Scatter(x=X_GRID, y=y_norm2, mode='lines',
                  line=dict(color='green', width=2)),
        row=1, col=2
    )

    # Method 3
    y_norm3 = normalize_by_complete_area(y_incomplete, params_incomplete)
    fig.add_trace(
        go.Scatter(x=X_GRID, y=y_norm3, mode='lines',
                  line=dict(color='red', width=2)),
        row=2, col=1
    )

    # Difference
    y_diff = y_norm3 - y_norm2
    fig.add_trace(
        go.Scatter(x=X_GRID, y=y_diff, mode='lines',
                  line=dict(color='purple', width=2),
                  fill='tozeroy'),
        row=2, col=2
    )

    # Add explanatory annotations
    stats_raw = calculate_stats(y_incomplete, params_incomplete)
    stats2 = calculate_stats(y_norm2, params_incomplete)
    stats3 = calculate_stats(y_norm3, params_incomplete)

    explanation = (
        f"<b>Raw Spectrum:</b><br>"
        f"Tail: {stats_raw['tail']:.3f} (>0.7 = incomplete)<br>"
        f"Obs Area: {stats_raw['obs_area']:.3f}<br>"
        f"Complete Area: {stats_raw['complete_area']:.3f}<br><br>"
        f"<b>Missing area:</b> {stats_raw['complete_area'] - stats_raw['obs_area']:.3f}"
    )

    fig.add_annotation(
        text=explanation,
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,200,0.9)",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=10)
    )

    fig.update_xaxes(title_text="Position")
    fig.update_yaxes(title_text="Intensity")

    fig.update_layout(
        title="Why Method 3 Matters: Accounting for Cut-off Peaks",
        height=800,
        showlegend=True
    )

    output_path = os.path.join(RESULTS_DIR, 'method3_demonstration.html')
    fig.write_html(output_path)
    print(f"Created: {output_path}")


if __name__ == '__main__':
    print("Generating normalization demonstration plots...")
    print("=" * 60)

    plot_normalization_comparison()
    plot_complete_vs_incomplete()
    plot_multicomponent_normalizations()
    plot_method3_demonstration()

    print("\n" + "=" * 60)
    print(f"All normalization demo files created in '{RESULTS_DIR}/' folder:")
    print("  1. normalization_comparison.html - Three methods on same spectrum")
    print("  2. complete_vs_incomplete.html - How peak completeness affects normalization")
    print("  3. multicomponent_normalizations.html - 2, 3, 5 components with all methods")
    print("  4. method3_demonstration.html - Why method 3 matters for incomplete peaks")
    print("=" * 60)
