"""
Demo visualization script for GEG peak generator

Creates interactive Plotly plots saved as HTML files for browser viewing.
All spectra are filtered to ensure complete peak tails (not cut off at edges).
"""

import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from peak_generator import (X_GRID, geg_peak, generate_random_params,
                            generate_spectrum, generate_complete_spectrum,
                            normalize_by_max)


# Create results directory if it doesn't exist
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_single_peak_demo():
    """Demonstrate effect of individual parameters"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Effect of α (alpha)', 'Effect of τ (tau)',
                       'Effect of μ (mu)', 'Effect of σ (sigma)')
    )

    # Effect of alpha
    for alpha in [0.5, 1.0, 2.0, 3.0]:
        y = geg_peak(X_GRID, alpha=alpha, tau=0.1, mu=0.5, sigma=0.05)
        fig.add_trace(
            go.Scatter(x=X_GRID, y=y, mode='lines', name=f'α={alpha}',
                      legendgroup='alpha', showlegend=True),
            row=1, col=1
        )

    # Effect of tau
    for tau in [0.05, 0.1, 0.2, 0.3]:
        y = geg_peak(X_GRID, alpha=1.5, tau=tau, mu=0.5, sigma=0.05)
        fig.add_trace(
            go.Scatter(x=X_GRID, y=y, mode='lines', name=f'τ={tau}',
                      legendgroup='tau', showlegend=True),
            row=1, col=2
        )

    # Effect of mu
    for mu in [0.2, 0.4, 0.6, 0.8]:
        y = geg_peak(X_GRID, alpha=1.5, tau=0.1, mu=mu, sigma=0.05)
        fig.add_trace(
            go.Scatter(x=X_GRID, y=y, mode='lines', name=f'μ={mu}',
                      legendgroup='mu', showlegend=True),
            row=2, col=1
        )

    # Effect of sigma
    for sigma in [0.02, 0.05, 0.1, 0.2, 0.4]:
        y = geg_peak(X_GRID, alpha=1.5, tau=0.1, mu=0.5, sigma=sigma)
        fig.add_trace(
            go.Scatter(x=X_GRID, y=y, mode='lines', name=f'σ={sigma}',
                      legendgroup='sigma', showlegend=True),
            row=2, col=2
        )

    fig.update_xaxes(title_text="Position", row=1, col=1)
    fig.update_xaxes(title_text="Position", row=1, col=2)
    fig.update_xaxes(title_text="Position", row=2, col=1)
    fig.update_xaxes(title_text="Position", row=2, col=2)

    fig.update_yaxes(title_text="Intensity", row=1, col=1)
    fig.update_yaxes(title_text="Intensity", row=1, col=2)
    fig.update_yaxes(title_text="Intensity", row=2, col=1)
    fig.update_yaxes(title_text="Intensity", row=2, col=2)

    fig.update_layout(
        title_text="GEG Parameter Effects",
        height=800,
        showlegend=True
    )

    output_path = os.path.join(RESULTS_DIR, 'parameter_effects.html')
    fig.write_html(output_path)
    print(f"Created: {output_path}")


def plot_multi_component_spectra():
    """Show spectra with 2, 3, 4, and 5 components (filtered for completeness)"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('2 Components', '3 Components',
                       '4 Components', '5 Components')
    )

    cases = [
        (2, 1, 1, 42),
        (3, 1, 2, 123),
        (4, 2, 1, 456),
        (5, 2, 2, 789)
    ]

    colors_list = ['red', 'blue', 'green', 'orange', 'purple', 'brown',
                   'pink', 'gray', 'olive', 'cyan']

    for n_comp, row, col, seed in cases:
        # Generate complete spectrum
        params, y_total = generate_complete_spectrum(n_comp, seed=seed)

        if params is None:
            print(f"Warning: Could not generate complete {n_comp}-component spectrum")
            continue

        # Plot total spectrum (normalized)
        y_total_norm = normalize_by_max(y_total)
        fig.add_trace(
            go.Scatter(x=X_GRID, y=y_total_norm, mode='lines',
                      name=f'Total ({n_comp} peaks)', line=dict(color='black', width=3)),
            row=row, col=col
        )

        # Plot individual components (normalized by their own max)
        for i, param_row in enumerate(params):
            alpha, tau, mu, sigma = param_row
            y = geg_peak(X_GRID, alpha, tau, mu, sigma)
            y_norm = normalize_by_max(y)  # Normalize each peak individually
            color = colors_list[i % len(colors_list)]
            fig.add_trace(
                go.Scatter(x=X_GRID, y=y_norm, mode='lines',
                          name=f'Peak {i+1}', line=dict(color=color, width=1, dash='dot'),
                          opacity=0.6),
                row=row, col=col
            )

    fig.update_xaxes(title_text="Position")
    fig.update_yaxes(title_text="Normalized Intensity")

    fig.update_layout(
        title_text="Multi-Component Spectra (Normalized, Complete Peaks)",
        height=800,
        showlegend=False
    )

    output_path = os.path.join(RESULTS_DIR, 'multi_component_spectra.html')
    fig.write_html(output_path)
    print(f"Created: {output_path}")


def plot_component_examples():
    """Show multiple examples for 2, 3, 4, and 5 components"""
    fig = make_subplots(
        rows=4, cols=3,
        subplot_titles=[f'{n}-comp Ex.{i+1}' for n in [2,3,4,5] for i in range(3)],
        vertical_spacing=0.08,
        horizontal_spacing=0.08
    )

    component_counts = [2, 3, 4, 5]

    for comp_idx, n_comp in enumerate(component_counts):
        for ex_idx in range(3):
            row = comp_idx + 1
            col = ex_idx + 1
            seed = n_comp * 100 + ex_idx * 10

            # Generate complete spectrum
            params, y_total = generate_complete_spectrum(n_comp, seed=seed)

            if params is None:
                continue

            # Normalize and plot total spectrum
            y_total_norm = normalize_by_max(y_total)
            fig.add_trace(
                go.Scatter(x=X_GRID, y=y_total_norm, mode='lines',
                          name=f'{n_comp}-comp', line=dict(color='blue', width=2)),
                row=row, col=col
            )

    fig.update_xaxes(title_text="Position")
    fig.update_yaxes(title_text="Normalized Intensity")

    fig.update_layout(
        title_text="Multiple Examples: 2-5 Components (Normalized, Complete Peaks)",
        height=1000,
        showlegend=False
    )

    output_path = os.path.join(RESULTS_DIR, 'component_examples_2to5.html')
    fig.write_html(output_path)
    print(f"Created: {output_path}")


def plot_training_samples():
    """Show batch of random training samples (1-10 components, filtered)"""
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=[f'Sample {i+1}' for i in range(9)]
    )

    np.random.seed(42)

    for idx in range(9):
        row = idx // 3 + 1
        col = idx % 3 + 1

        # Random number of components (1-10)
        n_comp = np.random.randint(1, 11)

        # Generate complete spectrum
        params, y_total = generate_complete_spectrum(n_comp, seed=idx*10)

        if params is None:
            print(f"Warning: Could not generate complete spectrum for sample {idx+1}")
            continue

        # Normalize total spectrum
        y_total_norm = normalize_by_max(y_total)

        fig.add_trace(
            go.Scatter(x=X_GRID, y=y_total_norm, mode='lines',
                      name=f'{n_comp} peaks', line=dict(color='blue')),
            row=row, col=col
        )

        # Update subplot title to include n_comp
        fig.layout.annotations[idx].text = f'Sample {idx+1} ({n_comp} peaks)'

    fig.update_xaxes(title_text="Position")
    fig.update_yaxes(title_text="Normalized Intensity")

    fig.update_layout(
        title_text="Random Training Data Samples (Normalized, 1-10 Components)",
        height=900,
        showlegend=False
    )

    output_path = os.path.join(RESULTS_DIR, 'training_samples.html')
    fig.write_html(output_path)
    print(f"Created: {output_path}")


def plot_detailed_example():
    """Show a detailed 3-component example with parameters"""
    # Generate 3-component spectrum
    params, y_total = generate_complete_spectrum(3, seed=42)

    if params is None:
        print("Warning: Could not generate complete 3-component spectrum")
        return

    # Create figure
    fig = go.Figure()

    # Plot individual peaks (normalized by their own max)
    colors = ['rgba(255,0,0,0.5)', 'rgba(0,0,255,0.5)', 'rgba(0,255,0,0.5)']

    for i, (param_row, color) in enumerate(zip(params, colors)):
        alpha, tau, mu, sigma = param_row
        y = geg_peak(X_GRID, alpha, tau, mu, sigma)
        y_norm = normalize_by_max(y)  # Normalize each peak individually

        fig.add_trace(
            go.Scatter(
                x=X_GRID, y=y_norm,
                mode='lines',
                name=f'Component {i+1} (μ={mu:.2f}, σ={sigma:.2f})',
                line=dict(color=color, width=2),
                fill='tozeroy',
                opacity=0.6
            )
        )

    # Plot total (normalized)
    y_total_norm = normalize_by_max(y_total)
    fig.add_trace(
        go.Scatter(
            x=X_GRID, y=y_total_norm,
            mode='lines',
            name='Total Spectrum',
            line=dict(color='black', width=3)
        )
    )

    # Add parameter table
    param_text = "<b>Ground Truth Parameters:</b><br>"
    param_text += "Component | α | τ | μ | σ<br>"
    for i, p in enumerate(params):
        param_text += f"{i+1} | {p[0]:.2f} | {p[1]:.2f} | {p[2]:.2f} | {p[3]:.2f}<br>"

    # Add tail completeness info
    y_norm = y_total / np.max(y_total)
    tail_value = y_norm[-1]
    param_text += f"<br>Tail value at x=1.0: {tail_value:.3f} (&lt; 0.7 ✓)"

    fig.add_annotation(
        text=param_text,
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1
    )

    fig.update_layout(
        title="3-Component Spectrum Example (Normalized, Complete Peaks)",
        xaxis_title="Position (normalized)",
        yaxis_title="Normalized Intensity",
        height=600,
        hovermode='x unified'
    )

    output_path = os.path.join(RESULTS_DIR, 'detailed_3component.html')
    fig.write_html(output_path)
    print(f"Created: {output_path}")


if __name__ == '__main__':
    print("Generating interactive Plotly visualizations...")
    print("All spectra filtered for complete peak tails (tail < 0.7 at x=1.0)")
    print("=" * 60)

    plot_single_peak_demo()
    plot_multi_component_spectra()
    plot_component_examples()
    plot_detailed_example()
    plot_training_samples()

    print("\n" + "=" * 60)
    print(f"All HTML files created in '{RESULTS_DIR}/' folder:")
    print("  1. parameter_effects.html - Parameter effect demonstration")
    print("  2. multi_component_spectra.html - 2, 3, 4, 5 component examples")
    print("  3. component_examples_2to5.html - Multiple examples for 2-5 components")
    print("  4. detailed_3component.html - Detailed 3-component example")
    print("  5. training_samples.html - Random training samples (1-10 components)")
    print("=" * 60)
