"""
Simple demonstration of GEG peak generation with 2, 6, and 10 components

Shows normalized spectra (max normalization) for different component counts.
"""

import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from peak_generator import (X_GRID, geg_peak, generate_complete_spectrum,
                            normalize_by_max)


RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_component_examples():
    """Show examples with 2, 6, and 10 components (3 examples each)"""

    component_counts = [2, 6, 10]

    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=[f'{n} Components - Example {i+1}'
                       for n in component_counts for i in range(3)],
        vertical_spacing=0.10,
        horizontal_spacing=0.08
    )

    colors_list = ['red', 'blue', 'green', 'orange', 'purple', 'brown',
                   'pink', 'gray', 'olive', 'cyan']

    for comp_idx, n_comp in enumerate(component_counts):
        row = comp_idx + 1

        for ex_idx in range(3):
            col = ex_idx + 1
            seed = n_comp * 100 + ex_idx * 50

            # Generate complete spectrum
            params, y_total = generate_complete_spectrum(n_comp, seed=seed)

            if params is None:
                print(f"Warning: Could not generate complete {n_comp}-component spectrum")
                continue

            # Normalize by total spectrum max
            total_max = np.max(y_total)
            y_normalized = y_total / total_max

            # Plot normalized total spectrum
            fig.add_trace(
                go.Scatter(x=X_GRID, y=y_normalized, mode='lines',
                          name=f'{n_comp} peaks',
                          line=dict(color='black', width=2.5),
                          showlegend=False),
                row=row, col=col
            )

            # Plot individual components (normalized by total max)
            for i, param_row in enumerate(params):
                alpha, tau, mu, sigma = param_row
                y_peak = geg_peak(X_GRID, alpha, tau, mu, sigma)
                y_peak_norm = y_peak / total_max  # Normalize by total max

                color = colors_list[i % len(colors_list)]
                fig.add_trace(
                    go.Scatter(x=X_GRID, y=y_peak_norm, mode='lines',
                              name=f'Peak {i+1}',
                              line=dict(color=color, width=1, dash='dot'),
                              opacity=0.5,
                              showlegend=False),
                    row=row, col=col
                )

    fig.update_xaxes(title_text="Position")
    fig.update_yaxes(title_text="Normalized Intensity")

    fig.update_layout(
        title_text="GEG Spectra: 2, 6, and 10 Components (Max Normalized)",
        height=900,
        showlegend=False
    )

    output_path = os.path.join(RESULTS_DIR, 'component_examples_2_6_10.html')
    fig.write_html(output_path)
    print(f"Created: {output_path}")


def plot_detailed_examples():
    """Show detailed view with parameters for 2, 6, 10 components"""

    component_counts = [2, 6, 10]
    seeds = [142, 256, 310]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[f'{n} Components' for n in component_counts]
    )

    colors_list = ['red', 'blue', 'green', 'orange', 'purple', 'brown',
                   'pink', 'gray', 'olive', 'cyan']

    for idx, (n_comp, seed) in enumerate(zip(component_counts, seeds)):
        col = idx + 1

        params, y_total = generate_complete_spectrum(n_comp, seed=seed)

        if params is None:
            continue

        # Normalize by total spectrum max
        total_max = np.max(y_total)
        y_normalized = y_total / total_max

        # Plot individual peaks (normalized by total max)
        for i, param_row in enumerate(params):
            alpha, tau, mu, sigma = param_row
            y_peak = geg_peak(X_GRID, alpha, tau, mu, sigma)
            y_peak_norm = y_peak / total_max  # Normalize by total max

            color = colors_list[i % len(colors_list)]
            fig.add_trace(
                go.Scatter(
                    x=X_GRID, y=y_peak_norm,
                    mode='lines',
                    name=f'Comp {i+1}',
                    line=dict(color=color, width=2),
                    fill='tozeroy',
                    opacity=0.5,
                    showlegend=False
                ),
                row=1, col=col
            )

        # Plot total spectrum on top
        fig.add_trace(
            go.Scatter(
                x=X_GRID, y=y_normalized,
                mode='lines',
                name='Total',
                line=dict(color='black', width=3),
                showlegend=False
            ),
            row=1, col=col
        )

        # Add parameter info
        param_text = "<b>Parameters:</b><br>"
        param_text += "# | α | τ | μ | σ<br>"
        for i, p in enumerate(params):
            param_text += f"{i+1} | {p[0]:.2f} | {p[1]:.2f} | {p[2]:.2f} | {p[3]:.2f}<br>"

        # Tail value
        tail_value = y_normalized[-1]
        param_text += f"<br>Tail@x=1: {tail_value:.3f}"

        fig.add_annotation(
            text=param_text,
            xref=f"x{col if col > 1 else ''}",
            yref=f"y{col if col > 1 else ''}",
            x=0.02, y=0.98,
            xanchor='left', yanchor='top',
            showarrow=False,
            align="left",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=9)
        )

    fig.update_xaxes(title_text="Position (normalized)")
    fig.update_yaxes(title_text="Normalized Intensity")

    fig.update_layout(
        title="Detailed Examples: 2, 6, and 10 Component Spectra",
        height=600,
        showlegend=False
    )

    output_path = os.path.join(RESULTS_DIR, 'detailed_2_6_10.html')
    fig.write_html(output_path)
    print(f"Created: {output_path}")


if __name__ == '__main__':
    print("Generating simple demo plots for 2, 6, and 10 components...")
    print("=" * 60)

    plot_component_examples()
    plot_detailed_examples()

    print("\n" + "=" * 60)
    print(f"Demo files created in '{RESULTS_DIR}/' folder:")
    print("  1. component_examples_2_6_10.html - 3 examples each of 2, 6, 10 components")
    print("  2. detailed_2_6_10.html - Detailed view with parameters")
    print("=" * 60)
