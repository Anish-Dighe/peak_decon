"""
Validation against published manuscript
Compare GEG implementation with manuscript plots showing effect of alpha
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from peak_generator import geg_peak

# Results directory at project root
RESULTS_DIR = Path(__file__).parent.parent / 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)


def test_manuscript_plot_1():
    """
    Plot 1: Effect of alpha with small tau
    tau = 0.05, mu = 7, sigma = 0.25
    alpha = [0.01, 0.008, 0.01]
    x range = -5 to 15

    Note: alpha values [0.01, 0.008, 0.01] might be a typo in specification
          (0.01 appears twice). Using as provided.
    """
    tau = 0.05
    mu = 7
    sigma = 0.25
    alphas = [0.01, 0.008, 0.01]  # As specified (may contain typo)

    x_range = np.linspace(-5, 15, 500)

    fig = go.Figure()

    colors = ['red', 'blue', 'green']

    for i, alpha in enumerate(alphas):
        y = geg_peak(x_range, alpha, tau, mu, sigma)

        fig.add_trace(
            go.Scatter(
                x=x_range, y=y,
                mode='lines',
                name=f'α = {alpha}',
                line=dict(color=colors[i], width=2)
            )
        )

    fig.update_layout(
        title=f"Manuscript Plot 1: Effect of α<br>τ={tau}, μ={mu}, σ={sigma}",
        xaxis_title="x",
        yaxis_title="Intensity",
        height=500,
        showlegend=True
    )

    output_path = RESULTS_DIR / 'manuscript_plot1.html'
    # Delete old file if exists
    if output_path.exists():
        output_path.unlink()
    fig.write_html(str(output_path))
    print(f"Saved: {output_path}")

    return {
        'tau': tau, 'mu': mu, 'sigma': sigma,
        'alphas': alphas, 'x_range': (-5, 15)
    }


def test_manuscript_plot_2():
    """
    Plot 2: Effect of alpha with moderate tau
    tau = 3.5, mu = 4.5, sigma = 1.75
    alpha = [0.25, 0.75, 2.75]
    x range = -10 to 15
    """
    tau = 3.5
    mu = 4.5
    sigma = 1.75
    alphas = [0.25, 0.75, 2.75]

    x_range = np.linspace(-10, 15, 500)

    fig = go.Figure()

    colors = ['red', 'blue', 'green']

    for i, alpha in enumerate(alphas):
        y = geg_peak(x_range, alpha, tau, mu, sigma)

        fig.add_trace(
            go.Scatter(
                x=x_range, y=y,
                mode='lines',
                name=f'α = {alpha}',
                line=dict(color=colors[i], width=2)
            )
        )

    fig.update_layout(
        title=f"Manuscript Plot 2: Effect of α<br>τ={tau}, μ={mu}, σ={sigma}",
        xaxis_title="x",
        yaxis_title="Intensity",
        height=500,
        showlegend=True
    )

    output_path = RESULTS_DIR / 'manuscript_plot2.html'
    # Delete old file if exists
    if output_path.exists():
        output_path.unlink()
    fig.write_html(str(output_path))
    print(f"Saved: {output_path}")

    return {
        'tau': tau, 'mu': mu, 'sigma': sigma,
        'alphas': alphas, 'x_range': (-10, 15)
    }


def test_manuscript_plot_3():
    """
    Plot 3: Effect of alpha with larger values
    tau = 0.5, mu = 2.25, sigma = 1
    alpha = [1, 4.75, 11.75]
    x range = -10 to 15 (assumed same as plot 2)
    """
    tau = 0.5
    mu = 2.25
    sigma = 1
    alphas = [1, 4.75, 11.75]

    x_range = np.linspace(-10, 15, 500)

    fig = go.Figure()

    colors = ['red', 'blue', 'green']

    for i, alpha in enumerate(alphas):
        y = geg_peak(x_range, alpha, tau, mu, sigma)

        fig.add_trace(
            go.Scatter(
                x=x_range, y=y,
                mode='lines',
                name=f'α = {alpha}',
                line=dict(color=colors[i], width=2)
            )
        )

    fig.update_layout(
        title=f"Manuscript Plot 3: Effect of α<br>τ={tau}, μ={mu}, σ={sigma}",
        xaxis_title="x",
        yaxis_title="Intensity",
        height=500,
        showlegend=True
    )

    output_path = RESULTS_DIR / 'manuscript_plot3.html'
    # Delete old file if exists
    if output_path.exists():
        output_path.unlink()
    fig.write_html(str(output_path))
    print(f"Saved: {output_path}")

    return {
        'tau': tau, 'mu': mu, 'sigma': sigma,
        'alphas': alphas, 'x_range': (-10, 15)
    }


def plot_all_three_combined():
    """
    Create combined figure with all 3 manuscript plots side-by-side
    """
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            'Plot 1: Small τ (0.05)',
            'Plot 2: Moderate τ (3.5)',
            'Plot 3: Medium τ (0.5), Large α'
        ],
        horizontal_spacing=0.10
    )

    colors = ['red', 'blue', 'green']

    # Plot 1
    tau, mu, sigma = 0.05, 7, 0.25
    alphas = [0.01, 0.008, 0.01]
    x_range = np.linspace(-5, 15, 500)

    for i, alpha in enumerate(alphas):
        y = geg_peak(x_range, alpha, tau, mu, sigma)
        fig.add_trace(
            go.Scatter(
                x=x_range, y=y,
                mode='lines',
                name=f'α={alpha}',
                line=dict(color=colors[i], width=2),
                showlegend=False
            ),
            row=1, col=1
        )

    # Plot 2
    tau, mu, sigma = 3.5, 4.5, 1.75
    alphas = [0.25, 0.75, 2.75]
    x_range = np.linspace(-10, 15, 500)

    for i, alpha in enumerate(alphas):
        y = geg_peak(x_range, alpha, tau, mu, sigma)
        fig.add_trace(
            go.Scatter(
                x=x_range, y=y,
                mode='lines',
                name=f'α={alpha}',
                line=dict(color=colors[i], width=2),
                showlegend=False
            ),
            row=1, col=2
        )

    # Plot 3
    tau, mu, sigma = 0.5, 2.25, 1
    alphas = [1, 4.75, 11.75]
    x_range = np.linspace(-10, 15, 500)

    for i, alpha in enumerate(alphas):
        y = geg_peak(x_range, alpha, tau, mu, sigma)
        fig.add_trace(
            go.Scatter(
                x=x_range, y=y,
                mode='lines',
                name=f'α={alpha}',
                line=dict(color=colors[i], width=2),
                showlegend=False
            ),
            row=1, col=3
        )

    # Add legend annotations
    annotations = [
        (1, 1, "α=[0.01, 0.008, 0.01]"),
        (1, 2, "α=[0.25, 0.75, 2.75]"),
        (1, 3, "α=[1, 4.75, 11.75]")
    ]

    for row, col, text in annotations:
        subplot_num = (row - 1) * 3 + col
        x_ref = f"x{subplot_num}" if subplot_num > 1 else "x"
        y_ref = f"y{subplot_num}" if subplot_num > 1 else "y"

        fig.add_annotation(
            text=text,
            xref=x_ref,
            yref=y_ref,
            x=0.98, y=0.95,
            xanchor='right', yanchor='top',
            showarrow=False,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=10)
        )

    fig.update_xaxes(title_text="x")
    fig.update_yaxes(title_text="Intensity")

    fig.update_layout(
        title="Manuscript Validation: Effect of α Parameter on GEG Peak Shape<br><sub>Red = lowest α, Blue = medium α, Green = highest α</sub>",
        height=500,
        showlegend=False
    )

    output_path = RESULTS_DIR / 'manuscript_combined.html'
    # Delete old file if exists
    if output_path.exists():
        output_path.unlink()
    fig.write_html(str(output_path))
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    print("="*60)
    print("MANUSCRIPT VALIDATION TEST")
    print("Comparing GEG implementation with published plots")
    print("="*60)

    print("\nGenerating individual plots...")
    result1 = test_manuscript_plot_1()
    result2 = test_manuscript_plot_2()
    result3 = test_manuscript_plot_3()

    print("\nGenerating combined plot...")
    plot_all_three_combined()

    print("\n" + "="*60)
    print("PARAMETER SUMMARY")
    print("="*60)

    print("\nPlot 1:")
    print(f"  τ={result1['tau']}, μ={result1['mu']}, σ={result1['sigma']}")
    print(f"  α={result1['alphas']}")
    print(f"  x range: {result1['x_range']}")
    print(f"  Note: α values contain 0.01 twice (possible typo)")

    print("\nPlot 2:")
    print(f"  τ={result2['tau']}, μ={result2['mu']}, σ={result2['sigma']}")
    print(f"  α={result2['alphas']}")
    print(f"  x range: {result2['x_range']}")

    print("\nPlot 3:")
    print(f"  τ={result3['tau']}, μ={result3['mu']}, σ={result3['sigma']}")
    print(f"  α={result3['alphas']}")
    print(f"  x range: {result3['x_range']}")

    print("\n" + "="*60)
    print("VALIDATION COMPLETE!")
    print("="*60)
    print("Generated files:")
    print("  - manuscript_plot1.html")
    print("  - manuscript_plot2.html")
    print("  - manuscript_plot3.html")
    print("  - manuscript_combined.html (all 3 plots side-by-side)")
    print("="*60)
    print("\nCompare these plots with the published manuscript to validate")
    print("the GEG equation implementation.")
    print("="*60)
