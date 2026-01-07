"""
Simple demonstration of peak deconvolution optimization

Shows how optimization separates mixed signals into individual components
when given the correct number of components.
"""

import numpy as np
import matplotlib.pyplot as plt
from peak_generator import generate_random_spectrum, GEGPeak
from deconvolve_opt_only import OptimizationDeconvolver

def demonstrate_deconvolution(n_components, seed=42):
    """
    Generate synthetic data and deconvolve it
    """
    print(f"\n{'='*60}")
    print(f"Deconvolving {n_components} component spectrum")
    print(f"{'='*60}")

    # Generate synthetic chromatogram
    print("1. Generating synthetic mixture...")
    x_norm, y_norm, true_params = generate_random_spectrum(
        n_components=n_components,
        seed=seed,
        num_points=1000
    )

    # Convert to time scale (0-20 minutes)
    time_min = x_norm * 20

    # Add realistic noise
    noise = np.random.normal(0, 0.02, size=y_norm.shape)
    y_noisy = y_norm + noise
    y_noisy = np.maximum(y_noisy, 0)

    print(f"   Generated mixture with {n_components} components")
    print(f"   True peak positions: {true_params[:, 2] * 20}")

    # Run deconvolution (tell it the correct number)
    print(f"\n2. Running optimization with n={n_components}...")
    deconvolver = OptimizationDeconvolver(method='differential_evolution')

    n_detected, pred_params, metrics = deconvolver.deconvolve(
        x_norm, y_noisy, n_components=n_components
    )

    print(f"   Optimization complete!")
    print(f"   Predicted positions: {pred_params[:, 2] * 20}")

    # Generate individual component curves
    print("\n3. Generating individual component curves...")
    individual_peaks = []
    for i, param in enumerate(pred_params):
        peak = GEGPeak(
            alpha=param[0],
            tau=param[1],
            mu=param[2],
            sigma=param[3],
            amplitude=param[4]
        )
        _, peak_y = peak.generate_peak(x_norm)
        individual_peaks.append(peak_y)
        print(f"   Component {i+1}: position={param[2]*20:.2f} min, amplitude={param[4]:.3f}")

    # Generate fitted total
    fitted_total = np.sum(individual_peaks, axis=0)

    return time_min, y_noisy, individual_peaks, fitted_total, true_params, pred_params


def plot_deconvolution_result(time_min, observed, individual_peaks, fitted_total,
                               true_params, pred_params, n_components):
    """
    Create a single clear plot showing the deconvolution
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot observed mixed signal (thick black line)
    ax.plot(time_min, observed, 'k-', linewidth=2.5, alpha=0.7,
            label='Observed Mixture', zorder=1)

    # Plot fitted total (dashed red line)
    ax.plot(time_min, fitted_total, 'r--', linewidth=2, alpha=0.8,
            label='Fitted Total', zorder=2)

    # Plot individual deconvoluted peaks (colored lines)
    colors = plt.cm.Set3(np.linspace(0, 1, len(individual_peaks)))
    for i, (peak_y, color) in enumerate(zip(individual_peaks, colors)):
        ax.plot(time_min, peak_y, '-', linewidth=2, color=color, alpha=0.8,
                label=f'Component {i+1}', zorder=3)
        # Fill under curve
        ax.fill_between(time_min, 0, peak_y, color=color, alpha=0.2)

    # Mark true peak positions (vertical green lines)
    for i, pos in enumerate(true_params[:, 2]):
        ax.axvline(x=pos*20, color='green', linestyle=':', linewidth=2,
                   alpha=0.5, label='True positions' if i == 0 else '')

    # Formatting
    ax.set_xlabel('Time (minutes)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Signal Intensity (normalized)', fontsize=14, fontweight='bold')
    ax.set_title(f'Peak Deconvolution: {n_components} Components',
                 fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, None)

    # Add text box with info
    info_text = f"Black line = Observed mixture\n"
    info_text += f"Red dashed = Fitted (sum of components)\n"
    info_text += f"Colored = Individual components\n"
    info_text += f"Green dots = True peak positions"

    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    print("="*60)
    print("SIMPLE PEAK DECONVOLUTION DEMONSTRATION")
    print("="*60)
    print("\nShowing how optimization separates mixed signals")
    print("when given the CORRECT number of components\n")

    # Test cases: 2, 5, and 10 components
    test_cases = [
        (2, 42),
        (5, 123),
        (10, 456)
    ]

    for n_comp, seed in test_cases:
        # Run deconvolution
        time_min, observed, individual_peaks, fitted_total, true_params, pred_params = \
            demonstrate_deconvolution(n_comp, seed)

        # Create plot
        fig = plot_deconvolution_result(
            time_min, observed, individual_peaks, fitted_total,
            true_params, pred_params, n_comp
        )

        # Save
        filename = f'simple_demo_{n_comp}components.png'
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\n   Saved: {filename}")
        plt.close(fig)

    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE!")
    print("="*60)
    print("\nGenerated 3 plots:")
    print("  - simple_demo_2components.png")
    print("  - simple_demo_5components.png")
    print("  - simple_demo_10components.png")
    print("\nEach plot shows:")
    print("  • Black line: The observed mixed signal")
    print("  • Colored lines: Individual deconvoluted components")
    print("  • Red dashed: Sum of deconvoluted components (should match black)")
    print("  • Green dots: True peak positions (ground truth)")
