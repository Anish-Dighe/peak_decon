#!/usr/bin/env python
"""
Peak Deconvolution Inference Script

This is the main script for running peak deconvolution on UV280 spectra.

Usage:
    python deconvolve.py input_spectrum.csv --output results/output.csv

Input CSV format:
    - Must have columns for position/wavelength and intensity/absorbance
    - Example: wavelength,absorbance

Output:
    - {output}_peaks.csv: Deconvoluted peak parameters
    - {output}_fitted_spectrum.csv: Fitted spectrum and residuals
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from data_utils import load_spectrum_from_csv, save_deconvolution_results
from model import HybridDeconvolutionModel


def deconvolve_spectrum(input_csv, output_csv,
                       position_col='wavelength',
                       intensity_col='absorbance',
                       model_path=None,
                       use_optimization=True,
                       plot=True):
    """
    Main deconvolution function

    Parameters:
    -----------
    input_csv : str
        Path to input CSV file
    output_csv : str
        Path for output CSV file (will create _peaks.csv and _fitted_spectrum.csv)
    position_col : str
        Name of position/wavelength column
    intensity_col : str
        Name of intensity/absorbance column
    model_path : str, optional
        Path to trained model weights (if None, uses untrained model)
    use_optimization : bool
        Whether to use optimization refinement
    plot : bool
        Whether to generate visualization plot

    Returns:
    --------
    dict
        Results dictionary with n_components, parameters, and metrics
    """
    print("=" * 70)
    print("PEAK DECONVOLUTION FOR UV280 SPECTRA")
    print("=" * 70)

    # Load spectrum
    print(f"\n1. Loading spectrum from: {input_csv}")
    try:
        position, intensity, metadata = load_spectrum_from_csv(
            input_csv,
            position_col=position_col,
            intensity_col=intensity_col,
            normalize=True,
            interpolate_points=1000
        )
        print(f"   ✓ Loaded {len(position)} points")
        print(f"   ✓ Range: {metadata['original_position_min']:.2f} - "
              f"{metadata['original_position_max']:.2f}")
    except Exception as e:
        print(f"   ✗ Error loading file: {e}")
        sys.exit(1)

    # Load model
    print("\n2. Loading deconvolution model...")
    if model_path and Path(model_path).exists():
        print(f"   ✓ Using trained model: {model_path}")
        model = HybridDeconvolutionModel(model_path=model_path)
    else:
        if model_path:
            print(f"   ⚠ Model not found: {model_path}")
        print("   ⚠ Using untrained model (predictions will be unreliable)")
        print("   → Train a model first using: python train.py")
        model = HybridDeconvolutionModel()

    # Predict
    print("\n3. Running deconvolution...")
    print(f"   - Using {'hybrid (NN + optimization)' if use_optimization else 'NN only'} approach")

    n_components, parameters, confidence, metrics = model.predict(
        spectrum_y=intensity,
        spectrum_x=position,
        use_optimization=use_optimization
    )

    print(f"\n   ✓ Detected {n_components} components (confidence: {confidence:.1%})")
    print(f"   ✓ Goodness of fit:")
    print(f"      - R² = {metrics['R2']:.4f}")
    print(f"      - RMSE = {metrics['RMSE']:.6f}")
    print(f"      - NRMSE = {metrics['NRMSE']:.4f}")

    # Generate fitted spectrum
    from peak_generator import GEGPeak
    fitted_intensity = np.zeros_like(intensity)
    for param in parameters:
        peak = GEGPeak(
            alpha=param[0],
            tau=param[1],
            mu=param[2],
            sigma=param[3],
            amplitude=param[4]
        )
        _, peak_y = peak.generate_peak(position)
        fitted_intensity += peak_y

    # Save results
    print("\n4. Saving results...")
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    peaks_file = save_deconvolution_results(
        output_path=output_path,
        n_components=n_components,
        params=parameters,
        metadata=metadata,
        fitted_spectrum=(position, fitted_intensity),
        original_spectrum=(position, intensity),
        goodness_of_fit=metrics
    )

    print(f"   ✓ Results saved!")

    # Plot
    if plot:
        print("\n5. Generating visualization...")
        fig = plot_deconvolution_results(
            position, intensity, fitted_intensity,
            parameters, n_components, metadata, metrics
        )
        plot_path = output_path.parent / f"{output_path.stem}_plot.png"
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"   ✓ Plot saved to: {plot_path}")
        plt.close(fig)

    print("\n" + "=" * 70)
    print("DECONVOLUTION COMPLETE!")
    print("=" * 70)

    return {
        'n_components': n_components,
        'parameters': parameters,
        'confidence': confidence,
        'metrics': metrics,
        'output_file': peaks_file
    }


def plot_deconvolution_results(position, original, fitted, parameters,
                               n_components, metadata, metrics):
    """
    Create visualization of deconvolution results
    """
    from peak_generator import GEGPeak

    # Denormalize position for plotting
    pos_range = metadata['original_position_max'] - metadata['original_position_min']
    pos_min = metadata['original_position_min']
    position_original = position * pos_range + pos_min

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Top plot: Full deconvolution
    ax = axes[0]

    # Plot individual components
    colors = plt.cm.tab10(np.linspace(0, 1, n_components))
    for i, (param, color) in enumerate(zip(parameters, colors)):
        peak = GEGPeak(
            alpha=param[0],
            tau=param[1],
            mu=param[2],
            sigma=param[3],
            amplitude=param[4]
        )
        _, peak_y = peak.generate_peak(position)

        peak_pos_denorm = param[2] * pos_range + pos_min
        ax.plot(position_original, peak_y, '--', color=color, alpha=0.7,
               label=f'Peak {i+1}: {peak_pos_denorm:.2f}')
        ax.fill_between(position_original, 0, peak_y, alpha=0.2, color=color)

    # Plot original and fitted
    ax.plot(position_original, original, 'ko-', alpha=0.5, markersize=2,
           linewidth=1, label='Original')
    ax.plot(position_original, fitted, 'r-', linewidth=2, label='Fitted')

    ax.set_xlabel('Position', fontsize=12)
    ax.set_ylabel('Normalized Intensity', fontsize=12)
    ax.set_title(f'Peak Deconvolution: {n_components} Components\n' +
                f'R² = {metrics["R2"]:.4f}, RMSE = {metrics["RMSE"]:.6f}',
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Bottom plot: Residuals
    ax = axes[1]
    residuals = original - fitted
    ax.plot(position_original, residuals, 'b-', linewidth=1)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.fill_between(position_original, 0, residuals, alpha=0.3)

    ax.set_xlabel('Position', fontsize=12)
    ax.set_ylabel('Residuals', fontsize=12)
    ax.set_title('Fit Residuals', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description='Peak Deconvolution for UV280 Spectra',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python deconvolve.py input.csv --output results/output.csv

  # Specify column names
  python deconvolve.py input.csv --position wavelength --intensity absorbance

  # Use trained model
  python deconvolve.py input.csv --model models/trained_model.pth

  # Disable optimization (faster but less accurate)
  python deconvolve.py input.csv --no-optimization

  # Batch process directory
  python deconvolve.py input_dir/ --batch --output results/
        """
    )

    parser.add_argument('input', type=str,
                       help='Input CSV file or directory (for batch mode)')
    parser.add_argument('--output', '-o', type=str, default='results/output.csv',
                       help='Output CSV path (default: results/output.csv)')
    parser.add_argument('--position', '-p', type=str, default='wavelength',
                       help='Position/wavelength column name (default: wavelength)')
    parser.add_argument('--intensity', '-i', type=str, default='absorbance',
                       help='Intensity/absorbance column name (default: absorbance)')
    parser.add_argument('--model', '-m', type=str, default=None,
                       help='Path to trained model weights')
    parser.add_argument('--no-optimization', action='store_true',
                       help='Disable optimization refinement (faster)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Disable plot generation')
    parser.add_argument('--batch', action='store_true',
                       help='Batch process all CSV files in input directory')

    args = parser.parse_args()

    # Single file or batch mode
    input_path = Path(args.input)

    if args.batch or input_path.is_dir():
        # Batch mode
        if not input_path.is_dir():
            print("Error: --batch requires input to be a directory")
            sys.exit(1)

        csv_files = list(input_path.glob('*.csv'))
        print(f"Found {len(csv_files)} CSV files to process\n")

        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, csv_file in enumerate(csv_files, 1):
            print(f"\n[{i}/{len(csv_files)}] Processing: {csv_file.name}")
            output_file = output_dir / f"{csv_file.stem}_deconvolved.csv"

            try:
                deconvolve_spectrum(
                    input_csv=str(csv_file),
                    output_csv=str(output_file),
                    position_col=args.position,
                    intensity_col=args.intensity,
                    model_path=args.model,
                    use_optimization=not args.no_optimization,
                    plot=not args.no_plot
                )
            except Exception as e:
                print(f"   ✗ Error: {e}")
                continue

    else:
        # Single file mode
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}")
            sys.exit(1)

        deconvolve_spectrum(
            input_csv=str(input_path),
            output_csv=args.output,
            position_col=args.position,
            intensity_col=args.intensity,
            model_path=args.model,
            use_optimization=not args.no_optimization,
            plot=not args.no_plot
        )


if __name__ == '__main__':
    main()
