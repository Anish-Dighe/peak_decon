"""
Data utilities for CSV input/output and data preprocessing

This module handles reading experimental UV280 spectra from CSV files
and writing deconvolution results back to CSV format.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def load_spectrum_from_csv(filepath, position_col='position', intensity_col='intensity',
                           normalize=True, interpolate_points=1000):
    """
    Load spectrum from CSV file

    Parameters:
    -----------
    filepath : str or Path
        Path to CSV file with spectrum data
    position_col : str
        Name of column containing position/wavelength values
    intensity_col : str
        Name of column containing intensity values
    normalize : bool
        If True, normalize position to [0, 1] and intensity to [0, max]
    interpolate_points : int or None
        If provided, interpolate spectrum to this many evenly spaced points

    Returns:
    --------
    tuple
        (position, intensity) arrays, and metadata dict
    """
    df = pd.read_csv(filepath)

    if position_col not in df.columns:
        raise ValueError(f"Column '{position_col}' not found in CSV. Available: {df.columns.tolist()}")
    if intensity_col not in df.columns:
        raise ValueError(f"Column '{intensity_col}' not found in CSV. Available: {df.columns.tolist()}")

    position = df[position_col].values
    intensity = df[intensity_col].values

    # Store original ranges
    metadata = {
        'original_position_min': float(position.min()),
        'original_position_max': float(position.max()),
        'original_intensity_min': float(intensity.min()),
        'original_intensity_max': float(intensity.max()),
        'n_original_points': len(position)
    }

    # Normalize position to [0, 1]
    if normalize:
        position_normalized = (position - position.min()) / (position.max() - position.min())
    else:
        position_normalized = position

    # Normalize intensity
    intensity_normalized = intensity - intensity.min()
    if intensity_normalized.max() > 0:
        intensity_normalized = intensity_normalized / intensity_normalized.max()

    # Interpolate to uniform grid if requested
    if interpolate_points is not None:
        position_interp = np.linspace(0, 1, interpolate_points)
        intensity_interp = np.interp(position_interp, position_normalized, intensity_normalized)
        return position_interp, intensity_interp, metadata

    return position_normalized, intensity_normalized, metadata


def denormalize_parameters(params, metadata):
    """
    Convert normalized parameters back to original scale

    Parameters:
    -----------
    params : array-like, shape (n_peaks, 5)
        Normalized parameters [alpha, tau, mu, sigma, amplitude]
    metadata : dict
        Metadata from load_spectrum_from_csv containing original ranges

    Returns:
    --------
    pandas.DataFrame
        Parameters in original scale with descriptive column names
    """
    params = np.atleast_2d(params)
    n_peaks = params.shape[0]

    # Denormalize mu (position) and sigma (width)
    pos_range = metadata['original_position_max'] - metadata['original_position_min']
    pos_min = metadata['original_position_min']

    results = []
    for i in range(n_peaks):
        alpha, tau, mu_norm, sigma_norm, amplitude = params[i]

        result = {
            'peak_id': i + 1,
            'alpha': alpha,
            'tau': tau,
            'position': mu_norm * pos_range + pos_min,  # Denormalize to original scale
            'position_normalized': mu_norm,  # Keep normalized for reference
            'width': sigma_norm * pos_range,  # Denormalize to original scale
            'width_normalized': sigma_norm,  # Keep normalized for reference
            'amplitude': amplitude,
            'relative_intensity': amplitude / params[:, 4].sum() if params[:, 4].sum() > 0 else 0
        }
        results.append(result)

    return pd.DataFrame(results)


def save_deconvolution_results(output_path, n_components, params, metadata,
                               fitted_spectrum=None, original_spectrum=None,
                               goodness_of_fit=None):
    """
    Save deconvolution results to CSV file

    Parameters:
    -----------
    output_path : str or Path
        Path for output CSV file
    n_components : int
        Number of detected components
    params : array-like, shape (n_components, 5)
        Peak parameters [alpha, tau, mu, sigma, amplitude]
    metadata : dict
        Metadata from input spectrum
    fitted_spectrum : tuple, optional
        (position, intensity) of fitted spectrum for validation
    original_spectrum : tuple, optional
        (position, intensity) of original spectrum for comparison
    goodness_of_fit : dict, optional
        Dictionary with fit quality metrics (R2, RMSE, etc.)
    """
    output_path = Path(output_path)

    # Save peak parameters
    params_df = denormalize_parameters(params, metadata)
    params_df.insert(0, 'n_components', n_components)

    # Add goodness of fit metrics if available
    if goodness_of_fit is not None:
        for key, value in goodness_of_fit.items():
            params_df[key] = value

    params_file = output_path.parent / f"{output_path.stem}_peaks.csv"
    params_df.to_csv(params_file, index=False)
    print(f"Saved peak parameters to: {params_file}")

    # Save fitted spectrum if provided
    if fitted_spectrum is not None:
        position_fit, intensity_fit = fitted_spectrum

        # Denormalize position
        pos_range = metadata['original_position_max'] - metadata['original_position_min']
        pos_min = metadata['original_position_min']
        position_original_scale = position_fit * pos_range + pos_min

        spectrum_df = pd.DataFrame({
            'position': position_original_scale,
            'fitted_intensity': intensity_fit
        })

        # Add original spectrum if provided
        if original_spectrum is not None:
            position_orig, intensity_orig = original_spectrum
            position_orig_scale = position_orig * pos_range + pos_min
            # Interpolate original to match fitted grid
            intensity_orig_interp = np.interp(position_original_scale, position_orig_scale, intensity_orig)
            spectrum_df['original_intensity'] = intensity_orig_interp
            spectrum_df['residual'] = intensity_orig_interp - intensity_fit

        spectrum_file = output_path.parent / f"{output_path.stem}_fitted_spectrum.csv"
        spectrum_df.to_csv(spectrum_file, index=False)
        print(f"Saved fitted spectrum to: {spectrum_file}")

    return params_file


def create_sample_csv(output_path='sample_spectrum.csv', n_components=3):
    """
    Create a sample CSV file with synthetic UV280 spectrum for testing

    Parameters:
    -----------
    output_path : str
        Path for output CSV file
    n_components : int
        Number of components to include
    """
    from peak_generator import generate_random_spectrum

    # Generate synthetic data
    x_norm, y_norm, params = generate_random_spectrum(n_components=n_components, seed=42)

    # Simulate realistic UV280 wavelength range (270-290 nm typical)
    wavelength = x_norm * 20 + 270  # 270-290 nm
    intensity = y_norm * 1000  # Scale to realistic absorbance units

    # Add some noise
    noise = np.random.normal(0, 5, size=intensity.shape)
    intensity = intensity + noise
    intensity = np.maximum(intensity, 0)  # Ensure non-negative

    # Create DataFrame
    df = pd.DataFrame({
        'wavelength': wavelength,
        'absorbance': intensity
    })

    df.to_csv(output_path, index=False)
    print(f"Created sample CSV: {output_path}")
    print(f"  - {n_components} components")
    print(f"  - Wavelength range: {wavelength.min():.1f} - {wavelength.max():.1f} nm")
    print(f"  - Ground truth parameters:")
    for i, p in enumerate(params):
        print(f"    Peak {i+1}: position={p[2]*20+270:.2f} nm, amplitude={p[4]:.3f}")

    return output_path, params


def load_batch_from_directory(directory, pattern='*.csv', **load_kwargs):
    """
    Load multiple spectra from a directory

    Parameters:
    -----------
    directory : str or Path
        Directory containing CSV files
    pattern : str
        Glob pattern to match CSV files
    **load_kwargs : dict
        Additional arguments passed to load_spectrum_from_csv

    Returns:
    --------
    list
        List of tuples (filepath, position, intensity, metadata)
    """
    directory = Path(directory)
    files = sorted(directory.glob(pattern))

    if len(files) == 0:
        raise ValueError(f"No files matching '{pattern}' found in {directory}")

    results = []
    for filepath in files:
        try:
            pos, intensity, metadata = load_spectrum_from_csv(filepath, **load_kwargs)
            results.append((filepath, pos, intensity, metadata))
            print(f"Loaded: {filepath.name}")
        except Exception as e:
            print(f"Error loading {filepath.name}: {e}")

    return results


if __name__ == '__main__':
    print("Data Utilities Demo")
    print("=" * 60)

    # Demo 1: Create sample CSV
    print("\n1. Creating sample CSV file...")
    sample_file, true_params = create_sample_csv('sample_spectrum.csv', n_components=4)

    # Demo 2: Load the CSV
    print("\n2. Loading spectrum from CSV...")
    position, intensity, metadata = load_spectrum_from_csv(
        'sample_spectrum.csv',
        position_col='wavelength',
        intensity_col='absorbance'
    )
    print(f"Loaded spectrum: {len(position)} points")
    print(f"Metadata: {metadata}")

    # Demo 3: Create mock deconvolution results
    print("\n3. Saving mock deconvolution results...")
    mock_params = np.array([
        [1.5, 0.08, 0.25, 0.04, 0.8],
        [1.0, 0.10, 0.50, 0.06, 1.0],
        [1.8, 0.12, 0.65, 0.05, 0.6],
        [2.0, 0.09, 0.80, 0.04, 0.4]
    ])

    mock_gof = {
        'R2': 0.985,
        'RMSE': 12.3,
        'chi_squared': 0.045
    }

    save_deconvolution_results(
        output_path='results/sample_deconvolution.csv',
        n_components=4,
        params=mock_params,
        metadata=metadata,
        fitted_spectrum=(position, intensity),
        original_spectrum=(position, intensity),
        goodness_of_fit=mock_gof
    )

    print("\n" + "=" * 60)
    print("Demo complete! Check the generated CSV files.")
