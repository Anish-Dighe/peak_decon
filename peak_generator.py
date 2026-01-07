"""
Peak Generator using Generalized Exponential-Gaussian (GEG) Distribution

This module implements the GEG distribution from:
https://pmc.ncbi.nlm.nih.gov/articles/PMC9871144/

The GEG distribution is useful for modeling asymmetric peaks in UV280 spectra.
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


class GEGPeak:
    """
    Generalized Exponential-Gaussian (GEG) Peak Generator

    Parameters:
    -----------
    alpha : float
        Shape parameter controlling skewness and kurtosis (α > 0)
    tau : float
        Exponential component parameter regulating kurtosis (τ > 0)
    mu : float
        Location parameter (center of peak, 0 < μ < 1 for normalized axis)
    sigma : float
        Scale parameter (width, σ > 0)
    amplitude : float
        Peak amplitude (for normalization to 0-1 range)
    """

    def __init__(self, alpha=1.0, tau=0.1, mu=0.5, sigma=0.05, amplitude=1.0):
        self.alpha = alpha
        self.tau = tau
        self.mu = mu
        self.sigma = sigma
        self.amplitude = amplitude

    def pdf(self, x):
        """
        Calculate the GEG probability density function

        φ(x) = ατe^(-(x-μ)/τ + σ²/(2τ²)) * Φ((x-μ)/σ - σ/τ) *
               [Φ((x-μ)/σ - σ/τ) - e^(-(x-μ)/τ + σ²/(2τ²)) * Φ((x-μ)/σ - σ/τ)]^(α-1)

        Parameters:
        -----------
        x : array-like
            Input values (typically wavelength or retention time)

        Returns:
        --------
        array-like
            GEG PDF values scaled by amplitude
        """
        x = np.asarray(x)

        # Calculate components
        z = (x - self.mu) / self.sigma
        exp_term = -(x - self.mu) / self.tau + (self.sigma**2) / (2 * self.tau**2)
        phi_arg = z - self.sigma / self.tau

        # Standard normal CDF
        Phi = norm.cdf(phi_arg)

        # Avoid numerical issues
        exp_val = np.exp(exp_term)

        # Main equation components
        term1 = self.alpha * self.tau * exp_val * Phi

        # Bracket term: [Φ(...) - e^(...)*Φ(...)]^(α-1)
        bracket = Phi - exp_val * Phi
        bracket = np.maximum(bracket, 1e-10)  # Avoid log(0)

        if self.alpha != 1.0:
            term2 = bracket**(self.alpha - 1)
        else:
            term2 = 1.0

        result = term1 * term2

        # Scale by amplitude and normalize
        return self.amplitude * result / np.max(result) if np.max(result) > 0 else result

    def generate_peak(self, x=None, num_points=1000):
        """
        Generate a peak over a normalized range [0, 1]

        Parameters:
        -----------
        x : array-like, optional
            Custom x values. If None, creates linearly spaced points from 0 to 1
        num_points : int
            Number of points to generate if x is not provided

        Returns:
        --------
        tuple
            (x_values, y_values) for the peak
        """
        if x is None:
            x = np.linspace(0, 1, num_points)

        y = self.pdf(x)
        return x, y


class MultiPeakSpectrum:
    """
    Generate UV280 spectra with multiple overlapping peaks
    """

    def __init__(self):
        self.peaks = []

    def add_peak(self, alpha=1.0, tau=0.1, mu=0.5, sigma=0.05, amplitude=1.0):
        """Add a GEG peak to the spectrum"""
        peak = GEGPeak(alpha, tau, mu, sigma, amplitude)
        self.peaks.append(peak)
        return peak

    def generate_spectrum(self, x=None, num_points=1000):
        """
        Generate combined spectrum from all peaks

        Returns:
        --------
        tuple
            (x_values, y_values) for the combined spectrum
        """
        if x is None:
            x = np.linspace(0, 1, num_points)

        y_total = np.zeros_like(x)

        for peak in self.peaks:
            _, y = peak.generate_peak(x, num_points)
            y_total += y

        return x, y_total

    def plot_spectrum(self, show_individual=True, figsize=(12, 6)):
        """
        Plot the spectrum with optional individual peak components

        Parameters:
        -----------
        show_individual : bool
            If True, show individual peaks in addition to combined spectrum
        figsize : tuple
            Figure size (width, height)
        """
        x = np.linspace(0, 1, 1000)

        fig, ax = plt.subplots(figsize=figsize)

        # Plot individual peaks
        if show_individual and len(self.peaks) > 1:
            for i, peak in enumerate(self.peaks):
                _, y = peak.generate_peak(x)
                ax.plot(x, y, '--', alpha=0.5,
                       label=f'Peak {i+1} (μ={peak.mu:.2f}, σ={peak.sigma:.3f}, α={peak.alpha:.2f})')

        # Plot combined spectrum
        x, y_total = self.generate_spectrum(x)
        ax.plot(x, y_total, 'k-', linewidth=2, label='Combined Spectrum')

        ax.set_xlabel('Normalized Position')
        ax.set_ylabel('Normalized Intensity')
        ax.set_title('UV280 Spectrum Simulation using GEG Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, None)

        plt.tight_layout()
        return fig, ax


def generate_spectrum_from_params(params, x=None, num_points=1000):
    """
    Generate spectrum from predefined parameter array - ML-ready function

    Parameters:
    -----------
    params : list of dict or 2D array
        If list of dict: [{'alpha': 1.0, 'tau': 0.1, 'mu': 0.5, 'sigma': 0.05, 'amplitude': 1.0}, ...]
        If 2D array: shape (n_peaks, 5) where columns are [alpha, tau, mu, sigma, amplitude]
    x : array-like, optional
        X values for spectrum. If None, creates linearly spaced points from 0 to 1
    num_points : int
        Number of points if x is not provided

    Returns:
    --------
    tuple
        (x_values, y_values, params_array) - spectrum and ground truth parameters
    """
    if x is None:
        x = np.linspace(0, 1, num_points)

    y_total = np.zeros_like(x)

    # Convert params to standard format
    if isinstance(params, np.ndarray):
        # params is 2D array: (n_peaks, 5)
        params_list = []
        for row in params:
            params_list.append({
                'alpha': row[0],
                'tau': row[1],
                'mu': row[2],
                'sigma': row[3],
                'amplitude': row[4]
            })
        params = params_list

    # Generate each peak and sum
    for peak_params in params:
        peak = GEGPeak(**peak_params)
        _, y = peak.generate_peak(x, num_points)
        y_total += y

    # Convert params to array for ML use
    params_array = np.array([[p['alpha'], p['tau'], p['mu'], p['sigma'], p['amplitude']]
                             for p in params])

    return x, y_total, params_array


def generate_random_spectrum(n_components=3, x=None, num_points=1000,
                            param_ranges=None, seed=None):
    """
    Generate random spectrum for ML training data

    Parameters:
    -----------
    n_components : int
        Number of peaks/components in the spectrum
    x : array-like, optional
        X values for spectrum
    num_points : int
        Number of points if x is not provided
    param_ranges : dict, optional
        Ranges for random parameter generation:
        {
            'alpha': (min, max),
            'tau': (min, max),
            'mu': (min, max),
            'sigma': (min, max),
            'amplitude': (min, max)
        }
        If None, uses sensible defaults for UV280 spectra
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    tuple
        (x_values, y_values, params_array) - spectrum and ground truth parameters
    """
    if seed is not None:
        np.random.seed(seed)

    if param_ranges is None:
        # Default ranges for UV280 spectra
        param_ranges = {
            'alpha': (0.5, 3.0),
            'tau': (0.05, 0.3),
            'mu': (0.15, 0.85),  # Keep peaks away from edges
            'sigma': (0.02, 0.12),
            'amplitude': (0.3, 1.0)
        }

    params = []
    for _ in range(n_components):
        peak_params = {
            'alpha': np.random.uniform(*param_ranges['alpha']),
            'tau': np.random.uniform(*param_ranges['tau']),
            'mu': np.random.uniform(*param_ranges['mu']),
            'sigma': np.random.uniform(*param_ranges['sigma']),
            'amplitude': np.random.uniform(*param_ranges['amplitude'])
        }
        params.append(peak_params)

    # Sort by mu to ensure consistent ordering
    params = sorted(params, key=lambda p: p['mu'])

    return generate_spectrum_from_params(params, x, num_points)


def generate_training_batch(batch_size=32, n_components_range=(1, 10),
                           num_points=1000, param_ranges=None,
                           add_noise=False, noise_level=0.01):
    """
    Generate batch of training samples for ML

    Parameters:
    -----------
    batch_size : int
        Number of spectra to generate
    n_components_range : tuple
        (min, max) number of components per spectrum
    num_points : int
        Number of points in each spectrum
    param_ranges : dict, optional
        Parameter ranges for random generation
    add_noise : bool
        Whether to add Gaussian noise to spectra
    noise_level : float
        Standard deviation of Gaussian noise (relative to max intensity)

    Returns:
    --------
    tuple
        (spectra, params_list) where:
        - spectra: array of shape (batch_size, num_points)
        - params_list: list of parameter arrays, one per spectrum
    """
    x = np.linspace(0, 1, num_points)
    spectra = []
    params_list = []

    for i in range(batch_size):
        # Random number of components
        n_components = np.random.randint(*n_components_range) + 1

        # Generate spectrum
        _, y, params = generate_random_spectrum(
            n_components=n_components,
            x=x,
            num_points=num_points,
            param_ranges=param_ranges,
            seed=None
        )

        # Add noise if requested
        if add_noise:
            noise = np.random.normal(0, noise_level * np.max(y), size=y.shape)
            y = y + noise
            y = np.maximum(y, 0)  # Ensure non-negative

        spectra.append(y)
        params_list.append(params)

    spectra = np.array(spectra)

    return x, spectra, params_list


def demo_single_peak_parameters():
    """
    Demonstrate the effect of different parameters on peak shape
    """
    x = np.linspace(0, 1, 1000)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Effect of alpha (shape parameter)
    ax = axes[0, 0]
    for alpha in [0.5, 1.0, 2.0, 3.0]:
        peak = GEGPeak(alpha=alpha, tau=0.1, mu=0.5, sigma=0.05)
        _, y = peak.generate_peak(x)
        ax.plot(x, y, label=f'α={alpha}')
    ax.set_title('Effect of α (shape parameter)')
    ax.set_xlabel('Normalized Position')
    ax.set_ylabel('Normalized Intensity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Effect of tau (exponential component)
    ax = axes[0, 1]
    for tau in [0.05, 0.1, 0.2, 0.3]:
        peak = GEGPeak(alpha=1.0, tau=tau, mu=0.5, sigma=0.05)
        _, y = peak.generate_peak(x)
        ax.plot(x, y, label=f'τ={tau}')
    ax.set_title('Effect of τ (exponential component)')
    ax.set_xlabel('Normalized Position')
    ax.set_ylabel('Normalized Intensity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Effect of mu (location parameter)
    ax = axes[1, 0]
    for mu in [0.3, 0.4, 0.5, 0.6, 0.7]:
        peak = GEGPeak(alpha=1.0, tau=0.1, mu=mu, sigma=0.05)
        _, y = peak.generate_peak(x)
        ax.plot(x, y, label=f'μ={mu}')
    ax.set_title('Effect of μ (location/center)')
    ax.set_xlabel('Normalized Position')
    ax.set_ylabel('Normalized Intensity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Effect of sigma (scale/width parameter)
    ax = axes[1, 1]
    for sigma in [0.02, 0.05, 0.08, 0.12]:
        peak = GEGPeak(alpha=1.0, tau=0.1, mu=0.5, sigma=sigma)
        _, y = peak.generate_peak(x)
        ax.plot(x, y, label=f'σ={sigma}')
    ax.set_title('Effect of σ (scale/width)')
    ax.set_xlabel('Normalized Position')
    ax.set_ylabel('Normalized Intensity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    # Demo 1: Show effect of different parameters
    print("Generating parameter effect demonstration...")
    fig1 = demo_single_peak_parameters()
    plt.savefig('parameter_effects.png', dpi=150, bbox_inches='tight')
    print("Saved: parameter_effects.png")

    # Demo 2: Create a multi-component spectrum
    print("\nGenerating multi-component spectrum...")
    spectrum = MultiPeakSpectrum()

    # Add multiple peaks simulating different protein components
    spectrum.add_peak(alpha=1.5, tau=0.08, mu=0.3, sigma=0.04, amplitude=0.8)   # Component 1
    spectrum.add_peak(alpha=1.0, tau=0.1, mu=0.5, sigma=0.06, amplitude=1.0)    # Component 2 (dominant)
    spectrum.add_peak(alpha=2.0, tau=0.12, mu=0.65, sigma=0.05, amplitude=0.4)  # Component 3 (weak)

    fig2, ax = spectrum.plot_spectrum(show_individual=True)
    plt.savefig('multi_component_spectrum.png', dpi=150, bbox_inches='tight')
    print("Saved: multi_component_spectrum.png")

    plt.show()
    print("\nDemo complete! Check the generated PNG files.")
