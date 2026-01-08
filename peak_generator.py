"""
Simple GEG Peak Generator

Pure function-based implementation of Generalized Exponential-Gaussian distribution
for chromatography peak modeling.
"""

import numpy as np
from scipy.stats import norm

# Fixed grid - same for all data generation
X_GRID = np.arange(0, 1.01, 0.01)  # 101 points from 0 to 1


def geg_peak(x, alpha, tau, mu, sigma):
    """
    Calculate pure GEG equation output

    Parameters:
    -----------
    x : array
        Position values (e.g., time, volume)
    alpha : float
        Shape parameter (0.5 to 3.0)
    tau : float
        Exponential component (0.05 to 0.3)
    mu : float
        Peak center position (0.0 to 1.0)
    sigma : float
        Peak width (0.01 to 0.4)

    Returns:
    --------
    y : array
        GEG probability density function values

    Equation:
    φ(x) = (α/τ) · E · Φ₂ · [Φ₁ - E · Φ₂]^(α-1)
    where:
        E = exp[-(x-μ)/τ + σ²/(2τ²)]
        Φ₁ = Φ((x-μ)/σ)
        Φ₂ = Φ((x-μ)/σ - σ/τ)
    """
    x = np.asarray(x)

    # Calculate z-score
    z = (x - mu) / sigma

    # Exponential term
    exp_term = -(x - mu) / tau + (sigma**2) / (2 * tau**2)
    E = np.exp(exp_term)

    # Two different CDF terms
    Phi_1 = norm.cdf(z)                      # Standard normal CDF
    Phi_2 = norm.cdf(z - sigma / tau)        # Shifted CDF

    # Main equation: (α/τ) · E · Φ₂ · [Φ₁ - E·Φ₂]^(α-1)
    coefficient = alpha / tau
    bracket = Phi_1 - E * Phi_2
    bracket = np.maximum(bracket, 1e-10)  # Avoid numerical issues

    if alpha != 1.0:
        bracket_term = bracket**(alpha - 1)
    else:
        bracket_term = 1.0

    result = coefficient * E * Phi_2 * bracket_term

    return result


def generate_random_params(n_components, seed=None):
    """
    Generate random parameters uniformly sampled from valid ranges

    Parameters:
    -----------
    n_components : int
        Number of peaks to generate parameters for
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    params : ndarray, shape (n_components, 4)
        Each row contains [alpha, tau, mu, sigma]

    Parameter ranges:
        alpha: 0.5 to 3.0
        tau: 0.05 to 0.3
        mu: 0.0 to 1.0
        sigma: 0.01 to 0.4
    """
    if seed is not None:
        np.random.seed(seed)

    params = []
    for _ in range(n_components):
        alpha = np.random.uniform(0.5, 3.0)
        tau = np.random.uniform(0.05, 0.3)
        mu = np.random.uniform(0.0, 1.0)
        sigma = np.random.uniform(0.01, 0.4)
        params.append([alpha, tau, mu, sigma])

    # Convert to array and sort by mu (peak position)
    params = np.array(params)
    params = params[params[:, 2].argsort()]  # Sort by column 2 (mu)

    return params


def generate_spectrum(params):
    """
    Generate summed spectrum from multiple peak parameters

    Parameters:
    -----------
    params : ndarray, shape (n_components, 4)
        Each row contains [alpha, tau, mu, sigma]

    Returns:
    --------
    y_total : ndarray, shape (101,)
        Summed spectrum evaluated at X_GRID points
    """
    y_total = np.zeros_like(X_GRID)

    for param_row in params:
        alpha, tau, mu, sigma = param_row
        y = geg_peak(X_GRID, alpha, tau, mu, sigma)
        y_total += y

    return y_total


def is_spectrum_complete(y_spectrum, tail_threshold=0.7):
    """
    Check if spectrum has complete peak tails (not cut off at edges)

    Parameters:
    -----------
    y_spectrum : ndarray
        Spectrum intensity values
    tail_threshold : float
        Maximum allowed normalized intensity at rightmost edge (default 0.7)

    Returns:
    --------
    is_complete : bool
        True if spectrum tail has decayed below threshold at x=1.0
        False if spectrum is cut off (tail >= threshold)
    """
    # Normalize spectrum to [0, 1]
    y_normalized = y_spectrum / np.max(y_spectrum)

    # Check if rightmost edge has decayed enough
    tail_value = y_normalized[-1]  # Value at x=1.0

    return tail_value < tail_threshold


def generate_complete_spectrum(n_components, tail_threshold=0.7, max_attempts=100, seed=None):
    """
    Generate spectrum with complete peak tails (filtering incomplete ones)

    Parameters:
    -----------
    n_components : int
        Number of peaks (max 10)
    tail_threshold : float
        Maximum normalized intensity at edge (default 0.7)
    max_attempts : int
        Maximum number of attempts before giving up
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    params : ndarray or None
        Parameters if successful, None if max_attempts exceeded
    y_total : ndarray or None
        Spectrum if successful, None if max_attempts exceeded
    """
    if n_components > 10:
        raise ValueError("n_components must be <= 10")

    if seed is not None:
        np.random.seed(seed)

    for attempt in range(max_attempts):
        params = generate_random_params(n_components)
        y_total = generate_spectrum(params)

        if is_spectrum_complete(y_total, tail_threshold):
            return params, y_total

    # Failed to generate complete spectrum
    return None, None


if __name__ == '__main__':
    # Quick test
    print("GEG Peak Generator")
    print("=" * 60)
    print(f"X_GRID: {len(X_GRID)} points from {X_GRID[0]} to {X_GRID[-1]}")

    # Test single peak
    y = geg_peak(X_GRID, alpha=1.5, tau=0.1, mu=0.5, sigma=0.05)
    print(f"\nSingle peak test:")
    print(f"  Max value: {y.max():.4f}")
    print(f"  Peak at x={X_GRID[np.argmax(y)]:.2f}")

    # Test random generation
    params = generate_random_params(n_components=3, seed=42)
    print(f"\nRandom parameters (3 components):")
    print(params)

    y_total = generate_spectrum(params)
    print(f"\nGenerated spectrum:")
    print(f"  Max value: {y_total.max():.4f}")
