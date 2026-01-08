"""
Peak Deconvolution using Optimization

Estimates GEG parameters given observed spectrum and number of components.
Uses Differential Evolution for global optimization.
"""

import numpy as np
from scipy.optimize import differential_evolution
from peak_generator import X_GRID, generate_spectrum, normalize_by_max


def deconvolve_spectrum(y_observed, n_components, max_iter=1000,
                       popsize=15, verbose=True, seed=None):
    """
    Deconvolve spectrum into n_components peaks using Differential Evolution

    Parameters:
    -----------
    y_observed : array, shape (101,)
        Observed spectrum (unnormalized, absolute intensities)
    n_components : int
        Number of peaks (1-10)
    max_iter : int
        Maximum iterations for optimization
    popsize : int
        Population size multiplier (default 15)
    verbose : bool
        Print progress information
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    params_estimated : ndarray, shape (n_components, 4)
        Estimated parameters [α, τ, μ, σ] for each peak, sorted by μ
    y_fitted : ndarray, shape (101,)
        Fitted spectrum from estimated parameters (unnormalized)
    result : OptimizeResult
        Full optimization result from scipy
    """
    # Validate inputs
    if n_components < 1 or n_components > 10:
        raise ValueError("n_components must be between 1 and 10")

    if len(y_observed) != 101:
        raise ValueError(f"y_observed must have length 101, got {len(y_observed)}")

    # NO normalization - work with absolute intensities

    if verbose:
        print(f"Deconvolving spectrum with {n_components} component(s)")
        print(f"Parameter space: {4 * n_components}D")
        print(f"Max iterations: {max_iter}")

    # Set up parameter bounds
    bounds = get_bounds(n_components)

    # Define loss function for this optimization
    def loss(params_flat):
        return loss_function(params_flat, y_observed, n_components)

    # Run Differential Evolution
    if verbose:
        print("Running Differential Evolution...")

    result = differential_evolution(
        func=loss,
        bounds=bounds,
        maxiter=max_iter,
        popsize=popsize,
        tol=1e-7,
        mutation=(0.5, 1.5),
        recombination=0.7,
        seed=seed,
        polish=True,           # Final local refinement with L-BFGS-B
        workers=1,             # Serial evaluation (avoids pickling issues)
        disp=verbose
    )

    # Extract and reshape parameters
    params_estimated = result.x.reshape(n_components, 4)

    # Sort by μ (peak position) for identifiability
    params_estimated = params_estimated[params_estimated[:, 2].argsort()]

    # Generate fitted spectrum (no normalization)
    y_fitted = generate_spectrum(params_estimated)

    if verbose:
        print(f"\nOptimization {'converged' if result.success else 'did not converge'}")
        print(f"Final MSE: {result.fun:.8f}")
        print(f"Function evaluations: {result.nfev}")
        print(f"\nEstimated parameters (sorted by μ):")
        print("Comp |   α    |   τ    |   μ    |   σ")
        print("-" * 45)
        for i, p in enumerate(params_estimated):
            print(f"  {i+1}  | {p[0]:.4f} | {p[1]:.4f} | {p[2]:.4f} | {p[3]:.4f}")

    return params_estimated, y_fitted, result


def get_bounds(n_components):
    """
    Generate parameter bounds for n_components

    Parameters:
    -----------
    n_components : int
        Number of peaks

    Returns:
    --------
    bounds : list of tuples
        List of (min, max) bounds for each parameter
        Length: 4 * n_components
    """
    single_bounds = [
        (0.5, 3.0),    # α (alpha) - shape parameter
        (0.05, 0.3),   # τ (tau) - exponential component
        (0.0, 1.0),    # μ (mu) - peak position
        (0.01, 0.4)    # σ (sigma) - peak width
    ]

    # Repeat for each component
    bounds = single_bounds * n_components

    return bounds


def loss_function(params_flat, y_observed, n_components):
    """
    Calculate Mean Squared Error between observed and predicted spectrum

    Parameters:
    -----------
    params_flat : array, shape (4 * n_components,)
        Flattened parameters [α₁, τ₁, μ₁, σ₁, α₂, ...]
    y_observed : array, shape (101,)
        Observed spectrum (unnormalized, absolute intensities)
    n_components : int
        Number of peaks

    Returns:
    --------
    mse : float
        Mean squared error
    """
    # Reshape from flat to (n_components, 4)
    params = params_flat.reshape(n_components, 4)

    # Sort by μ (peak position) for identifiability
    # This ensures consistent ordering regardless of optimization path
    params = params[params[:, 2].argsort()]

    # Generate predicted spectrum (no normalization)
    y_predicted = generate_spectrum(params)

    # Calculate Mean Squared Error on absolute intensities
    mse = np.mean((y_observed - y_predicted) ** 2)

    return mse


def calculate_r2_score(y_true, y_pred):
    """
    Calculate R² (coefficient of determination) score

    Parameters:
    -----------
    y_true : array
        True/observed values
    y_pred : array
        Predicted values

    Returns:
    --------
    r2 : float
        R² score (1.0 = perfect fit, 0.0 = mean baseline)
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


def calculate_parameter_mae(params_true, params_estimated):
    """
    Calculate Mean Absolute Error for parameter estimates

    Parameters:
    -----------
    params_true : ndarray, shape (n_components, 4)
        Ground truth parameters
    params_estimated : ndarray, shape (n_components, 4)
        Estimated parameters

    Returns:
    --------
    mae : float
        Mean absolute error across all parameters
    mae_per_param : ndarray, shape (4,)
        MAE for each parameter type [α, τ, μ, σ]
    """
    # Sort both by μ for comparison
    params_true = params_true[params_true[:, 2].argsort()]
    params_estimated = params_estimated[params_estimated[:, 2].argsort()]

    # Calculate absolute errors
    abs_errors = np.abs(params_true - params_estimated)

    # Overall MAE
    mae = np.mean(abs_errors)

    # Per-parameter MAE
    mae_per_param = np.mean(abs_errors, axis=0)

    return mae, mae_per_param


if __name__ == '__main__':
    # Quick test with synthetic data
    print("=" * 60)
    print("Quick Test: Deconvolving a 3-component spectrum")
    print("=" * 60)

    from peak_generator import generate_complete_spectrum

    # Generate synthetic test data
    params_true, y_true = generate_complete_spectrum(3, seed=42)

    if params_true is None:
        print("Failed to generate test spectrum")
    else:
        print("\nGround Truth Parameters:")
        print("Comp |   α    |   τ    |   μ    |   σ")
        print("-" * 45)
        for i, p in enumerate(params_true):
            print(f"  {i+1}  | {p[0]:.4f} | {p[1]:.4f} | {p[2]:.4f} | {p[3]:.4f}")

        # Normalize for comparison
        y_true = normalize_by_max(y_true)

        print("\n" + "=" * 60)

        # Deconvolve
        params_est, y_fit, result = deconvolve_spectrum(
            y_true,
            n_components=3,
            max_iter=500,
            verbose=True,
            seed=123
        )

        # Calculate metrics
        mae, mae_per_param = calculate_parameter_mae(params_true, params_est)
        r2 = calculate_r2_score(y_true, y_fit)

        print("\n" + "=" * 60)
        print("Validation Metrics:")
        print("=" * 60)
        print(f"Overall Parameter MAE: {mae:.6f}")
        print(f"Per-parameter MAE:")
        print(f"  α: {mae_per_param[0]:.6f}")
        print(f"  τ: {mae_per_param[1]:.6f}")
        print(f"  μ: {mae_per_param[2]:.6f}")
        print(f"  σ: {mae_per_param[3]:.6f}")
        print(f"Spectrum R²: {r2:.6f}")
        print("=" * 60)
