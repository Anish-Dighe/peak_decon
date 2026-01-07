"""
Validation Test Suite for Peak Deconvolution

Generates synthetic chromatography data with known ground truth,
runs deconvolution, and compares results.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from peak_generator import generate_random_spectrum
from deconvolve_opt_only import OptimizationDeconvolver
from data_utils import load_spectrum_from_csv, save_deconvolution_results

# Create test directories
Path("test_data").mkdir(exist_ok=True)
Path("test_results").mkdir(exist_ok=True)


def generate_chromatogram(n_components, test_id, add_noise=True, noise_level=0.02):
    """
    Generate synthetic chromatography data

    Returns:
    --------
    filepath, true_params, x, y
    """
    # Generate spectrum
    x_norm, y_norm, true_params = generate_random_spectrum(
        n_components=n_components,
        seed=test_id * 100 + n_components,
        num_points=1000
    )

    # Convert to chromatography units
    time_min = x_norm * 20  # 0-20 minutes
    signal_mAU = y_norm * 1000  # 0-1000 mAU scale

    # Add realistic noise
    if add_noise:
        noise = np.random.normal(0, noise_level * 1000, size=signal_mAU.shape)
        signal_mAU = signal_mAU + noise
        signal_mAU = np.maximum(signal_mAU, 0)  # Ensure non-negative

    # Save to CSV
    filename = f"test_data/test_{n_components}comp_run{test_id}.csv"
    df = pd.DataFrame({
        'time_minutes': time_min,
        'UV280_mAU': signal_mAU
    })
    df.to_csv(filename, index=False)

    return filename, true_params, x_norm, y_norm


def calculate_peak_matching_error(true_params, pred_params):
    """
    Calculate error between true and predicted parameters
    Matches peaks by position (mu)
    """
    n_true = len(true_params)
    n_pred = len(pred_params)

    if n_true != n_pred:
        return None, f"Component mismatch: true={n_true}, predicted={n_pred}"

    # Sort both by position (mu is column 2)
    true_sorted = true_params[np.argsort(true_params[:, 2])]
    pred_sorted = pred_params[np.argsort(pred_params[:, 2])]

    # Calculate errors for each parameter
    errors = {
        'position_error': np.abs(true_sorted[:, 2] - pred_sorted[:, 2]),
        'width_error': np.abs(true_sorted[:, 3] - pred_sorted[:, 3]),
        'amplitude_error': np.abs(true_sorted[:, 4] - pred_sorted[:, 4]),
        'mean_position_error': np.mean(np.abs(true_sorted[:, 2] - pred_sorted[:, 2])),
        'mean_width_error': np.mean(np.abs(true_sorted[:, 3] - pred_sorted[:, 3])),
        'mean_amplitude_error': np.mean(np.abs(true_sorted[:, 4] - pred_sorted[:, 4]))
    }

    return errors, None


def run_test(n_components, test_id):
    """
    Run a single test
    """
    print(f"\n{'='*70}")
    print(f"TEST: {n_components} components, Run #{test_id}")
    print(f"{'='*70}")

    # Generate synthetic data
    print("  Generating synthetic chromatogram...")
    filename, true_params, x_norm, y_norm = generate_chromatogram(
        n_components, test_id
    )

    print(f"  Ground truth parameters:")
    print(f"    Positions (normalized): {true_params[:, 2]}")
    print(f"    Amplitudes: {true_params[:, 4]}")

    # Load data
    position, intensity, metadata = load_spectrum_from_csv(
        filename,
        position_col='time_minutes',
        intensity_col='UV280_mAU',
        normalize=True,
        interpolate_points=1000
    )

    # Run deconvolution
    print("  Running deconvolution...")
    deconvolver = OptimizationDeconvolver(method='differential_evolution')

    # Test with correct number of components
    n_detected, pred_params, metrics = deconvolver.deconvolve(
        position, intensity, n_components=n_components
    )

    print(f"  Detected {n_detected} components")
    print(f"  Predicted positions: {pred_params[:, 2]}")
    print(f"  Predicted amplitudes: {pred_params[:, 4]}")
    print(f"  R² = {metrics['R2']:.4f}, RMSE = {metrics['RMSE']:.6f}")

    # Calculate errors
    errors, error_msg = calculate_peak_matching_error(true_params, pred_params)

    if errors is None:
        print(f"  ERROR: {error_msg}")
        return {
            'n_components': n_components,
            'test_id': test_id,
            'success': False,
            'error': error_msg,
            'R2': metrics['R2'],
            'RMSE': metrics['RMSE']
        }

    print(f"  Mean position error: {errors['mean_position_error']:.4f}")
    print(f"  Mean amplitude error: {errors['mean_amplitude_error']:.4f}")

    # Save results
    output_file = f"test_results/test_{n_components}comp_run{test_id}.csv"

    fitted = deconvolver._generate_spectrum(position, pred_params)
    save_deconvolution_results(
        output_path=output_file,
        n_components=n_detected,
        params=pred_params,
        metadata=metadata,
        fitted_spectrum=(position, fitted),
        original_spectrum=(position, intensity),
        goodness_of_fit=metrics
    )

    return {
        'n_components': n_components,
        'test_id': test_id,
        'success': True,
        'n_detected': n_detected,
        'R2': metrics['R2'],
        'RMSE': metrics['RMSE'],
        'NRMSE': metrics['NRMSE'],
        'mean_position_error': errors['mean_position_error'],
        'mean_width_error': errors['mean_width_error'],
        'mean_amplitude_error': errors['mean_amplitude_error'],
        'true_positions': true_params[:, 2].tolist(),
        'pred_positions': pred_params[:, 2].tolist(),
        'true_amplitudes': true_params[:, 4].tolist(),
        'pred_amplitudes': pred_params[:, 4].tolist()
    }


def run_test_suite():
    """
    Run complete test suite
    """
    print("="*70)
    print("PEAK DECONVOLUTION VALIDATION TEST SUITE")
    print("="*70)
    print("\nTesting with 1, 3, and 6 components (10 tests each)")
    print("Total: 30 tests\n")

    all_results = []

    # Test with 1, 3, and 6 components
    for n_components in [1, 3, 6]:
        print(f"\n{'#'*70}")
        print(f"# TESTING {n_components} COMPONENT(S)")
        print(f"{'#'*70}")

        for test_id in range(1, 11):
            result = run_test(n_components, test_id)
            all_results.append(result)

    # Save all results to CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('test_results/all_test_results.csv', index=False)

    # Generate summary statistics
    print("\n" + "="*70)
    print("TEST SUITE SUMMARY")
    print("="*70)

    for n_comp in [1, 3, 6]:
        subset = results_df[results_df['n_components'] == n_comp]
        successful = subset[subset['success'] == True]

        print(f"\n{n_comp} Component(s): {len(successful)}/10 successful")

        if len(successful) > 0:
            print(f"  Mean R²: {successful['R2'].mean():.4f} ± {successful['R2'].std():.4f}")
            print(f"  Mean RMSE: {successful['RMSE'].mean():.6f} ± {successful['RMSE'].std():.6f}")
            print(f"  Mean Position Error: {successful['mean_position_error'].mean():.4f} ± {successful['mean_position_error'].std():.4f}")
            print(f"  Mean Amplitude Error: {successful['mean_amplitude_error'].mean():.4f} ± {successful['mean_amplitude_error'].std():.4f}")

    # Create visualization
    create_summary_plots(results_df)

    return results_df


def create_summary_plots(results_df):
    """
    Create summary visualization plots
    """
    successful = results_df[results_df['success'] == True]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: R² by component count
    ax = axes[0, 0]
    for n_comp in [1, 3, 6]:
        subset = successful[successful['n_components'] == n_comp]
        ax.scatter([n_comp] * len(subset), subset['R2'], alpha=0.6, s=100)
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('R² Score')
    ax.set_title('Goodness of Fit by Component Count')
    ax.set_xticks([1, 3, 6])
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.95, color='r', linestyle='--', label='R²=0.95')
    ax.legend()

    # Plot 2: RMSE by component count
    ax = axes[0, 1]
    for n_comp in [1, 3, 6]:
        subset = successful[successful['n_components'] == n_comp]
        ax.scatter([n_comp] * len(subset), subset['RMSE'], alpha=0.6, s=100)
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('RMSE')
    ax.set_title('Root Mean Square Error')
    ax.set_xticks([1, 3, 6])
    ax.grid(True, alpha=0.3)

    # Plot 3: Position error by component count
    ax = axes[0, 2]
    for n_comp in [1, 3, 6]:
        subset = successful[successful['n_components'] == n_comp]
        ax.scatter([n_comp] * len(subset), subset['mean_position_error'], alpha=0.6, s=100)
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Mean Position Error')
    ax.set_title('Peak Position Accuracy')
    ax.set_xticks([1, 3, 6])
    ax.grid(True, alpha=0.3)

    # Plot 4: Amplitude error by component count
    ax = axes[1, 0]
    for n_comp in [1, 3, 6]:
        subset = successful[successful['n_components'] == n_comp]
        ax.scatter([n_comp] * len(subset), subset['mean_amplitude_error'], alpha=0.6, s=100)
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Mean Amplitude Error')
    ax.set_title('Peak Amplitude Accuracy')
    ax.set_xticks([1, 3, 6])
    ax.grid(True, alpha=0.3)

    # Plot 5: R² distribution
    ax = axes[1, 1]
    for n_comp, color in zip([1, 3, 6], ['blue', 'green', 'red']):
        subset = successful[successful['n_components'] == n_comp]
        ax.hist(subset['R2'], alpha=0.5, label=f'{n_comp} comp', color=color, bins=10)
    ax.set_xlabel('R² Score')
    ax.set_ylabel('Frequency')
    ax.set_title('R² Score Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: Success rate
    ax = axes[1, 2]
    success_rates = []
    for n_comp in [1, 3, 6]:
        subset = results_df[results_df['n_components'] == n_comp]
        success_rate = (subset['success'].sum() / len(subset)) * 100
        success_rates.append(success_rate)
    ax.bar([1, 3, 6], success_rates, color=['blue', 'green', 'red'], alpha=0.7)
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Deconvolution Success Rate')
    ax.set_xticks([1, 3, 6])
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3, axis='y')
    for i, (x, y) in enumerate(zip([1, 3, 6], success_rates)):
        ax.text(x, y + 2, f'{y:.0f}%', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('test_results/validation_summary.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved summary plots to: test_results/validation_summary.png")
    plt.close()


def show_individual_test_details(results_df, n_components, test_id):
    """
    Show detailed results for a specific test
    """
    result = results_df[
        (results_df['n_components'] == n_components) &
        (results_df['test_id'] == test_id)
    ].iloc[0]

    print(f"\n{'='*70}")
    print(f"DETAILED RESULTS: {n_components} components, Test #{test_id}")
    print(f"{'='*70}")
    print(f"Success: {result['success']}")
    print(f"R² Score: {result['R2']:.4f}")
    print(f"RMSE: {result['RMSE']:.6f}")

    if result['success']:
        print(f"\nTrue vs Predicted Positions:")
        for i, (true_pos, pred_pos) in enumerate(zip(result['true_positions'], result['pred_positions'])):
            error = abs(true_pos - pred_pos)
            print(f"  Peak {i+1}: True={true_pos:.4f}, Pred={pred_pos:.4f}, Error={error:.4f}")

        print(f"\nTrue vs Predicted Amplitudes:")
        for i, (true_amp, pred_amp) in enumerate(zip(result['true_amplitudes'], result['pred_amplitudes'])):
            error = abs(true_amp - pred_amp)
            print(f"  Peak {i+1}: True={true_amp:.4f}, Pred={pred_amp:.4f}, Error={error:.4f}")


if __name__ == '__main__':
    # Run complete test suite
    results_df = run_test_suite()

    # Show detailed results for first test of each component count
    print("\n" + "#"*70)
    print("# DETAILED RESULTS FOR SELECTED TESTS")
    print("#"*70)

    for n_comp in [1, 3, 6]:
        show_individual_test_details(results_df, n_comp, 1)

    print("\n" + "="*70)
    print("VALIDATION COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: test_results/")
    print(f"  - all_test_results.csv: Complete results table")
    print(f"  - validation_summary.png: Summary plots")
    print(f"  - Individual test results in test_results/")
