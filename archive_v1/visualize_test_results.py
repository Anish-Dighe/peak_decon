"""
Visualize test results with original and deconvoluted data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from peak_generator import GEGPeak
from data_utils import load_spectrum_from_csv

def plot_test_result(n_components, test_id, results_df):
    """
    Create detailed plot for a specific test
    """
    # Get test result
    result = results_df[
        (results_df['n_components'] == n_components) &
        (results_df['test_id'] == test_id)
    ].iloc[0]

    # Load data
    data_file = f"test_data/test_{n_components}comp_run{test_id}.csv"
    position, intensity, metadata = load_spectrum_from_csv(
        data_file,
        position_col='time_minutes',
        intensity_col='UV280_mAU',
        normalize=True
    )

    # Denormalize for plotting (convert to time)
    time_minutes = position * 20  # 0-20 minutes

    # Get true parameters (only if successful)
    if result['success']:
        true_params = np.array(eval(result['true_positions']))
        true_amps = np.array(eval(result['true_amplitudes']))
    else:
        # For failed tests, we need to regenerate ground truth
        from peak_generator import generate_random_spectrum
        _, _, params_array = generate_random_spectrum(
            n_components=n_components,
            seed=test_id * 100 + n_components,
            num_points=1000
        )
        true_params = params_array[:, 2]  # mu (position)
        true_amps = params_array[:, 4]    # amplitude

    # Generate true individual peaks
    true_peaks_y = []
    for i in range(len(true_params)):
        # We need to reconstruct the peak from the test data generator
        # For visualization, we'll just mark the positions
        pass

    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Main plot - top row spanning both columns
    ax_main = fig.add_subplot(gs[0, :])

    # Plot original data
    ax_main.plot(time_minutes, intensity, 'ko-', alpha=0.5, markersize=2,
                linewidth=1, label='Observed Signal', zorder=1)

    # Mark true peak positions
    for i, (pos, amp) in enumerate(zip(true_params, true_amps)):
        time_pos = pos * 20
        ax_main.axvline(x=time_pos, color='green', linestyle='--',
                       alpha=0.7, linewidth=2, label=f'True Peak {i+1}' if i == 0 else '')
        ax_main.plot(time_pos, amp, 'g^', markersize=15, alpha=0.7, zorder=3)

    # If successful deconvolution, plot predicted peaks
    if result['success']:
        pred_params = np.array(eval(result['pred_positions']))
        pred_amps = np.array(eval(result['pred_amplitudes']))

        # Plot predicted peak positions
        for i, (pos, amp) in enumerate(zip(pred_params, pred_amps)):
            time_pos = pos * 20
            ax_main.axvline(x=time_pos, color='red', linestyle=':',
                           alpha=0.7, linewidth=2, label=f'Predicted Peak {i+1}' if i == 0 else '')
            ax_main.plot(time_pos, amp, 'rs', markersize=12, alpha=0.7, zorder=3)

        # Load fitted spectrum from results
        fitted_file = f"test_results/test_{n_components}comp_run{test_id}_fitted_spectrum.csv"
        if Path(fitted_file).exists():
            fitted_df = pd.read_csv(fitted_file)
            ax_main.plot(fitted_df['position'], fitted_df['fitted_intensity'],
                        'r-', linewidth=2.5, label='Fitted Spectrum', zorder=2)

    ax_main.set_xlabel('Time (minutes)', fontsize=12, fontweight='bold')
    ax_main.set_ylabel('Normalized Intensity', fontsize=12, fontweight='bold')

    title = f'{n_components} Component Test #{test_id}'
    if result['success']:
        title += f" - SUCCESS (R²={result['R2']:.4f})"
        ax_main.set_title(title, fontsize=14, fontweight='bold', color='green')
    else:
        title += f" - FAILED ({result.get('error', 'Unknown error')})"
        ax_main.set_title(title, fontsize=14, fontweight='bold', color='red')

    ax_main.legend(loc='best', fontsize=10)
    ax_main.grid(True, alpha=0.3)
    ax_main.set_xlim(0, 20)

    # Position comparison plot
    ax_pos = fig.add_subplot(gs[1, 0])
    if result['success']:
        x = np.arange(len(true_params))
        width = 0.35
        ax_pos.bar(x - width/2, true_params * 20, width, label='True',
                  color='green', alpha=0.7)
        ax_pos.bar(x + width/2, pred_params * 20, width, label='Predicted',
                  color='red', alpha=0.7)
        ax_pos.set_xlabel('Peak Number', fontsize=11, fontweight='bold')
        ax_pos.set_ylabel('Retention Time (min)', fontsize=11, fontweight='bold')
        ax_pos.set_title('Peak Position Comparison', fontsize=12, fontweight='bold')
        ax_pos.set_xticks(x)
        ax_pos.set_xticklabels([f'{i+1}' for i in range(len(true_params))])
        ax_pos.legend()
        ax_pos.grid(True, alpha=0.3, axis='y')
    else:
        ax_pos.text(0.5, 0.5, 'Deconvolution Failed\n' + result.get('error', ''),
                   ha='center', va='center', fontsize=12, color='red',
                   transform=ax_pos.transAxes)
        ax_pos.set_xticks([])
        ax_pos.set_yticks([])

    # Amplitude comparison plot
    ax_amp = fig.add_subplot(gs[1, 1])
    if result['success']:
        x = np.arange(len(true_amps))
        ax_amp.bar(x - width/2, true_amps, width, label='True',
                  color='green', alpha=0.7)
        ax_amp.bar(x + width/2, pred_amps, width, label='Predicted',
                  color='red', alpha=0.7)
        ax_amp.set_xlabel('Peak Number', fontsize=11, fontweight='bold')
        ax_amp.set_ylabel('Amplitude (normalized)', fontsize=11, fontweight='bold')
        ax_amp.set_title('Peak Amplitude Comparison', fontsize=12, fontweight='bold')
        ax_amp.set_xticks(x)
        ax_amp.set_xticklabels([f'{i+1}' for i in range(len(true_amps))])
        ax_amp.legend()
        ax_amp.grid(True, alpha=0.3, axis='y')
    else:
        ax_amp.text(0.5, 0.5, 'Deconvolution Failed',
                   ha='center', va='center', fontsize=12, color='red',
                   transform=ax_amp.transAxes)
        ax_amp.set_xticks([])
        ax_amp.set_yticks([])

    # Error metrics plot
    ax_error = fig.add_subplot(gs[2, 0])
    if result['success']:
        metrics = ['Position\nError', 'Amplitude\nError', 'R²', 'RMSE']
        values = [
            result['mean_position_error'],
            result['mean_amplitude_error'],
            result['R2'],
            result['RMSE']
        ]
        colors = ['red' if v > 0.05 else 'green' for v in values[:2]] + ['blue', 'orange']

        bars = ax_error.bar(metrics, values, color=colors, alpha=0.7)
        ax_error.set_ylabel('Value', fontsize=11, fontweight='bold')
        ax_error.set_title('Error Metrics', fontsize=12, fontweight='bold')
        ax_error.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax_error.text(bar.get_x() + bar.get_width()/2., height,
                         f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    else:
        ax_error.text(0.5, 0.5, 'Deconvolution Failed',
                     ha='center', va='center', fontsize=12, color='red',
                     transform=ax_error.transAxes)
        ax_error.set_xticks([])
        ax_error.set_yticks([])

    # Residuals plot
    ax_resid = fig.add_subplot(gs[2, 1])
    if result['success'] and Path(fitted_file).exists():
        residuals = fitted_df['original_intensity'] - fitted_df['fitted_intensity']
        ax_resid.plot(fitted_df['position'], residuals, 'b-', linewidth=1, alpha=0.7)
        ax_resid.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax_resid.fill_between(fitted_df['position'], 0, residuals, alpha=0.3)
        ax_resid.set_xlabel('Time (minutes)', fontsize=11, fontweight='bold')
        ax_resid.set_ylabel('Residuals', fontsize=11, fontweight='bold')
        ax_resid.set_title('Fit Residuals', fontsize=12, fontweight='bold')
        ax_resid.grid(True, alpha=0.3)
        ax_resid.set_xlim(0, 20)
    else:
        ax_resid.text(0.5, 0.5, 'No Residuals\n(Deconvolution Failed)',
                     ha='center', va='center', fontsize=12, color='red',
                     transform=ax_resid.transAxes)
        ax_resid.set_xticks([])
        ax_resid.set_yticks([])

    return fig


def create_gallery_plot():
    """
    Create a gallery showing multiple test results
    """
    results_df = pd.read_csv('test_results/all_test_results.csv')

    # Select representative tests
    tests_to_show = [
        (1, 1, 'success'),   # 1 component - successful
        (1, 6, 'success'),   # 1 component - successful but challenging
        (3, 3, 'success'),   # 3 components - successful
        (3, 1, 'failed'),    # 3 components - failed (detected 2)
        (3, 4, 'failed'),    # 3 components - failed (detected 1)
        (6, 1, 'failed'),    # 6 components - failed
    ]

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle('Peak Deconvolution Test Gallery', fontsize=18, fontweight='bold', y=0.995)

    for idx, (n_comp, test_id, expected) in enumerate(tests_to_show):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]

        # Load data
        data_file = f"test_data/test_{n_comp}comp_run{test_id}.csv"
        position, intensity, metadata = load_spectrum_from_csv(
            data_file,
            position_col='time_minutes',
            intensity_col='UV280_mAU',
            normalize=True
        )
        time_minutes = position * 20

        # Get result
        result = results_df[
            (results_df['n_components'] == n_comp) &
            (results_df['test_id'] == test_id)
        ].iloc[0]

        # Plot original
        ax.plot(time_minutes, intensity, 'ko-', alpha=0.5, markersize=2, linewidth=1)

        # Get true peaks (regenerate from seed)
        from peak_generator import generate_random_spectrum
        _, _, params_array = generate_random_spectrum(
            n_components=n_comp,
            seed=test_id * 100 + n_comp,
            num_points=1000
        )
        true_params = params_array[:, 2]  # mu (position)

        # Mark true peaks
        for pos in true_params:
            ax.axvline(x=pos * 20, color='green', linestyle='--', alpha=0.5, linewidth=1.5)

        # Plot fitted if successful
        if result['success']:
            fitted_file = f"test_results/test_{n_comp}comp_run{test_id}_fitted_spectrum.csv"
            if Path(fitted_file).exists():
                fitted_df = pd.read_csv(fitted_file)
                ax.plot(fitted_df['position'], fitted_df['fitted_intensity'],
                       'r-', linewidth=2, label='Fitted')

                # Mark predicted peaks
                pred_params = np.array(eval(result['pred_positions']))
                for pos in pred_params:
                    ax.axvline(x=pos * 20, color='red', linestyle=':', alpha=0.7, linewidth=2)

        # Title
        if result['success']:
            title = f"{n_comp} Component, Test #{test_id}\nSUCCESS - R²={result['R2']:.3f}"
            color = 'green'
        else:
            title = f"{n_comp} Component, Test #{test_id}\nFAILED - {result.get('error', 'Error')[:30]}"
            color = 'red'

        ax.set_title(title, fontsize=10, fontweight='bold', color=color)
        ax.set_xlabel('Time (minutes)', fontsize=9)
        ax.set_ylabel('Intensity', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 20)
        if result['success']:
            ax.legend(fontsize=8)

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    print("Creating detailed test visualizations...")
    print("="*70)

    # Load results
    results_df = pd.read_csv('test_results/all_test_results.csv')

    # Create output directory
    Path('test_results/plots').mkdir(exist_ok=True)

    # Plot individual detailed results
    tests_to_plot = [
        (1, 1),   # 1 component - perfect success
        (1, 6),   # 1 component - lower amplitude (harder)
        (3, 3),   # 3 components - successful
        (3, 8),   # 3 components - successful but harder
        (3, 1),   # 3 components - failed (detected 2)
        (3, 4),   # 3 components - failed (detected 1)
        (6, 1),   # 6 components - failed
        (6, 4),   # 6 components - failed (detected 1)
    ]

    for n_comp, test_id in tests_to_plot:
        print(f"Creating plot for {n_comp} component(s), test #{test_id}...")
        fig = plot_test_result(n_comp, test_id, results_df)
        filename = f"test_results/plots/detail_{n_comp}comp_test{test_id}.png"
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {filename}")

    # Create gallery view
    print("\nCreating gallery plot...")
    fig_gallery = create_gallery_plot()
    fig_gallery.savefig('test_results/plots/gallery.png', dpi=150, bbox_inches='tight')
    plt.close(fig_gallery)
    print("  Saved: test_results/plots/gallery.png")

    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE!")
    print("="*70)
    print(f"Plots saved to: test_results/plots/")
    print(f"  - detail_*.png: Individual detailed plots")
    print(f"  - gallery.png: Overview of multiple tests")
