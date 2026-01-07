"""
Demo: ML-Ready Data Generation for Peak Deconvolution

This demonstrates the new functions that make it easy to generate
training data for machine learning models.
"""

import numpy as np
import matplotlib.pyplot as plt
from peak_generator import (
    generate_spectrum_from_params,
    generate_random_spectrum,
    generate_training_batch
)


def demo1_predefined_params():
    """Demo 1: Generate spectrum from predefined parameter array"""
    print("=" * 60)
    print("Demo 1: Generate Spectrum from Predefined Parameters")
    print("=" * 60)

    # Method 1: Using list of dictionaries
    params_dict = [
        {'alpha': 1.5, 'tau': 0.08, 'mu': 0.3, 'sigma': 0.04, 'amplitude': 0.8},
        {'alpha': 1.0, 'tau': 0.1, 'mu': 0.5, 'sigma': 0.06, 'amplitude': 1.0},
        {'alpha': 2.0, 'tau': 0.12, 'mu': 0.7, 'sigma': 0.05, 'amplitude': 0.6}
    ]

    x, y, params_array = generate_spectrum_from_params(params_dict)

    print(f"Generated spectrum with {len(params_dict)} components")
    print(f"Spectrum shape: {y.shape}")
    print(f"Parameters array shape: {params_array.shape}")
    print(f"\nGround truth parameters:\n{params_array}")

    # Method 2: Using numpy array directly
    params_array2 = np.array([
        [1.2, 0.09, 0.35, 0.05, 0.7],  # alpha, tau, mu, sigma, amplitude
        [1.8, 0.11, 0.6, 0.07, 0.9]
    ])

    x2, y2, params_return = generate_spectrum_from_params(params_array2)

    # Plot both
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(x, y, 'b-', linewidth=2)
    axes[0].set_title('Spectrum from Dict Parameters (3 components)')
    axes[0].set_xlabel('Normalized Position')
    axes[0].set_ylabel('Intensity')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x2, y2, 'r-', linewidth=2)
    axes[1].set_title('Spectrum from Array Parameters (2 components)')
    axes[1].set_xlabel('Normalized Position')
    axes[1].set_ylabel('Intensity')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('demo_predefined_params.png', dpi=150, bbox_inches='tight')
    print("\nSaved: demo_predefined_params.png\n")

    return fig


def demo2_random_spectra():
    """Demo 2: Generate random spectra for training"""
    print("=" * 60)
    print("Demo 2: Generate Random Spectra")
    print("=" * 60)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for idx, n_comp in enumerate([1, 3, 5, 7, 10, 8]):
        row = idx // 3
        col = idx % 3

        x, y, params = generate_random_spectrum(n_components=n_comp, seed=42 + idx)

        axes[row, col].plot(x, y, 'k-', linewidth=1.5)
        axes[row, col].set_title(f'{n_comp} Component(s)')
        axes[row, col].set_xlabel('Normalized Position')
        axes[row, col].set_ylabel('Intensity')
        axes[row, col].grid(True, alpha=0.3)

        print(f"Generated {n_comp}-component spectrum:")
        print(f"  Parameters shape: {params.shape}")
        print(f"  Peak centers (μ): {params[:, 2]}")
        print()

    plt.tight_layout()
    plt.savefig('demo_random_spectra.png', dpi=150, bbox_inches='tight')
    print("Saved: demo_random_spectra.png\n")

    return fig


def demo3_training_batch():
    """Demo 3: Generate batch of training data"""
    print("=" * 60)
    print("Demo 3: Generate Training Batch")
    print("=" * 60)

    # Generate batch without noise
    x, spectra_clean, params_list = generate_training_batch(
        batch_size=16,
        n_components_range=(2, 8),
        num_points=500,
        add_noise=False
    )

    # Generate batch with noise
    x_noisy, spectra_noisy, params_list_noisy = generate_training_batch(
        batch_size=16,
        n_components_range=(2, 8),
        num_points=500,
        add_noise=True,
        noise_level=0.02
    )

    print(f"Generated training batch:")
    print(f"  Batch size: {len(spectra_clean)}")
    print(f"  Spectrum shape: {spectra_clean[0].shape}")
    print(f"  Number of components per sample: {[len(p) for p in params_list[:5]]}...")

    # Visualize a subset
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for i in range(4):
        # Clean spectra
        axes[0, i].plot(x, spectra_clean[i], 'b-', linewidth=1.5)
        axes[0, i].set_title(f'Clean - {len(params_list[i])} peaks')
        axes[0, i].set_ylabel('Intensity')
        axes[0, i].grid(True, alpha=0.3)

        # Noisy spectra
        axes[1, i].plot(x_noisy, spectra_noisy[i], 'r-', linewidth=1.5, alpha=0.7)
        axes[1, i].set_title(f'Noisy - {len(params_list_noisy[i])} peaks')
        axes[1, i].set_xlabel('Normalized Position')
        axes[1, i].set_ylabel('Intensity')
        axes[1, i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('demo_training_batch.png', dpi=150, bbox_inches='tight')
    print("\nSaved: demo_training_batch.png\n")

    return fig, spectra_clean, params_list


def demo4_ml_workflow_example():
    """Demo 4: Example ML workflow with generated data"""
    print("=" * 60)
    print("Demo 4: Example ML Training Workflow")
    print("=" * 60)

    # Simulate generating a dataset
    print("Generating training dataset...")
    n_samples = 100
    all_spectra = []
    all_params = []

    for i in range(n_samples):
        n_comp = np.random.randint(1, 11)  # 1-10 components
        x, y, params = generate_random_spectrum(
            n_components=n_comp,
            num_points=500,
            seed=i
        )
        all_spectra.append(y)
        all_params.append(params)

    all_spectra = np.array(all_spectra)

    print(f"Dataset shape: {all_spectra.shape}")
    print(f"Number of samples: {len(all_params)}")

    # Analyze the dataset
    n_components_dist = [len(p) for p in all_params]

    # Visualize dataset statistics
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: Distribution of number of components
    axes[0].hist(n_components_dist, bins=range(1, 12), edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Number of Components')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Peak Counts')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Sample spectra from dataset
    for i in range(10):
        axes[1].plot(x, all_spectra[i], alpha=0.5, linewidth=1)
    axes[1].set_xlabel('Normalized Position')
    axes[1].set_ylabel('Intensity')
    axes[1].set_title('Sample Spectra from Dataset')
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Peak positions distribution
    all_mus = []
    for params in all_params:
        all_mus.extend(params[:, 2])  # mu is column 2

    axes[2].hist(all_mus, bins=30, edgecolor='black', alpha=0.7)
    axes[2].set_xlabel('Peak Position (μ)')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Distribution of Peak Centers')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('demo_ml_workflow.png', dpi=150, bbox_inches='tight')
    print("\nSaved: demo_ml_workflow.png")

    print("\nDataset Statistics:")
    print(f"  Mean components per spectrum: {np.mean(n_components_dist):.2f}")
    print(f"  Min/Max components: {min(n_components_dist)}/{max(n_components_dist)}")
    print(f"  Total peaks in dataset: {len(all_mus)}")
    print(f"  Mean peak position: {np.mean(all_mus):.3f}")

    print("\nNext steps for ML:")
    print("  1. Define model architecture (CNN, transformer, etc.)")
    print("  2. Decide on task: regression (predict params) or classification")
    print("  3. Create train/val/test splits")
    print("  4. Train model with this synthetic data")
    print("  5. Validate on real UV280 spectra")

    return fig


if __name__ == '__main__':
    print("\nML-READY PEAK DECONVOLUTION DATA GENERATION")
    print("=" * 60)
    print()

    # Run all demos
    demo1_predefined_params()
    demo2_random_spectra()
    demo3_training_batch()
    demo4_ml_workflow_example()

    print("\n" + "=" * 60)
    print("All demos complete!")
    print("Check the generated PNG files for visualizations.")
    print("=" * 60)
