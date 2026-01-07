# Peak Deconvolution for UV280 Spectra

A machine learning-based approach for peak deconvolution of UV280 spectra using Generalized Exponential-Gaussian (GEG) distribution.

## Overview

This project aims to analyze UV280 spectra containing 1-10 components, including low-signal components that are difficult to analyze experimentally. The approach uses mathematical modeling with the GEG distribution to generate training data and perform deconvolution.

## Installation

The code runs in a conda Python environment. Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

Run the demo to see the effect of various parameters:

```bash
python peak_generator.py
```

This will generate:
- `parameter_effects.png` - Shows how α, τ, μ, and σ parameters affect peak shape
- `multi_component_spectrum.png` - Example of a 3-component UV280 spectrum

## GEG Distribution Parameters

The Generalized Exponential-Gaussian distribution has 4 key parameters:

- **α (alpha)**: Shape parameter controlling skewness and kurtosis
- **τ (tau)**: Exponential component parameter regulating kurtosis
- **μ (mu)**: Location parameter (center position of peak, 0-1 normalized)
- **σ (sigma)**: Scale parameter (width of peak)

## Usage Examples

### Generate a single peak

```python
from peak_generator import GEGPeak
import matplotlib.pyplot as plt

peak = GEGPeak(alpha=1.5, tau=0.1, mu=0.5, sigma=0.05, amplitude=1.0)
x, y = peak.generate_peak()

plt.plot(x, y)
plt.show()
```

### Generate multi-component spectrum

```python
from peak_generator import MultiPeakSpectrum

spectrum = MultiPeakSpectrum()
spectrum.add_peak(alpha=1.5, tau=0.08, mu=0.3, sigma=0.04, amplitude=0.8)
spectrum.add_peak(alpha=1.0, tau=0.1, mu=0.5, sigma=0.06, amplitude=1.0)
spectrum.add_peak(alpha=2.0, tau=0.12, mu=0.65, sigma=0.05, amplitude=0.4)

spectrum.plot_spectrum(show_individual=True)
```

## ML-Ready Functions

The package includes specialized functions for machine learning workflows that return both spectra and ground truth parameters.

### Generate spectrum from predefined parameters

```python
from peak_generator import generate_spectrum_from_params
import numpy as np

# Method 1: List of dictionaries
params = [
    {'alpha': 1.5, 'tau': 0.08, 'mu': 0.3, 'sigma': 0.04, 'amplitude': 0.8},
    {'alpha': 1.0, 'tau': 0.1, 'mu': 0.5, 'sigma': 0.06, 'amplitude': 1.0}
]
x, y, params_array = generate_spectrum_from_params(params)

# Method 2: NumPy array (n_peaks, 5) - [alpha, tau, mu, sigma, amplitude]
params_array = np.array([
    [1.5, 0.08, 0.3, 0.04, 0.8],
    [1.0, 0.1, 0.5, 0.06, 1.0]
])
x, y, params_array = generate_spectrum_from_params(params_array)
```

### Generate random training samples

```python
from peak_generator import generate_random_spectrum

# Generate a random 5-component spectrum
x, y, params = generate_random_spectrum(n_components=5, seed=42)
# Returns: x values, spectrum, and ground truth parameters (5, 5) array
```

### Generate training batch

```python
from peak_generator import generate_training_batch

# Generate 100 training samples with 1-10 components each
x, spectra, params_list = generate_training_batch(
    batch_size=100,
    n_components_range=(1, 10),
    num_points=1000,
    add_noise=True,
    noise_level=0.01
)

# spectra shape: (100, 1000)
# params_list: list of 100 parameter arrays
```

### Run ML-ready demo

```bash
python demo_ml_ready.py
```

This generates:
- `demo_predefined_params.png` - Spectra from predefined parameters
- `demo_random_spectra.png` - Random spectra with varying components
- `demo_training_batch.png` - Training batch with/without noise
- `demo_ml_workflow.png` - Example ML dataset statistics

## Reference

GEG distribution from: https://pmc.ncbi.nlm.nih.gov/articles/PMC9871144/

## Next Steps

- Generate synthetic training data with known peak parameters
- Implement ML-based deconvolution algorithm
- Validate against experimental UV280 spectra
