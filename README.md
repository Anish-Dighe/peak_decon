# Peak Deconvolution for Chromatography Data

A tool for generating and analyzing overlapping peaks in chromatography using Generalized Exponential-Gaussian (GEG) distribution.

## Overview

This project generates synthetic chromatography data with multiple overlapping components using the GEG distribution from:
https://pmc.ncbi.nlm.nih.gov/articles/PMC9871144/

## Installation

```bash
pip install -r requirements.txt
```

## GEG Distribution Parameters

Each peak is described by 5 parameters:

- **α (alpha)**: Shape parameter controlling skewness and kurtosis (α > 0)
- **τ (tau)**: Exponential component parameter regulating kurtosis (τ > 0)
- **μ (mu)**: Location parameter (peak center position, 0-1 normalized)
- **σ (sigma)**: Scale parameter (peak width, σ > 0)
- **amplitude**: Peak height

## Usage

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

### Generate training data

```python
from peak_generator import generate_spectrum_from_params, generate_random_spectrum

# From predefined parameters
params = [
    {'alpha': 1.5, 'tau': 0.08, 'mu': 0.3, 'sigma': 0.04, 'amplitude': 0.8},
    {'alpha': 1.0, 'tau': 0.1, 'mu': 0.5, 'sigma': 0.06, 'amplitude': 1.0}
]
x, y, params_array = generate_spectrum_from_params(params)

# Random spectrum with 5 components
x, y, params = generate_random_spectrum(n_components=5, seed=42)
```

### Generate batch training data

```python
from peak_generator import generate_training_batch

# Generate 100 spectra with 1-10 components each
x, spectra, params_list = generate_training_batch(
    batch_size=100,
    n_components_range=(1, 10),
    num_points=1000,
    add_noise=True,
    noise_level=0.01
)

# spectra.shape = (100, 1000)
# params_list = list of parameter arrays
```

## Demo

Run the built-in demo:

```bash
python peak_generator.py
```

This generates:
- `parameter_effects.png` - Shows how α, τ, μ, and σ parameters affect peak shape
- `multi_component_spectrum.png` - Example of a 3-component spectrum

## Next Steps

- Implement deconvolution algorithm
- Train ML model for parameter prediction
- Add CSV input/output for real chromatography data

## Archive

Previous development work (deconvolution attempts, tests, etc.) is archived in `archive_v1/`
