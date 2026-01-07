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

## Reference

GEG distribution from: https://pmc.ncbi.nlm.nih.gov/articles/PMC9871144/

## Next Steps

- Generate synthetic training data with known peak parameters
- Implement ML-based deconvolution algorithm
- Validate against experimental UV280 spectra
