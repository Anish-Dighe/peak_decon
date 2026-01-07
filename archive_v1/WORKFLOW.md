# Complete Peak Deconvolution Workflow

## Overview

This package provides a complete pipeline for UV280 peak deconvolution:

**Input**: CSV file with position and intensity columns
**Output**: CSV files with deconvoluted peak parameters and fitted spectrum

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Deconvolution

```bash
# Optimization-only (works immediately, no training needed)
python deconvolve_opt_only.py your_data.csv --output results/output.csv

# Specify number of components (3-5 is typical)
python deconvolve_opt_only.py your_data.csv -n 3 --output results/output.csv

# Auto-detect number of components
python deconvolve_opt_only.py your_data.csv --output results/output.csv
```

### 3. Check Results

Two CSV files are generated:

**`results/output_peaks.csv`** - Peak parameters:
```csv
n_components,peak_id,alpha,tau,position,width,amplitude,relative_intensity,R2,RMSE
3,1,2.33,0.15,277.73,0.98,0.25,0.20,0.96,0.051
3,2,3.00,0.05,281.03,1.73,0.79,0.63,0.96,0.051
3,3,1.49,0.12,286.30,1.81,0.20,0.16,0.96,0.051
```

**`results/output_fitted_spectrum.csv`** - Fitted spectrum:
```csv
position,fitted_intensity,original_intensity,residual
270.0,0.001,0.003,0.002
270.5,0.002,0.004,0.002
...
```

## Input CSV Format

Your CSV must have two columns for position and intensity:

```csv
wavelength,absorbance
270.0,0.123
270.5,0.145
271.0,0.167
...
```

Column names can be customized:
```bash
python deconvolve_opt_only.py data.csv --position nm --intensity intensity
```

## Two Approaches Available

### Approach 1: Optimization Only (Current - Works Immediately)

**File**: `deconvolve_opt_only.py`

✅ **Pros**:
- Works immediately, no training required
- Good for small batches
- Interpretable results
- Auto-detects peaks

❌ **Cons**:
- Slower for large batches
- May get stuck in local minima
- Requires good initial guess

**Usage**:
```bash
python deconvolve_opt_only.py input.csv -n 4 --output results/output.csv
```

### Approach 2: Hybrid (NN + Optimization) - Future

**File**: `deconvolve.py` (requires trained model)

✅ **Pros**:
- Fast inference after training
- Better for large batches
- More robust to noise
- Learns from data

❌ **Cons**:
- Requires training data generation
- Needs GPU for efficient training
- More complex setup

**Training** (coming soon):
```bash
# Generate training data
python generate_training_data.py --samples 10000 --output data/training.npz

# Train model
python train.py --data data/training.npz --epochs 100 --output models/model.pth

# Use trained model
python deconvolve.py input.csv --model models/model.pth --output results/output.csv
```

## Parameter Interpretation

Each peak has 5 parameters from the Generalized Exponential-Gaussian (GEG) distribution:

| Parameter | Symbol | Range | Meaning |
|-----------|--------|-------|---------|
| **alpha** | α | 0.5-3.0 | Shape (skewness/kurtosis) |
| **tau** | τ | 0.05-0.3 | Exponential component |
| **position** | μ | data range | Peak center location |
| **width** | σ | 0.01-0.15 | Peak width |
| **amplitude** | A | 0.0-2.0 | Peak height |

**Additional outputs**:
- **relative_intensity**: Fraction of total signal (sums to 1.0)
- **R²**: Goodness of fit (0-1, higher is better)
- **RMSE**: Root mean squared error (lower is better)
- **NRMSE**: Normalized RMSE (lower is better)

## Batch Processing

Process multiple files at once:

```bash
# Create a directory with your CSV files
mkdir my_data/
cp spectrum1.csv spectrum2.csv spectrum3.csv my_data/

# Process all files
for file in my_data/*.csv; do
    python deconvolve_opt_only.py "$file" --output "results/$(basename $file .csv)_out.csv"
done
```

## Validation and Quality Control

### Check Goodness of Fit

- **R² > 0.95**: Excellent fit
- **R² = 0.90-0.95**: Good fit
- **R² < 0.90**: Poor fit (try different n_components)

### Visual Inspection

Plot results to verify:
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
spectrum = pd.read_csv('results/output_fitted_spectrum.csv')

# Plot
plt.figure(figsize=(10, 6))
plt.plot(spectrum['position'], spectrum['original_intensity'], 'ko-', label='Original')
plt.plot(spectrum['position'], spectrum['fitted_intensity'], 'r-', label='Fitted')
plt.plot(spectrum['position'], spectrum['residual'], 'b-', alpha=0.5, label='Residual')
plt.legend()
plt.show()
```

## Troubleshooting

### Problem: Poor fit (low R²)

**Solutions**:
1. Try different number of components: `--n-components 2`, `--n-components 5`, etc.
2. Check if data is noisy - smooth before deconvolution
3. Verify correct column names with `--position` and `--intensity`

### Problem: Too many/few peaks detected

**Solutions**:
1. Explicitly set `--n-components N`
2. Adjust peak detection threshold (edit `prominence` in code)
3. Pre-process data to remove baseline

### Problem: Optimization is slow

**Solutions**:
1. Reduce number of components
2. Use `--method L-BFGS-B` instead of `differential_evolution`
3. Wait for hybrid model training (faster inference)

## Example Workflow

```bash
# 1. Create sample data for testing
python data_utils.py

# 2. Run deconvolution on sample
python deconvolve_opt_only.py sample_spectrum.csv \
    --output results/sample_out.csv \
    --n-components 4

# 3. Check results
cat results/sample_out_peaks.csv

# 4. Run on your real data
python deconvolve_opt_only.py my_uv280_data.csv \
    --position wavelength \
    --intensity absorbance \
    --output results/my_results.csv
```

## Next Steps

1. ✅ **Current**: Use optimization-only deconvolution
2. 🚧 **In Progress**: Train hybrid model on synthetic data
3. 📋 **Planned**:
   - Add baseline correction
   - Implement peak merging for close peaks
   - Add uncertainty quantification
   - Web interface for easier use

## Citation

If you use this code, please cite the GEG distribution paper:
https://pmc.ncbi.nlm.nih.gov/articles/PMC9871144/
