# Peak Deconvolution Project - Overview

## Project Goal

Build a machine learning-based peak deconvolution system for chromatography data (SEC, IEX, etc.) using the Generalized Exponential-Gaussian (GEG) distribution to model overlapping peaks.

## Current Phase: Forward Model (Training Data Generation)

The current implementation focuses on generating synthetic chromatography data with known ground truth parameters to train ML models.

---

## Core Components

### 1. **peak_generator.py** - GEG Peak Generation Engine

**Pure function-based implementation** of the corrected GEG equation:

```
φ(x) = (α/τ) · E · Φ₂ · [Φ₁ - E · Φ₂]^(α-1)
where:
  E = exp[-(x-μ)/τ + σ²/(2τ²)]
  Φ₁ = Φ((x-μ)/σ)
  Φ₂ = Φ((x-μ)/σ - σ/τ)
```

**Key Functions:**
- `geg_peak(x, alpha, tau, mu, sigma)` - Pure GEG equation evaluation
- `generate_random_params(n_components)` - Uniform random parameter sampling
- `generate_spectrum(params)` - Sum multiple peaks into composite spectrum
- `generate_complete_spectrum(n_components)` - Generate spectra with complete peak tails
- `is_spectrum_complete(y_spectrum)` - Filter incomplete peaks (tail < 0.7 threshold)
- `normalize_by_max(y_spectrum)` - Max normalization (all spectra scaled to [0,1])

**Fixed Constants:**
- `X_GRID = np.arange(0, 1.01, 0.01)` - 101 points from 0 to 1

**Parameter Ranges:**
- α (alpha): 0.5 to 3.0 - Shape parameter (controls skewness)
- τ (tau): 0.05 to 0.3 - Exponential parameter (controls tailing)
- μ (mu): 0.0 to 1.0 - Peak center position
- σ (sigma): 0.01 to 0.4 - Peak width (standard deviation)
- Components: 1 to 10 peaks per spectrum

---

### 2. **GEG_EQUATION.md** - Mathematical Reference

Complete documentation of the corrected GEG equation including:
- Full mathematical derivation
- Component breakdown
- Parameter descriptions
- Corrected implementation notes (fixing two critical mistakes from initial version)

---

### 3. **Visualization Scripts**

#### **demo_visualize.py** - Comprehensive Demonstrations
Generates 5 HTML files showing:
1. Parameter effects (α, τ, μ, σ)
2. Multi-component spectra (2, 3, 4, 5 components)
3. Multiple examples for 2-5 components
4. Detailed 3-component example
5. Random training samples (1-10 components)

#### **demo_simple.py** - Focused Examples
Generates 2 HTML files:
1. Three examples each of 2, 6, and 10 components
2. Detailed view with parameter tables

#### **demo_normalizations.py** - Normalization Experiments (Archived)
Initial experiments with area-based normalization methods.
**Note:** Area normalization removed - only max normalization used.

---

## Data Quality Control

### Peak Completeness Filtering
- Spectra are filtered to ensure complete peak tails
- Threshold: Normalized intensity at x=1.0 must be < 0.7
- Prevents training on artificially truncated data
- Resembles real chromatography where incomplete peaks indicate issues

### Normalization Strategy
- **Method:** Max normalization only
- All spectra scaled so max intensity = 1.0
- User will manually normalize both x and y axes of real data before ML inference
- Simple, robust, and avoids issues with area-based methods

---

## File Organization

```
peak_decon/
├── peak_generator.py          # Core GEG generation engine
├── GEG_EQUATION.md            # Mathematical reference
├── demo_simple.py             # Simple 2, 6, 10 component demos
├── demo_visualize.py          # Comprehensive visualizations
├── demo_normalizations.py     # Normalization experiments (archived)
├── README.md                  # User documentation
├── requirements.txt           # Dependencies
├── PROJECT_OVERVIEW.md        # This file
├── results/                   # Generated HTML visualizations
│   ├── component_examples_2_6_10.html
│   ├── detailed_2_6_10.html
│   ├── parameter_effects.html
│   ├── multi_component_spectra.html
│   └── ... (other visualization outputs)
└── archive_v1/                # Previous deconvolution attempts
    └── ... (archived code from earlier iterations)
```

---

## Workflow

### Current: Training Data Generation
1. Define parameter ranges for GEG peaks
2. Generate random parameter sets
3. Calculate composite spectra (sum of individual peaks)
4. Filter for complete peaks (tail < 0.7)
5. Normalize by max value
6. Export ground truth parameters and spectra

### Future: ML Model Development
1. **Model Input:** Normalized composite spectrum (101 points)
2. **Model Output:**
   - Number of components (1-10)
   - Parameters for each component (α, τ, μ, σ)
3. **Training:** Supervised learning with ground truth parameters
4. **Inference:** User provides normalized chromatography data → model predicts components

---

## Key Design Decisions

### Why Pure Functions?
- GEG is a mathematical equation, not an object
- Simpler, more composable code
- Easier to understand and debug

### Why Fixed X Grid?
- Consistent data shape for ML training
- All real data will be resampled to same grid
- Simplifies implementation

### Why Filter Incomplete Peaks?
- Real chromatography data with incomplete peaks is problematic
- Training on complete peaks ensures model learns proper peak shapes
- User will handle incomplete data separately (different analysis)

### Why Max Normalization Only?
- Simple and robust
- No assumptions about underlying distribution
- Area normalization had numerical issues
- User controls x and y normalization manually

---

## Next Steps (Not Yet Implemented)

1. **ML Model Architecture**
   - Peak count classifier (1-10 components)
   - Parameter regression network
   - Uncertainty estimation

2. **Training Pipeline**
   - Generate large training dataset (10k-100k spectra)
   - Train/validation/test split
   - Hyperparameter tuning

3. **Real Data Interface**
   - CSV/Excel import for chromatography data
   - Automatic resampling to X_GRID
   - Batch processing

4. **Deconvolution Validation**
   - Compare predicted vs ground truth on test set
   - Metrics: parameter MAE, peak position accuracy, etc.

---

## References

- GEG Distribution Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC9871144/
- Application: Chromatography peak modeling (SEC, IEX, RP-HPLC)
- Detector: UV280 (fixed wavelength, variable time/volume)

---

## Dependencies

```
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.4.0
plotly>=5.0.0
```

---

## Version History

- **v0.1** - Initial class-based implementation (archived)
- **v0.2** - Corrected GEG equation, pure function approach
- **v0.3** - Added peak completeness filtering
- **v0.4** - Simplified to max normalization only (current)

---

*Last Updated: January 8, 2026*
