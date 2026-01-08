# Tests Directory

All test scripts for the peak deconvolution system.

## Running Tests

Tests can be run from the **project root directory**:

```bash
# From project root
python tests/test_clean.py                  # ⭐ RECOMMENDED - clean 2x2 layout
python tests/test_manuscript_validation.py  # Validate against published manuscript
python tests/test_no_normalization.py       # Multiple separate plots
python tests/test_deconvolve.py             # Legacy (with normalization)
python tests/test_detailed_comparison.py    # Legacy (with normalization)
```

## Available Tests

### 1. `test_clean.py` ⭐ **RECOMMENDED**
Tests deconvolution **WITHOUT normalization** with clean 2x2 layout.

**What it does:**
- Tests n_comp = 6, 7, 8, 9, 10
- No normalization in optimization (better parameter accuracy)
- Clean 2x2 layout per test

**Plot layout for each test:**
```
┌─────────────────────┬─────────────────────┐
│ Row 1, Col 1:       │ Row 1, Col 2:       │
│ True Individual     │ True Total          │
│ Peaks (colored)     │ Spectrum (black)    │
├─────────────────────┼─────────────────────┤
│ Row 2, Col 1:       │ Row 2, Col 2:       │
│ Fitted Individual   │ Fitted Total        │
│ Peaks (colored)     │ Spectrum (red)      │
└─────────────────────┴─────────────────────┘
```

**Colors:** Consistent across true and fitted
- Red = Peak 1 (leftmost, smallest μ)
- Blue = Peak 2
- Green = Peak 3, etc.

**Output files** (in `results/` folder):
- `test_6_components.html` - 2x2 plot for 6 components
- `test_7_components.html` - 2x2 plot for 7 components
- `test_8_components.html` - 2x2 plot for 8 components
- `test_9_components.html` - 2x2 plot for 9 components
- `test_10_components.html` - 2x2 plot for 10 components
- `parameter_comparison.html` - True vs estimated parameters with x=y reference line

**Key findings:**
- **Better parameter accuracy** (MAE ~0.12-0.26 vs 0.20-0.34 with normalization)
- R² > 0.999 for most cases
- Easy visual comparison of true vs fitted decompositions

### 2. `test_manuscript_validation.py`
Validate GEG implementation against published manuscript plots.

**What it does:**
- Reproduces 3 plots from published manuscript showing effect of α parameter
- Tests GEG equation with specific parameter combinations from paper
- Uses extended X ranges (beyond 0-1) to match manuscript

**Test cases:**
- **Plot 1:** Small τ (0.05) - narrow peak behavior
  - Parameters: τ=0.05, μ=7, σ=0.25, α=[0.01, 0.008, 0.01]
  - X range: -5 to 15

- **Plot 2:** Moderate τ (3.5) - typical asymmetric peaks
  - Parameters: τ=3.5, μ=4.5, σ=1.75, α=[0.25, 0.75, 2.75]
  - X range: -10 to 15

- **Plot 3:** Large α values - extreme asymmetry
  - Parameters: τ=0.5, μ=2.25, σ=1, α=[1, 4.75, 11.75]
  - X range: -10 to 15

**Output files** (in `results/` folder):
- `manuscript_plot1.html` - Individual plot 1
- `manuscript_plot2.html` - Individual plot 2
- `manuscript_plot3.html` - Individual plot 3
- `manuscript_combined.html` - All 3 plots side-by-side for easy comparison

**Purpose:**
- Verify GEG equation implementation is correct
- Compare visual appearance with published results
- Validate peak shape behavior across parameter ranges

### 3. `test_no_normalization.py`
Tests deconvolution **WITHOUT normalization** (uses absolute intensities).

**What it does:**
- Tests n_comp = 6, 7, 8, 9, 10
- No normalization in optimization
- Better parameter identifiability

**Output files** (in `results/` folder):
- `n{X}_true_total.html` - True total spectrum only
- `n{X}_fitted_total.html` - Fitted total spectrum only
- `n{X}_true_individual.html` - All true individual peaks
- `n{X}_fitted_individual.html` - All fitted individual peaks
- `n{X}_true_sidebyside.html` - Individual peaks | Total (TRUE)
- `n{X}_fitted_sidebyside.html` - Individual peaks | Total (FITTED)

**Key findings:**
- **Better parameter accuracy** (MAE ~0.12-0.26 vs 0.20-0.34 with normalization)
- Absolute intensities constrain the solution space
- R² > 0.999 for most cases

### 4. `test_deconvolve.py`
Basic test suite with normalized data (legacy).

**What it does:**
- Tests n_comp = 1, 2, 3, 5
- Uses normalization (may have identifiability issues)
- Creates parameter comparison plots with x=y reference lines

**Output files:**
- `deconvolution_test_results.html` - Total spectra comparison
- `parameter_comparison.html` - True vs estimated parameters

### 5. `test_detailed_comparison.py`
Detailed verification with normalized data (legacy).

**What it does:**
- Tests n_comp = 6, 7, 8, 9, 10
- Uses normalization
- Creates overlay comparison plots

**Output files:**
- `detailed_total_comparison.html` - Total spectra overlay
- `detailed_individual_comparison.html` - Individual peaks overlay
- `side_by_side_decomposition.html` - True vs Fitted side-by-side

## Results Location

All test results are saved to the **`results/`** folder in the project root, NOT in the tests folder.

```
peak_decon/
├── tests/              ← Test scripts (run from root)
└── results/            ← All output HTML files
```

## Key Insight: Normalization vs No Normalization

**Without Normalization** (`test_no_normalization.py`):
- Loss function: MSE on absolute intensities
- Optimizer must match both **shape** and **scale**
- Better parameter identifiability
- Example: 6 components → **MAE = 0.115** (parameter accuracy)

**With Normalization** (legacy tests):
- Loss function: MSE on normalized intensities [0, 1]
- Optimizer only matches relative **shape**, not scale
- Multiple parameter sets can give same normalized curve
- Example: 6 components → **MAE = 0.344** (worse accuracy)

**Conclusion:** Use unnormalized optimization for better parameter recovery!
