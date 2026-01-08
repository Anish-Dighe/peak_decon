# Recent Changes Summary

## Key Changes

### 1. ✅ Removed Normalization from Optimization

**Files changed:**
- `deconvolve.py` - Removed all `normalize_by_max()` calls

**Impact:**
- Parameter accuracy improved by **30-66%**
- Example (6 components): MAE dropped from **0.344 → 0.115**
- Absolute intensities constrain solution space → better identifiability

**Before (with normalization):**
```python
y_observed = normalize_by_max(y_observed)  # Loses scale info
y_predicted = normalize_by_max(y_predicted)
```

**After (no normalization):**
```python
# Work directly with absolute intensities
loss = MSE(y_true, y_pred)  # Preserves scale info
```

### 2. ✅ Reorganized Project Structure

```
peak_decon/
├── tests/              ← All test scripts (run from root)
│   ├── test_clean.py   ← ⭐ RECOMMENDED
│   ├── test_no_normalization.py
│   ├── test_deconvolve.py
│   ├── test_detailed_comparison.py
│   └── README.md
│
├── results/            ← All HTML output files
│   ├── test_6_components.html
│   ├── test_7_components.html
│   ├── ...
│   └── parameter_comparison.html
│
├── deconvolve.py      ← Core optimization (no normalization)
├── peak_generator.py  ← GEG peak generation
├── docs/
│   ├── NORMALIZATION_COMPARISON.md  ← Detailed analysis
│   └── ...
```

### 3. ✅ Created Clean Visualization (test_clean.py)

**2x2 Layout per test:**
```
┌─────────────────────┬─────────────────────┐
│ True Individual     │ True Total          │
│ Peaks (colored)     │ Spectrum (black)    │
├─────────────────────┼─────────────────────┤
│ Fitted Individual   │ Fitted Total        │
│ Peaks (colored)     │ Spectrum (red)      │
└─────────────────────┴─────────────────────┘
```

**Features:**
- Consistent colors across true and fitted
- Peak 1 (leftmost) = red, Peak 2 = blue, etc.
- All peaks sorted by μ position
- Metrics shown on fitted total plot

### 4. ✅ Parameter Comparison Plots

- True vs estimated for each parameter (α, τ, μ, σ)
- Gray dashed line shows x=y (perfect fit)
- Equal x and y axes for easy visual comparison
- MAE shown for each parameter

---

## Results Summary (NO Normalization)

```
n_comp   MSE          R²         MAE        Time(s)
------------------------------------------------------------
6        0.00000713   0.999998   0.115283   64.12    ← Excellent!
7        0.00000410   0.999999   0.267748   95.79
8        0.02599219   0.998760   0.185205   115.79
9        0.00008653   0.999994   0.241793   141.05
10       0.06770319   0.992509   0.257990   175.74
```

## How to Use

### Run Tests (from project root)
```bash
python tests/test_clean.py              # ⭐ Recommended
python tests/test_no_normalization.py   # Separate plots
```

### View Results
Open HTML files in `results/` folder:
- `test_6_components.html` - Full 2x2 comparison
- `test_7_components.html` - Full 2x2 comparison
- ... (one per n_comp)
- `parameter_comparison.html` - Parameter accuracy plots

### Check Documentation
- `docs/NORMALIZATION_COMPARISON.md` - Why no normalization is better
- `tests/README.md` - Test descriptions and usage

---

## Key Insight

**Normalization was causing parameter identifiability issues!**

When optimizing on normalized [0,1] curves:
- Multiple different parameter sets can produce the same normalized shape
- Scale information is lost
- Optimizer accepts wrong parameters that happen to match after normalization

When optimizing on absolute intensities:
- Optimizer must match both shape AND scale
- Solution space is more constrained
- Better parameter recovery (MAE reduced by 30-66%)

**Example (6 components):**
- With normalization: MAE = 0.344, but R² = 0.9997
- Without normalization: MAE = 0.115, and R² = 0.999998

Both have excellent spectrum fits, but without normalization the **parameters are much more accurate**!

---

## Recommendation

✅ **Always use unnormalized optimization** for peak deconvolution

This is critical for:
- Training data generation (need accurate ground truth parameters)
- Scientific analysis (need to recover true peak parameters)
- ML model validation (need reliable reference values)

The updated `deconvolve.py` now works without normalization by default.
