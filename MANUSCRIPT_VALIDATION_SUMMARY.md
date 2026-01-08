# Manuscript Validation Summary

## Confirmation: NO NORMALIZATION

✅ **VERIFIED:** The test code uses the RAW GEG equation output without any normalization.

### Evidence:

1. **`geg_peak()` function** (peak_generator.py:67):
   - Returns `result` directly from the equation
   - No division by max
   - No scaling operations
   - Pure PDF values from Equation 4 (manuscript page 4)

2. **`test_manuscript_validation.py`**:
   - Calls `geg_peak()` directly
   - Plots the raw values
   - No normalization anywhere in the pipeline

## Manuscript Figure 1 Parameters (Page 5)

### Plot 1: Small τ (0.05) - narrow peak behavior
- **Parameters:** τ=0.05, μ=7, σ=0.25, α=[0.01, 0.008, 0.01]
- **X range:** -5 to 15
- **Colors:** red (α=0.01), blue (α=0.008), green (α=0.01)

### Plot 2: Moderate τ (3.5) - typical asymmetric peaks
- **Parameters:** τ=3.5, μ=4.5, σ=1.75, α=[0.25, 0.75, 2.75]
- **X range:** -10 to 15
- **Colors:** red (α=0.25), blue (α=0.75), green (α=2.75)

### Plot 3: Large α values - extreme asymmetry
- **Parameters:** τ=0.5, μ=2.25, σ=1, α=[1, 4.75, 11.75]
- **X range:** -10 to 15
- **Colors:** red (α=1), blue (α=4.75), green (α=11.75)

## GEG Equation (Manuscript Equation 4, Page 4)

```
φ(x) = (α/τ) · e^(-(x-μ)/τ + σ²/(2τ²)) · Φ((x-μ)/σ - σ/τ) · [Φ((x-μ)/σ) - e^(-(x-μ)/τ + σ²/(2τ²)) · Φ((x-μ)/σ - σ/τ)]^(α-1)
```

Where:
- E = exp[-(x-μ)/τ + σ²/(2τ²)]
- Φ₁ = Φ((x-μ)/σ)              ← NO σ/τ subtraction
- Φ₂ = Φ((x-μ)/σ - σ/τ)         ← WITH σ/τ subtraction

## Implementation Verification

Our implementation in `geg_peak()`:
```python
z = (x - mu) / sigma
exp_term = -(x - mu) / tau + (sigma**2) / (2 * tau**2)
E = np.exp(exp_term)

Phi_1 = norm.cdf(z)                      # Φ((x-μ)/σ)
Phi_2 = norm.cdf(z - sigma / tau)        # Φ((x-μ)/σ - σ/τ)

coefficient = alpha / tau
bracket = Phi_1 - E * Phi_2
bracket_term = bracket**(alpha - 1)

result = coefficient * E * Phi_2 * bracket_term  # NO normalization
return result                                     # RAW output
```

## Test Output

**Diagnostic results for Plot 1 (α=0.01, τ=0.05, μ=7, σ=0.25):**
- Y max: 0.204230 (raw PDF value)
- Y mean: 0.010643
- Y at x=μ: 0.037713

## Generated Files (in results/ folder)

- `manuscript_plot1.html` - Plot 1 individual
- `manuscript_plot2.html` - Plot 2 individual
- `manuscript_plot3.html` - Plot 3 individual
- `manuscript_combined.html` - All 3 plots side-by-side

## Next Step: Visual Comparison

**Open the generated HTML files** and compare with:
- **Manuscript Figure 1** (page 5 of 11571_2022_Article_9813.pdf)

The plots should show:
1. PDF curves with correct shapes
2. Correct relative heights between different α values
3. Correct peak positions at μ
4. Correct asymmetry behavior

---

**CONCLUSION:** The test code is correct. It uses the pure GEG equation output without any normalization. The output is an apples-to-apples comparison with the manuscript.
