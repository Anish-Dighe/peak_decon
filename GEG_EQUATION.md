# Generalized Exponential-Gaussian (GEG) Distribution

## Mathematical Form - CORRECTED

The GEG probability density function:

```
φ(x) = (α/τ)·e^(-(x-μ)/τ + σ²/(2τ²)) · Φ((x-μ)/σ - σ/τ) · [Φ((x-μ)/σ) - e^(-(x-μ)/τ + σ²/(2τ²))·Φ((x-μ)/σ - σ/τ)]^(α-1)
```

Where:
- **Φ(z)** = standard normal cumulative distribution function (CDF)
- **e** = Euler's number (exponential function base)

---

## Components Breakdown

### Define intermediate terms:

**z-score:**
```
z = (x - μ) / σ
```

**Exponential term:**
```
E = exp[-(x - μ)/τ + σ²/(2τ²)]
```

**Two different CDF terms:**
```
Φ₁ = Φ(z) = Φ((x - μ)/σ)                    [used inside bracket]

Φ₂ = Φ(z - σ/τ) = Φ((x - μ)/σ - σ/τ)        [used outside bracket and inside bracket]
```

### Rewritten equation:

```
φ(x) = (α/τ) · E · Φ₂ · [Φ₁ - E · Φ₂]^(α-1)
```

Expanded:
```
φ(x) = (α/τ) · E · Φ((x-μ)/σ - σ/τ) · [Φ((x-μ)/σ) - E · Φ((x-μ)/σ - σ/τ)]^(α-1)
```

---

## Key Observations

**Note the TWO different Φ terms:**

1. **Φ₁ = Φ(z) = Φ((x-μ)/σ)**
   - Standard normal CDF of z-score
   - Appears ONLY as first term inside the bracket

2. **Φ₂ = Φ(z - σ/τ) = Φ((x-μ)/σ - σ/τ)**
   - Shifted CDF
   - Appears outside the bracket AND as second term inside bracket

---

## Parameters

| Parameter | Symbol | Description | Role | Range |
|-----------|--------|-------------|------|-------|
| **alpha** | α | Shape parameter | Controls peak skewness and kurtosis | 0.5 - 3.0 |
| **tau** | τ | Exponential parameter | Controls tail length and asymmetry | 0.05 - 0.3 |
| **mu** | μ | Location parameter | Peak center position | 0.0 - 1.0 (normalized) |
| **sigma** | σ | Scale parameter | Peak width (standard deviation) | 0.01 - 0.15 |
| **amplitude** | A | Height scaling | Peak maximum intensity | 0.0 - 2.0 |

---

## Correct Python Implementation

```python
def pdf(self, x):
    # Calculate z-score
    z = (x - self.mu) / self.sigma

    # Exponential term
    exp_term = -(x - self.mu) / self.tau + (self.sigma**2) / (2 * self.tau**2)
    E = np.exp(exp_term)

    # TWO different CDF terms
    Phi_1 = norm.cdf(z)                      # Φ(z) - for inside bracket
    Phi_2 = norm.cdf(z - self.sigma / self.tau)  # Φ(z - σ/τ) - for outside & inside bracket

    # Main equation components
    term1 = (self.alpha / self.tau) * E * Phi_2  # Note: α/τ not α·τ

    # Bracket term: [Φ₁ - E·Φ₂]^(α-1)
    bracket = Phi_1 - E * Phi_2  # Note: Phi_1 (not Phi_2) for first term
    bracket = np.maximum(bracket, 1e-10)  # Avoid division by zero

    if self.alpha != 1.0:
        term2 = bracket**(self.alpha - 1)
    else:
        term2 = 1.0

    # Final result
    result = term1 * term2

    # Normalize and scale by amplitude
    return self.amplitude * result / np.max(result)
```

---

## Step-by-Step Calculation

Given input `x` and parameters (α, τ, μ, σ):

1. Calculate **z-score**: `z = (x - μ) / σ`

2. Calculate **exponential term**: `E = exp[-(x-μ)/τ + σ²/(2τ²)]`

3. Calculate **two CDF values**:
   - `Φ₁ = Φ(z)`
   - `Φ₂ = Φ(z - σ/τ)`

4. Calculate **outside term**: `(α/τ) · E · Φ₂`

5. Calculate **bracket**: `[Φ₁ - E · Φ₂]`

6. Calculate **bracket power**: `bracket^(α-1)`

7. **Multiply**: `φ(x) = (α/τ) · E · Φ₂ · [Φ₁ - E · Φ₂]^(α-1)`

---

## Special Cases

### When α = 1:
The bracket term becomes 1:
```
φ(x) = (1/τ) · E · Φ₂
φ(x) = (1/τ) · exp[-(x-μ)/τ + σ²/(2τ²)] · Φ((x-μ)/σ - σ/τ)
```
This is the **Exponential-Gaussian (EG) distribution**.

---

## What's Different from Original Implementation

**Mistake 1: Coefficient**
- ❌ Original: `α · τ` (alpha times tau)
- ✅ Correct: `α / τ` (alpha divided by tau)

**Mistake 2: CDF inside bracket**
- ❌ Original: `Φ(z - σ/τ) - E · Φ(z - σ/τ)` (same Φ for both terms)
- ✅ Correct: `Φ(z) - E · Φ(z - σ/τ)` (different Φ for first term)

---

## Reference

Original paper: [Generalized Exponential-Gaussian Distribution](https://pmc.ncbi.nlm.nih.gov/articles/PMC9871144/)

---

## Multi-Component Mixture

For a mixture of N components, the total signal is:

```
y_total(x) = Σ φ_i(x)
            i=1 to N

where each φ_i has its own parameters: (α_i, τ_i, μ_i, σ_i, A_i)
```

This creates overlapping peaks that need to be deconvoluted to recover individual components.

---

## IMPORTANT NOTE

**The current code implementation has BOTH mistakes.**

Before using this for training data or deconvolution, the code needs to be corrected to match this equation.
