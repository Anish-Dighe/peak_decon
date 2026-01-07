# Generalized Exponential-Gaussian (GEG) Distribution

## Mathematical Form

The GEG probability density function implemented in this code:

```
φ(x) = α·τ·e^(-(x-μ)/τ + σ²/(2τ²)) · Φ((x-μ)/σ - σ/τ) · [Φ((x-μ)/σ - σ/τ) - e^(-(x-μ)/τ + σ²/(2τ²))·Φ((x-μ)/σ - σ/τ)]^(α-1)
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

**CDF argument:**
```
Φ_arg = Φ(z - σ/τ) = Φ((x - μ)/σ - σ/τ)
```

### Rewritten equation:

```
φ(x) = α · τ · E · Φ_arg · [Φ_arg - E · Φ_arg]^(α-1)
```

Or equivalently:

```
φ(x) = α · τ · E · Φ_arg · [Φ_arg · (1 - E)]^(α-1)
```

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

## Parameter Effects

### α (alpha) - Shape Parameter
- **α = 1.0**: Exponential-Gaussian (standard case)
- **α < 1.0**: More skewed, sharper peak
- **α > 1.0**: Less skewed, broader distribution
- Controls both **skewness** and **kurtosis**

### τ (tau) - Exponential Component
- **Small τ (0.05)**: Short tail, more symmetric
- **Large τ (0.3)**: Long tail, more asymmetric
- Controls the **tailing** behavior common in chromatography
- Represents adsorption/desorption effects in separation

### μ (mu) - Location
- **μ = 0.5**: Peak centered at midpoint
- Determines **retention time** or **elution position**
- Range: 0 to 1 (normalized coordinates)
- In real chromatography: maps to actual time/volume

### σ (sigma) - Scale/Width
- **Small σ (0.02)**: Narrow, sharp peak
- **Large σ (0.12)**: Wide, broad peak
- Represents **band broadening** in chromatography
- Related to column efficiency

---

## Python Implementation

```python
def pdf(self, x):
    # Calculate z-score
    z = (x - self.mu) / self.sigma

    # Exponential term
    exp_term = -(x - self.mu) / self.tau + (self.sigma**2) / (2 * self.tau**2)
    exp_val = np.exp(exp_term)

    # CDF argument
    phi_arg = z - self.sigma / self.tau
    Phi = norm.cdf(phi_arg)  # Standard normal CDF

    # Main equation components
    term1 = self.alpha * self.tau * exp_val * Phi

    # Bracket term: [Φ(...) - e^(...)·Φ(...)]^(α-1)
    bracket = Phi - exp_val * Phi  # = Φ(1 - E)
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

## Special Cases

### When α = 1:
The equation simplifies (the bracket term becomes 1):
```
φ(x) = τ · E · Φ_arg
```
This is the **Exponential-Gaussian (EG) distribution**.

### When τ → 0:
As tau approaches zero, the exponential component vanishes and the distribution approaches a **Gaussian**.

### When τ → ∞:
As tau approaches infinity, the distribution becomes more exponential-like with a strong tail.

---

## Normalization

**Important**: The implemented function applies two normalizations:

1. **Peak normalization**: Divides by `max(result)` to normalize peak height to 1.0
2. **Amplitude scaling**: Multiplies by the `amplitude` parameter

Final output:
```
output(x) = amplitude · [φ(x) / max(φ(x))]
```

This ensures all peaks have comparable heights regardless of parameter combinations.

---

## Reference

Original paper: [Generalized Exponential-Gaussian Distribution](https://pmc.ncbi.nlm.nih.gov/articles/PMC9871144/)

The GEG distribution is particularly useful for modeling:
- **Chromatography peaks** with tailing
- **Asymmetric peaks** in separation science
- **Real-world peak shapes** that deviate from ideal Gaussian behavior

---

## Multi-Component Mixture

For a mixture of N components, the total signal is:

```
y_total(x) = Σ φ_i(x)
            i=1 to N

where each φ_i has its own parameters: (α_i, τ_i, μ_i, σ_i, A_i)
```

This creates overlapping peaks that need to be deconvoluted to recover individual components.
