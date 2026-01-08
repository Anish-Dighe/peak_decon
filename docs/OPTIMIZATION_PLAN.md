# Optimization Algorithm Design Plan

## Goal
Given a spectrum and the number of components (n_comp), estimate the GEG parameters for each component.

**Input:**
- `y_observed`: Spectrum intensity (101 points, normalized)
- `n_components`: Number of peaks (1-10, user-specified)

**Output:**
- `params_estimated`: Array of shape (n_components, 4) containing [α, τ, μ, σ] for each peak

---

## Parameter Space Scaling

### Problem Dimensions
| n_comp | Parameters | Dimension | Complexity |
|--------|-----------|-----------|------------|
| 1      | 4         | 4D        | Easy       |
| 2      | 8         | 8D        | Moderate   |
| 3      | 12        | 12D       | Moderate   |
| 5      | 20        | 20D       | Hard       |
| 10     | 40        | 40D       | Very Hard  |

**Parameter vector structure:**
```python
# For n_comp = 3:
params = [α₁, τ₁, μ₁, σ₁, α₂, τ₂, μ₂, σ₂, α₃, τ₃, μ₃, σ₃]
# Total: 4 × n_comp parameters
```

### Parameter Bounds
```python
bounds = [
    (0.5, 3.0),   # α₁
    (0.05, 0.3),  # τ₁
    (0.0, 1.0),   # μ₁
    (0.01, 0.4),  # σ₁
    # Repeat for each component...
]
```

---

## Optimization Strategy

### 1. Loss Function: Mean Squared Error (MSE)

```python
def loss_function(params_flat, y_observed, n_components):
    """
    Calculate MSE between observed and predicted spectrum

    Parameters:
    -----------
    params_flat : array, shape (4 * n_components,)
        Flattened parameters [α₁, τ₁, μ₁, σ₁, α₂, ...]
    y_observed : array, shape (101,)
        Observed spectrum (normalized)
    n_components : int
        Number of peaks

    Returns:
    --------
    mse : float
        Mean squared error
    """
    # Reshape params_flat to (n_components, 4)
    params = params_flat.reshape(n_components, 4)

    # Sort by μ (peak position) for identifiability
    params = params[params[:, 2].argsort()]

    # Generate predicted spectrum
    y_predicted = generate_spectrum(params)

    # Calculate MSE
    mse = np.mean((y_observed - y_predicted) ** 2)

    return mse
```

**Why MSE?**
- Differentiable (supports gradient-based optimization)
- Intuitive interpretation
- Standard metric for regression

---

## 2. Optimization Algorithm: Differential Evolution

**Choice: `scipy.optimize.differential_evolution`**

**Why Differential Evolution?**
✅ Global optimization (avoids local minima)
✅ Handles bounded constraints natively
✅ Robust to initial guess
✅ Works well for 4-40 dimensional problems
✅ No gradient required (GEG equation is complex)

**Algorithm Parameters:**
```python
result = differential_evolution(
    func=loss_function,
    bounds=bounds,
    args=(y_observed, n_components),
    strategy='best1bin',          # DE strategy
    maxiter=1000,                  # Maximum iterations
    popsize=15,                    # Population size multiplier
    tol=1e-7,                      # Tolerance for convergence
    mutation=(0.5, 1.5),           # Mutation constant range
    recombination=0.7,             # Crossover probability
    seed=None,                     # For reproducibility
    polish=True,                   # Final local optimization
    workers=1,                     # Parallel workers (or -1 for all CPUs)
    updating='deferred'            # Update strategy
)
```

**Computational Cost:**
- Population size: 15 × (4 × n_comp) = 60 for n_comp=1, 600 for n_comp=10
- Function evaluations: ~15k-100k depending on n_comp
- Time: Seconds for n_comp=1-3, minutes for n_comp=10

---

## 3. Initial Guess Strategy

While Differential Evolution doesn't require initial guess, we can seed the population with smart guesses to speed up convergence.

### Strategy A: Uniform Random Sampling (Default)
```python
# DE samples uniformly within bounds
# No manual initial guess needed
```

### Strategy B: Peak Detection + Intelligent Guessing (Optional Enhancement)
```python
def generate_initial_guess(y_observed, n_components):
    """Generate smart initial guess using peak detection"""
    from scipy.signal import find_peaks

    # Find peaks in observed spectrum
    peak_indices, properties = find_peaks(y_observed,
                                          prominence=0.05,
                                          distance=5)

    # Sort by height, take top n_components
    heights = y_observed[peak_indices]
    top_indices = np.argsort(heights)[-n_components:]
    mu_guesses = X_GRID[peak_indices[top_indices]]

    # Generate guesses for other parameters
    guesses = []
    for mu in sorted(mu_guesses):
        guesses.extend([
            1.5,      # α (moderate shape)
            0.15,     # τ (moderate tail)
            mu,       # μ (from peak detection)
            0.05      # σ (moderate width)
        ])

    return np.array(guesses)
```

---

## 4. Handling Multiple Components

### Parameter Identifiability Issue
**Problem:** Peaks can be in any order, leading to multiple equivalent solutions.

**Solution:** Sort by μ (peak position)
```python
# Always sort params by column 2 (μ) before generating spectrum
params = params[params[:, 2].argsort()]
```

This ensures:
- Peak 1 is always leftmost (smallest μ)
- Peak n is always rightmost (largest μ)
- Unique parameter representation

### Scaling Strategy for Different n_comp

**n_comp = 1 (4 parameters):**
- Fast convergence (~5-10 seconds)
- Simple 4D optimization
- High success rate

**n_comp = 2-3 (8-12 parameters):**
- Moderate convergence (~30-60 seconds)
- Can use strategy B for initial guess
- Good success rate with DE

**n_comp = 4-6 (16-24 parameters):**
- Slower convergence (~2-5 minutes)
- May need increased maxiter
- Consider increasing popsize to 20

**n_comp = 7-10 (28-40 parameters):**
- Slow convergence (~5-15 minutes)
- Increase maxiter=2000, popsize=20
- May require multiple runs
- Consider multi-start strategy

---

## 5. Implementation Structure

```python
# deconvolve.py

import numpy as np
from scipy.optimize import differential_evolution
from peak_generator import X_GRID, geg_peak, generate_spectrum, normalize_by_max


def deconvolve_spectrum(y_observed, n_components, method='differential_evolution',
                       max_iter=1000, verbose=True):
    """
    Deconvolve spectrum into n_components peaks using optimization

    Parameters:
    -----------
    y_observed : array, shape (101,)
        Observed spectrum (will be normalized if not already)
    n_components : int
        Number of peaks (1-10)
    method : str
        Optimization method ('differential_evolution', 'minimize', etc.)
    max_iter : int
        Maximum iterations
    verbose : bool
        Print progress

    Returns:
    --------
    params_estimated : ndarray, shape (n_components, 4)
        Estimated parameters [α, τ, μ, σ] for each peak
    y_fitted : ndarray, shape (101,)
        Fitted spectrum
    result : OptimizeResult
        Full optimization result
    """
    # Normalize observed spectrum
    y_observed = normalize_by_max(y_observed)

    # Set up bounds
    bounds = get_bounds(n_components)

    # Define loss function
    def loss(params_flat):
        return loss_function(params_flat, y_observed, n_components)

    # Run optimization
    if method == 'differential_evolution':
        result = differential_evolution(
            func=loss,
            bounds=bounds,
            maxiter=max_iter,
            popsize=15,
            polish=True,
            workers=-1,  # Use all CPUs
            updating='deferred'
        )
    else:
        raise ValueError(f"Method {method} not implemented")

    # Extract and reshape parameters
    params_estimated = result.x.reshape(n_components, 4)
    params_estimated = params_estimated[params_estimated[:, 2].argsort()]

    # Generate fitted spectrum
    y_fitted = generate_spectrum(params_estimated)

    if verbose:
        print(f"Optimization converged: {result.success}")
        print(f"Final MSE: {result.fun:.6f}")
        print(f"Function evaluations: {result.nfev}")

    return params_estimated, y_fitted, result


def get_bounds(n_components):
    """Generate parameter bounds for n_components"""
    single_bounds = [
        (0.5, 3.0),   # α
        (0.05, 0.3),  # τ
        (0.0, 1.0),   # μ
        (0.01, 0.4)   # σ
    ]
    return single_bounds * n_components


def loss_function(params_flat, y_observed, n_components):
    """Calculate MSE between observed and predicted spectrum"""
    # Reshape to (n_components, 4)
    params = params_flat.reshape(n_components, 4)

    # Sort by μ for identifiability
    params = params[params[:, 2].argsort()]

    # Generate predicted spectrum
    y_predicted = generate_spectrum(params)

    # Calculate MSE
    mse = np.mean((y_observed - y_predicted) ** 2)

    return mse
```

---

## 6. Validation Strategy

### Test Cases

**Test 1: Single Peak Recovery (n_comp=1)**
```python
# Generate ground truth
params_true = np.array([[1.5, 0.1, 0.5, 0.05]])
y_true = generate_spectrum(params_true)

# Deconvolve
params_est, y_fit, result = deconvolve_spectrum(y_true, n_components=1)

# Validate
param_error = np.mean(np.abs(params_true - params_est))
print(f"Parameter MAE: {param_error:.6f}")
```

**Test 2: Multiple Peaks (n_comp=2,3,5)**
- Use generate_complete_spectrum() to create test data
- Compare estimated vs ground truth parameters
- Calculate metrics: parameter MAE, spectrum MSE, R²

**Test 3: Noisy Data**
```python
y_noisy = y_true + np.random.normal(0, 0.01, size=y_true.shape)
# Test robustness to noise
```

### Metrics
- **Parameter MAE**: Mean absolute error in parameter estimates
- **Spectrum MSE**: Mean squared error between fitted and true spectrum
- **R² Score**: Coefficient of determination
- **Success Rate**: Fraction of tests with MSE < 1e-4

---

## 7. Computational Considerations

### Time Complexity
- **Function evaluation**: O(n_comp × 101) - generating spectrum
- **DE iterations**: O(maxiter × popsize × 4 × n_comp)
- **Total**: O(maxiter × popsize × n_comp)

### Expected Runtime (single CPU)
| n_comp | Dimension | Expected Time |
|--------|-----------|---------------|
| 1      | 4D        | 5-10 sec      |
| 2      | 8D        | 15-30 sec     |
| 3      | 12D       | 30-60 sec     |
| 5      | 20D       | 2-5 min       |
| 10     | 40D       | 5-15 min      |

### Optimization Tips
1. Use `workers=-1` for parallel evaluation (speeds up 4-8x)
2. Reduce maxiter for quick tests
3. Increase maxiter for n_comp > 5
4. Consider caching spectrum evaluations

---

## 8. Alternative Approaches (Future Consideration)

### A. Basin Hopping (scipy.optimize.basinhopping)
- Pros: Good for escaping local minima
- Cons: Slower than DE for high dimensions

### B. Multi-start Local Optimization
- Pros: Fast if good initial guess
- Cons: Requires many random starts for reliability

### C. BFGS with Constraints (scipy.optimize.minimize)
- Pros: Very fast convergence with good guess
- Cons: Sensitive to initial conditions, may get stuck

### D. Two-Stage Optimization
1. **Stage 1**: DE for global search (coarse)
2. **Stage 2**: Local refinement with L-BFGS-B (fine)

Already implemented via `polish=True` in DE!

---

## 9. Implementation Checklist

- [ ] Create `deconvolve.py` module
- [ ] Implement `deconvolve_spectrum()` function
- [ ] Implement `loss_function()`
- [ ] Implement `get_bounds()`
- [ ] Create validation script `test_deconvolve.py`
- [ ] Test with n_comp = 1, 2, 3, 5
- [ ] Generate performance benchmarks
- [ ] Create visualization of fitted vs observed spectra
- [ ] Add progress callback (optional)
- [ ] Documentation and examples

---

## 10. Expected Challenges

1. **Local Minima**: Overlapping peaks can cause convergence issues
   - **Solution**: DE is robust to local minima

2. **High Dimensionality**: 40 parameters for n_comp=10
   - **Solution**: Increase maxiter, use parallel workers

3. **Peak Identifiability**: Multiple solutions with reordered peaks
   - **Solution**: Always sort by μ

4. **Computational Cost**: Minutes for large n_comp
   - **Solution**: Use parallel workers, optimize code

5. **Convergence Criteria**: How to know when to stop?
   - **Solution**: Use tolerance on MSE improvement

---

## Summary

**Optimization Approach:**
- Algorithm: Differential Evolution (global optimizer)
- Loss: Mean Squared Error (MSE)
- Parameters: 4 × n_comp (scales linearly)
- Bounds: Enforced via DE
- Identifiability: Sort peaks by μ
- Runtime: Seconds to minutes depending on n_comp

**Next Steps:**
1. Implement `deconvolve.py`
2. Create test suite
3. Validate on synthetic data
4. Benchmark performance
5. Create visualization tools
