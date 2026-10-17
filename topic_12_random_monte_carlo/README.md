# Topic 12: Random Numbers and Monte Carlo Methods

## Objective
Generate random numbers and sample from target distributions, use Monte Carlo
simulation to approximate quantities (pi, an integral, a rare-event probability),
study how the error shrinks with the number of simulations, reduce the variance of
estimates, and use simulation to examine the properties of estimators.

## Dataset
No external data is needed. Every sample is generated in `analysis.py` (random seed
42). A 1,000-value sample from each generator is saved as `data/random_samples.csv`.

## How to run
```
python analysis.py
```

## Results

### 1a. Linear congruential generator (LCG)
The LCG uses x(k+1) = (a x(k) + c) mod m, with the uniform value x / m.

| Toy generator (seed 1) | Period |
|------------------------|--------|
| x -> (5x + 3) mod 16 | 16 (full period) |
| x -> (4x + 2) mod 16 | 1 (poor constants) |

Tests on 10,000 values (target for Uniform(0,1): mean 0.5000, variance 0.0833):

| Generator | Mean | Variance | Chi-square p | K-S p | Lag-1 autocorrelation |
|-----------|------|----------|--------------|-------|-----------------------|
| Own LCG | 0.5016 | 0.0838 | 0.6743 | 0.5270 | -0.0037 |
| numpy generator | 0.4971 | 0.0831 | 0.7600 | 0.6343 | 0.0041 |

### 1b to 1d. Sampling from target distributions
| Method | Target | Result | K-S p |
|--------|--------|--------|-------|
| Inverse transform, n = 5000 | Exponential(rate 0.5) | mean 1.9988 (theory 2.0000), variance 4.0018 (theory 4.0000) | 0.7038 |
| Box-Muller, n = 5000 | Standard normal | mean 0.0000, sd 1.0134, skewness 0.0345; correlation of the two halves -0.0098 | 0.4894 |
| Acceptance-rejection | Beta(2, 2) | 13,327 of 20,000 accepted (rate 0.6663, theory 1/M = 0.6667); mean 0.5027 (theory 0.5000), variance 0.0500 (theory 0.0500) | 0.2541 |

### 2. Monte Carlo estimate of pi
| N | Estimate | Error | Standard error |
|---|----------|-------|----------------|
| 100 | 3.08000 | -0.06159 | 0.16833 |
| 1,000 | 3.16800 | +0.02641 | 0.05134 |
| 10,000 | 3.16920 | +0.02761 | 0.01623 |
| 100,000 | 3.14788 | +0.00629 | 0.00518 |
| 1,000,000 | 3.14197 | +0.00038 | 0.00164 |

95% confidence interval at N = 1,000,000: (3.13875, 3.14519), which contains pi
(3.14159).

Error scaling over 200 repeated runs:

| N | Empirical sd | Theoretical sqrt(pi (4 - pi) / N) |
|---|--------------|-----------------------------------|
| 100 | 0.15869 | 0.16422 |
| 1,000 | 0.04925 | 0.05193 |
| 10,000 | 0.01737 | 0.01642 |
| 100,000 | 0.00543 | 0.00519 |

### 3. Monte Carlo integration of exp(-x^2) on [0, 1]
Exact value = (sqrt(pi) / 2) erf(1) = 0.746824.

One run with n = 1000: estimate 0.73677, standard error 0.00650, 95% CI
(0.72403, 0.74951), which contains the exact value.

Variance reduction (5,000 repeated estimates, 1,000 function evaluations each):

| Method | Mean | Bias | Standard deviation | Variance reduction factor |
|--------|------|------|--------------------|---------------------------|
| Simple Monte Carlo | 0.746843 | +1.89e-05 | 0.006203 | 1.0 |
| Antithetic variates | 0.746840 | +1.62e-05 | 0.001244 | 24.9 |
| Control variate | 0.746831 | +7.03e-06 | 0.000787 | 62.1 |

The control variate was g(x) = 1 - x^2 (known mean 2/3), with coefficient 0.6690.

### 4. Importance sampling for a rare event, P(Z > 3)
Exact value = 0.001350. 1,000 repeated estimates with 10,000 draws each.

| Method | Mean | Standard deviation | Relative sd |
|--------|------|--------------------|-------------|
| Naive Monte Carlo | 0.001347 | 0.000361 | 26.7% |
| Importance sampling (proposal N(3, 1)) | 0.001351 | 0.000025 | 1.8% |

Variance reduction factor: 214.

### 5. Properties of estimators by simulation
Bias of the variance estimator (true variance 4, n = 10, 100,000 samples):

| Divisor | Mean estimate | Bias | Theoretical bias |
|---------|---------------|------|------------------|
| n | 3.5992 | -0.4008 | -0.4000 |
| n - 1 | 3.9991 | -0.0009 | 0.0000 |

Mean against median as estimators of the centre (n = 25, 20,000 samples):

| Distribution | Variance of mean | Variance of median |