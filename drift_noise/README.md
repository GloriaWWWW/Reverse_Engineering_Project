# Drift & Noise: Modeling and Simulation

**Author:** Gloria Wu

## Model

The background voltage is modeled as the sum of two components:

`background(t) = drift(t) + noise(t)`

- **Drift:** A slowly varying baseline modeled by a sine function.
- **Noise:** Random background fluctuations modeled by a Gaussian distribution.

### Parameters from Original Data

| Parameter | Value |
|---|---:|
| Drift offset (B) | 0.0032 V |
| Drift amplitude (A) | 0.6475 V |
| Drift frequency (f) | 0.0909 Hz |
| Drift phase (φ) | 0.2826 rad |
| Noise mean (μ) | ≈ 0 V |
| Noise SD (σ) | 0.2990 V |

The estimated parameters are saved in `drift_noise_params.json`.

## 1. Drift Simulation

The drift is generated using a sine function:

`D(t) = B + A sin(2πft + φ)`

![Drift Estimation](figures/drift_fit.png)

The fitted sine wave reproduces the slow baseline variation observed in the original recording.

## 2. Noise Simulation

Background noise is generated using a zero-mean Gaussian distribution:

`N(t) ~ Normal(0, 0.2990²)`

![Noise Distribution](figures/noise_distribution.png)

The estimated noise standard deviation is approximately 0.299 V.

## 3. Simulated Background

The simulated drift and noise are combined to generate the background signal:

`background(t) = D(t) + N(t)`

![Simulated Background](figures/simulation.png)

The simulated background reproduces the slow baseline drift and random voltage fluctuations.

## Files

| File | Description |
|---|---|
| `drift_noise.py` | Functions for drift and noise estimation and generation |
| `run_analysis.py` | Runs the analysis and saves results |
| `drift_noise_params.json` | Estimated parameters |
| `figures/` | Figures for drift, noise, and simulation |

Run from the repository root:

```bash
python -m drift_noise.run_analysis
```

## Use in Full Simulation

```python
import json
from drift_noise.drift_noise import generate

with open("drift_noise/drift_noise_params.json") as f:
    p = json.load(f)

drift, noise = generate(t, p, seed=42)

raw = drift + noise + spike_signal
```

## Limitations

- The drift is approximated by a sine function.
- The noise is assumed to be independent Gaussian noise.
- The model reproduces background signal characteristics but does not include spike generation.
